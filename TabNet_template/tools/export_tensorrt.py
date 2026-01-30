#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict

import onnx


def get_data_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
    """graph.input 중 initializer(상수)로 쓰이는 것 빼고 '진짜' 입력만 남기기."""
    graph = model.graph
    init_names = {init.name for init in graph.initializer}
    return [vi for vi in graph.input if vi.name not in init_names]


def get_input_shapes(model_path: Path) -> Dict[str, List[int]]:
    """
    ONNX 모델에서 입력 텐서 이름과 shape를 추출.
    - batch dim(0번 축)은 동적일 수 있다고 보고, 나머지 축은 정수여야 한다고 가정.
    """
    model = onnx.load(str(model_path))
    data_inputs = get_data_inputs(model)

    if not data_inputs:
        raise RuntimeError(f"{model_path}: data input 을 찾지 못했습니다.")

    shapes: Dict[str, List[int]] = {}

    for vi in data_inputs:
        tensor_type = vi.type.tensor_type
        dims = []
        for d in tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                # dim_param이나 0/음수인 경우는 동적이라고 보고 -1로 표시
                dims.append(-1)
        shapes[vi.name] = dims

    return shapes


def build_shape_arg(
    name: str,
    dims: List[int],
    batch_size: int,
) -> str:
    """
    trtexec용 shape 문자열 생성: name:BxD1xD2...
    - dims[0]은 batch라고 보고 무시하고 batch_size로 대체
    - 나머지 dims는 양의 정수여야 함
    """
    if not dims:
        raise ValueError(f"Input {name} has empty shape")

    print(f"[debug] Input {name} dims: {dims}")
    # 0번째 축은 batch
    spatial = dims[1:]
    # for d in spatial:
    #     if d <= 0:
    #         raise ValueError(
    #             f"Input {name} has non-positive/unknown non-batch dim {d}. "
    #             f"수동으로 --minShapes/--maxShapes를 지정해야 할 수도 있습니다."
    #         )

    all_dims = [batch_size] + spatial
    return f"{name}:" + "x".join(str(d) for d in all_dims)


def run_trtexec_for_model(
    onnx_path: Path,
    trtexec: str,
    fp16: bool,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    extra_args: List[str],
) -> None:
    shapes = get_input_shapes(onnx_path)
    if not shapes:
        raise RuntimeError(f"{onnx_path}: 입력 텐서를 찾을 수 없습니다.")

    # 여러 입력이 있을 경우, 모든 입력에 동일한 batch 크기를 사용
    min_shapes = ",".join(
        build_shape_arg(name, dims, min_batch) for name, dims in shapes.items()
    )
    opt_shapes = ",".join(
        build_shape_arg(name, dims, opt_batch) for name, dims in shapes.items()
    )
    max_shapes = ",".join(
        build_shape_arg(name, dims, max_batch) for name, dims in shapes.items()
    )

    engine_path = onnx_path.with_suffix(".plan")
    log_path = onnx_path.with_suffix(".trtexec.log")

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={max_shapes}",
        "--verbose",
    ]

    if fp16:
        cmd.append("--fp16")

    cmd.extend(extra_args)

    print("=" * 80)
    print(f"[info] Building TensorRT engine for {onnx_path.name}")
    print(f"       Engine : {engine_path.name}")
    print(f"       minShapes: {min_shapes}")
    print(f"       optShapes: {opt_shapes}")
    print(f"       maxShapes: {max_shapes}")
    print(f"[info] Running: {' '.join(cmd)}")
    print(f"[info] Log will be saved to: {log_path}")
    print("=" * 80)

    with log_path.open("w") as log_f:
        proc = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"trtexec failed for {onnx_path.name} (exit code {proc.returncode}). "
            f"로그를 확인하세요: {log_path}"
        )


def main():
    ap = argparse.ArgumentParser(
        description="현재 디렉토리의 ONNX 모델들을 trtexec으로 자동 변환 (입력 shape 자동 감지)"
    )
    ap.add_argument('--onnx-dir', type=str, default='.', help='ONNX 모델이 있는 디렉토리 (기본: 현재 디렉토리)')
    ap.add_argument(
        "--trtexec",
        default="trtexec",
        help="trtexec 실행 파일 경로 (기본: PATH에서 찾음)",
    )
    ap.add_argument(
        "--no-fp16",
        action="store_true",
        help="FP16 비활성화 (기본은 --fp16 켬)",
    )
    ap.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="minShapes에 사용할 batch size (기본: 1)",
    )
    ap.add_argument(
        "--opt-batch",
        type=int,
        default=8192,
        help="optShapes에 사용할 batch size (기본: 8192)",
    )
    ap.add_argument(
        "--max-batch",
        type=int,
        default=8192*2,
        help="maxShapes에 사용할 batch size (기본: 8192*2)",
    )
    ap.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="trtexec에 전달할 추가 플래그 (예: --extra-arg=--int8, --extra-arg=--calib=foo.cache)",
    )
    args = ap.parse_args()

    onnx_files = sorted(Path(args.onnx_dir).glob("*.onnx"))
    if not onnx_files:
        print("[warn] 현재 디렉토리에 .onnx 파일이 없습니다.")
        return

    fp16 = not args.no_fp16

    for onnx_path in onnx_files:
        try:
            run_trtexec_for_model(
                onnx_path=onnx_path,
                trtexec=args.trtexec,
                fp16=fp16,
                min_batch=args.min_batch,
                opt_batch=args.opt_batch,
                max_batch=args.max_batch,
                extra_args=args.extra_arg,
            )
        except Exception as e:
            print(f"[error] {onnx_path.name}: {e}")


if __name__ == "__main__":
    main()