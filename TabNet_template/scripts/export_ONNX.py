#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from input_preprocessing import (
    build_tabnet_export_wrapper,
    preprocess_metadata_props,
)
from tabnet_compat import (
    load_tabnet_model,
    suppress_tabnet_onnx_warnings,
    verify_tabnet_onnx_export,
)

DEFAULT_MODEL_FOLDER = Path(
    "/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/TabNET_model/2024_Cat_embed_2"
)


@dataclass
class ModelInfo:
    path: Path
    model: TabNetClassifier | None
    fold: int
    preprocess_info: dict | None = None
    converted_onnx_model: onnx.ModelProto | None = None
    quantized_path: Path | None = None


def _load_fold_info(fold_path: Path) -> dict:
    info_path = fold_path / "info.npy"
    if not info_path.exists():
        return {}
    return np.load(str(info_path), allow_pickle=True)[()]


def _set_model_from_fold(model_info: ModelInfo) -> None:
    zips = list(model_info.path.glob("*.zip"))
    assert len(zips) == 1, f"Expected exactly one zip file in {model_info.path}, found {zips}"
    model = TabNetClassifier()
    load_tabnet_model(model, str(zips[0]))
    model_info.model = model


def _find_fold(model_folder: Path) -> List[ModelInfo]:
    model_infos: List[ModelInfo] = []
    for file in sorted(model_folder.iterdir()):
        if file.name.startswith("fold") and file.is_dir():
            fold_info = _load_fold_info(file)
            m = ModelInfo(
                path=file,
                model=None,
                fold=int(file.name.replace("fold", "")),
                preprocess_info=fold_info.get("preprocess_info"),
            )
            _set_model_from_fold(m)
            model_infos.append(m)
    print(f"Discovered {len(model_infos)} folds: {[mi.fold for mi in model_infos]}")
    return model_infos


def _set_onnx_preprocess_metadata(
    model: onnx.ModelProto, preprocess_info: dict | None
) -> onnx.ModelProto:
    props = {prop.key: prop.value for prop in model.metadata_props}
    props.update(preprocess_metadata_props(preprocess_info))
    onnx.helper.set_model_props(model, props)
    return model


def export_tabnet_to_onnx(
    mi: ModelInfo,
    out_dir: Path,
    opset: int = 13,
    verify: bool = True,
    verify_batch_size: int = 64,
    verify_seed: int = 0,
    verify_atol: float = 2e-3,
    verify_rtol: float = 1e-4,
) -> Path:
    assert mi.model is not None

    clf = mi.model
    input_dim = getattr(clf, "input_dim", getattr(clf.network, "input_dim"))
    device = getattr(clf, "device", "cpu")
    torch_model = build_tabnet_export_wrapper(
        clf.network,
        mi.preprocess_info,
        input_dim=input_dim,
    ).to(device)
    torch_model.eval()
    dummy_input = torch.ones((1, input_dim), dtype=torch.float32, device=device)

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"tabnet_fold{mi.fold}.onnx"

    with suppress_tabnet_onnx_warnings():
        torch.onnx.export(
            torch_model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
        )

    mi.converted_onnx_model = onnx.load(str(onnx_path))
    _set_onnx_preprocess_metadata(mi.converted_onnx_model, mi.preprocess_info)
    onnx.save(mi.converted_onnx_model, str(onnx_path))
    onnx.checker.check_model(mi.converted_onnx_model)

    if verify:
        stats = verify_tabnet_onnx_export(
            clf,
            onnx_path,
            data_root=mi.path.parent,
            fold_idx=mi.fold,
            batch_size=verify_batch_size,
            seed=verify_seed,
            atol=verify_atol,
            rtol=verify_rtol,
            reference_model=torch_model,
        )
        print(
            f"[verify fold {mi.fold}] "
            f"max_abs_diff={stats['max_abs_diff']:.3e}, "
            f"mean_abs_diff={stats['mean_abs_diff']:.3e}, "
            f"p99_abs_diff={stats['p99_abs_diff']:.3e}, "
            f"batch={stats['batch_size']}, seed={stats['seed']}, "
            f"source={stats['sample_source']}"
        )

    print(f"[fold {mi.fold}] Exported ONNX -> {onnx_path}")
    return onnx_path


def quantize_onnx(input_path: Path, preprocess_info: dict | None = None) -> Path:
    print(f"[QNT] Quantizing {input_path.name}")

    out_path = input_path.with_suffix(".int8.onnx")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )

    quantized_model = onnx.load(str(out_path))
    _set_onnx_preprocess_metadata(quantized_model, preprocess_info)
    onnx.save(quantized_model, str(out_path))

    print(f"[QNT] Saved INT8 model -> {out_path}")
    return out_path


def export_all_folds_to_onnx(
    model_folder: Path,
    out_dir: Path,
    opset: int = 13,
    do_quant: bool = True,
    verify: bool = True,
    verify_batch_size: int = 64,
    verify_seed: int = 0,
    verify_atol: float = 2e-3,
    verify_rtol: float = 1e-4,
) -> Tuple[List[ModelInfo], List[Tuple[Path, Path | None]]]:
    infos = _find_fold(model_folder)

    paths: List[Tuple[Path, Path | None]] = []
    for mi in infos:
        original = export_tabnet_to_onnx(
            mi,
            out_dir,
            opset=opset,
            verify=verify,
            verify_batch_size=verify_batch_size,
            verify_seed=verify_seed,
            verify_atol=verify_atol,
            verify_rtol=verify_rtol,
        )
        quantized: Path | None = None
        if do_quant:
            quantized = quantize_onnx(original, preprocess_info=mi.preprocess_info)
            mi.quantized_path = quantized
        paths.append((original, quantized))

    return infos, paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TabNet fold 디렉토리에서 ONNX(+INT8) 모델들을 일괄 export 하는 CLI"
    )
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=DEFAULT_MODEL_FOLDER,
        help=f"foldN 서브디렉토리들을 포함한 루트 모델 폴더 (기본: {DEFAULT_MODEL_FOLDER})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="ONNX/INT8 출력 디렉토리 (기본: <model-folder>/onnx)",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="INT8 dynamic quantization 수행하지 않음 (기본은 수행)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (기본: 13)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="소스 TabNet과 exported ONNX의 logits 비교 검증을 생략",
    )
    parser.add_argument(
        "--verify-batch-size",
        type=int,
        default=64,
        help="무결성 검증에 사용할 배치 크기 (기본: 64)",
    )
    parser.add_argument(
        "--verify-seed",
        type=int,
        default=0,
        help="무결성 검증 입력 생성용 시드 (기본: 0)",
    )
    parser.add_argument(
        "--verify-atol",
        type=float,
        default=2e-3,
        help="무결성 검증 절대 오차 허용치 (기본: 2e-3)",
    )
    parser.add_argument(
        "--verify-rtol",
        type=float,
        default=1e-4,
        help="무결성 검증 상대 오차 허용치 (기본: 1e-4)",
    )

    args = parser.parse_args()

    model_folder: Path = args.model_folder.resolve()
    if args.out_dir is None:
        out_dir: Path = (model_folder / "onnx").resolve()
    else:
        out_dir: Path = args.out_dir.resolve()

    print(f"[cfg] model_folder = {model_folder}")
    print(f"[cfg] out_dir      = {out_dir}")
    print(f"[cfg] opset        = {args.opset}")
    print(f"[cfg] quantization = {not args.no_quant}")
    print(f"[cfg] verification = {not args.no_verify}")
    print(f"[cfg] verify_batch = {args.verify_batch_size}")
    print(f"[cfg] verify_seed  = {args.verify_seed}")
    print(f"[cfg] verify_atol  = {args.verify_atol}")
    print(f"[cfg] verify_rtol  = {args.verify_rtol}")

    infos, paths = export_all_folds_to_onnx(
        model_folder=model_folder,
        out_dir=out_dir,
        opset=args.opset,
        do_quant=not args.no_quant,
        verify=not args.no_verify,
        verify_batch_size=args.verify_batch_size,
        verify_seed=args.verify_seed,
        verify_atol=args.verify_atol,
        verify_rtol=args.verify_rtol,
    )

    print("\n=== Summary ===")
    for mi, (onnx_path, q_path) in zip(infos, paths):
        print(f"fold {mi.fold}:")
        print(f"  ONNX : {onnx_path}")
        if q_path is not None:
            print(f"  INT8 : {q_path}")
        else:
            print("  INT8 : (not generated)")


if __name__ == "__main__":
    main()
