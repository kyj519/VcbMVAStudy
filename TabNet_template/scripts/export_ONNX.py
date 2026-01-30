#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np  # 현재는 안 쓰지만 나중에 써도 되게 남겨둠
import onnx
import onnxruntime as ort  # quantization backend
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

# 기본값 (CLI에서 override 가능)
DEFAULT_MODEL_FOLDER = Path(
    "/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/TabNET_model/2024_Cat_embed_2"
)


@dataclass
class ModelInfo:
    path: Path
    model: TabNetClassifier | None
    fold: int
    converted_onnx_model: onnx.ModelProto | None = None
    quantized_path: Path | None = None


def _set_model_from_fold(model_info: ModelInfo) -> None:
    """폴더 안의 zip 하나를 TabNetClassifier로 로드"""
    zips = list(model_info.path.glob("*.zip"))
    assert len(zips) == 1, f"Expected exactly one zip file in {model_info.path}, found {zips}"
    model = TabNetClassifier()
    model.load_model(str(zips[0]))
    model_info.model = model


def _find_fold(model_folder: Path) -> List[ModelInfo]:
    """foldN 디렉토리들을 찾아서 ModelInfo 리스트 생성"""
    model_infos: List[ModelInfo] = []
    for file in sorted(model_folder.iterdir()):
        if file.name.startswith("fold") and file.is_dir():
            m = ModelInfo(
                path=file,
                model=None,
                fold=int(file.name.replace("fold", "")),
            )
            _set_model_from_fold(m)
            model_infos.append(m)
    print(f"Discovered {len(model_infos)} folds: {[mi.fold for mi in model_infos]}")
    return model_infos


def export_tabnet_to_onnx(mi: ModelInfo, out_dir: Path, opset: int = 13) -> Path:
    """단일 fold TabNet을 ONNX로 export"""
    assert mi.model is not None

    clf = mi.model
    torch_model = clf.network
    torch_model.eval()

    input_dim = getattr(clf, "input_dim", getattr(torch_model, "input_dim"))
    device = getattr(clf, "device", "cpu")
    dummy_input = torch.ones((1, input_dim), dtype=torch.float32, device=device)

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"tabnet_fold{mi.fold}.onnx"

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
    onnx.checker.check_model(mi.converted_onnx_model)

    print(f"[fold {mi.fold}] Exported ONNX → {onnx_path}")
    return onnx_path


def quantize_onnx(input_path: Path) -> Path:
    """INT8 dynamic quantization (weights only)"""
    print(f"[QNT] Quantizing {input_path.name}")

    out_path = input_path.with_suffix(".int8.onnx")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )

    print(f"[QNT] Saved INT8 model → {out_path}")
    return out_path


def export_all_folds_to_onnx(
    model_folder: Path,
    out_dir: Path,
    opset: int = 13,
    do_quant: bool = True,
) -> Tuple[List[ModelInfo], List[Tuple[Path, Path | None]]]:
    infos = _find_fold(model_folder)

    paths: List[Tuple[Path, Path | None]] = []
    for mi in infos:
        original = export_tabnet_to_onnx(mi, out_dir, opset=opset)
        quantized: Path | None = None
        if do_quant:
            quantized = quantize_onnx(original)
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

    infos, paths = export_all_folds_to_onnx(
        model_folder=model_folder,
        out_dir=out_dir,
        opset=args.opset,
        do_quant=not args.no_quant,
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