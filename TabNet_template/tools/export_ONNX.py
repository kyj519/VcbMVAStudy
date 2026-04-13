#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

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


@dataclass
class FoldModel:
    fold: int
    fold_dir: Path
    zip_path: Path
    preprocess_info: dict | None = None


def _parse_fold_index(dirname: str) -> int:
    match = re.fullmatch(r"fold(\d+)", dirname)
    if not match:
        raise ValueError(
            f"Invalid fold directory name: {dirname} (expected format: foldN)"
        )
    return int(match.group(1))


def _resolve_model_zip(fold_dir: Path, zip_name: str) -> Path:
    explicit = fold_dir / zip_name
    if explicit.exists():
        return explicit

    zip_candidates = sorted(fold_dir.glob("*.zip"))
    if len(zip_candidates) == 1:
        return zip_candidates[0]
    if len(zip_candidates) == 0:
        raise FileNotFoundError(
            f"No zip model found in {fold_dir}. Expected {zip_name} or exactly one *.zip."
        )
    raise RuntimeError(
        f"Multiple zip files found in {fold_dir}: {zip_candidates}. "
        f"Specify --zip-name explicitly."
    )


def _load_fold_info(fold_dir: Path) -> dict:
    info_path = fold_dir / "info.npy"
    if not info_path.exists():
        return {}
    return np.load(str(info_path), allow_pickle=True)[()]


def discover_folds(model_root: Path, zip_name: str) -> List[FoldModel]:
    fold_dirs = [
        p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("fold")
    ]
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found under {model_root}")

    found: List[FoldModel] = []
    for fold_dir in sorted(fold_dirs):
        fold_idx = _parse_fold_index(fold_dir.name)
        zip_path = _resolve_model_zip(fold_dir, zip_name)
        fold_info = _load_fold_info(fold_dir)
        found.append(
            FoldModel(
                fold=fold_idx,
                fold_dir=fold_dir,
                zip_path=zip_path,
                preprocess_info=fold_info.get("preprocess_info"),
            )
        )

    return sorted(found, key=lambda item: item.fold)


def _input_dim_from_model(clf: object) -> int:
    dim = getattr(clf, "input_dim", None)
    if dim is None:
        dim = getattr(clf.network, "input_dim", None)
    if dim is None:
        raise RuntimeError("Could not resolve input_dim from TabNetClassifier.")
    return int(dim)


def _set_onnx_preprocess_metadata(model, preprocess_info: dict | None):
    import onnx

    props = {prop.key: prop.value for prop in model.metadata_props}
    props.update(preprocess_metadata_props(preprocess_info))
    onnx.helper.set_model_props(model, props)
    return model


def export_fold_to_onnx(
    fold_model: FoldModel,
    out_dir: Path,
    opset: int,
    overwrite: bool,
    verify: bool,
    verify_batch_size: int,
    verify_seed: int,
    verify_atol: float,
    verify_rtol: float,
) -> Path:
    try:
        import onnx
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError as exc:
        raise RuntimeError(
            "Dependencies are missing. Install torch, pytorch-tabnet, and onnx."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"tabnet_fold{fold_model.fold}.onnx"
    if onnx_path.exists() and not overwrite:
        print(f"[skip] {onnx_path} already exists (use --overwrite to replace)")
        return onnx_path

    clf = TabNetClassifier()
    load_tabnet_model(clf, str(fold_model.zip_path))

    network = clf.network
    network.eval()

    device = getattr(clf, "device", None)
    if device is None:
        try:
            device = next(network.parameters()).device
        except StopIteration:
            device = "cpu"

    input_dim = _input_dim_from_model(clf)
    export_model = build_tabnet_export_wrapper(
        network,
        fold_model.preprocess_info,
        input_dim=input_dim,
    ).to(device)
    export_model.eval()
    dummy_input = torch.zeros((1, input_dim), dtype=torch.float32, device=device)

    with suppress_tabnet_onnx_warnings():
        torch.onnx.export(
            export_model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
        )

    onnx_model = onnx.load(str(onnx_path))
    _set_onnx_preprocess_metadata(onnx_model, fold_model.preprocess_info)
    onnx.save(onnx_model, str(onnx_path))
    onnx.checker.check_model(onnx_model)

    if verify:
        stats = verify_tabnet_onnx_export(
            clf,
            onnx_path,
            data_root=fold_model.fold_dir.parent,
            fold_idx=fold_model.fold,
            batch_size=verify_batch_size,
            seed=verify_seed,
            atol=verify_atol,
            rtol=verify_rtol,
            reference_model=export_model,
        )
        print(
            f"[verify] fold{fold_model.fold}: "
            f"max_abs_diff={stats['max_abs_diff']:.3e}, "
            f"mean_abs_diff={stats['mean_abs_diff']:.3e}, "
            f"p99_abs_diff={stats['p99_abs_diff']:.3e}, "
            f"batch={stats['batch_size']}, seed={stats['seed']}, "
            f"source={stats['sample_source']}"
        )

    print(f"[ok] fold{fold_model.fold}: {fold_model.zip_path} -> {onnx_path}")
    return onnx_path


def quantize_int8(
    onnx_path: Path,
    overwrite: bool,
    preprocess_info: dict | None = None,
) -> Path:
    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for INT8 quantization. "
            "Install onnxruntime or run without --quantize-int8."
        ) from exc

    out_path = onnx_path.with_suffix(".int8.onnx")
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} already exists (use --overwrite to replace)")
        return out_path

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )
    quantized_model = onnx.load(str(out_path))
    _set_onnx_preprocess_metadata(quantized_model, preprocess_info)
    onnx.save(quantized_model, str(out_path))
    print(f"[ok] int8: {onnx_path.name} -> {out_path.name}")
    return out_path


def run_export(
    model_root: Path,
    out_dir: Path,
    zip_name: str,
    opset: int,
    quantize: bool,
    overwrite: bool,
    dry_run: bool,
    verify: bool,
    verify_batch_size: int,
    verify_seed: int,
    verify_atol: float,
    verify_rtol: float,
) -> List[Tuple[Path, Path | None]]:
    folds = discover_folds(model_root, zip_name=zip_name)
    print(f"[info] found folds: {[fold.fold for fold in folds]}")

    if dry_run:
        for fold in folds:
            print(f"[dry-run] fold{fold.fold}: {fold.zip_path}")
        return []

    produced: List[Tuple[Path, Path | None]] = []
    for fold in folds:
        onnx_path = export_fold_to_onnx(
            fold_model=fold,
            out_dir=out_dir,
            opset=opset,
            overwrite=overwrite,
            verify=verify,
            verify_batch_size=verify_batch_size,
            verify_seed=verify_seed,
            verify_atol=verify_atol,
            verify_rtol=verify_rtol,
        )
        int8_path = (
            quantize_int8(
                onnx_path,
                overwrite=overwrite,
                preprocess_info=fold.preprocess_info,
            )
            if quantize
            else None
        )
        produced.append((onnx_path, int8_path))
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export TabNet fold models (fold*/model.zip) to ONNX."
    )
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path.cwd(),
        help="Model root containing fold directories (default: current directory)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for ONNX files (default: <model-folder>/onnx)",
    )
    parser.add_argument(
        "--zip-name",
        default="model.zip",
        help="Zip filename inside each fold directory (default: model.zip)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13)",
    )
    parser.add_argument(
        "--quantize-int8",
        action="store_true",
        help="Also generate dynamic INT8 ONNX files (*.int8.onnx)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ONNX/INT8 files if present",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered folds and zip paths",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX integrity verification against the source TabNet model",
    )
    parser.add_argument(
        "--verify-batch-size",
        type=int,
        default=64,
        help="Batch size used for ONNX integrity verification (default: 64)",
    )
    parser.add_argument(
        "--verify-seed",
        type=int,
        default=0,
        help="Random seed used to generate verification inputs (default: 0)",
    )
    parser.add_argument(
        "--verify-atol",
        type=float,
        default=2e-3,
        help="Absolute tolerance for ONNX integrity verification (default: 2e-3)",
    )
    parser.add_argument(
        "--verify-rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for ONNX integrity verification (default: 1e-4)",
    )
    args = parser.parse_args()

    model_root = args.model_folder.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else (model_root / "onnx")

    print(f"[cfg] model_folder = {model_root}")
    print(f"[cfg] out_dir      = {out_dir}")
    print(f"[cfg] zip_name     = {args.zip_name}")
    print(f"[cfg] opset        = {args.opset}")
    print(f"[cfg] quantize     = {args.quantize_int8}")
    print(f"[cfg] overwrite    = {args.overwrite}")
    print(f"[cfg] dry_run      = {args.dry_run}")
    print(f"[cfg] verify       = {not args.no_verify}")
    print(f"[cfg] verify_batch = {args.verify_batch_size}")
    print(f"[cfg] verify_seed  = {args.verify_seed}")
    print(f"[cfg] verify_atol  = {args.verify_atol}")
    print(f"[cfg] verify_rtol  = {args.verify_rtol}")

    produced = run_export(
        model_root=model_root,
        out_dir=out_dir,
        zip_name=args.zip_name,
        opset=args.opset,
        quantize=args.quantize_int8,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        verify=not args.no_verify,
        verify_batch_size=args.verify_batch_size,
        verify_seed=args.verify_seed,
        verify_atol=args.verify_atol,
        verify_rtol=args.verify_rtol,
    )

    if args.dry_run:
        return

    print("\n=== Summary ===")
    for onnx_path, int8_path in produced:
        print(f"ONNX : {onnx_path}")
        if int8_path is not None:
            print(f"INT8 : {int8_path}")


if __name__ == "__main__":
    main()
