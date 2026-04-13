#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from input_preprocessing import (
    apply_feature_preprocess,
    build_tabnet_export_wrapper,
    preprocess_has_effect,
)
from tabnet_compat import (
    build_tabnet_verification_input,
    load_tabnet_model,
    load_tabnet_verification_input_from_dataset,
)

DEFAULT_MODEL_FOLDER = Path(
    "/gv0/Users/yeonjoon/TabNET_model/2024_QuadJet_pseudocont_newWP_v2_rescale"
)

RAW_COLOR = "#1f77b4"
PROC_COLOR = "#ff7f0e"
ZIP_COLOR = "#1f77b4"
ONNX_COLOR = "#d62728"
DIFF_COLOR = "#2ca02c"


@dataclass
class FoldPaths:
    fold: int
    fold_dir: Path
    zip_path: Path
    onnx_path: Path
    preprocess_info: dict[str, Any] | None
    feature_names: list[str]


def _load_info(path: Path) -> dict[str, Any]:
    info_path = path / "info.npy"
    if not info_path.exists():
        return {}
    return np.load(str(info_path), allow_pickle=True)[()]


def _resolve_onnx_path(model_root: Path, fold: int, prefer_int8: bool = False) -> Path:
    base = f"tabnet_fold{fold}"
    onnx_dir = model_root / "onnx"
    candidates: list[Path] = []
    if prefer_int8:
        candidates.extend(
            [
                onnx_dir / f"{base}.int8.onnx",
                onnx_dir / f"{base}_int8.onnx",
                model_root / f"{base}.int8.onnx",
            ]
        )
    candidates.extend([onnx_dir / f"{base}.onnx", model_root / f"{base}.onnx"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"ONNX model for fold {fold} not found. Looked for: {', '.join(str(x) for x in candidates)}"
    )


def _resolve_feature_names(
    info: dict[str, Any],
    preprocess_info: dict[str, Any] | None,
    input_dim: int,
) -> list[str]:
    for names in (
        (preprocess_info or {}).get("feature_names"),
        info.get("data_info", {}).get("varlist"),
    ):
        if names and len(names) == input_dim:
            return [str(name) for name in names]
    return [f"feature_{idx}" for idx in range(input_dim)]


def _discover_folds(
    model_root: Path,
    folds: Sequence[int] | None,
    *,
    zip_name: str,
    prefer_int8: bool,
) -> list[FoldPaths]:
    requested = set(int(x) for x in folds) if folds else None
    found: list[FoldPaths] = []

    for fold_dir in sorted(model_root.iterdir()):
        if not fold_dir.is_dir() or not fold_dir.name.startswith("fold"):
            continue
        fold_idx = int(fold_dir.name.replace("fold", ""))
        if requested is not None and fold_idx not in requested:
            continue
        zip_path = fold_dir / zip_name
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing {zip_name} in {fold_dir}")
        info = _load_info(fold_dir)
        preprocess_info = info.get("preprocess_info")
        input_dim = len((preprocess_info or {}).get("mean", [])) or len(
            (preprocess_info or {}).get("feature_names", [])
        )
        feature_names = _resolve_feature_names(info, preprocess_info, input_dim)
        found.append(
            FoldPaths(
                fold=fold_idx,
                fold_dir=fold_dir,
                zip_path=zip_path,
                onnx_path=_resolve_onnx_path(model_root, fold_idx, prefer_int8=prefer_int8),
                preprocess_info=preprocess_info,
                feature_names=feature_names,
            )
        )

    if requested is not None:
        missing = sorted(requested - {item.fold for item in found})
        if missing:
            raise FileNotFoundError(f"Requested folds not found: {missing}")

    if not found:
        raise FileNotFoundError(f"No fold directories found in {model_root}")

    return found


def _load_model(zip_path: Path, *, device_name: str) -> TabNetClassifier:
    model = TabNetClassifier(device_name=device_name)
    load_tabnet_model(model, str(zip_path))
    return model


def _sample_inputs(
    model_root: Path,
    fold_idx: int,
    model,
    *,
    sample_size: int,
    seed: int,
    sample_source: str,
) -> tuple[np.ndarray, str]:
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")

    if sample_source in {"auto", "dataset"}:
        try:
            sample = load_tabnet_verification_input_from_dataset(
                model_root,
                fold_idx=fold_idx,
                batch_size=sample_size,
                seed=seed,
            )
            return sample, "dataset"
        except FileNotFoundError:
            if sample_source == "dataset":
                raise

    if sample_source == "dataset":
        raise RuntimeError(f"Could not load dataset samples from {model_root}")

    return (
        build_tabnet_verification_input(model, batch_size=sample_size, seed=seed),
        "synthetic",
    )


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shift = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shift)
    denom = np.sum(exp, axis=1, keepdims=True)
    return np.asarray(exp / denom, dtype=np.float32)


def _run_zip_reference(
    model: TabNetClassifier,
    raw_input: np.ndarray,
    preprocess_info: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    input_dim = getattr(model, "input_dim", getattr(model.network, "input_dim"))
    device = getattr(model, "device", "cpu")
    wrapped = build_tabnet_export_wrapper(
        model.network,
        preprocess_info,
        input_dim=input_dim,
    ).to(device)
    wrapped.eval()

    raw_tensor = torch.from_numpy(np.asarray(raw_input, dtype=np.float32)).to(device)
    with torch.no_grad():
        logits = wrapped(raw_tensor)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    processed = apply_feature_preprocess(raw_input, preprocess_info)
    return np.asarray(logits.detach().cpu().numpy(), dtype=np.float32), processed


def _embedded_preprocess_from_metadata(model_meta) -> bool:
    custom = getattr(model_meta, "custom_metadata_map", None) or {}
    flag = str(custom.get("tabnet_preprocess_embedded", "0")).strip().lower()
    return flag in {"1", "true", "yes"}


def _run_onnx(
    onnx_path: Path,
    raw_input: np.ndarray,
    preprocess_info: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for ONNX comparison.") from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1:
        raise RuntimeError(f"Expected one ONNX input in {onnx_path}")
    input_name = session.get_inputs()[0].name
    embedded_preprocess = _embedded_preprocess_from_metadata(session.get_modelmeta())
    effective_input = (
        np.asarray(raw_input, dtype=np.float32, order="C")
        if embedded_preprocess
        else apply_feature_preprocess(raw_input, preprocess_info)
    )
    logits = session.run(None, {input_name: effective_input})[0]
    return (
        np.asarray(logits, dtype=np.float32),
        np.asarray(effective_input, dtype=np.float32, order="C"),
        embedded_preprocess,
    )


def _setup_style(style_name: str) -> None:
    plt.style.use("default")
    style_obj = getattr(hep.style, style_name, hep.style.CMS)
    hep.style.use(style_obj)


def _maybe_add_hep_label(ax, *, label: str) -> None:
    try:
        hep.cms.label(label=label, data=False, ax=ax)
    except Exception:
        pass


def _flatten_axes(axes) -> list[Any]:
    if isinstance(axes, np.ndarray):
        return list(axes.ravel())
    return [axes]


def _finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _make_bins(*arrays: np.ndarray, bins: int = 50) -> np.ndarray:
    parts = [_finite_values(arr) for arr in arrays]
    parts = [part for part in parts if part.size]
    if not parts:
        return np.linspace(-0.5, 0.5, 11)
    merged = np.concatenate(parts)
    lo = float(np.min(merged))
    hi = float(np.max(merged))
    if lo == hi:
        width = 1.0 if lo == 0.0 else max(abs(lo) * 0.1, 1e-3)
        return np.linspace(lo - width, hi + width, bins + 1)
    return np.linspace(lo, hi, bins + 1)


def _hist_density(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = _finite_values(values)
    if finite.size == 0:
        return np.zeros(len(bins) - 1, dtype=np.float64), bins
    counts, edges = np.histogram(finite, bins=bins, density=True)
    return np.asarray(counts, dtype=np.float64), edges


def _draw_hist(
    ax,
    counts: np.ndarray,
    edges: np.ndarray,
    *,
    label: str,
    color: str,
) -> None:
    hep.histplot(
        counts,
        edges,
        histtype="step",
        ax=ax,
        label=label,
        color=color,
        linewidth=1.6,
    )


def _compute_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    ratio = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=ratio, where=denominator > 0)
    return ratio


def _ratio_ylim(ratio: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(ratio[np.isfinite(ratio)], dtype=np.float64)
    if finite.size == 0:
        return 0.5, 1.5
    q_lo, q_hi = np.quantile(finite, [0.05, 0.95])
    lo = min(float(q_lo), 1.0)
    hi = max(float(q_hi), 1.0)
    span = max(hi - lo, 0.2)
    pad = max(0.12 * span, 0.08)
    return max(0.0, lo - pad), hi + pad


def _save_figure(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay_ratio_pages(
    lhs: np.ndarray,
    rhs: np.ndarray,
    names: Sequence[str],
    out_dir: Path,
    *,
    stem: str,
    lhs_label: str,
    rhs_label: str,
    lhs_color: str,
    rhs_color: str,
    plot_formats: Sequence[str],
    items_per_page: int,
    ncols: int,
) -> list[Path]:
    outputs: list[Path] = []
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape != rhs.shape or lhs.shape[1] == 0:
        return outputs

    ratio_label = f"{rhs_label}/{lhs_label}"
    n_items = lhs.shape[1]
    for start in range(0, n_items, items_per_page):
        end = min(n_items, start + items_per_page)
        subset = list(range(start, end))
        nrows = math.ceil(len(subset) / ncols)
        height_ratios: list[float] = []
        for _ in range(nrows):
            height_ratios.extend([3.0, 1.0])

        fig = plt.figure(figsize=(ncols * 4.3, nrows * 4.0), dpi=150)
        gs = fig.add_gridspec(
            nrows * 2,
            ncols,
            height_ratios=height_ratios,
            hspace=0.15,  # 수정됨: 서브플롯 위아래 간격 확보 (0.08 -> 0.15)
            wspace=0.35,  # 수정됨: 서브플롯 좌우 간격 확보 (0.28 -> 0.35)
        )

        for local_idx, item_idx in enumerate(subset):
            row = local_idx // ncols
            col = local_idx % ncols
            ax = fig.add_subplot(gs[2 * row, col])
            rax = fig.add_subplot(gs[2 * row + 1, col], sharex=ax)

            bins = _make_bins(lhs[:, item_idx], rhs[:, item_idx])
            lhs_counts, edges = _hist_density(lhs[:, item_idx], bins)
            rhs_counts, _ = _hist_density(rhs[:, item_idx], bins)
            ratio = _compute_ratio(rhs_counts, lhs_counts)

            _draw_hist(ax, lhs_counts, edges, label=lhs_label, color=lhs_color)
            _draw_hist(ax, rhs_counts, edges, label=rhs_label, color=rhs_color)
            hep.histplot(
                ratio,
                edges,
                histtype="step",
                ax=rax,
                color=rhs_color,
                linewidth=1.4,
            )
            rax.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.7)

            # 수정됨: 타이틀, Y축 라벨, 틱(tick) 폰트 크기 축소
            ax.set_title(str(names[item_idx]), fontsize=11)
            ax.set_ylabel('Density', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.grid(alpha=0.18)
            
            # 수정됨: 첫 번째 행(맨 위쪽 플롯들)에만 CMS 라벨을 붙여 공간 확보
            if row == 0:
                _maybe_add_hep_label(ax, label='Private work')
                
            if local_idx == 0:
                ax.legend(frameon=False, fontsize=9)

            # 수정됨: Ratio Y축/X축 라벨 및 틱 폰트 크기 축소
            rax.set_ylabel('Ratio', fontsize=10)
            rax.set_xlabel('Value', fontsize=10)
            rax.tick_params(axis='both', which='major', labelsize=9)
            rax.set_ylim(*_ratio_ylim(ratio))
            rax.grid(alpha=0.18)
            plt.setp(ax.get_xticklabels(), visible=False)
            if local_idx == 0:
                rax.text(
                    0.98,
                    0.86,
                    ratio_label,
                    transform=rax.transAxes,
                    ha='right',
                    va='top',
                    fontsize=7,  # 수정됨: 범례 텍스트 겹침 방지 (8 -> 7)
                    color=rhs_color,
                )

        for local_idx in range(len(subset), nrows * ncols):
            row = local_idx // ncols
            col = local_idx % ncols
            ax = fig.add_subplot(gs[2 * row, col])
            rax = fig.add_subplot(gs[2 * row + 1, col])
            ax.axis('off')
            rax.axis('off')

        page = start // items_per_page + 1
        for fmt in plot_formats:
            out_path = out_dir / f"{stem}_page{page:02d}.{fmt}"
            _save_figure(fig, out_path)
            outputs.append(out_path)

    return outputs

def _plot_abs_diff(
    diff: np.ndarray,
    out_dir: Path,
    *,
    stem: str,
    plot_formats: Sequence[str],
) -> list[Path]:
    outputs: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=150)
    bins_all = _make_bins(diff.ravel())
    counts_all, edges_all = _hist_density(diff.ravel(), bins_all)
    _draw_hist(axes[0], counts_all, edges_all, label="|zip-onnx|", color=DIFF_COLOR)
    axes[0].set_title("All classes")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.18)
    _maybe_add_hep_label(axes[0], label="Private work")
    axes[0].legend(frameon=False, fontsize=9)

    max_per_row = np.max(diff, axis=1)
    bins_row = _make_bins(max_per_row)
    counts_row, edges_row = _hist_density(max_per_row, bins_row)
    _draw_hist(
        axes[1],
        counts_row,
        edges_row,
        label="max |zip-onnx| per event",
        color=DIFF_COLOR,
    )
    axes[1].set_title("Per-event max diff")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.18)
    _maybe_add_hep_label(axes[1], label="Private work")
    axes[1].legend(frameon=False, fontsize=9)

    for fmt in plot_formats:
        out_path = out_dir / f"{stem}.{fmt}"
        _save_figure(fig, out_path)
        outputs.append(out_path)

    return outputs


def _write_stats_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_summary(
    *,
    fold: int,
    sample_size: int,
    sample_source: str,
    embedded_preprocess: bool,
    raw_input: np.ndarray,
    preprocessed_input: np.ndarray,
    torch_logits: np.ndarray,
    onnx_logits: np.ndarray,
) -> dict[str, Any]:
    diff = np.abs(torch_logits - onnx_logits)
    torch_prob = _softmax_np(torch_logits)
    onnx_prob = _softmax_np(onnx_logits)
    prob_diff = np.abs(torch_prob - onnx_prob)

    per_class: list[dict[str, Any]] = []
    for cls_idx in range(torch_logits.shape[1]):
        class_diff = diff[:, cls_idx]
        class_prob_diff = prob_diff[:, cls_idx]
        per_class.append(
            {
                "class_index": cls_idx,
                "logit_max_abs_diff": float(np.max(class_diff)),
                "logit_mean_abs_diff": float(np.mean(class_diff)),
                "logit_p99_abs_diff": float(np.quantile(class_diff, 0.99)),
                "prob_max_abs_diff": float(np.max(class_prob_diff)),
                "prob_mean_abs_diff": float(np.mean(class_prob_diff)),
                "prob_p99_abs_diff": float(np.quantile(class_prob_diff, 0.99)),
            }
        )

    return {
        "fold": fold,
        "sample_size": int(sample_size),
        "sample_source": sample_source,
        "embedded_preprocess": bool(embedded_preprocess),
        "input_dim": int(raw_input.shape[1]),
        "output_dim": int(torch_logits.shape[1]),
        "preprocess_effective": bool(
            np.any(np.abs(preprocessed_input - raw_input) > 0.0)
        ),
        "global": {
            "logit_max_abs_diff": float(np.max(diff)),
            "logit_mean_abs_diff": float(np.mean(diff)),
            "logit_p99_abs_diff": float(np.quantile(diff, 0.99)),
            "prob_max_abs_diff": float(np.max(prob_diff)),
            "prob_mean_abs_diff": float(np.mean(prob_diff)),
            "prob_p99_abs_diff": float(np.quantile(prob_diff, 0.99)),
        },
        "per_class": per_class,
    }


def _compare_fold(
    fold_paths: FoldPaths,
    model_root: Path,
    out_dir: Path,
    *,
    device_name: str,
    sample_size: int,
    seed: int,
    sample_source: str,
    plot_formats: Sequence[str],
    features_per_page: int,
    classes_per_page: int,
    ncols: int,
    save_arrays: bool,
) -> dict[str, Any]:
    print(f"[fold {fold_paths.fold}] loading {fold_paths.zip_path.name} and {fold_paths.onnx_path.name}")
    model = _load_model(fold_paths.zip_path, device_name=device_name)
    raw_input, actual_source = _sample_inputs(
        model_root,
        fold_paths.fold,
        model,
        sample_size=sample_size,
        seed=seed,
        sample_source=sample_source,
    )

    torch_logits, preprocessed_input = _run_zip_reference(
        model,
        raw_input,
        fold_paths.preprocess_info,
    )
    onnx_logits, onnx_input, embedded_preprocess = _run_onnx(
        fold_paths.onnx_path,
        raw_input,
        fold_paths.preprocess_info,
    )

    if torch_logits.shape != onnx_logits.shape:
        raise RuntimeError(
            f"Shape mismatch in fold {fold_paths.fold}: zip={torch_logits.shape}, onnx={onnx_logits.shape}"
        )

    diff = np.abs(torch_logits - onnx_logits)
    torch_probs = _softmax_np(torch_logits)
    onnx_probs = _softmax_np(onnx_logits)
    fold_out_dir = out_dir / f"fold{fold_paths.fold}"
    fold_out_dir.mkdir(parents=True, exist_ok=True)

    summary = _collect_summary(
        fold=fold_paths.fold,
        sample_size=sample_size,
        sample_source=actual_source,
        embedded_preprocess=embedded_preprocess,
        raw_input=raw_input,
        preprocessed_input=preprocessed_input,
        torch_logits=torch_logits,
        onnx_logits=onnx_logits,
    )
    summary["zip_path"] = str(fold_paths.zip_path)
    summary["onnx_path"] = str(fold_paths.onnx_path)
    summary["onnx_effective_input"] = "raw" if embedded_preprocess else "preprocessed"

    with (fold_out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    per_class_rows = [
        {"fold": fold_paths.fold, **row} for row in summary["per_class"]
    ]
    _write_stats_csv(fold_out_dir / "per_class_stats.csv", per_class_rows)

    if save_arrays:
        np.savez_compressed(
            fold_out_dir / "compare_arrays.npz",
            raw_input=raw_input,
            zip_network_input=preprocessed_input,
            onnx_input=onnx_input,
            zip_logits=torch_logits,
            onnx_logits=onnx_logits,
            zip_probs=torch_probs,
            onnx_probs=onnx_probs,
            abs_diff=diff,
        )

    feature_names = fold_paths.feature_names
    if len(feature_names) != raw_input.shape[1]:
        feature_names = [f"feature_{idx}" for idx in range(raw_input.shape[1])]

    plotted_files: list[str] = []
    plotted_files.extend(
        str(path)
        for path in _plot_overlay_ratio_pages(
            raw_input,
            onnx_input,
            feature_names,
            fold_out_dir,
            stem="input_compare",
            lhs_label="model.zip input",
            rhs_label="onnx input",
            lhs_color=RAW_COLOR,
            rhs_color=ONNX_COLOR,
            plot_formats=plot_formats,
            items_per_page=features_per_page,
            ncols=ncols,
        )
    )
    if preprocess_has_effect(fold_paths.preprocess_info):
        plotted_files.extend(
            str(path)
            for path in _plot_overlay_ratio_pages(
                raw_input,
                preprocessed_input,
                feature_names,
                fold_out_dir,
                stem="input_transform",
                lhs_label="raw input",
                rhs_label="network input",
                lhs_color=RAW_COLOR,
                rhs_color=PROC_COLOR,
                plot_formats=plot_formats,
                items_per_page=features_per_page,
                ncols=ncols,
            )
        )
    class_names = [f"class {idx}" for idx in range(torch_logits.shape[1])]
    plotted_files.extend(
        str(path)
        for path in _plot_overlay_ratio_pages(
            torch_logits,
            onnx_logits,
            class_names,
            fold_out_dir,
            stem="output_logits",
            lhs_label="model.zip",
            rhs_label="onnx",
            lhs_color=ZIP_COLOR,
            rhs_color=ONNX_COLOR,
            plot_formats=plot_formats,
            items_per_page=classes_per_page,
            ncols=ncols,
        )
    )
    plotted_files.extend(
        str(path)
        for path in _plot_overlay_ratio_pages(
            torch_probs,
            onnx_probs,
            class_names,
            fold_out_dir,
            stem="output_probs",
            lhs_label="model.zip",
            rhs_label="onnx",
            lhs_color=ZIP_COLOR,
            rhs_color=ONNX_COLOR,
            plot_formats=plot_formats,
            items_per_page=classes_per_page,
            ncols=ncols,
        )
    )
    plotted_files.extend(
        str(path)
        for path in _plot_abs_diff(
            diff,
            fold_out_dir,
            stem="output_abs_diff",
            plot_formats=plot_formats,
        )
    )
    summary["plots"] = plotted_files
    with (fold_out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"[fold {fold_paths.fold}] "
        f"max_abs_diff={summary['global']['logit_max_abs_diff']:.3e}, "
        f"mean_abs_diff={summary['global']['logit_mean_abs_diff']:.3e}, "
        f"p99_abs_diff={summary['global']['logit_p99_abs_diff']:.3e}, "
        f"source={actual_source}, embedded_preprocess={embedded_preprocess}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fold-wise model.zip against exported ONNX and save mplhep distribution plots."
    )
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=DEFAULT_MODEL_FOLDER,
        help=f"Root model folder containing fold*/model.zip and onnx/ (default: {DEFAULT_MODEL_FOLDER})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for summaries and plots (default: <model-folder>/compare_zip_onnx)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Specific fold indices to compare. Default: all folds found.",
    )
    parser.add_argument(
        "--zip-name",
        type=str,
        default="model.zip",
        help="Zip filename inside each fold directory (default: model.zip)",
    )
    parser.add_argument(
        "--prefer-int8",
        action="store_true",
        help="Prefer *.int8.onnx when both FP32 and INT8 ONNX files exist.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=8192,
        help="Number of events to compare per fold (default: 4096)",
    )
    parser.add_argument(
        "--sample-source",
        choices=("auto", "dataset", "synthetic"),
        default="auto",
        help="Where to sample comparison inputs from (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for input sampling (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="TabNet device_name for loading model.zip (default: auto)",
    )
    parser.add_argument(
        "--hep-style",
        type=str,
        default="CMS",
        help="mplhep style name (default: CMS)",
    )
    parser.add_argument(
        "--plot-formats",
        nargs="+",
        default=("png",),
        help="Plot file formats to save (default: png)",
    )
    parser.add_argument(
        "--features-per-page",
        type=int,
        default=12,
        help="Number of input features per figure page (default: 12)",
    )
    parser.add_argument(
        "--classes-per-page",
        type=int,
        default=9,
        help="Number of output classes per figure page (default: 9)",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of subplot columns per figure page (default: 3)",
    )
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="Also save sampled inputs, logits, probabilities, and diffs as NPZ.",
    )
    args = parser.parse_args()

    model_root = args.model_folder.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else (model_root / "compare_zip_onnx")
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_style(args.hep_style)

    folds = _discover_folds(
        model_root,
        args.folds,
        zip_name=args.zip_name,
        prefer_int8=args.prefer_int8,
    )
    print(f"[cfg] model_folder  = {model_root}")
    print(f"[cfg] out_dir       = {out_dir}")
    print(f"[cfg] folds         = {[item.fold for item in folds]}")
    print(f"[cfg] sample_size   = {args.sample_size}")
    print(f"[cfg] sample_source = {args.sample_source}")
    print(f"[cfg] device        = {args.device}")
    print(f"[cfg] hep_style     = {args.hep_style}")

    summaries = [
        _compare_fold(
            fold_paths,
            model_root,
            out_dir,
            device_name=args.device,
            sample_size=args.sample_size,
            seed=args.seed,
            sample_source=args.sample_source,
            plot_formats=args.plot_formats,
            features_per_page=args.features_per_page,
            classes_per_page=args.classes_per_page,
            ncols=args.ncols,
            save_arrays=args.save_arrays,
        )
        for fold_paths in folds
    ]

    summary_rows = []
    for item in summaries:
        summary_rows.append(
            {
                "fold": item["fold"],
                "sample_size": item["sample_size"],
                "sample_source": item["sample_source"],
                "embedded_preprocess": item["embedded_preprocess"],
                "logit_max_abs_diff": item["global"]["logit_max_abs_diff"],
                "logit_mean_abs_diff": item["global"]["logit_mean_abs_diff"],
                "logit_p99_abs_diff": item["global"]["logit_p99_abs_diff"],
                "prob_max_abs_diff": item["global"]["prob_max_abs_diff"],
                "prob_mean_abs_diff": item["global"]["prob_mean_abs_diff"],
                "prob_p99_abs_diff": item["global"]["prob_p99_abs_diff"],
                "zip_path": item["zip_path"],
                "onnx_path": item["onnx_path"],
            }
        )

    _write_stats_csv(out_dir / "summary.csv", summary_rows)
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)

    print("[done] Wrote summary to", out_dir / "summary.json")


if __name__ == "__main__":
    main()
