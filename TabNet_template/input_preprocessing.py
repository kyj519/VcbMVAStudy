from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn


_LEGACY_PREPROCESS_MODES = {
    "log1p",
    "norm",
    "log1p_norm",
    "winsorize_norm",
    "winsorize_log1p_norm",
}
_ALL_PREPROCESS_MODES = _LEGACY_PREPROCESS_MODES | {"explicit"}


def normalize_preprocess_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None

    text = str(mode).strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text not in _ALL_PREPROCESS_MODES:
        raise ValueError(
            "Unsupported preprocess mode "
            f"{mode!r}. Choose from None, 'explicit', 'log1p', 'norm', "
            "'log1p_norm', 'winsorize_norm', 'winsorize_log1p_norm'."
        )
    return text


def preprocess_uses_log(mode: Optional[str]) -> bool:
    return normalize_preprocess_mode(mode) in {"log1p", "log1p_norm", "winsorize_log1p_norm"}


def preprocess_uses_norm(mode: Optional[str]) -> bool:
    return normalize_preprocess_mode(mode) in {"norm", "log1p_norm", "winsorize_norm", "winsorize_log1p_norm"}


def preprocess_uses_winsorize(mode: Optional[str]) -> bool:
    return normalize_preprocess_mode(mode) in {"winsorize_norm", "winsorize_log1p_norm"}


def preprocess_has_effect(preprocess_info: Optional[Dict[str, Any]]) -> bool:
    if not preprocess_info:
        return False

    keys = (
        "log_indices",
        "norm_indices",
        "winsorize_indices",
        "log_columns",
        "norm_columns",
        "log_norm_columns",
        "winsorize_columns",
        "winsorize_log_columns",
        "winsorize_norm_columns",
        "winsorize_log_norm_columns",
    )
    return any(preprocess_info.get(key) for key in keys)


def _preprocess_mode_label(preprocess_info: Optional[Dict[str, Any]]) -> str:
    if not preprocess_has_effect(preprocess_info):
        return "none"
    mode = None if not preprocess_info else preprocess_info.get("mode")
    normalized = normalize_preprocess_mode(mode)
    if normalized is not None:
        return normalized
    return "custom"


def preprocess_metadata_props(
    preprocess_info: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    return {
        "tabnet_preprocess_embedded": "1" if preprocess_has_effect(preprocess_info) else "0",
        "tabnet_preprocess_mode": _preprocess_mode_label(preprocess_info),
    }


def _resolve_feature_indices(
    feature_names: Sequence[str],
    columns: Optional[Sequence[str]],
) -> list[int]:
    if not columns:
        return []

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    missing = [name for name in columns if name not in name_to_idx]
    if missing:
        raise KeyError(f"Preprocess columns not found in features: {missing}")

    return [int(name_to_idx[name]) for name in columns]


def _normalize_name_columns(columns: Optional[Sequence[str]]) -> list[str]:
    return [str(name) for name in (columns or [])]


def _normalize_winsorize_columns(
    columns: Optional[Sequence[tuple[str, tuple[float, float]]]],
) -> list[tuple[str, tuple[float, float]]]:
    normalized = []
    for name, bounds in list(columns or []):
        low, high = bounds
        low = float(low)
        high = float(high)
        if low > high:
            raise ValueError(
                f"Invalid winsorize bounds for {name!r}: low={low} > high={high}"
            )
        normalized.append((str(name), (low, high)))
    return normalized


def _build_winsorize_spec(
    feature_names: Sequence[str],
    columns: Sequence[tuple[str, tuple[float, float]]],
) -> tuple[list[tuple[str, tuple[float, float]]], list[int], list[float], list[float]]:
    normalized = _normalize_winsorize_columns(columns)
    indices: list[int] = []
    lows: list[float] = []
    highs: list[float] = []
    for name, (low, high) in normalized:
        idx = _resolve_feature_indices(feature_names, [name])[0]
        indices.append(int(idx))
        lows.append(float(low))
        highs.append(float(high))
    return normalized, indices, lows, highs


def _validate_explicit_groups(
    categorical_columns: Sequence[str],
    *,
    log_columns: Sequence[str],
    norm_columns: Sequence[str],
    log_norm_columns: Sequence[str],
    winsorize_columns: Sequence[tuple[str, tuple[float, float]]],
    winsorize_log_columns: Sequence[tuple[str, tuple[float, float]]],
    winsorize_norm_columns: Sequence[tuple[str, tuple[float, float]]],
    winsorize_log_norm_columns: Sequence[tuple[str, tuple[float, float]]],
) -> None:
    categorical = set(categorical_columns or [])
    groups = {
        "log_columns": list(log_columns),
        "norm_columns": list(norm_columns),
        "log_norm_columns": list(log_norm_columns),
        "winsorize_columns": [name for name, _ in winsorize_columns],
        "winsorize_log_columns": [name for name, _ in winsorize_log_columns],
        "winsorize_norm_columns": [name for name, _ in winsorize_norm_columns],
        "winsorize_log_norm_columns": [name for name, _ in winsorize_log_norm_columns],
    }

    seen: dict[str, str] = {}
    for group_name, names in groups.items():
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise ValueError(f"Duplicate columns in {group_name}: {duplicates}")
        overlap_with_categorical = sorted(name for name in names if name in categorical)
        if overlap_with_categorical:
            raise ValueError(
                f"Categorical columns cannot be preprocessed in {group_name}: {overlap_with_categorical}"
            )
        for name in names:
            if name in seen:
                raise ValueError(
                    f"Column {name!r} is assigned to both {seen[name]} and {group_name}."
                )
            seen[name] = group_name


def _build_explicit_spec(
    feature_names: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    log_columns: Optional[Sequence[str]] = None,
    norm_columns: Optional[Sequence[str]] = None,
    log_norm_columns: Optional[Sequence[str]] = None,
    winsorize_columns: Optional[Sequence[tuple[str, tuple[float, float]]]] = None,
    winsorize_log_columns: Optional[Sequence[tuple[str, tuple[float, float]]]] = None,
    winsorize_norm_columns: Optional[Sequence[tuple[str, tuple[float, float]]]] = None,
    winsorize_log_norm_columns: Optional[Sequence[tuple[str, tuple[float, float]]]] = None,
) -> Dict[str, Any]:
    log_columns = _normalize_name_columns(log_columns)
    norm_columns = _normalize_name_columns(norm_columns)
    log_norm_columns = _normalize_name_columns(log_norm_columns)
    winsorize_columns = _normalize_winsorize_columns(winsorize_columns)
    winsorize_log_columns = _normalize_winsorize_columns(winsorize_log_columns)
    winsorize_norm_columns = _normalize_winsorize_columns(winsorize_norm_columns)
    winsorize_log_norm_columns = _normalize_winsorize_columns(winsorize_log_norm_columns)

    _validate_explicit_groups(
        categorical_columns,
        log_columns=log_columns,
        norm_columns=norm_columns,
        log_norm_columns=log_norm_columns,
        winsorize_columns=winsorize_columns,
        winsorize_log_columns=winsorize_log_columns,
        winsorize_norm_columns=winsorize_norm_columns,
        winsorize_log_norm_columns=winsorize_log_norm_columns,
    )

    log_only_indices = _resolve_feature_indices(feature_names, log_columns)
    norm_only_indices = _resolve_feature_indices(feature_names, norm_columns)
    log_norm_indices = _resolve_feature_indices(feature_names, log_norm_columns)
    winsorize_columns, winsorize_indices, winsorize_lows, winsorize_highs = _build_winsorize_spec(
        feature_names, winsorize_columns
    )
    winsorize_log_columns, winsorize_log_indices, winsorize_log_lows, winsorize_log_highs = _build_winsorize_spec(
        feature_names, winsorize_log_columns
    )
    winsorize_norm_columns, winsorize_norm_indices, winsorize_norm_lows, winsorize_norm_highs = _build_winsorize_spec(
        feature_names, winsorize_norm_columns
    )
    winsorize_log_norm_columns, winsorize_log_norm_indices, winsorize_log_norm_lows, winsorize_log_norm_highs = _build_winsorize_spec(
        feature_names, winsorize_log_norm_columns
    )

    return {
        "log_columns": log_columns,
        "log_only_indices": [int(idx) for idx in log_only_indices],
        "norm_columns": norm_columns,
        "norm_only_indices": [int(idx) for idx in norm_only_indices],
        "log_norm_columns": log_norm_columns,
        "log_norm_indices": [int(idx) for idx in log_norm_indices],
        "winsorize_columns": winsorize_columns,
        "winsorize_only_indices": [int(idx) for idx in winsorize_indices],
        "winsorize_only_lows": winsorize_lows,
        "winsorize_only_highs": winsorize_highs,
        "winsorize_log_columns": winsorize_log_columns,
        "winsorize_log_indices": [int(idx) for idx in winsorize_log_indices],
        "winsorize_log_lows": winsorize_log_lows,
        "winsorize_log_highs": winsorize_log_highs,
        "winsorize_norm_columns": winsorize_norm_columns,
        "winsorize_norm_indices": [int(idx) for idx in winsorize_norm_indices],
        "winsorize_norm_lows": winsorize_norm_lows,
        "winsorize_norm_highs": winsorize_norm_highs,
        "winsorize_log_norm_columns": winsorize_log_norm_columns,
        "winsorize_log_norm_indices": [int(idx) for idx in winsorize_log_norm_indices],
        "winsorize_log_norm_lows": winsorize_log_norm_lows,
        "winsorize_log_norm_highs": winsorize_log_norm_highs,
    }


def _build_legacy_spec(
    feature_names: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    mode: Optional[str],
    log_columns: Optional[Sequence[str]] = None,
    winsorize_columns: Optional[Sequence[tuple[str, tuple[float, float]]]] = None,
) -> Dict[str, Any]:
    mode = normalize_preprocess_mode(mode)
    categorical = set(categorical_columns or [])
    all_non_categorical = [name for name in feature_names if name not in categorical]

    log_columns = _normalize_name_columns(log_columns)
    winsorize_columns = _normalize_winsorize_columns(winsorize_columns)
    winsorize_map = {name: bounds for name, bounds in winsorize_columns}
    winsorize_names = list(winsorize_map.keys())
    log_name_set = set(log_columns)
    wins_name_set = set(winsorize_names)

    explicit_log_columns: list[str] = []
    explicit_norm_columns: list[str] = []
    explicit_log_norm_columns: list[str] = []
    explicit_winsorize_columns: list[tuple[str, tuple[float, float]]] = []
    explicit_winsorize_log_columns: list[tuple[str, tuple[float, float]]] = []
    explicit_winsorize_norm_columns: list[tuple[str, tuple[float, float]]] = []
    explicit_winsorize_log_norm_columns: list[tuple[str, tuple[float, float]]] = []

    if mode == "log1p":
        explicit_log_columns = list(log_columns)
    elif mode == "norm":
        explicit_norm_columns = list(all_non_categorical)
    elif mode == "log1p_norm":
        non_categorical_log = [name for name in log_columns if name not in categorical]
        categorical_log = [name for name in log_columns if name in categorical]
        explicit_log_columns = categorical_log
        explicit_log_norm_columns = non_categorical_log
        explicit_norm_columns = [
            name for name in all_non_categorical if name not in set(non_categorical_log)
        ]
    elif mode == "winsorize_norm":
        non_categorical_wins = [name for name in winsorize_names if name not in categorical]
        categorical_wins = [name for name in winsorize_names if name in categorical]
        explicit_winsorize_columns = [
            (name, winsorize_map[name]) for name in categorical_wins
        ]
        explicit_winsorize_norm_columns = [
            (name, winsorize_map[name]) for name in non_categorical_wins
        ]
        explicit_norm_columns = [
            name for name in all_non_categorical if name not in set(non_categorical_wins)
        ]
    elif mode == "winsorize_log1p_norm":
        non_categorical_log = {name for name in log_columns if name not in categorical}
        categorical_log = {name for name in log_columns if name in categorical}
        for name in winsorize_names:
            bounds = winsorize_map[name]
            if name in categorical:
                if name in categorical_log:
                    explicit_winsorize_log_columns.append((name, bounds))
                else:
                    explicit_winsorize_columns.append((name, bounds))
            else:
                if name in non_categorical_log:
                    explicit_winsorize_log_norm_columns.append((name, bounds))
                else:
                    explicit_winsorize_norm_columns.append((name, bounds))

        explicit_log_columns = [
            name for name in log_columns if name in categorical_log and name not in wins_name_set
        ]
        explicit_log_norm_columns = [
            name for name in log_columns if name in non_categorical_log and name not in wins_name_set
        ]
        normalized_names = {
            *(name for name, _ in explicit_winsorize_norm_columns),
            *(name for name, _ in explicit_winsorize_log_norm_columns),
            *explicit_log_norm_columns,
        }
        explicit_norm_columns = [
            name for name in all_non_categorical if name not in normalized_names
        ]

    return _build_explicit_spec(
        feature_names,
        categorical_columns,
        log_columns=explicit_log_columns,
        norm_columns=explicit_norm_columns,
        log_norm_columns=explicit_log_norm_columns,
        winsorize_columns=explicit_winsorize_columns,
        winsorize_log_columns=explicit_winsorize_log_columns,
        winsorize_norm_columns=explicit_winsorize_norm_columns,
        winsorize_log_norm_columns=explicit_winsorize_log_norm_columns,
    )


def fit_preprocess_info(
    train_features: np.ndarray,
    feature_names: Sequence[str],
    *,
    mode: Optional[str],
    log_columns: Optional[Sequence[str]] = None,
    norm_columns: Optional[Sequence[str]] = None,
    log_norm_columns: Optional[Sequence[str]] = None,
    winsorize_columns: Optional[Sequence[tuple[str, tuple[float, float]]]] = None,
    winsorize_log_columns: Optional[Sequence[tuple[str, tuple[float, float]]]] = None,
    winsorize_norm_columns: Optional[Sequence[tuple[str, tuple[float, float]]]] = None,
    winsorize_log_norm_columns: Optional[Sequence[tuple[str, tuple[float, float]]]] = None,
    categorical_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    mode = normalize_preprocess_mode(mode)
    feature_names = list(feature_names)
    num_features = len(feature_names)

    if train_features.ndim != 2:
        raise ValueError(
            f"Expected 2D training features, got shape={train_features.shape}"
        )
    if train_features.shape[1] != num_features:
        raise ValueError(
            "Feature name count does not match training matrix width: "
            f"{num_features} vs {train_features.shape[1]}"
        )

    categorical = list(categorical_columns or [])
    explicit_only_requested = any(
        [
            norm_columns,
            log_norm_columns,
            winsorize_log_columns,
            winsorize_norm_columns,
            winsorize_log_norm_columns,
        ]
    )
    use_explicit = mode == "explicit" or explicit_only_requested
    if mode not in {None, "explicit"} and use_explicit:
        raise ValueError(
            "Do not mix legacy preprocess_mode with explicit column groups. "
            "Use preprocess_mode='explicit' or set preprocess_mode=None."
        )

    if use_explicit:
        spec = _build_explicit_spec(
            feature_names,
            categorical,
            log_columns=log_columns,
            norm_columns=norm_columns,
            log_norm_columns=log_norm_columns,
            winsorize_columns=winsorize_columns,
            winsorize_log_columns=winsorize_log_columns,
            winsorize_norm_columns=winsorize_norm_columns,
            winsorize_log_norm_columns=winsorize_log_norm_columns,
        )
        stored_mode = "explicit"
        source = "explicit"
    else:
        spec = _build_legacy_spec(
            feature_names,
            categorical,
            mode=mode,
            log_columns=log_columns,
            winsorize_columns=winsorize_columns,
        )
        stored_mode = mode
        source = "legacy_mode"

    log_indices = [
        *spec.get("log_only_indices", []),
        *spec.get("log_norm_indices", []),
        *spec.get("winsorize_log_indices", []),
        *spec.get("winsorize_log_norm_indices", []),
    ]
    winsorize_indices = [
        *spec.get("winsorize_only_indices", []),
        *spec.get("winsorize_log_indices", []),
        *spec.get("winsorize_norm_indices", []),
        *spec.get("winsorize_log_norm_indices", []),
    ]
    winsorize_lows = [
        *spec.get("winsorize_only_lows", []),
        *spec.get("winsorize_log_lows", []),
        *spec.get("winsorize_norm_lows", []),
        *spec.get("winsorize_log_norm_lows", []),
    ]
    winsorize_highs = [
        *spec.get("winsorize_only_highs", []),
        *spec.get("winsorize_log_highs", []),
        *spec.get("winsorize_norm_highs", []),
        *spec.get("winsorize_log_norm_highs", []),
    ]
    norm_indices = [
        *spec.get("norm_only_indices", []),
        *spec.get("log_norm_indices", []),
        *spec.get("winsorize_norm_indices", []),
        *spec.get("winsorize_log_norm_indices", []),
    ]

    mean = np.zeros(num_features, dtype=np.float32)
    scale = np.ones(num_features, dtype=np.float32)
    work = np.asarray(train_features, dtype=np.float32, order="C").copy()

    if winsorize_indices:
        work[:, winsorize_indices] = np.clip(
            work[:, winsorize_indices],
            np.asarray(winsorize_lows, dtype=np.float32),
            np.asarray(winsorize_highs, dtype=np.float32),
        )

    if log_indices:
        work[:, log_indices] = np.log1p(work[:, log_indices])

    if norm_indices:
        mu = work[:, norm_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
        sigma = work[:, norm_indices].std(axis=0, dtype=np.float64).astype(np.float32)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, 1.0).astype(
            np.float32
        )
        mean[norm_indices] = mu
        scale[norm_indices] = sigma

    preprocess_info = {
        "version": 2,
        "schema": "column_groups_v2",
        "source": source,
        "mode": stored_mode,
        "feature_names": feature_names,
        "categorical_columns": sorted(set(categorical)),
        **spec,
        "log_indices": [int(idx) for idx in log_indices],
        "winsorize_indices": [int(idx) for idx in winsorize_indices],
        "winsorize_lows": [float(v) for v in winsorize_lows],
        "winsorize_highs": [float(v) for v in winsorize_highs],
        "norm_indices": [int(idx) for idx in norm_indices],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    return preprocess_info


def apply_feature_preprocess(
    features: np.ndarray,
    preprocess_info: Optional[Dict[str, Any]],
) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape={arr.shape}")
    if not preprocess_has_effect(preprocess_info):
        return np.ascontiguousarray(arr, dtype=np.float32)

    out = np.array(arr, dtype=np.float32, copy=True, order="C")
    log_indices = [int(idx) for idx in preprocess_info.get("log_indices", [])]
    winsorize_indices = [int(idx) for idx in preprocess_info.get("winsorize_indices", [])]
    norm_indices = [int(idx) for idx in preprocess_info.get("norm_indices", [])]

    if winsorize_indices:
        out[:, winsorize_indices] = np.clip(
            out[:, winsorize_indices],
            np.asarray(preprocess_info.get("winsorize_lows", []), dtype=np.float32),
            np.asarray(preprocess_info.get("winsorize_highs", []), dtype=np.float32),
        )

    if log_indices:
        out[:, log_indices] = np.log1p(out[:, log_indices])

    if norm_indices:
        mean = np.asarray(preprocess_info.get("mean", []), dtype=np.float32)
        scale = np.asarray(preprocess_info.get("scale", []), dtype=np.float32)
        if mean.shape != (out.shape[1],) or scale.shape != (out.shape[1],):
            raise ValueError(
                "Preprocess stats shape mismatch: "
                f"mean={mean.shape}, scale={scale.shape}, input_dim={out.shape[1]}"
            )
        out = (out - mean) / scale

    return np.ascontiguousarray(out, dtype=np.float32)


class InputPreprocessModule(nn.Module):
    def __init__(self, preprocess_info: Optional[Dict[str, Any]], input_dim: int):
        super().__init__()
        info = preprocess_info or {}

        log_mask = torch.zeros(input_dim, dtype=torch.bool)
        for idx in info.get("log_indices", []):
            log_mask[int(idx)] = True

        winsorize_mask = torch.zeros(input_dim, dtype=torch.bool)
        winsorize_low = np.full(input_dim, -np.inf, dtype=np.float32)
        winsorize_high = np.full(input_dim, np.inf, dtype=np.float32)
        for idx, low, high in zip(
            info.get("winsorize_indices", []),
            info.get("winsorize_lows", []),
            info.get("winsorize_highs", []),
        ):
            winsorize_mask[int(idx)] = True
            winsorize_low[int(idx)] = float(low)
            winsorize_high[int(idx)] = float(high)

        mean = np.asarray(info.get("mean", np.zeros(input_dim)), dtype=np.float32)
        scale = np.asarray(info.get("scale", np.ones(input_dim)), dtype=np.float32)
        if mean.shape != (input_dim,):
            mean = np.zeros(input_dim, dtype=np.float32)
        if scale.shape != (input_dim,):
            scale = np.ones(input_dim, dtype=np.float32)

        self._has_log = bool(info.get("log_indices"))
        self._has_winsorize = bool(info.get("winsorize_indices"))
        self._has_norm = bool(info.get("norm_indices"))
        self.register_buffer("log_mask", log_mask)
        self.register_buffer("winsorize_mask", winsorize_mask)
        self.register_buffer("winsorize_low", torch.as_tensor(winsorize_low, dtype=torch.float32))
        self.register_buffer("winsorize_high", torch.as_tensor(winsorize_high, dtype=torch.float32))
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()

        if self._has_winsorize:
            mask = self.winsorize_mask.unsqueeze(0).expand_as(out)
            clipped = torch.minimum(torch.maximum(out, self.winsorize_low), self.winsorize_high)
            out = torch.where(mask, clipped, out)

        if self._has_log:
            mask = self.log_mask.unsqueeze(0).expand_as(out)
            safe = torch.where(mask, out, torch.zeros_like(out))
            logged = torch.log1p(safe)
            out = torch.where(mask, logged, out)

        if self._has_norm:
            out = (out - self.mean) / self.scale

        return out


class TabNetExportWrapper(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        preprocess_info: Optional[Dict[str, Any]],
        input_dim: int,
    ):
        super().__init__()
        self.preprocess = InputPreprocessModule(preprocess_info, input_dim=input_dim)
        self.network = network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(self.preprocess(x))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def build_tabnet_export_wrapper(
    network: nn.Module,
    preprocess_info: Optional[Dict[str, Any]],
    input_dim: int,
) -> TabNetExportWrapper:
    return TabNetExportWrapper(
        network=network,
        preprocess_info=preprocess_info,
        input_dim=input_dim,
    )
