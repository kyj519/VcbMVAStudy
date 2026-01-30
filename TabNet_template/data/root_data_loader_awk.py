"""
Awkward/Uproot-based ROOT → ML loader with **reproducible K-folds** (no train/val/test ratios)
===========================================================================================

What's new vs previous refactor
-------------------------------
- **K-fold split**: deterministic, stratified by class. No val/train/test ratios anymore.
- **Weight handling**: the loader **never reads a `weight` leaf**. It **always computes** per-event
  weights from configured fields (defaults to `weight_*` leaves) with optional RF patch via callback.
- Same vectorized pipeline: `uproot` + `awkward` → NumPy.
- Returns a compact dataset + a ready-to-use list of K folds `(train_idx, val_idx)`.

Quickstart
----------

    dataset = load_root_as_dataset_kfold(
        tree_path_filter_str=[
            ("/path/Vcb.root",  "Mu/Central/Result_Tree", "n_jets >= 6 && n_bjets >= 2"),  # class 0 (signal)
            ("/path/TTLJ.root", "Mu/Central/Result_Tree", "n_jets >= 6 && n_bjets >= 2"),  # class 1 (bkg)
        ],
        varlist=[
            "pt_w_u","pt_w_d","m_had_t","n_jets","n_bjets","year_index"
        ],
        categorical_columns=["n_jets","n_bjets"],
        n_splits=5,
        seed=123,
    )

    # Use fold k
    from root_data_loader_awk import get_fold
    Xtr, ytr, wtr, Xval, yval, wval = get_fold(dataset, k=0)

Notes / limitations
-------------------
- Columns must be **event-level scalars**. If you need jet-level reductions, compute them upstream.
- `weight_RFPatch` from your C++ JetSFProducer is modeled via a Python hook (`rf_patch_provider`).
  If unset, it defaults to 1.0.
- "Don't accept weight column" requirement is enforced: if `"weight"` is in `varlist`, a
  `ValueError` is raised.

"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import re
from pathlib import Path
import numpy as np
import awkward as ak
import uproot
from dataclasses import dataclass
import ROOT
from typing import Callable, Dict, Iterable, List, Optional, OrderedDict, Sequence, Tuple, Any
import sys
sys.path.append(os.environ["DIR_PATH"] + "/Corrections")
from RFHelper import make_rf_provider_fast, RFProvider
from helpers import task
import json

# ----------------------------
# Short-name mapping (kept)
# ----------------------------
map_short_name: Dict[str, List[str]] = {}
map_short_name["WJets"] = ["WJets"]
map_short_name["DYJets"] = ["DYJets"]
#map_short_name["VV"] = ["WW", "WZ", "ZZ"]
map_short_name["WW"] = ["WW"]
map_short_name["WZ"] = ["WZ"]
map_short_name["ZZ"] = ["ZZ"]
map_short_name["TTLL"] = ["TTLL"]
map_short_name["TTLJ"] = ["TTLJ"]
#map_short_name["TTJJ"] = ["TTJJ"]
#map_short_name["ttV"] = ["ttH", "ttW", "ttZ"]
map_short_name["ttHTobb"] = ["ttHTobb"]
map_short_name["ttHToNonbb"] = ["ttHToNonbb"]
map_short_name["ttWToLNu"] = ["ttWToLNu"]
map_short_name["ttWToQQ"] = ["ttWToQQ"]
map_short_name["ttZToLLNuNu"] = ["ttZToLLNuNu"]
map_short_name["ttZToQQ"] = ["ttZToQQ","ttZToQQ_ll"]

map_short_name["QCD_bEn"] = ["QCD_bEn"]

#map_short_name["ST"] = ["SingleTop"]
map_short_name["ST_tw"] = ["SingleTop_tW"]
map_short_name["ST_tch"] = ["SingleTop_tch"]
map_short_name["ST_sch"] = ["SingleTop_sch"]
map_short_name["DATA"] = ["SingleMuon", "SingleElectron", "EGamma"]


def getShortName(file: str) -> str:
    for short, keys in map_short_name.items():
        if any(k in file for k in keys):
            return short
    raise ValueError(f"getShortName: No short name matched for file {file}. Check the file name.")

# ----------------------------
# Year / syst helpers
# ----------------------------
YEAR_MAP: Dict[str, int] = {
    "2016preVFP": 0,
    "2016postVFP": 1,
    "2017": 2,
    "2018": 3,
}

IDX_TO_YEAR: Dict[int, str] = {v: k for k, v in YEAR_MAP.items()}

LUMI_MAP: Dict[str, float] = {
    "2016preVFP": 16.81,
    "2016postVFP": 19.52,
    "2017": 41.53,
    "2018": 59.74,
}


def _detect_year_index(path: str) -> int:
    for k, v in YEAR_MAP.items():
        if k in path:
            return v
    raise ValueError(f"_detect_year_index: cannot infer year from path: {path}")


# ----------------------------
# Filter expression translator
# ----------------------------
_BOOL_SYMS = (("&&", "&"), ("||", "|"))


def _translate_filter_expr(expr: str) -> str:
    if not expr:
        return expr
    out = expr
    for a, b in _BOOL_SYMS:
        out = out.replace(a, b)
    out = re.sub(r'!(?!=)', '~', out)
    out = out.replace("abs(", "np.abs(")
    return out



# ----------------------------
# Weight construction (never read a `weight` leaf)
# ----------------------------
_DEFAULT_WEIGHT_FIELDS_BT = [
    "weight_b_tag",
    "weight_el_id",
    "sign(weight_mc)",
    #"weight_lumi",
    "weight_mu_id",
    "weight_mu_iso",
    "weight_el_reco",
    "weight_pileup",
    "weight_prefire",
    "weight_sl_trig",
    "weight_top_pt_mva",
    "weight_b_frag_mva_nominal",
    "weight_hem_veto"
]
_DEFAULT_WEIGHT_FIELDS_CT = [
    "weight_c_tag",
] + [f for f in _DEFAULT_WEIGHT_FIELDS_BT if f != "weight_b_tag"]


def _build_weight_vector(
    arrays: Dict[str, ak.Array],
    use_btag: bool,
    include_rf: Optional[np.ndarray],
    fields: Optional[Sequence[str]] = None,
) -> np.ndarray:
    fields = list(fields) if fields is not None else (
        _DEFAULT_WEIGHT_FIELDS_BT if use_btag else _DEFAULT_WEIGHT_FIELDS_CT
    )
    n = len(next(iter(arrays.values()))) if arrays else 0
    w = np.ones(n, dtype=np.float32)
    for name in fields:
        if name.startswith("sign(") and name.endswith(")"):
            inner = name[5:-1].strip()
            if inner in arrays:
                w *= np.sign(ak.to_numpy(arrays[inner]))
        elif name in arrays:
            w *= ak.to_numpy(arrays[name])
    if include_rf is not None:
        w *= include_rf.astype(np.float32, copy=False)
    return w


def _resolve_weight_fields(
    assume_btag_mode: bool,
    weight_fields: Optional[Sequence[str]] = None,
    weight_fields_bt: Optional[Sequence[str]] = None,
    weight_fields_ct: Optional[Sequence[str]] = None,
) -> List[str]:
    if weight_fields is not None:
        return list(weight_fields)
    if assume_btag_mode:
        return list(weight_fields_bt) if weight_fields_bt is not None else list(_DEFAULT_WEIGHT_FIELDS_BT)
    return list(weight_fields_ct) if weight_fields_ct is not None else list(_DEFAULT_WEIGHT_FIELDS_CT)


# ----------------------------
# I/O helpers
# ----------------------------

def _resolve_spanet_partner(root_path: str) -> Tuple[str, Optional[str]]:
    path = Path(root_path)
    # If this is a TABNET merged file, map back to the original base/SPANET pair.
    if path.parent.name == "TABNET" and path.suffix == ".root":
        stem = path.stem
        if stem.endswith("__merged") and "__" in stem:
            try:
                base_stem, _tag, _merged = stem.rsplit("__", 2)
            except ValueError:
                base_stem = None
            if base_stem:
                base_dir = path.parent.parent
                orig_name = base_stem + path.suffix
                base_path = base_dir / orig_name
                spanet_path = base_dir / "SPANET" / orig_name
                if base_path.is_file():
                    if spanet_path.is_file():
                        print(
                            f"[info] TABNET merged input detected; using base+SPANET: "
                            f"{base_path} + {spanet_path}"
                        )
                        return str(base_path), str(spanet_path)
                    return str(base_path), None
    if path.parent.name == "SPANET":
        base_path = path.parent.parent / path.name
        if base_path.is_file():
            return str(base_path), str(path)
        return str(path), None
    spanet_path = path.parent / "SPANET" / path.name
    if spanet_path.is_file():
        return str(path), str(spanet_path)
    return str(path), None


def _branch_set(tree) -> set[str]:
    return {str(k).split(";")[0] for k in tree.keys()}


def _read_uproot(
    root_path: str,
    tree_path: str,
    want_columns: List[str],
    log_columns: Optional[Sequence[str]] = None,
    winsorize_columns: Optional[Sequence[Tuple[str, Tuple[float, float]]]] = None,
) -> Dict[str, ak.Array]:
    if "year_index" in want_columns:
        want_columns.remove("year_index")

    base_path, spanet_partner = _resolve_spanet_partner(root_path)

    arrays: Dict[str, ak.Array] = {}
    missing: List[str] = []
    n_base = 0
    with uproot.open(base_path, object_cache=uproot.cache.LRUCache(800_000)) as f:
        t_base = f[tree_path]
        n_base = t_base.num_entries
        base_keys = _branch_set(t_base)
        base_want = [c for c in want_columns if c in base_keys]
        missing = [c for c in want_columns if c not in base_keys]
        if base_want:
            arrays.update(t_base.arrays(list(base_want), library="ak", how=dict))

    if missing:
        if not spanet_partner:
            raise KeyError(
                f"Missing columns in {base_path} (no SPANET partner): {missing}"
            )
        with uproot.open(
            spanet_partner, object_cache=uproot.cache.LRUCache(800_000)
        ) as f:
            t_spanet = f[tree_path]
            n_spanet = t_spanet.num_entries
            if n_spanet != n_base:
                raise RuntimeError(
                    f"Entry mismatch for {tree_path}: {base_path} ({n_base}) vs "
                    f"{spanet_partner} ({n_spanet})"
                )
            spanet_keys = _branch_set(t_spanet)
            spanet_want = [c for c in missing if c in spanet_keys]
            still_missing = [c for c in missing if c not in spanet_keys]
            if still_missing:
                raise KeyError(
                    f"Missing columns in base+SPANET for {tree_path}: {still_missing}"
                )
            if spanet_want:
                spanet_arrays = t_spanet.arrays(
                    list(spanet_want), library="ak", how=dict
                )
                for k, v in spanet_arrays.items():
                    if k not in arrays:
                        arrays[k] = v

    if log_columns:
        for col in log_columns:
            arrays[col] = np.log1p(arrays[col])

    if winsorize_columns:
        for col, (low, high) in winsorize_columns:
            arr = ak.to_numpy(arrays[col])
            arr = np.clip(arr, low, high)
            arrays[col] = ak.Array(arr)

    arrays["__idx"] = ak.Array(np.arange(n_base, dtype=np.uint64))  # 0..n-1
    return arrays

def _apply_filter(arrays: Dict[str, ak.Array], fstr: Optional[str]) -> np.ndarray:
    if not fstr:
        n = len(next(iter(arrays.values()))) if arrays else 0
        return np.ones(n, dtype=bool)
    expr = _translate_filter_expr(fstr)
    localns = {k: ak.to_numpy(v) for k, v in arrays.items()}
    localns["np"] = np
    mask = eval(expr, {"__builtins__": {}}, localns)
    return np.asarray(mask, dtype=bool)


def _ensure_numpy_scalars(d: Dict[str, ak.Array]) -> Dict[str, np.ndarray]:
    """
    Accept only event-level scalar columns:
      - If an inner list-dimension exists (ak.num(..., axis=1) succeeds) → reject (jagged).
      - Otherwise convert to 1-D numpy array.
    """
    out: Dict[str, np.ndarray] = {}
    for k, v in d.items():
        try:
            _ = ak.num(v, axis=1)  # succeeds only if inner list axis exists
        except Exception:
            pass # no inner list axis, good 
        else:
            raise ValueError(f"Column '{k}' appears jagged; provide event-level scalars.")

        arr = ak.to_numpy(v)
        if arr.ndim != 1:
            raise ValueError(f"Column '{k}' is not 1-D after conversion (ndim={arr.ndim}).")
        out[k] = arr
    return out


def _concat_dicts(dicts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {k: np.concatenate([d[k] for d in dicts], axis=0) for k in keys}


# ----------------------------
# RF patch provider
# ----------------------------

def compute_rf_patch(arrays: Dict[str, ak.Array], file_path: str, rf_provider: RFProvider) -> np.ndarray:
    if rf_provider is None:
        n = len(next(iter(arrays.values()))) if arrays else 0
        return np.ones(n, dtype=np.float32)
    short_mc = getShortName(file_path)
    year_tag = next((k for k in YEAR_MAP if k in file_path), "")
    return rf_provider(arrays, short_mc, year_tag)


def _year_fractions(present_year_idxs: np.ndarray) -> Dict[int, float]:
    lumis = {yi: LUMI_MAP.get(IDX_TO_YEAR.get(int(yi), ""), 0.0) for yi in present_year_idxs}
    total = float(sum(lumis.values()))
    if total <= 0:
        # fallback to uniform if not found
        return {int(yi): 1.0 / len(present_year_idxs) for yi in present_year_idxs}
    return {int(yi): lumis[int(yi)] / total for yi in present_year_idxs}


def _downsample_to_lumi_ratio(years: np.ndarray, seed: int, *, return_info: bool = False):
    """Return boolean mask over entries **within one class** so that the kept samples
    are downsampled to match LUMI_MAP ratios across present years.

    Strategy: choose the **largest feasible total** T = min_y floor(n_avail[y] / frac[y]) and
    set target[y] ≈ round(frac[y] * T) while not exceeding availability, using a stable order.

    If `return_info=True`, also return a dict with `target`, `avail`, `fracs`, and `T`.
    """
    n = years.size
    if n == 0:
        return (np.zeros(0, dtype=bool), {"target": {}, "avail": {}, "fracs": {}, "T": 0}) if return_info else np.zeros(0, dtype=bool)

    uniq_years, counts = np.unique(years, return_counts=True)
    fracs = _year_fractions(uniq_years)

    # compute maximum feasible total T
    T_candidates = []
    for yi in uniq_years:
        c = counts[np.where(uniq_years == yi)[0][0]]
        f = fracs[int(yi)]
        if f <= 0:
            continue
        T_candidates.append(int(np.floor(c / f)))
    if not T_candidates:
        keep = np.ones(n, dtype=bool)
        return (keep, {"target": {}, "avail": {}, "fracs": fracs, "T": int(np.sum(keep))}) if return_info else keep
    T = min(T_candidates)

    # initial targets by floor
    raw = {int(yi): fracs[int(yi)] * T for yi in uniq_years}
    target = {int(yi): int(np.floor(raw[int(yi)])) for yi in uniq_years}

    # distribute remainder based on largest fractional part, capped by availability
    remainder = T - sum(target.values())
    fractional = sorted([(raw[i] - target[i], i) for i in target.keys()], reverse=True)

    avail = {int(yi): counts[np.where(uniq_years == yi)[0][0]] for yi in uniq_years}

    for _ in range(remainder):
        for _, yi in fractional:
            if target[yi] < avail[yi]:
                target[yi] += 1
                break

    # select per-year using stable permutation
    keep = np.zeros(n, dtype=bool)
    for yi in uniq_years:
        yi = int(yi)
        idx = np.flatnonzero(years == yi)
        shuffle_idx = np.random.default_rng(seed + yi).permutation(idx)
        k = min(target.get(yi, 0), idx.size)
        keep[shuffle_idx[:k]] = True

    if return_info:
        return keep, {"target": target, "avail": avail, "fracs": fracs, "T": int(T)}
    return keep

def _guard_nan_weights(w: np.ndarray) -> np.ndarray:
    w_clean = np.copy(w)
    nan_mask = np.isnan(w_clean) | np.isinf(w_clean)
    if np.any(nan_mask):
        w_clean[nan_mask] = 0.0
    print(f"_guard_nan_weights: Replaced {int(np.sum(nan_mask))} NaN/Inf weights with 0.0")
    return w_clean

# ----------------------------
# Public API
# ----------------------------
def load_root_as_dataset_kfold(
    tree_path_filter_str: Sequence[Tuple[str, str, Optional[str]]],
    varlist: Sequence[str],
    log_columns: Optional[Sequence[str]] = None,
    winsorize_columns: Optional[Sequence[Tuple[str, Tuple[float, float]]]] = None,
    categorical_columns: Optional[Sequence[str]] = None,
    categorical_dims: Optional[Dict[str, int]] = None,
    *,
    weight_fields: Optional[Sequence[str]] = None,
    weight_fields_bt: Optional[Sequence[str]] = None,
    weight_fields_ct: Optional[Sequence[str]] = None,
    sample_bkg: Optional[float] = None,
    rf_patch_provider: RFProvider = make_rf_provider_fast,
    n_splits: int = 3,
    seed: int = 42,
    assume_btag_mode: bool = True,
    infer_mode: bool = False
) -> Dict[str, np.ndarray]:
    """Load multiple ROOT trees into a single dataset and build reproducible K-fold splits.

    Parameters
    ----------
    tree_path_filter_str : list of (root_path, tree_path, filter_string)
        Each item is treated as a separate class (y = 0..C-1).
    varlist : list[str]
        Feature names to include. **Must NOT contain 'weight'** (weights are always computed).
    log_columns : list[str] | None
        If set, columns to apply `np.log1p` transformation before returning.
    weight_fields : list[str] | None
        Override default weight columns (applies to both tag modes).
    weight_fields_bt : list[str] | None
        Override default weight columns for b-tag mode only.
    weight_fields_ct : list[str] | None
        Override default weight columns for c-tag mode only.
    sample_bkg : float | None
        If set, for **each year** and for **each background class (1..C-1)**, keep at most
        `round(sample_bkg * N_sig_year)` events after year balancing.
    n_splits : int
        Number of folds.
    seed : int
        Seed used for deterministic selections.
    assume_btag_mode : bool
        If True, use b-tag weight set; otherwise use c-tag variant.
    """
    #print varlist in red
    print("\033[31m" + "[load_root_as_dataset_kfold] get varlist:" + str(varlist) + "\033[0m")
    if any(v.lower() == "weight" for v in varlist):
        raise ValueError("Do not pass 'weight' in varlist. We always compute weights internally.")

    # extra columns needed for weight computation + filters (+ year_index for balancing)
    tag_mode = "btag" if assume_btag_mode else "ctag"
    print("\033[31m" + f"[load_root_as_dataset_kfold] assume {tag_mode} mode" + "\033[0m")
    weight_fields_eff = _resolve_weight_fields(
        assume_btag_mode,
        weight_fields=weight_fields,
        weight_fields_bt=weight_fields_bt,
        weight_fields_ct=weight_fields_ct,
    )
    weight_cols = set(weight_fields_eff)

    # per-class accumulation (first pass)
    shards = []
    tree_cache: Dict[Tuple[str, str], ak.Array] = {}
    with task("Preparing to load data..."):
        want_union: dict[tuple[str, str], set[str]] = {}
        for group in tree_path_filter_str:
            sources = [group] if (isinstance(group, tuple) and len(group) == 3) else list(group)
            for root_path, tree_path, fstr in sources:
                key = (root_path, tree_path)
                want = set(varlist)
                want.add("index_fold")

                if not infer_mode:
                    want.update(weight_cols)
                for col in list(want):
                    if col.startswith("sign(") and col.endswith(")"):
                        inner = col[5:-1].strip()
                        want.add(inner); want.discard(col)
                if fstr:
                    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", fstr):
                        if tok not in {"and", "or", "not", "abs", "np", "True", "False"}:
                            want.add(tok)
                want_union.setdefault(key, set()).update(want)
        print(want_union)

        # 2) 실제 로딩 (key당 1회)
        tree_cache = {}
        with task("Preloading trees (union of required columns)..."):
            for (root_path, tree_path), want in want_union.items():
                with task(f"  File: {root_path}, Tree: {tree_path}"):
                    tree_cache[(root_path, tree_path)] = _read_uproot(
                        root_path, tree_path, sorted(want),
                        log_columns=log_columns,
                        winsorize_columns=winsorize_columns,
                    )
                
    for class_idx, group in enumerate(tree_path_filter_str):
        with task(f"Loading class {class_idx}"):
            # group: (root, tree, fstr) 하나이거나, [(root, tree, fstr), ...] 목록
            sources = [group] if (isinstance(group, tuple) and len(group) == 3) else list(group)

            feat_chunks, y_chunks, w_chunks = [], [], []
            fold_chunks, year_chunks = [], []
            stage0_counts = {}
            class_labels = []

            for (root_path, tree_path, fstr) in sources:
                with task(f"  File: {root_path}, Tree: {tree_path}, Filter: {fstr or '(none)'}"):

                    arrays = tree_cache[(root_path, tree_path)]
                    
                    if "year_index" in varlist and "year_index" not in arrays:
                        yi = _detect_year_index(root_path)
                        print("\033[31m" + f"[load_root_as_dataset_kfold] inferred year_index {yi} for {root_path}" + "\033[0m")
                        arrays["year_index"] = ak.Array(np.full(len(next(iter(arrays.values()))), yi, dtype=np.int64))
                        
                    rf_provider = rf_patch_provider(assume_btag_mode)
                    if infer_mode:
                        rf_vec = None
                    else:
                        rf_vec = compute_rf_patch(arrays, root_path, rf_provider)
                    w_full = _build_weight_vector(
                        arrays,
                        use_btag=assume_btag_mode,
                        include_rf=rf_vec,
                        fields=weight_fields_eff,
                    )
                    w_full = _guard_nan_weights(w_full)

                    mask = _apply_filter(arrays, fstr)
                    arrays_f = OrderedDict((k, arrays[k][mask]) for k in varlist)
                    data_np = _ensure_numpy_scalars(arrays_f)
                    if not data_np:
                        continue

                    n = len(next(iter(data_np.values())))
                    y = np.full(n, class_idx, dtype=np.int64)
                    w = w_full[mask].astype(np.float32, copy=False)

                    #__idx = ak.to_numpy(arrays["__idx"][mask]).astype(np.uint64, copy=False)
                    #folds_local = (__idx % np.uint64(n_splits)).astype(np.int64, copy=False)
                    folds_local = ak.to_numpy(arrays["index_fold"][mask]).astype(np.int64, copy=False)
                    years_src = ak.to_numpy(arrays["year_index"][mask]) if "year_index" in arrays else None
                    if years_src is not None:
                        u0, c0 = np.unique(years_src, return_counts=True)
                        for u, c in zip(u0, c0):
                            stage0_counts[int(u)] = stage0_counts.get(int(u), 0) + int(c)
                        year_chunks.append(years_src.astype(np.int64, copy=False))

                    try:
                        class_labels.append(getShortName(root_path))
                    except Exception:
                        pass

                    feat_chunks.append(data_np)
                    y_chunks.append(y)
                    w_chunks.append(w)
                    fold_chunks.append(folds_local)

                if not feat_chunks:
                    continue

            features_cat = _concat_dicts(feat_chunks) if len(feat_chunks) > 1 else feat_chunks[0]
            y_cat      = np.concatenate(y_chunks, axis=0)
            w_cat      = np.concatenate(w_chunks, axis=0)
            folds_cat  = np.concatenate(fold_chunks, axis=0)
            years_cat  = np.concatenate(year_chunks, axis=0) if year_chunks else None

            stage0 = {int(k): int(v) for k, v in stage0_counts.items()} if years_cat is not None else {-1: int(y_cat.size)}
            class_label = class_labels[0] if class_labels else f"class{class_idx}"

            shards.append({
                "class_idx": class_idx,
                "class_label": class_label,
                "features": features_cat,
                "y": y_cat,
                "w": w_cat,
                "fold_label": folds_cat,
                "years": years_cat,
                "stats": {"stage0": stage0},
            })
    # ----- END NEW ingest loop -----
    if not shards:
        raise RuntimeError("No data loaded. Check file paths / trees / filters.")

    # ----------------------------
    # Step A: per-class year balancing to LUMI_MAP ratios (if year_index exists)
    # ----------------------------
    for s in shards:
        years = s["years"]
        if years is None:
            continue
        keep_mask, info = _downsample_to_lumi_ratio(
            years, seed=seed, return_info=True
        )

        # --- per-year kept count & sumW (AFTER year balancing) ---
        keptA = {}
        sumW_A = {}
        for yi in np.unique(years):
            yi = int(yi)
            sel = (years == yi) & keep_mask
            keptA[yi] = int(np.sum(sel))
            # float64로 누적해 숫자 안정성 ↑
            sumW_A[yi] = float(np.sum(s["w"][sel], dtype=np.float64))

        s["stats"]["year_balance"] = {
            "target": {int(k): int(v) for k, v in info["target"].items()},
            "avail":  {int(k): int(v) for k, v in info["avail"].items()},
            "kept":   keptA,
            "fracs":  {int(k): float(v) for k, v in info["fracs"].items()},
            "T":      int(info["T"]),
            "sumW":   sumW_A,   # 👈 추가: per-year sum of weights (kept)
        }

        # apply selection in-place
        for k in list(s["features"].keys()):
            s["features"][k] = s["features"][k][keep_mask]
        s["y"] = s["y"][keep_mask]
        s["w"] = s["w"][keep_mask]
        s["fold_label"] = s["fold_label"][keep_mask]
        s["years"] = s["years"][keep_mask] if years is not None else None

    # ----------------------------
    # Step B: per-year background downsampling vs signal, per background class
    # ----------------------------
    if sample_bkg is not None and len(shards) >= 2:
        # collect per-year signal counts AFTER year balancing
        sig_shard = shards[0]
        sig_years = sig_shard["years"]
        if sig_years is not None:
            uniq, cnt = np.unique(sig_years, return_counts=True)
            n_sig_year = {int(u): int(c) for u, c in zip(uniq, cnt)}
        else:
            n_sig_year = {-1: sig_shard["y"].size}  # single bucket

        # apply per background shard
        for s in shards[1:]:
            years = s["years"]
            if years is None:
                # global bucket
                target_total = int(round(sample_bkg * n_sig_year.get(-1, 0)))
                idx_all = np.arange(s["y"].size, dtype=np.int64)
                k = min(target_total, idx_all.size)
                keep_idx = idx_all[:k]
                keep_mask = np.zeros(idx_all.size, dtype=bool)
                keep_mask[keep_idx] = True
                keptB = {-1: int(np.sum(keep_mask))}
                availB = {-1: int(idx_all.size)}
                targetB = {-1: int(target_total)}
            else:
                keep_mask = np.zeros(s["y"].size, dtype=bool)
                keptB, availB, targetB = {}, {}, {}
                for yi in np.unique(years):
                    yi = int(yi)
                    idx = np.flatnonzero(years == yi)
                    availB[yi] = int(idx.size)
                    target = int(round(sample_bkg * n_sig_year.get(yi, 0)))
                    targetB[yi] = int(target)
                    if idx.size == 0 or target <= 0:
                        continue
                    k = min(target, idx.size)
                    sel = idx[:k]
                    keep_mask[sel] = True
                for yi in np.unique(years):
                    yi = int(yi)
                    keptB[yi] = int(np.sum(keep_mask[years == yi]))
                    
            sumW_B = {}
            if s["years"] is None:
                sumW_B[-1] = float(np.sum(s["w"][keep_mask], dtype=np.float64))
            else:
                years = s["years"]
                for yi in np.unique(years):
                    yi = int(yi)
                    sel = (years == yi) & keep_mask
                    sumW_B[yi] = float(np.sum(s["w"][sel], dtype=np.float64))

            s.setdefault("stats", {})["bkg_sample"] = {
                "sig_counts": n_sig_year,
                "target": targetB,
                "avail":  availB,
                "kept":   keptB,
                "sample_bkg": float(sample_bkg),
                "sumW":   sumW_B,   # 👈 추가: per-year sum of weights (kept after Step B)
            }

            # apply selection
            for kf in list(s["features"].keys()):
                s["features"][kf] = s["features"][kf][keep_mask]
            s["y"] = s["y"][keep_mask]
            s["w"] = s["w"][keep_mask]
            s["fold_label"] = s["fold_label"][keep_mask]
            s["years"] = s["years"][keep_mask] if s["years"] is not None else None
    # ----------------------------
    # Concatenate shards (features aligned by column name order)
    # ----------------------------
    if not shards:
        raise RuntimeError("No data remaining after selections.")

    features_all = list(shards[0]["features"].keys())
    feat_chunks = [{k: s["features"][k] for k in features_all} for s in shards]
    X_all = _concat_dicts(feat_chunks) if len(feat_chunks) > 1 else feat_chunks[0]
    y_all = np.concatenate([s["y"] for s in shards], axis=0)
    w_all = np.concatenate([s["w"] for s in shards], axis=0)
    folds_lab = np.concatenate([s["fold_label"] for s in shards], axis=0)

    # categorical encoding discovery (lightweight heuristic)
    features = list(X_all.keys())
    # feature matrix (float32)
    Xmat = np.column_stack([X_all[c] for c in features]).astype(np.float32, copy=False)
    if categorical_columns is None:
        categorical_columns = []
        categorical_dims = {}
    if "year_index" in features and "year_index" not in categorical_columns:
        categorical_columns.append("year_index")
        categorical_dims["year_index"] = len(YEAR_MAP)
        
    print("\033[31m" + f"[load_root_as_dataset_kfold] Categorical columns: {categorical_columns}" + "\033[0m")
    print("\033[31m" + f"[load_root_as_dataset_kfold] Categorical dims: {categorical_dims}" + "\033[0m")
    
    # class weights (sum of event weights per class → inverse proportional)
    total_w = float(np.sum(w_all))+1e-12
    sumW = []
    count = []
    C = len(tree_path_filter_str)
    final_counts_by_class = {}
    for cls in range(C):
        mask_c = (y_all == cls)
        n_c = int(np.sum(mask_c))
        final_counts_by_class[cls] = n_c
        count.append(n_c)
        sumW.append(float(np.sum(w_all[mask_c])))
    class_weight = (total_w / np.asarray(sumW, dtype=np.float64)).astype(np.float32)

    # Build explicit (train_idx, val_idx) lists for K folds (deterministic)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    N = y_all.size
    idx_all = np.arange(N, dtype=np.int64)
    for k in range(n_splits):
        val_idx = idx_all[folds_lab == k]
        train_idx = idx_all[folds_lab != k]
        folds.append((train_idx, val_idx))

    data = {
        "X": Xmat,
        "y": y_all,
        "weight": w_all.astype(np.float32, copy=False),
        "features": features,
        "folds": folds,  
        "seed": seed,
        "n_splits": n_splits,
        "cat_idxs": [i for i, f in enumerate(features) if f in categorical_columns],
        "cat_dims": [int(categorical_dims[f]) for f in features if f in categorical_columns],
        "cat_columns": list(categorical_columns),
        "sumW": sumW,
        "count": count,
        "class_weight": class_weight,
        # debug payload for rich summary
        "_debug": {
            "LUMI_MAP": LUMI_MAP,
            "YEAR_MAP": YEAR_MAP,
            "IDX_TO_YEAR": IDX_TO_YEAR,
            "per_class": [
                {
                    "class_idx": s["class_idx"],
                    "class_label": s.get("class_label", f"class{s['class_idx']}") ,
                    "stage0": s.get("stats", {}).get("stage0", {}),
                    "year_balance": s.get("stats", {}).get("year_balance", None),
                    "bkg_sample": s.get("stats", {}).get("bkg_sample", None),
                }
                for s in shards
            ],
            "final_counts_by_class": final_counts_by_class,
            "sample_bkg": float(sample_bkg) if sample_bkg is not None else None,
        },
    }
    render_downsample_summary(data)
    return data

# ----------------------------
# Save and load dataset
# ----------------------------

def _make_folds_lab_from_folds(folds: List[Tuple[np.ndarray, np.ndarray]], N: int) -> np.ndarray:
    """folds -> 샘플별 폴드 라벨(0..K-1). 누락 인덱스가 있으면 에러."""
    lab = np.full(N, -1, dtype=np.int16)
    for k, (_, val_idx) in enumerate(folds):
        lab[val_idx] = k
    if (lab < 0).any():
        miss = int(np.sum(lab < 0))
        raise RuntimeError(f"folds_lab 생성 실패: 라벨 미지정 인덱스 {miss}개")
    return lab

def save_dataset_npz_json(path_prefix: str, data: Dict[str, Any], *, include_debug_minimal: bool = False) -> None:
    """
    핵심 숫자 배열은 {prefix}.npz, 문자열/메타는 {prefix}.json 저장.
    - folds(가변 리스트)는 저장하지 않고 folds_lab(int16)만 저장 → 로드시 복원
    """
    N = int(data["y"].shape[0])

    # folds_lab 준비 (이미 있으면 사용, 없으면 folds로부터 생성)
    folds_lab = data.get("folds_lab", None)
    if folds_lab is None:
        folds = data.get("folds", None)
        if folds is None:
            raise KeyError("data에 'folds'도 'folds_lab'도 없습니다.")
        folds_lab = _make_folds_lab_from_folds(folds, N)

    # ---- .npz: 숫자 배열만 저장 (pickle 불필요)
    np.savez_compressed(
        path_prefix + "/data.npz",
        X=data["X"].astype(np.float32, copy=False),
        y=data["y"].astype(np.int64, copy=False),
        weight=data["weight"].astype(np.float32, copy=False),
        folds_lab=folds_lab.astype(np.int16, copy=False),
        n_splits=np.int16(data["n_splits"]),
        seed=np.int64(data["seed"]),
        cat_idxs=np.asarray(data["cat_idxs"], dtype=np.int32),
        cat_dims=np.asarray(data["cat_dims"], dtype=np.int32),
        class_weight=np.asarray(data["class_weight"], dtype=np.float32),
        sumW=np.asarray(data["sumW"], dtype=np.float64),
        count=np.asarray(data["count"], dtype=np.int64),
    )

    # ---- .json: 문자열/리스트 메타만 저장
    meta = {
        "features": list(data["features"]),
        "cat_columns": list(data.get("cat_columns", [])),
    }
    if include_debug_minimal:
        dbg = data.get("_debug", {})
        meta["_debug"] = {
            "final_counts_by_class": dbg.get("final_counts_by_class", {}),
            "sample_bkg": dbg.get("sample_bkg", None),
        }

    with open(path_prefix + "/data.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def load_dataset_npz_json(path_prefix: str) -> Dict[str, Any]:
    """
    {prefix}.npz + {prefix}.json 로딩 후, folds를 재구성해 원래 스키마로 반환.
    """
    npz = np.load(path_prefix + "/data.npz", allow_pickle=False)
    with open(path_prefix + "/data.json", "r") as f:
        meta = json.load(f)

    X = npz["X"]
    y = npz["y"]
    w = npz["weight"]
    folds_lab = npz["folds_lab"]
    n_splits = int(npz["n_splits"])
    seed = int(npz["seed"])
    cat_idxs = npz["cat_idxs"]
    cat_dims = npz["cat_dims"]
    class_weight = npz["class_weight"]
    sumW = npz["sumW"]
    count = npz["count"]

    idx_all = np.arange(y.size, dtype=np.int64)
    folds = [(idx_all[folds_lab != k], idx_all[folds_lab == k]) for k in range(n_splits)]

    data = {
        "X": X,
        "y": y,
        "weight": w,
        "features": meta["features"],
        "folds": folds,
        "seed": seed,
        "n_splits": n_splits,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
        "cat_columns": meta.get("cat_columns", []),
        "class_weight": class_weight,
        "sumW": sumW,
        "count": count,
    }
    return data

# ----------------------------
# Pretty Rich summary
# ----------------------------

def render_downsample_summary(dataset: Dict[str, np.ndarray], console=None) -> None:
    """Pretty-print a two-part summary (Year balancing & Per-year bkg sampling) with **rich**.

    Usage:
        >>> from root_data_loader_awk import render_downsample_summary
        >>> render_downsample_summary(dataset)
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.columns import Columns
        from rich import box
    except Exception as e:
        # graceful fallback
        print("[rich not available]", str(e))
        dbg = dataset.get("_debug", {})
        print("Year balancing targets/kept per class:")
        for c in dbg.get("per_class", []):
            yb = c.get("year_balance")
            if not yb:
                continue
            print(f" - class {c['class_idx']} ({c.get('class_label','')}):")
            for yi, tgt in yb.get("target", {}).items():
                kept = yb.get("kept", {}).get(yi, 0)
                print(f"    year {yi}: kept {kept} / target {tgt} (avail={yb['avail'].get(yi,0)})")
        print("Per-year bkg sampling:")
        for c in dbg.get("per_class", []):
            bk = c.get("bkg_sample")
            if not bk:
                continue
            print(f" - class {c['class_idx']} ({c.get('class_label','')}): sample_bkg={bk.get('sample_bkg')}")
            for yi, tgt in bk.get("target", {}).items():
                kept = bk.get("kept", {}).get(yi, 0)
                sig = bk.get("sig_counts", {}).get(yi, 0)
                print(f"    year {yi}: kept {kept} / target {tgt} (sig={sig}, avail={bk['avail'].get(yi,0)})")
        return

    if console is None:
        console = Console(width=120)

    dbg = dataset.get("_debug", {})
    IDX_TO_YEAR = dbg.get("IDX_TO_YEAR", {})

    # ---------- Table A: Year balancing ----------
    tA = Table(title="Year Balancing to LUMI ratios", box=box.SIMPLE_HEAVY)
    tA.add_column("Class", justify="left")
    tA.add_column("Year", justify="center")
    tA.add_column("Lumi frac", justify="right")
    tA.add_column("Avail", justify="right")
    tA.add_column("Target", justify="right")
    tA.add_column("Kept", justify="right")
    tA.add_column("Keep%", justify="right")
    tA.add_column("sumW", justify="right")

    for c in dbg.get("per_class", []):
        yb = c.get("year_balance")
        if not yb:
            continue
        label = f"[{c['class_idx']}] {c.get('class_label','')}"
        targets = yb.get("target", {})
        for yi in sorted(targets.keys()):
            year_name = IDX_TO_YEAR.get(int(yi), str(yi))
            frac = yb.get("fracs", {}).get(int(yi), 0.0)
            avail = yb.get("avail", {}).get(int(yi), 0)
            tgt = targets.get(int(yi), 0)
            kept = yb.get("kept", {}).get(int(yi), 0)
            keep_pct = (100.0 * kept / avail) if avail else 0.0
            sumW = float(b) if (b := yb.get("sumW", {}).get(int(yi), None)) is not None else 0.0
            tA.add_row(label, year_name, f"{frac*100:.1f}%", f"{avail:,}", f"{tgt:,}", f"{kept:,}", f"{keep_pct:5.1f}%", f"{sumW:,.1f}")
            label = ""  # only show for first row per class

    # ---------- Table B: Per-year background sampling ----------
    tB = Table(title="Per-year Background Sampling vs Signal", box=box.SIMPLE_HEAVY)
    tB.add_column("Bkg Class", justify="left")
    tB.add_column("Year", justify="center")
    tB.add_column("sample_bkg", justify="right")
    tB.add_column("Sig N(year)", justify="right")
    tB.add_column("Avail(bkg)", justify="right")
    tB.add_column("Target", justify="right")
    tB.add_column("Kept", justify="right")
    tB.add_column("Keep%", justify="right")
    tB.add_column("sumW", justify="right")

    for c in dbg.get("per_class", [])[1:]:  # only backgrounds
        bk = c.get("bkg_sample")
        if not bk:
            continue
        label = f"[{c['class_idx']}] {c.get('class_label','')}"
        targets = bk.get("target", {})
        for yi in sorted(targets.keys()):
            year_name = IDX_TO_YEAR.get(int(yi), str(yi))
            sigN = bk.get("sig_counts", {}).get(int(yi), 0)
            avail = bk.get("avail", {}).get(int(yi), 0)
            tgt = targets.get(int(yi), 0)
            kept = bk.get("kept", {}).get(int(yi), 0)
            keep_pct = (100.0 * kept / avail) if avail else 0.0
            sumW = float(b) if (b := bk.get("sumW", {}).get(int(yi), None)) is not None else 0.0
            tB.add_row(label, year_name, f"{bk.get('sample_bkg', 0.0):.2f}", f"{sigN:,}", f"{avail:,}", f"{tgt:,}", f"{kept:,}", f"{keep_pct:5.1f}%", f"{sumW:,.1f}")
            label = ""

    # ---------- Header panel with totals ----------
    totals = Table.grid(padding=(0,1))
    totals.add_column(justify="left")
    totals.add_column(justify="right")
    totals.add_row("Classes", str(len(dbg.get("per_class", []))))
    totals.add_row("Total events (final)", f"{dataset['y'].size:,}")
    finals = dbg.get("final_counts_by_class", {})
    txt = ", ".join([f"[{k}]={v:,}" for k, v in finals.items()])
    totals.add_row("Per-class final N", txt)

    panel_totals = Panel(totals, title="Dataset Totals", box=box.SQUARE, padding=(1,2))

    console.print(panel_totals)
    console.print(Columns([tA, tB], expand=True))
