#!/usr/bin/env python3
"""
Refactored Multi-class weight optimizer for Vcb vs TTLJ.

Added now:
- Keeps **top-K** configurations (default K=5) and prints them in score order.
- **Plots class-wise scores** (per-class probabilities from your log-softmax) for Vcb vs TTLJ.

Usage examples
--------------
Grid search with plots:
  python MultiClassOptimizer_refactored.py \
    --mode c \
    --ttlj "/gv0/.../Vcb_TTLJ_powheg.root:Mu/Central/Result_Tree" \
    --vcb  "/gv0/.../Vcb_TTLJ_WtoCB_powheg.root:Mu/Central/Result_Tree" \
    --grid --grid-steps 25 --plot-dir plots --topk 5

Ray+Optuna (if available):
  python MultiClassOptimizer_refactored.py \
    --mode b \
    --ttlj "/gv0/.../Vcb_TTLJ_powheg.root:Mu/Central/Result_Tree" \
    --vcb  "/gv0/.../Vcb_TTLJ_WtoCB_powheg.root:Mu/Central/Result_Tree" \
    --samples 800 --num-cpus 16 --plot-dir plots_b --topk 5
"""
from __future__ import annotations
import os
import sys
import math
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Optional, List
import heapq

import numpy as np

# --- Optional deps ---
try:
    import ray
    from ray import tune
    from ray.tune import Tuner, TuneConfig
    from ray.tune.search.optuna import OptunaSearch
    RAY_OK = True
except Exception:
    RAY_OK = False

try:
    import uproot
    import awkward as ak
except Exception as e:
    print("[ERROR] This script requires uproot and awkward. pip install uproot awkward", file=sys.stderr)
    raise

# ----------------------------
# Math helpers
# ----------------------------

def _logsumexp(a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    s = np.log(np.sum(np.exp(a - m_safe), axis=axis, keepdims=True)) + m_safe
    return s if keepdims else np.squeeze(s, axis=axis)


def _expit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def log_prob_to_weighted_prob(log_prob: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Convert K-class log-probabilities (log-softmax) to a single binary-like "signal" score P_w(sig|x)
    using class-importance weights w (w0 corresponds to class 0 being the signal class).

    P_w(sig|x) = σ( log(w0 p0) - log Σ_{j≠0} (wj pj) )
    """
    logp = np.asarray(log_prob, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if logp.ndim != 2:
        raise ValueError("log_prob must be 2-D (N, K)")
    N, K = logp.shape
    if w.shape != (K,):
        raise ValueError(f"weights must have shape ({K},)")
    logw = np.log(np.maximum(w, 1e-300))
    s_num = logw[0] + logp[:, 0]
    s_den = _logsumexp(logp[:, 1:] + logw[1:][None, :], axis=1)
    s = s_num - s_den
    return _expit(s)


def asimov_Z(s, b, *, rel_b: float | np.ndarray = 0.0, include_systematics: bool = False,
             clip_nonneg: bool = True, eps: float = 1e-12) -> np.ndarray:
    """Vectorized Asimov Z.
    - include_systematics=False: Z0 = sqrt(2 * [(s+b) ln(1+s/b) - s])
    - include_systematics=True : Cowan+ (2011) eq. (71) with σ_b = rel_b * b
    """
    s = np.asarray(s, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if clip_nonneg:
        s = np.maximum(s, 0.0)
        b = np.maximum(b, 0.0)

    def _Z0(ss, bb):
        with np.errstate(divide='ignore', invalid='ignore'):
            val = 2.0 * ((ss + bb) * np.log(1.0 + ss / (bb + eps)) - ss)
        return np.sqrt(np.maximum(val, 0.0))

    if not include_systematics:
        Z = _Z0(s, b)
        Z[~np.isfinite(Z)] = 0.0
        return Z

    rb = np.asarray(rel_b, dtype=np.float64)
    sigma2 = (rb * b) ** 2

    Z2 = np.zeros_like(s, dtype=np.float64)
    mask = sigma2 > eps
    if np.any(mask):
        ss = s[mask]; bb = b[mask]; sig2 = sigma2[mask]
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = (ss + bb) * np.log(((ss + bb) * (bb + sig2)) / (bb * bb + (ss + bb) * sig2) + eps)
            term2 = (bb * bb / sig2) * np.log(1.0 + (sig2 * ss) / (bb * (bb + sig2) + eps))
            Z2[mask] = 2.0 * np.maximum(0.0, term1 - term2)
    if np.any(~mask):
        Z2[~mask] = (_Z0(s[~mask], b[~mask])) ** 2

    Z = np.sqrt(np.maximum(Z2, 0.0))
    Z[~np.isfinite(Z)] = 0.0
    return Z


def _cum_from_right(h: np.ndarray) -> np.ndarray:
    return np.cumsum(h[::-1], dtype=np.float64)[::-1]

# ----------------------------
# Data & weights
# ----------------------------

CLASSIF_KEYS = [
    "template_score_MultiClass_log_prob_0",
    "template_score_MultiClass_log_prob_1",
    "template_score_MultiClass_log_prob_2",
    "template_score_MultiClass_log_prob_3",
]

BASE_BRANCHES_COMMON = [
    "weight_el_id", "weight_mc", "weight_lumi", "weight_mu_id", "weight_mu_iso",
    "weight_el_reco", "weight_pileup", "weight_prefire", "weight_sl_trig", "weight_top_pt",
    "genTtbarId",
]

B_MODE_BRANCHES = [
    "weight_b_tag", "weight_b_tag_up_cferr1", "weight_b_tag_up_cferr2", "weight_b_tag_up_hf",
]

C_MODE_BRANCHES = [
    "weight_c_tag",
    "weight_c_tag_up_stat",
    "weight_c_tag_up_xsec_brunc_dyjets_b",
    "weight_c_tag_up_xsec_brunc_dyjets_c",
    "weight_c_tag_up_xsec_brunc_wjets_c",
]

@dataclass
class Paths:
    ttlj: str
    vcb: str


def need_branches(mode: str) -> Iterable[str]:
    out = list(BASE_BRANCHES_COMMON) + list(CLASSIF_KEYS)
    if mode == 'b':
        out += B_MODE_BRANCHES
    elif mode == 'c':
        out += C_MODE_BRANCHES
    else:
        raise ValueError("mode must be 'b' or 'c'")
    return out


def open_tree(spec: str):
    """Open a tree given "file.root:tree/path" or separate args."""
    if ":" not in spec:
        raise ValueError("Path must look like '/path/file.root:Tree/Path'")
    return uproot.open(spec)


def load_arrays(spec: str, branches: Iterable[str]) -> Dict[str, np.ndarray]:
    tree = open_tree(spec)
    arrs = tree.arrays(list(branches), library="ak")
    out = {}
    for k in arrs.fields:
        out[k] = arrs[k].to_numpy()
    return out


def build_weights_dict(arr: Dict[str, np.ndarray], mode: str) -> Dict[str, np.ndarray]:
    d = dict(arr)
    # masks for heavy-flavor components
    if "genTtbarId" in d:
        ttbar = d["genTtbarId"]
        d["TTBB_Mask"] = (ttbar % 100 >= 51) & (ttbar % 100 <= 55)
        d["TTCC_Mask"] = (ttbar % 100 >= 41) & (ttbar % 100 <= 45)
    else:
        any_key = next(iter(d))
        d["TTBB_Mask"] = d["TTCC_Mask"] = np.zeros_like(d[any_key], dtype=bool)

    # base
    base_common = (
        d["weight_el_id"] * d["weight_mc"] * d["weight_lumi"] * d["weight_mu_id"] * d["weight_mu_iso"] *
        d["weight_el_reco"] * d["weight_pileup"] * d["weight_prefire"] * d["weight_sl_trig"] * d["weight_top_pt"]
    )

    if mode == 'b':
        base = d["weight_b_tag"] * base_common
        d["weight_base"] = base
        d["weight_cferr1_up"] = base * d["weight_b_tag_up_cferr1"] / np.maximum(d["weight_b_tag"], 1e-12)
        d["weight_cferr2_up"] = base * d["weight_b_tag_up_cferr2"] / np.maximum(d["weight_b_tag"], 1e-12)
        d["weight_hf_up"]     = base * d["weight_b_tag_up_hf"]    / np.maximum(d["weight_b_tag"], 1e-12)
        d["weight_top_pt_up"]  = base / np.maximum(d["weight_top_pt"], 1e-12)
    else:
        base = d["weight_c_tag"] * base_common
        d["weight_base"] = base
        d["weight_c_tag_stat_up"] = base / np.maximum(d["weight_c_tag"], 1e-12) * d["weight_c_tag_up_stat"]
        d["weight_c_tag_xsec_dyjets_b_up"] = base / np.maximum(d["weight_c_tag"], 1e-12) * d["weight_c_tag_up_xsec_brunc_dyjets_b"]
        d["weight_c_tag_xsec_dyjets_c_up"] = base / np.maximum(d["weight_c_tag"], 1e-12) * d["weight_c_tag_up_xsec_brunc_dyjets_c"]
        d["weight_c_tag_xsec_wjets_c_up"] = base / np.maximum(d["weight_c_tag"], 1e-12) * d["weight_c_tag_up_xsec_brunc_wjets_c"]

    # flavor fractions
    d["weight_ttbb_up"] = base * np.where(d["TTBB_Mask"], 1.1, 1.0)
    d["weight_ttcc_up"] = base * np.where(d["TTCC_Mask"], 1.15, 1.0)

    # classif logits
    for i, key in enumerate(CLASSIF_KEYS):
        d[f"classif_{i}"] = d[key]

    return d

# ----------------------------
# Evaluation core
# ----------------------------

def make_signal_score(arr: Dict[str, np.ndarray], w: np.ndarray) -> np.ndarray:
    logp = np.vstack([arr['classif_0'], arr['classif_1'], arr['classif_2'], arr['classif_3']]).T
    return log_prob_to_weighted_prob(logp, w)


def cum_hists(prob: np.ndarray, bins: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, _ = np.histogram(prob, bins=bins, weights=weights)
    h2, _ = np.histogram(prob, bins=bins, weights=weights**2)
    nmc, _ = np.histogram(prob, bins=bins, weights=np.ones_like(prob))
    return _cum_from_right(h), _cum_from_right(h2), _cum_from_right(nmc)


def rel_b_from_systematics(mode: str, TTLJ: Dict[str, np.ndarray], TTLJ_prob: np.ndarray, bins: np.ndarray,
                            B_cum: np.ndarray) -> np.ndarray:
    # Build list of systematic variations that affect the BACKGROUND (TTLJ)
    if mode == 'b':
        systs = [
            ("cferr1", TTLJ.get("weight_cferr1_up")),
            ("cferr2", TTLJ.get("weight_cferr2_up")),
            ("hf",     TTLJ.get("weight_hf_up")),
            ("top_pt", TTLJ.get("weight_top_pt_up")),
            ("ttbb",   TTLJ.get("weight_ttbb_up")),
            ("ttcc",   TTLJ.get("weight_ttcc_up")),
        ]
    else:
        systs = [
            ("c_stat",             TTLJ.get("weight_c_tag_stat_up")),
            ("c_brunc_dyjets_b",   TTLJ.get("weight_c_tag_xsec_dyjets_b_up")),
            ("c_brunc_dyjets_c",   TTLJ.get("weight_c_tag_xsec_dyjets_c_up")),
            ("c_brunc_wjets_c",    TTLJ.get("weight_c_tag_xsec_wjets_c_up")),
            ("ttbb",               TTLJ.get("weight_ttbb_up")),
            ("ttcc",               TTLJ.get("weight_ttcc_up")),
        ]

    diffs2 = []
    for name, wvar in systs:
        if wvar is None:
            continue
        h, _ = np.histogram(TTLJ_prob, bins=bins, weights=wvar)
        B_var = _cum_from_right(h)
        diffs2.append((B_var - B_cum)**2)
    if len(diffs2) == 0:
        return np.zeros_like(B_cum, dtype=np.float64)
    sigma_b = np.sqrt(np.sum(diffs2, axis=0))
    rel_b = sigma_b / np.maximum(B_cum, 1e-12)
    return rel_b


def evaluate_config_np(config: Dict[str, float], Vcb: Dict[str, np.ndarray], TTLJ: Dict[str, np.ndarray],
                       bins: np.ndarray, mode: str) -> Dict[str, float]:
    w = np.array([1.0, config["w1"], config["w2"], config["w3"]], dtype=np.float64)
    Vcb_prob  = make_signal_score(Vcb, w)
    TTLJ_prob = make_signal_score(TTLJ, w)

    S_cum, S2_cum, S_nmc = cum_hists(Vcb_prob,  bins, Vcb['weight_base'])
    B_cum, B2_cum, B_nmc = cum_hists(TTLJ_prob, bins, TTLJ['weight_base'])

    rel_b = rel_b_from_systematics(mode, TTLJ, TTLJ_prob, bins, B_cum)

    Z_curve = asimov_Z(S_cum, B_cum, rel_b=rel_b, include_systematics=True)
    valid = np.isfinite(Z_curve) & (S_cum > 0) & (B_cum > 0)
    if not np.any(valid):
        return {"best_Z": -np.inf}

    i_best = int(np.nanargmax(np.where(valid, Z_curve, -np.inf)))
    return {
        "best_Z": float(Z_curve[i_best]),
        "thr":    float(bins[i_best]),
        "S":      float(S_cum[i_best]),
        "B":      float(B_cum[i_best]),
        "rel_b":  float(rel_b[i_best]),
        "w0":     1.0,
        "w1":     float(w[1]),
        "w2":     float(w[2]),
        "w3":     float(w[3]),
    }

# ----------------------------
# Ray-compatible wrapper
# ----------------------------

def evaluate_config(config, Vcb_ref=None, TTLJ_ref=None, bins_ref=None, working_mode='b'):
    Vcb  = ray.get(Vcb_ref)  if RAY_OK and hasattr(ray, 'ObjectRef') and isinstance(Vcb_ref,  ray.ObjectRef) else Vcb_ref
    TTLJ = ray.get(TTLJ_ref) if RAY_OK and hasattr(ray, 'ObjectRef') and isinstance(TTLJ_ref, ray.ObjectRef) else TTLJ_ref
    bins = ray.get(bins_ref) if RAY_OK and hasattr(ray, 'ObjectRef') and isinstance(bins_ref, ray.ObjectRef) else bins_ref
    res = evaluate_config_np(config, Vcb, TTLJ, bins, working_mode)
    if RAY_OK:
        from ray.air import session
        session.report(res)
    return res

# ----------------------------
# Search utilities (now keep top-K)
# ----------------------------

class TopK:
    def __init__(self, k: int):
        self.k = k
        self.heap: List[Tuple[float, Dict[str, float]]] = []  # min-heap by best_Z
    def push(self, item: Dict[str, float]):
        key = item.get("best_Z", -np.inf)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (key, item))
        else:
            if key > self.heap[0][0]:
                heapq.heapreplace(self.heap, (key, item))
    def results(self) -> List[Dict[str, float]]:
        return [x[1] for x in sorted(self.heap, key=lambda t: -t[0])]


def get_grids(dim: int, low: float, high: float, res: int = 10) -> np.ndarray:
    base_grid = np.linspace(low, high, res)
    from itertools import product
    return np.array(list(product(base_grid, repeat=dim)))


def run_grid_search(Vcb: Dict[str, np.ndarray], TTLJ: Dict[str, np.ndarray], bins: np.ndarray, mode: str,
                    low: float = 0.0, high: float = 10.0, steps: int = 21, topk: int = 5) -> Dict[str, object]:
    grid = get_grids(3, low, high, steps)
    best = None
    tk = TopK(topk)
    for w1, w2, w3 in grid:
        res = evaluate_config_np({"w1": float(w1), "w2": float(w2), "w3": float(w3)}, Vcb, TTLJ, bins, mode)
        if best is None or res["best_Z"] > best["best_Z"]:
            best = res
        tk.push(res)
    return {"best": best, "topk": tk.results()}


def run_ray_optuna(Vcb: Dict[str, np.ndarray], TTLJ: Dict[str, np.ndarray], bins: np.ndarray, mode: str,
                   samples: int = 500, num_cpus: Optional[int] = None, topk: int = 5) -> Dict[str, object]:
    if not RAY_OK:
        raise RuntimeError("Ray/Optuna not available")
    ray.shutdown()
    tmp = os.environ.get("RAY_TMPDIR", None)
    if tmp is None:
        os.environ["RAY_TMPDIR"] = "/tmp/ray"
    ray.init(include_dashboard=False, num_cpus=(num_cpus or os.cpu_count() or 2))

    Vcb_ref  = ray.put(Vcb)
    TTLJ_ref = ray.put(TTLJ)
    bins_ref = ray.put(bins)

    A = 20.0
    param_space = {"w1": tune.uniform(0, A), "w2": tune.uniform(0, A), "w3": tune.uniform(0, A)}

    def trainable(config):
        return evaluate_config(config, Vcb_ref=Vcb_ref, TTLJ_ref=TTLJ_ref, bins_ref=bins_ref, working_mode=mode)

    tuner = Tuner(
        tune.with_parameters(trainable),
        param_space=param_space,
        tune_config=TuneConfig(
            search_alg=OptunaSearch(metric="best_Z", mode="max"),
            num_samples=samples,
            metric="best_Z",
            mode="max",
        ),
    )
    results = tuner.fit()
    # Collect all results and pick top-k
    all_res: List[Dict[str, float]] = []
    for r in results:
        # r.metrics holds our returned dict
        m = r.metrics
        if isinstance(m, dict) and "best_Z" in m:
            all_res.append({k: m.get(k) for k in ("best_Z","thr","S","B","rel_b","w0","w1","w2","w3")})
    all_res = [x for x in all_res if x.get("best_Z", -np.inf) is not None]
    all_res.sort(key=lambda d: d.get("best_Z", -np.inf), reverse=True)
    topk_res = all_res[:topk]

    best = topk_res[0] if topk_res else results.get_best_result(metric="best_Z", mode="max").metrics
    ray.shutdown()

    return {"best": best, "topk": topk_res}

# ----------------------------
# Plotting: class-wise score (per-class probabilities)
# ----------------------------

def _ensure_dir(p: str):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


def _class_probs(arr: Dict[str, np.ndarray]) -> np.ndarray:
    """Return (N,4) probabilities from stored log-probs."""
    logp = np.vstack([arr['classif_0'], arr['classif_1'], arr['classif_2'], arr['classif_3']]).T
    # logp are log-softmax outputs; exponentiate to get probs
    p = np.exp(logp)
    # numerical guard: renormalize small drift
    s = np.sum(p, axis=1, keepdims=True)
    s = np.where(s>0, s, 1.0)
    return p / s


def plot_class_scores(Vcb: Dict[str, np.ndarray], TTLJ: Dict[str, np.ndarray], outdir: str, bins: int = 50,
                      tag: str = "") -> List[str]:
    import matplotlib.pyplot as plt
    _ensure_dir(outdir)
    paths = []
    p_v = _class_probs(Vcb)
    p_b = _class_probs(TTLJ)
    w_v = Vcb['weight_base']
    w_b = TTLJ['weight_base']

    # One figure per class (no subplots)
    for k in range(4):
        fig = plt.figure()
        # Hist overlay: TTLJ vs Vcb
        # Use step histograms; do not set explicit colors (follow default policy)
        plt.hist(p_b[:,k], bins=bins, weights=w_b, histtype='step', label=f'Bkg TTLJ - class{k}', density=True)
        plt.hist(p_v[:,k], bins=bins, weights=w_v, histtype='step', label=f'Sig Vcb - class{k}', density=True)
        plt.xlabel(f'Class-{k} probability')
        plt.ylabel('Weighted yield')
        ttl = f'Class-{k} score distribution' + (f' [{tag}]' if tag else '')
        plt.title(ttl)
        plt.legend()
        out = os.path.join(outdir, f'class_{k}_score{("_"+tag) if tag else ""}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths.append(out)
    return paths

# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Optimize class weights to maximize Asimov Z with systematics.")
    ap.add_argument('--mode', choices=['b','c'], required=True, help="Choose b-tag or c-tag systematic set")
    ap.add_argument('--ttlj', required=True, help="Path to TTLJ tree, e.g. file.root:Tree/Path")
    ap.add_argument('--vcb',  required=True, help="Path to Vcb  tree, e.g. file.root:Tree/Path")
    ap.add_argument('--bins', type=int, default=1000, help="Number of bins in [0,1] for score histogram")
    ap.add_argument('--samples', type=int, default=500, help="Optuna samples (Ray only)")
    ap.add_argument('--grid', action='store_true', help="Use pure grid search instead of Ray/Optuna")
    ap.add_argument('--grid-steps', type=int, default=21, help="Grid steps per dimension if --grid")
    ap.add_argument('--grid-low', type=float, default=0.0)
    ap.add_argument('--grid-high', type=float, default=10.0)
    ap.add_argument('--num-cpus', type=int, default=None, help="Ray num_cpus override")
    ap.add_argument('--topk', type=int, default=5, help='How many best configs to print')
    ap.add_argument('--plot-dir', type=str, default=None, help='If set, write per-class score plots here')
    return ap.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    # Load
    branches = need_branches(mode)
    ttlj_arr = load_arrays(args.ttlj, branches)
    vcb_arr  = load_arrays(args.vcb,  branches)

    TTLJ = build_weights_dict(ttlj_arr, mode)
    Vcb  = build_weights_dict(vcb_arr,  mode)

    # Binning for score in [0,1]
    bins = np.linspace(0, 1, max(args.bins, 10))

    if args.grid or not RAY_OK:
        out = run_grid_search(Vcb, TTLJ, bins, mode, low=args.grid_low, high=args.grid_high, steps=args.grid_steps, topk=args.topk)
    else:
        out = run_ray_optuna(Vcb, TTLJ, bins, mode, samples=args.samples, num_cpus=args.num_cpus, topk=args.topk)

    best = out["best"]
    topk = out["topk"]

    # Print top-K neatly
    print("\n=== Top {} configs (by best_Z) ===".format(len(topk)))
    print(json.dumps(topk, indent=2))
    print("\n=== Best ===")
    print(json.dumps(best, indent=2))

    # Optional plotting: per-class score histograms (no seaborn, one figure per class)
    if args.plot_dir:
        tag = f"w=[1,{best['w1']:.3g},{best['w2']:.3g},{best['w3']:.3g}]"
        files = plot_class_scores(Vcb, TTLJ, args.plot_dir, bins=50, tag=tag)
        print("\nWrote class-wise plots:")
        for f in files:
            print(f" - {f}")


if __name__ == '__main__':
    main()
