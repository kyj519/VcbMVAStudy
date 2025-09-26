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

# >>> 파일 맨 위쪽, numpy/torch를 import 하기 전에 추가 <<<

from __future__ import annotations
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import os
import sys
import math
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Optional, List
import heapq
from datetime import datetime
import csv 
import numpy as np
from tqdm.rich import tqdm
from mcdata_io import load_signal_and_background
from helpers import log_prob_to_weighted_prob, asimov_Z, _cum_from_idx, _bin_indices, _zshape_asimov, _merge_until_valid, build_w_from_params, extract_logp



# --- Optional deps ---
try:
    import ray
    from ray import tune
    from ray.tune import Tuner, TuneConfig
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search import ConcurrencyLimiter  
    RAY_OK = True
except Exception:
    RAY_OK = False

try:
    import uproot
    import awkward as ak
except Exception as e:
    print("[ERROR] This script requires uproot and awkward. pip install uproot awkward", file=sys.stderr)
    raise

@dataclass
class Paths:
    ttlj: str
    vcb: str

def _to_py(obj):
    """JSON/CSV 저장용: numpy/awkward 타입을 파이썬 기본형으로 변환"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def save_summary(out_dir: str, args, best: dict, topk: list, tag: str = "") -> dict:
    _ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"summary_{ts}")

    summary_json = base + ".json"
    payload = {
        "meta": {"timestamp": ts, "tag": tag, "args": {k: _to_py(v) for k,v in vars(args).items()}},
        "best": {k: _to_py(v) for k,v in (best or {}).items()},
        "topk": [{k: _to_py(v) for k,v in d.items()} for d in (topk or [])],
    }
    with open(summary_json, "w") as f:
        json.dump(payload, f, indent=2)

    topk_csv = base + "_topk.csv"
    if topk:
        # w 길이가 항목마다 동일하다고 가정
        K = len(topk[0].get("w", []))
        cols = ["best_Z","thr","S","B","rel_b"] + [f"w{i}" for i in range(K)]
        with open(topk_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for row in topk:
                flat = {c: row.get(c, "") for c in ["best_Z","thr","S","B","rel_b"]}
                for i, wi in enumerate(row.get("w", [])):
                    flat[f"w{i}"] = _to_py(wi)
                w.writerow(flat)
    else:
        with open(topk_csv, "w", newline="") as f:
            csv.writer(f).writerow(["no_results"])

    best_txt = base + "_best.txt"
    with open(best_txt, "w") as f:
        if best:
            w = best.get("w", [])
            f.write(
                "Best Z={:.6g} @ thr={:.6g} | S={:.6g}, B={:.6g}, rel_b={:.6g} | w={}\n".format(
                    best.get("best_Z", float("nan")),
                    best.get("thr", float("nan")),
                    best.get("S", float("nan")),
                    best.get("B", float("nan")),
                    best.get("rel_b", float("nan")),
                    ",".join(f"{x:.6g}" for x in w)
                )
            )
        else:
            f.write("No best result.\n")
    return {"json": summary_json, "csv": topk_csv, "txt": best_txt}



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

# ----------------------------
# Evaluation core
# ----------------------------

def make_signal_score(arr: Dict[str, np.ndarray], w: np.ndarray) -> np.ndarray:
    logp = np.vstack([arr['classif_0'], arr['classif_1'], arr['classif_2'], arr['classif_3']]).T
    return log_prob_to_weighted_prob(logp, w)

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

# def evaluate_config_np(config, Vcb, TTLJ, bins, mode):
#     w = np.array([1.0, config["w1"], config["w2"], config["w3"]], dtype=np.float64)

#     Vcb_prob  = make_signal_score(Vcb,  w)   # (N_v,)
#     TTLJ_prob = make_signal_score(TTLJ, w)   # (N_b,)

#     nb = len(bins) - 1
#     idx_v = _bin_indices(Vcb_prob,  bins)
#     idx_b = _bin_indices(TTLJ_prob, bins)

#     # base S,B 누적합 (가중치 1개씩만 bincount)
#     S_cum = _cum_from_idx(idx_v, Vcb['weight_base'].astype(np.float32, copy=False), nb)
#     B_cum = _cum_from_idx(idx_b, TTLJ['weight_base'].astype(np.float32, copy=False), nb)
    
#     S_W2_cum = _cum_from_idx(idx_v, (Vcb['weight_base']**2).astype(np.float32, copy=False), nb)
#     B_W2_cum = _cum_from_idx(idx_b, (TTLJ['weight_base']**2).astype(np.float32, copy=False), nb)
    
#     S_neff_cum = (S_cum**2) / np.maximum(S_W2_cum, 1e-12)
#     B_neff_cum = (B_cum**2) / np.maximum(B_W2_cum, 1e-12)

#     # systematics: 같은 idx_b 재사용해서 변형 가중치만 바꿔 누적합
#     rel_b = np.zeros_like(B_cum, dtype=np.float64)
#     syst_names, syst_w = [], []
#     if mode == 'b':
#         syst_w = [
#           TTLJ.get("weight_cferr1_up"),
#           TTLJ.get("weight_cferr2_up"),
#           TTLJ.get("weight_hf_up"),
#           TTLJ.get("weight_top_pt_up"),
#           TTLJ.get("weight_ttbb_up"),
#           TTLJ.get("weight_ttcc_up"),
#         ]
#     else:
#         syst_w = [
#           TTLJ.get("weight_c_tag_stat_up"),
#           TTLJ.get("weight_c_tag_xsec_brunc_dyjets_b_up"),
#           TTLJ.get("weight_c_tag_xsec_brunc_dyjets_c_up"),
#           TTLJ.get("weight_c_tag_xsec_brunc_wjets_c_up"),
#           TTLJ.get("weight_ttbb_up"),
#           TTLJ.get("weight_ttcc_up"),
#         ]

#     diffs2 = []
#     for wv in syst_w:
#         if wv is None: 
#             continue
#         B_var = _cum_from_idx(idx_b, wv.astype(np.float32, copy=False), nb)
#         diffs2.append((B_var - B_cum)**2)

#     if diffs2:
#         sigma_b = np.sqrt(np.sum(diffs2, axis=0))
#         rel_b = sigma_b / np.maximum(B_cum, 1e-12)

#     Z_curve = asimov_Z(S_cum, B_cum, rel_b=rel_b, include_systematics=True)
#     valid = np.isfinite(Z_curve) & (S_cum > 0) & (B_cum > 0) & (S_neff_cum >= 20) & (B_neff_cum >= 20)

#     zshape = _zshape_asimov(Vcb_prob, TTLJ_prob, bins,
#                            Vcb['weight_base'], TTLJ['weight_base'], syst_w)
    
#     if not np.any(valid): 
#         return {"best_Z": -np.inf}

#     i_best = int(np.nanargmax(np.where(valid, Z_curve, -np.inf)))
#     return {
#         "best_Z":zshape,
#         #"zshape": float(zshape[i_best]),
#         "thr":    float(bins[i_best]),
#         "S":      float(S_cum[i_best]),
#         "B":      float(B_cum[i_best]),
#         "rel_b":  float(rel_b[i_best]),
#         "w0": 1.0, "w1": float(w[1]), "w2": float(w[2]), "w3": float(w[3]),
#     }
    
    
def evaluate_config_np(config, Vcb, TTLJ, bins, mode, sig_idx: int):
    # config는 다음 둘 중 하나를 허용:
    #  (a) full w: config["w"] (길이 K)
    #  (b) params: config["params"] (길이 K-1, 비신호 클래스 가중치)
    logp_V, _, _ = extract_logp(Vcb)
    logp_B, _, _ = extract_logp(TTLJ)
    K = logp_V.shape[1]

    if "w" in config:
        w = np.asarray(config["w"], dtype=np.float64)
        if w.size != K: 
            raise ValueError(f"w size {w.size} != K {K}")
    else:
        params = np.asarray(config["params"], dtype=np.float64)
        if params.size != K-1:
            raise ValueError(f"params size {params.size} != K-1 ({K-1})")
        w = build_w_from_params(params, K, sig_idx)

    Vcb_prob  = log_prob_to_weighted_prob(logp_V, w, sig_idx)
    TTLJ_prob = log_prob_to_weighted_prob(logp_B, w, sig_idx)

    nb = len(bins) - 1
    idx_v = _bin_indices(Vcb_prob,  bins)
    idx_b = _bin_indices(TTLJ_prob, bins)

    S_cum = _cum_from_idx(idx_v, Vcb['weight_base'].astype(np.float32, copy=False), nb)
    B_cum = _cum_from_idx(idx_b, TTLJ['weight_base'].astype(np.float32, copy=False), nb)
    S_W2_cum = _cum_from_idx(idx_v, (Vcb['weight_base']**2).astype(np.float32, copy=False), nb)
    B_W2_cum = _cum_from_idx(idx_b, (TTLJ['weight_base']**2).astype(np.float32, copy=False), nb)
    S_neff_cum = (S_cum**2) / np.maximum(S_W2_cum, 1e-12)
    B_neff_cum = (B_cum**2) / np.maximum(B_W2_cum, 1e-12)

    # systematics (원래 코드 그대로)
    syst_w = ([
        TTLJ.get("weight_cferr1_up"),
        TTLJ.get("weight_cferr2_up"),
        TTLJ.get("weight_hf_up"),
        TTLJ.get("weight_top_pt_up"),
        TTLJ.get("weight_ttbb_up"),
        TTLJ.get("weight_ttcc_up"),
    ] if mode=='b' else [
        TTLJ.get("weight_c_tag_stat_up"),
        TTLJ.get("weight_c_tag_xsec_brunc_dyjets_b_up"),
        TTLJ.get("weight_c_tag_xsec_brunc_dyjets_c_up"),
        TTLJ.get("weight_c_tag_xsec_brunc_wjets_c_up"),
        TTLJ.get("weight_ttbb_up"),
        TTLJ.get("weight_ttcc_up"),
    ])

    diffs2 = []
    for wv in syst_w:
        if wv is None: 
            continue
        B_var = _cum_from_idx(idx_b, wv.astype(np.float32, copy=False), nb)
        diffs2.append((B_var - B_cum)**2)
    rel_b = np.zeros_like(B_cum, dtype=np.float64) if not diffs2 else (
            np.sqrt(np.sum(diffs2, axis=0)) / np.maximum(B_cum, 1e-12))

    Z_curve = asimov_Z(S_cum, B_cum, rel_b=rel_b, include_systematics=True)
    valid = np.isfinite(Z_curve) & (S_cum > 0) & (B_cum > 0) & (S_neff_cum >= 20) & (B_neff_cum >= 20)

    zshape = _zshape_asimov(Vcb_prob, TTLJ_prob, bins,
                            Vcb['weight_base'], TTLJ['weight_base'], syst_w)

    if not np.any(valid):
        return {"best_Z": float('-inf'), "w": w.tolist()}

    i_best = int(np.nanargmax(np.where(valid, Z_curve, -np.inf)))

    return {
        "best_Z": zshape,#float(Z_curve[i_best]),
        "thr":    float(bins[i_best]),
        "S":      float(S_cum[i_best]),
        "B":      float(B_cum[i_best]),
        "rel_b":  float(rel_b[i_best]),
        "w":      w.tolist(),   # ⬅️ 고정 키
        "zshape": float(np.nanmax(zshape)) if np.ndim(zshape) else float(zshape),
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
        try:
            from ray.air import session
            session.report(res)            # Ray >= 2.x (AIR)
        except Exception:
            from ray import tune
            try:
                tune.report(res)           # 일부 버전은 dict도 허용
            except TypeError:
                tune.report(**res)         # 아주 구버전 호환
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


def run_grid_search(Vcb, TTLJ, bins, mode, *, low=0.0, high=10.0, steps=21, topk=5, sig_idx=0):
    # K-1 차원 그리드
    logp, _, _ = extract_logp(Vcb)
    K = logp.shape[1]
    grid = get_grids(K-1, low, high, steps)

    best = None; tk = TopK(topk)
    for params in grid:
        res = evaluate_config_np({"params": params}, Vcb, TTLJ, bins, mode, sig_idx)
        if (best is None) or (res["best_Z"] > best["best_Z"]):
            best = res
        tk.push(res)
    return {"best": best, "topk": tk.results()}

def _parse_alpha(alpha_str: str) -> Tuple[float, float, float, float]:
    vals = [float(x) for x in alpha_str.split(',')]
    if len(vals) != 4:
        raise ValueError("--alpha must have 4 comma-separated values (for g0,g1,g2,g3)")
    return tuple(vals)

@ray.remote(num_cpus=1)
class _EvalActor:
    def __init__(self, Vcb, TTLJ, bins, mode, sig_idx=0):
        self.Vcb, self.TTLJ, self.bins, self.mode, self.sig_idx = Vcb, TTLJ, bins, mode, sig_idx
    def eval_w(self, *w_params):
        # w_params: 길이 K-1
        return evaluate_config_np({"params": np.array(w_params, dtype=np.float64)},
                                  self.Vcb, self.TTLJ, self.bins, self.mode, self.sig_idx)

def dirichlet_to_w_samples(G: np.ndarray, sig_idx: int) -> np.ndarray:
    """
    G: (N,K) 감마 샘플. 반환: (N,K) w 벡터 (w[sig]=1, 나머지 = g_j/g_sig)
    """
    g_sig = np.maximum(G[:, [sig_idx]], 1e-30)
    W = G / g_sig
    W[:, sig_idx] = 1.0
    return W

def run_ray_cem(Vcb, TTLJ, bins, mode, *,
                samples=1000,
                num_cpus=None,
                topk=5,
                elite_frac=0.2,
                init_sigma=1.0,
                min_sigma=0.05,
                patience=5,
                w_bounds=(1e-3, 1e3),
                seed=None,
                ray_address=None,
                sig_idx=0):
    """
    Cross-Entropy Method in log-space for variable K classes.
    We optimize K-1 log-weights (signal index fixed to 1.0 in evaluator).
    """
    if not RAY_OK:
        raise RuntimeError("Ray not available. Install ray or choose --search grid.")

    import numpy as np
    from tqdm import tqdm

    # --- 클래스 수(K) 추출 → 탐색 차원 D = K-1 ---
    logp_V, _, _ = extract_logp(Vcb)
    K = logp_V.shape[1]
    if K < 2:
        raise ValueError(f"Need at least 2 classes, got K={K}")
    D = K - 1

    # --- Ray init ---
    ray.shutdown()
    parallel = int(num_cpus or os.cpu_count())
    if ray_address:
        ray.init(address=ray_address)
    else:
        ray.init(include_dashboard=False, num_cpus=parallel)

    Vcb_ref  = ray.put(Vcb)
    TTLJ_ref = ray.put(TTLJ)
    bins_ref = ray.put(bins)

    n_actors = parallel if not ray_address else max(parallel, 8)
    actors = [
        _EvalActor.remote(Vcb_ref, TTLJ_ref, bins_ref, mode, sig_idx)
        for _ in range(max(1, n_actors))
    ]

    # --- CEM 초기화: r = log(w_params) in R^D (w_params = exp(r)) ---
    rng   = np.random.default_rng(seed)
    mu    = np.zeros(D, dtype=np.float64)   # 초기 평균: w≈1 → r≈0
    sigma = float(init_sigma)
    wmin, wmax = w_bounds

    # 배치/반복
    batch    = max(4*len(actors), 64)
    iters    = max(1, int(np.ceil(samples / batch)))
    elite_k  = lambda m: max(2, int(np.ceil(elite_frac * m)))

    tk = TopK(topk)
    best_so_far = -np.inf
    stale = 0

    bar = tqdm(total=samples, desc="CEM search", dynamic_ncols=True, mininterval=0.2)

    for it in range(iters):
        m = batch if it < iters-1 else (samples - (iters-1)*batch) or batch

        # (1) 샘플링: r ~ N(mu, sigma^2 I) → params = clip(exp(r), [wmin,wmax])
        r       = rng.normal(loc=mu, scale=sigma, size=(m, D))
        params  = np.exp(r)
        params  = np.clip(params, wmin, wmax)

        # (2) 병렬 평가: 액터에 가변 파라미터 전달
        in_flight = []
        for i in range(m):
            a = actors[i % len(actors)]
            in_flight.append(a.eval_w.remote(*params[i]))

        results = [ray.get(ref) for ref in in_flight]
        for res in results:
            tk.push(res)
            z = res.get("best_Z", -np.inf)
            if z > best_so_far:
                best_so_far = z
                stale = 0

        bar.update(m); bar.set_postfix(best_Z=f"{best_so_far:.3f}")

        # (3) 엘리트 선택 & 파라미터 업데이트 (log-space 평균/분산)
        zs   = np.array([res.get("best_Z", -np.inf) for res in results])
        idx  = np.argsort(-zs)[:elite_k(m)]
        elites   = params[idx]                   # (k, D)
        r_elite  = np.log(elites + 1e-300)      # 안정 로그

        mu    = r_elite.mean(axis=0)
        sigma = max(min_sigma, float(r_elite.std(axis=0).mean()))
        sigma *= 0.9  # 가벼운 수축

        stale += 1
        if stale >= patience:
            sigma = max(min_sigma, sigma * 0.5)
            stale = 0
            if sigma <= min_sigma + 1e-12:
                break

    bar.close()
    ray.shutdown()

    top = tk.results()
    best = top[0] if top else {"best_Z": -np.inf}
    return {"best": best, "topk": top}

def run_ray_dirichlet(Vcb, TTLJ, bins, mode, *, samples=2000, alpha=None,
                      num_cpus=None, topk=5, seed=None, ray_address=None, sig_idx=0):
    if not RAY_OK:
        raise RuntimeError("Ray not available")

    ray.shutdown()
    parallel = int(num_cpus or os.cpu_count())
    ray.init(address=ray_address) if ray_address else ray.init(include_dashboard=False, num_cpus=parallel)

    Vcb_ref, TTLJ_ref, bins_ref = ray.put(Vcb), ray.put(TTLJ), ray.put(bins)
    logp, _, _ = extract_logp(Vcb); K = logp.shape[1]

    n_actors = parallel if not ray_address else max(parallel, 8)
    actors = [_EvalActor.remote(Vcb_ref, TTLJ_ref, bins_ref, mode) for _ in range(max(1, n_actors))]

    rng = np.random.default_rng(seed)
    alpha_arr = np.ones(K, dtype=np.float64) if alpha is None else np.asarray(alpha, dtype=np.float64)
    if alpha_arr.size != K:
        raise ValueError(f"--alpha needs {K} values for Dirichlet/Gamma sampling")
    G = rng.gamma(shape=alpha_arr, scale=1.0, size=(samples, K))
    W = dirichlet_to_w_samples(G, sig_idx)   # (N,K)

    tk = TopK(topk); best_so_far = float('-inf')
    from tqdm import tqdm; bar = tqdm(total=samples, desc="Dirichlet search", dynamic_ncols=True, mininterval=0.2)

    in_flight = []
    for i in range(min(samples, max(1, len(actors)*4))):
        in_flight.append(actors[i % len(actors)].eval_w.remote(*W[i, np.arange(K)!=sig_idx]))  # 아래 eval_w 수정 필요
    next_i = len(in_flight)

    while in_flight:
        done, in_flight = ray.wait(in_flight, num_returns=1)
        res = ray.get(done[0])
        tk.push(res); best_so_far = max(best_so_far, res.get("best_Z", float('-inf')))
        bar.update(1); bar.set_postfix(best_Z=f"{best_so_far:.3f}")
        if next_i < samples:
            in_flight.append(actors[next_i % len(actors)].eval_w.remote(*W[next_i, np.arange(K)!=sig_idx]))
            next_i += 1

    bar.close(); ray.shutdown()
    top = tk.results()
    return {"best": top[0] if top else {"best_Z": -np.inf}, "topk": top}

def run_ray_optuna(Vcb, TTLJ, bins, mode, samples=500, num_cpus=None, topk=5):
    if not RAY_OK:
        raise RuntimeError("Ray/Optuna not available")

    # 로컬 강제 (원격 클러스터 붙으려는 환경변수 제거)
    os.environ.pop("RAY_ADDRESS", None)

    import ray
    ray.shutdown()

    parallel = int(num_cpus or os.cpu_count())

    # ★ 구버전 호환: system_config 없이 깨끗하게 시작
    try:
        ray.init(include_dashboard=False, num_cpus=parallel)
    except Exception:
        # 혹시 이전 세션 잔재 등으로 실패 시 안전 재시도
        ray.shutdown()
        ray.init(include_dashboard=False, num_cpus=parallel)

    Vcb_ref  = ray.put(Vcb)
    TTLJ_ref = ray.put(TTLJ)
    bins_ref = ray.put(bins)

    EPS = 0.05
    logp, _, _ = extract_logp(Vcb); K = logp.shape[1]
    param_space = { f"u{i}": tune.uniform(1e-6, 1-1e-6) for i in range(K) }  # K개의 U(0,1)
    # 샘플 → 감마: g_i = -log(u_i); 비신호는 g_i/g_sig 로 변환
    def trainable(config):
        us = np.array([config[f"u{i}"] for i in range(K)], dtype=np.float64)
        gs = -np.log(us)
        params = (gs/gs[0])[np.arange(K)!=0]  # 길이 K-1
        return evaluate_config({"params": params}, Vcb_ref=Vcb_ref, TTLJ_ref=TTLJ_ref,
                            bins_ref=bins_ref, working_mode=mode, sig_idx=sig_idx)

    # ★ 버전 호환 리소스 지정(Trial당 CPU 1개) + 동시성 = parallel
    trainable = tune.with_resources(tune.with_parameters(trainable), {"cpu": 1})
    algo = OptunaSearch(metric="best_Z", mode="max")
    algo = ConcurrencyLimiter(algo, max_concurrent=parallel, batch=False)

    tuner = Tuner(
        trainable,
        param_space=param_space,
        tune_config=TuneConfig(
            search_alg=algo,
            num_samples=samples,
            metric="best_Z",
            mode="max",
            reuse_actors=False,
        ),
    )

    results = tuner.fit()

    all_res = []
    for r in results:
        m = r.metrics
        if isinstance(m, dict) and "best_Z" in m:
            all_res.append({k: m.get(k) for k in ("best_Z","thr","S","B","rel_b","w0","w1","w2","w3")})
    all_res = [x for x in all_res if x.get("best_Z") is not None]
    all_res.sort(key=lambda d: d["best_Z"], reverse=True)
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
    logp, _, _ = extract_logp(arr)
    p = np.exp(logp)
    s = np.maximum(p.sum(axis=1, keepdims=True), 1e-30)
    return p / s


def plot_class_scores(Vcb, TTLJ, outdir: str, bins: int = 50, tag: str = "") -> List[str]:
    import matplotlib.pyplot as plt
    _ensure_dir(outdir)
    paths = []
    p_v = _class_probs(Vcb)
    p_b = _class_probs(TTLJ)
    w_v = Vcb['weight_base']; w_b = TTLJ['weight_base']
    K = p_v.shape[1]

    for k in range(K):
        fig = plt.figure()
        plt.hist(p_b[:,k], bins=bins, weights=w_b, histtype='step', density=True,
                 label=f'Bkg TTLJ - class{k}')
        plt.hist(p_v[:,k], bins=bins, weights=w_v, histtype='step', density=True,
                 label=f'Sig Vcb - class{k}')
        plt.xlabel(f'Class-{k} probability'); plt.ylabel('Weighted yield')
        plt.title(f'Class-{k} score distribution' + (f' [{tag}]' if tag else ''))
        plt.legend()
        out = os.path.join(outdir, f'class_{k}_score{("_"+tag) if tag else ""}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
        paths.append(out)
    return paths

def plot_signal_score(Vcb: Dict[str, np.ndarray], TTLJ: Dict[str, np.ndarray],
                      w: np.ndarray, outdir: str, bins: int = 100,
                      thr: Optional[float] = None, tag: str = "") -> str:
    """P_w(sig|x) 분포를 Vcb vs TTLJ로 겹쳐 그리고(best_thr 수직선 표시). density=True 유지."""
    import matplotlib.pyplot as plt
    _ensure_dir(outdir)

    # 최적화 웨이트 적용된 신호 스코어
    p_sig_v = make_signal_score(Vcb,  w)
    p_sig_b = make_signal_score(TTLJ, w)

    w_v = Vcb['weight_base']
    w_b = TTLJ['weight_base']

    fig = plt.figure()
    # density=True → 가중-정규화 분포(면적 1)
    plt.hist(p_sig_b, bins=bins, weights=w_b, histtype='step',
             density=True, label='Bkg TTLJ (P_w(sig|x))')
    plt.hist(p_sig_v, bins=bins, weights=w_v, histtype='step',
             density=True, label='Sig Vcb (P_w(sig|x))')

    plt.xlabel('P_w(sig | x)')
    plt.ylabel('Weighted density')  # 정확한 축 라벨
    ttl = f'Signal score distribution' + (f' [{tag}]' if tag else '')
    plt.title(ttl)
    plt.yscale('log')

    # best_thr 수직선
    if thr is not None and 0.0 <= thr <= 1.0:
        ymin, ymax = plt.ylim()
        plt.axvline(thr, linestyle='--')
        # 간단한 주석(옵션)
        plt.text(thr, ymax*0.95, f'thr={thr:.3g}', rotation=90,
                 va='top', ha='right')

    plt.legend()
    out = os.path.join(outdir, f'signal_score{("_"+tag) if tag else ""}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out



# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Optimize class weights to maximize Asimov Z with systematics.")
    ap.add_argument('--mode', choices=['b','c'], required=True, help="Choose b-tag or c-tag systematic set")
    ap.add_argument('--ttlj', required=True, help="Path to TTLJ tree, e.g. file.root:Tree/Path")
    ap.add_argument('--vcb',  required=True, help="Path to Vcb  tree, e.g. file.root:Tree/Path")
    ap.add_argument('--bins', type=int, default=100, help="Number of bins in [0,1] for score histogram")

    # --- 검색 전략 선택 ---

    ap.add_argument('--search', choices=['grid','dirichlet','cem','optuna'],  # ← cem 추가
                default='dirichlet',
                help="grid | dirichlet | cem | optuna")
    # 공통 파라미터
    ap.add_argument('--samples', type=int, default=500, help="Number of samples to evaluate (per search)")
    ap.add_argument('--grid', action='store_true', help=argparse.SUPPRESS)  # (하위호환; 쓰면 grid로 인식)
    ap.add_argument('--grid-steps', type=int, default=21, help="Grid steps per dimension if --search grid")
    ap.add_argument('--grid-low', type=float, default=0.0)
    ap.add_argument('--grid-high', type=float, default=10.0)

    # 동시성 제어
    ap.add_argument('--num-cpus', type=int, default=None, help="Override Ray num_cpus (defaults to all cores)")
    ap.add_argument('--ray-address', type=str, default=None,
                    help='Ray cluster address (e.g. "auto" or "ray://host:10001"). If unset, runs local.')

    # 디리클레 샘플링 설정
    ap.add_argument('--alpha', type=str, default='1,1,1,1',
                    help='Dirichlet α for [g0,g1,g2,g3] (comma-separated). Uniform=1, peaked>1, spiky<1')
    ap.add_argument('--seed', type=int, default=None, help='Random seed')

    ap.add_argument('--topk', type=int, default=5, help='How many best configs to print')
    ap.add_argument('--plot-dir', type=str, default=None, help='If set, write per-class score plots here')
    ap.add_argument('--branch-name', type=str, default='template_score_MultiClass', help='Branch name for the analysis')

    ap.add_argument('--elite-frac', type=float, default=0.2, help='CEM: elite 상위 비율')
    ap.add_argument('--init-sigma', type=float, default=1.0, help='CEM: 초기 log-공간 표준편차')
    ap.add_argument('--min-sigma', type=float, default=0.05, help='CEM: 하한 표준편차')
    ap.add_argument('--patience', type=int, default=5, help='CEM: 개선 없을 때 조기 종료 patience(반복)')
    ap.add_argument('--wmin', type=float, default=1e-3, help='CEM: w 하한(클리핑)')
    ap.add_argument('--wmax', type=float, default=1e+3, help='CEM: w 상한(클리핑)')
    ap.add_argument('--metric', choices=['zmax','zshape'], default='zmax')
    return ap.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    # Load
    TTLJ, Vcb = load_signal_and_background(
    ttlj_spec=args.ttlj,
    vcb_spec=args.vcb,
    mode=args.mode,
    branch_name=args.branch_name,
    num_classes=4,
    )

    # Binning for score in [0,1]
    bins = np.linspace(0, 1, max(args.bins, 10))

        # --- 선택된 검색 전략 ---
    if args.grid or args.search == 'grid' or not RAY_OK:
        out = run_grid_search(Vcb, TTLJ, bins, mode,
                              low=args.grid_low, high=args.grid_high,
                              steps=args.grid_steps, topk=args.topk)
    elif args.search == 'dirichlet':
        alpha = _parse_alpha(args.alpha)
        out = run_ray_dirichlet(Vcb, TTLJ, bins, mode,
                                samples=args.samples, alpha=alpha,
                                num_cpus=args.num_cpus, topk=args.topk,
                                seed=args.seed, ray_address=args.ray_address)
    elif args.search == 'cem':
        out = run_ray_cem(Vcb, TTLJ, bins, mode,
                          samples=args.samples, num_cpus=args.num_cpus,
                          topk=args.topk, elite_frac=args.elite_frac,
                          init_sigma=args.init_sigma, min_sigma=args.min_sigma,
                          patience=args.patience, w_bounds=(args.wmin, args.wmax),
                          seed=args.seed, ray_address=args.ray_address)
    elif args.search == 'optuna':
        out = run_ray_optuna(Vcb, TTLJ, bins, mode,
                             samples=args.samples, num_cpus=args.num_cpus, topk=args.topk)
    else:
        raise ValueError(f"Unknown --search {args.search}")

    # --- metric 선택: zmax(default) vs zshape ---
    if args.metric == 'zshape':
        topk = sorted(out["topk"], key=lambda d: d.get("zshape", float('-inf')), reverse=True)
        out["topk"] = topk
        out["best"] = topk[0] if topk else out["best"]
    else:
        # ensure best/topk are sorted by 'best_Z' (zmax)
        topk = sorted(out["topk"], key=lambda d: d.get("best_Z", float('-inf')), reverse=True)
        out["topk"] = topk
        out["best"] = topk[0] if topk else out["best"]

    best = out["best"]
    topk = out["topk"]
    
    # Print top-K neatly
    print("\n=== Top {} configs (by best_Z) ===".format(len(topk)))
    print(json.dumps(topk, indent=2))
    print("\n=== Best ===")
    print(json.dumps(best, indent=2))


    if args.plot_dir and best:
        tag = "w=[" + ",".join(f"{x:.3g}" for x in best["w"]) + "]"
        files = plot_class_scores(Vcb, TTLJ, args.plot_dir, bins=50, tag=tag)
        w_vec = np.asarray(best["w"], dtype=np.float64)
        sigplot = plot_signal_score(Vcb, TTLJ, w_vec, args.plot_dir,
                                    bins=100, thr=best.get('thr'), tag=tag)
        saved = save_summary(args.plot_dir, args, best, topk, tag=tag)


if __name__ == '__main__':
    main()
