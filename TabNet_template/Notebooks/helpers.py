from collections.abc import Iterable
from typing import Dict, Optional
import numpy as np
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


def log_prob_to_weighted_prob(log_prob: np.ndarray, weights: np.ndarray, sig_idx: Optional[int] = 0) -> np.ndarray:
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
    s_num = logw[sig_idx] + logp[:, sig_idx]
    idx_except_sig = np.arange(K) != sig_idx
    s_den = _logsumexp(logp[:, idx_except_sig] + logw[idx_except_sig][None, :], axis=1)
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

def _bin_indices(prob: np.ndarray, bins: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(bins, prob, side='right') - 1
    return np.clip(idx, 0, len(bins)-2).astype(np.int64, copy=False)

def _cum_from_idx(idx: np.ndarray, weights: np.ndarray, nbins: int) -> np.ndarray:
    h = np.bincount(idx, weights=weights, minlength=nbins)
    return _cum_from_right(h)


def _zshape_asimov(Vcb_prob, TTLJ_prob, bins, wv, wb, syst_ws, eps=1e-12):

    bins_merged = _merge_until_valid(bins,
                                    Vcb_prob, TTLJ_prob,
                                    wv, wb,
                                    min_neff_s=10, min_neff_b=20,
                                    max_iter=10000)
                                
    
    # 1) 빈별 S_i, B_i
    S_bin, _ = np.histogram(Vcb_prob,  bins=bins_merged, weights=wv)
    B_bin, _ = np.histogram(TTLJ_prob, bins=bins_merged, weights=wb)

    # 2) 빈별 배경 상대 오차 rel_b_i (여기선 up-변동들의 RMS → 보수적)
    nb = len(bins_merged) - 1
    idx_b = np.digitize(TTLJ_prob, bins_merged) - 1
    idx_b = np.clip(idx_b, 0, nb-1)

    # 기준 B_bin과 일치하도록 동일한 방식으로 산출
    B_ref = np.bincount(idx_b, weights=wb, minlength=nb).astype(float)

    diffs2 = []
    for wv_var in syst_ws:
        if wv_var is None:
            continue
        B_var = np.bincount(idx_b, weights=wv_var, minlength=nb).astype(float)
        diffs2.append((B_var - B_ref)**2)

    if diffs2:
        sigma = np.sqrt(np.sum(diffs2, axis=0))
        rel_b = sigma / np.maximum(B_ref, eps)
    else:
        rel_b = np.zeros_like(B_ref)

    # 3) 빈별 Asimov Z_i (시스템틱 포함) → 제곱합 후 제곱근
    Zi = asimov_Z(S_bin, B_bin, rel_b=rel_b, include_systematics=True)
    # 방어: NaN/inf 처리 + 아주 작은 분모 회피
    Zi = np.where(np.isfinite(Zi), Zi, 0.0)
    return float(np.sqrt(np.sum(Zi**2)))

# ----------------------------
# Bin validation/merge (stats only)
# ----------------------------

def _neff_from_hist(sumw: np.ndarray, sumw2: np.ndarray) -> np.ndarray:
    return (sumw * sumw) / np.maximum(sumw2, 1e-12)

def _merge_until_valid(edges: np.ndarray,
                       u_sig_v: np.ndarray, u_sig_b: np.ndarray,
                       w_v: np.ndarray, w_b: np.ndarray,
                       min_neff_s: float = 0.0, min_neff_b: float = 0.0,
                       max_iter: int = 10_000) -> np.ndarray:
    if (min_neff_s <= 0) and (min_neff_b <= 0):
        return edges

    def _stats(e):
        S  = np.histogram(u_sig_v, bins=e, weights=w_v)[0].astype(np.float64)
        B  = np.histogram(u_sig_b, bins=e, weights=w_b)[0].astype(np.float64)
        S2 = np.histogram(u_sig_v, bins=e, weights=w_v**2)[0].astype(np.float64)
        B2 = np.histogram(u_sig_b, bins=e, weights=w_b**2)[0].astype(np.float64)
        Neff_S = _neff_from_hist(S, S2)
        Neff_B = _neff_from_hist(B, B2)
        return S, B, Neff_S, Neff_B

    it = 0
    while it < max_iter and len(edges) > 2:
        S, B, Neff_S, Neff_B = _stats(edges)
        bad = np.where((Neff_S < min_neff_s) | (Neff_B < min_neff_b))[0]
        if bad.size == 0:
            break
        i = int(bad[0])

        # merge direction: prefer side with larger S 
        if i == 0:
            drop_idx = 1
        elif i == len(S) - 1:
            drop_idx = len(edges) - 2
        else:
            drop_idx = (i + 1) if (S[i+1] >= S[i-1]) else i

        edges = np.delete(edges, drop_idx)
        it += 1

    return edges

# ----------------------------
# Utils
# ---------------------------

import re
CLASSIF_RE = re.compile(r'^classif_(\d+)$')

def extract_logp(arr: Dict[str, np.ndarray]):
    """
    arr에 들어있는 classif_* 컬럼들을 자동 탐지하고 (N,K) logp와 인덱스 배열을 돌려준다.
    반환: logp:(N,K), cls_idx:(K,), names:Tuple[str,...]
    """
    found = []
    for k in arr.keys():
        m = CLASSIF_RE.match(k)
        if m:
            found.append((int(m.group(1)), k))
    if not found:
        raise KeyError("No classif_* columns found in arrays")
    found.sort(key=lambda t: t[0])   # 인덱스 순으로 정렬
    cls_idx, names = zip(*found)
    logp = np.vstack([arr[name] for name in names]).T
    return logp, np.asarray(cls_idx, dtype=int), names

def build_w_from_params(params: Iterable[float], K: int, sig: int) -> np.ndarray:
    """
    탐색 파라미터(params: 길이 K-1)를 받아 전체 길이 K의 w 벡터를 만든다.
    신호 인덱스 sig는 1.0으로 고정.
    """
    w = np.ones(K, dtype=np.float64)
    j = 0
    for k in range(K):
        if k == sig: 
            continue
        w[k] = float(params[j]); j += 1
    return w