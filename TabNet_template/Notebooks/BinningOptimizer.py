#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BinningFromOptimizedScore_mplhep.py
-----------------------------------
- 최적화된 class-weights (w0=1, w1, w2, w3)로 P_w(sig|x) 만들기
- 순수 NumPy 비닝 (시스템틱 *미사용*)
- 비닝 방법:
  * uniform_score : 점수 축 등간격
  * equal_b       : 배경 면적 균등 (B-quantile)
  * equal_s       : 신호 면적 균등 (S-quantile)
  * equal_info    : 정보 균등 ~ s^2/(s+b)   [권장]
  * t_equal       : t=log(s/b) 등간격 (단조화+꼬리 클립)
- mplhep 스타일로 플롯 저장 (CMS 라벨 옵션)
- bins는 **항상 JSON**(메타데이터 포함)으로 저장

사용 예시는 파일 끝의 주석 참고.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from typing import Dict, Tuple, List, Optional
from helpers import _merge_until_valid

import numpy as np

# --------- I/O helpers (uproot/awkward optional) ----------
try:
    import uproot
    import awkward as ak
except Exception:
    uproot = None
    ak = None

try:
    from mcdata_io import load_signal_and_background
    USE_MCDATA_IO = True
except Exception:
    USE_MCDATA_IO = False

# --------- mplhep style ----------
try:
    import matplotlib.pyplot as plt
    import mplhep as hep
except Exception as e:
    print("[ERROR] This script needs matplotlib and mplhep. Try: pip install matplotlib mplhep", file=sys.stderr)
    raise

# ----------------------------
# Generic utilities
# ----------------------------

def _ensure_dir(p: str):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _open_tree(spec: str):
    if uproot is None:
        raise RuntimeError("uproot not available; install 'uproot awkward' or use mcdata_io loader.")
    if ":" not in spec:
        raise ValueError("Path must be '/path/file.root:Tree/Path'")
    return uproot.open(spec)

def _load_arrays(spec: str, need: List[str]) -> Dict[str, np.ndarray]:
    tree = _open_tree(spec)
    arrs = tree.arrays(need, library="ak")
    out = {}
    for k in arrs.fields:
        out[k] = arrs[k].to_numpy()
    return out

def _class_probs_from_logp(logp: np.ndarray) -> np.ndarray:
    p = np.exp(logp)
    s = p.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return p / s

def weighted_sig_prob_from_logp(logp: np.ndarray, w: np.ndarray, sig_class: int = 0) -> np.ndarray:
    """P_w(sig|x) = (w_sig * p_sig) / sum_j (w_j * p_j)"""
    p = _class_probs_from_logp(logp)
    num = w[sig_class] * p[:, sig_class]
    den = (p * w.reshape((1, -1))).sum(axis=1)
    den = np.where(den > 0, den, 1.0)
    return num / den  # [0,1]

# ----------------------------
# Fine histograms & smoothing
# ----------------------------

def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1:
        return x
    k = int(k)
    ker = np.ones(k, dtype=np.float64) / k
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    return np.convolve(xpad, ker, mode='valid')

def _fine_SB(u_sig_v: np.ndarray, u_sig_b: np.ndarray,
             w_v: np.ndarray, w_b: np.ndarray,
             fine: int = 1024, smooth: int = 3,
             # --- NEW: fine-bin 최소 통계 요구치 ---
             min_neff_s: float = 0.0,
             min_neff_b: float = 0.0,
             min_sumw:  float = 0.0,
             max_iter:  int = 100_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    신호 분위수 기반 초기 엣지를 만든 뒤, 각 fine-bin이
    Neff_S >= min_neff_s, Neff_B >= min_neff_b, (S+B) >= min_sumw
    를 만족할 때까지 인접 병합을 반복한다.
    """
    # 1) 초기 엣지: (비가중) 신호 분위수로 시작
    if fine < 2 or u_sig_v.size == 0:
        edges = np.linspace(0.0, 1.0, max(fine, 2) + 1)
    else:
        edges = np.quantile(u_sig_v, np.linspace(0.0, 1.0, fine + 1))
        # 경계/단조/유니크 보정
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.maximum.accumulate(edges)
        edges = np.unique(edges)
        if edges.size < 3:
            edges = np.linspace(0.0, 1.0, fine + 1)

    # 내부 헬퍼: 현재 엣지에서 S,B,Neff 계산
    def _stats(e):
        S  = np.histogram(u_sig_v, bins=e, weights=w_v)[0].astype(np.float64)
        B  = np.histogram(u_sig_b, bins=e, weights=w_b)[0].astype(np.float64)
        S2 = np.histogram(u_sig_v, bins=e, weights=w_v**2)[0].astype(np.float64)
        B2 = np.histogram(u_sig_b, bins=e, weights=w_b**2)[0].astype(np.float64)
        Neff_S = (S*S) / np.maximum(S2, 1e-12)
        Neff_B = (B*B) / np.maximum(B2, 1e-12)
        return S, B, Neff_S, Neff_B

    # 2) 통계 확보될 때까지 인접 병합
    it = 0
    while it < max_iter and edges.size > 3:
        S, B, Neff_S, Neff_B = _stats(edges)
        bad = np.where((Neff_S < min_neff_s) | (Neff_B < min_neff_b) | ((S + B) < min_sumw))[0]
        if bad.size == 0:
            break
        i = int(bad[0])

        # 병합 방향: 통상 더 "안정"한 쪽(B 또는 S+B가 큰 쪽)으로 합친다
        if i == 0:
            drop_idx = 1
        elif i == len(S) - 1:
            drop_idx = len(edges) - 2
        else:
            right_mass = S[i+1] + B[i+1]
            left_mass  = S[i]   + B[i]
            drop_idx = i if left_mass <= right_mass else (i + 1)

        edges = np.delete(edges, drop_idx)
        it += 1

    # 마지막 통계
    S, B, _, _ = _stats(edges)

    # (선택) 매끈함을 원하면 이동평균으로 완만하게
    # if smooth and smooth > 1:
    #     S = _moving_average(S, smooth)
    #     B = _moving_average(B, smooth)

    return edges, S, B

# ----------------------------
# Binning constructors (NO systematics)
# ----------------------------

def _equal_area_bins(edges_fine: np.ndarray, dens: np.ndarray, nbins: int) -> np.ndarray:
    assert len(edges_fine) == len(dens) + 1
    c = np.cumsum(dens)
    tot = c[-1]
    if tot <= 0:
        return np.linspace(0.0, 1.0, nbins + 1)
    targets = np.linspace(0, tot, nbins + 1)
    x = edges_fine[:-1]
    out = np.interp(targets, c, x, left=x[0], right=edges_fine[-1])
    out[0], out[-1] = 0.0, 1.0
    out = np.maximum.accumulate(out)          # non-decreasing
    out = np.unique(out)
    if len(out) < 3:
        out = np.linspace(0.0, 1.0, nbins + 1)
    return out

def _equal_info_bins(edges_fine: np.ndarray, S: np.ndarray, B: np.ndarray, nbins: int) -> np.ndarray:
    w = S * S / np.maximum(S + B, 1e-12)
    return _equal_area_bins(edges_fine, w, nbins)

def _equal_B_bins(edges_fine: np.ndarray, B: np.ndarray, nbins: int) -> np.ndarray:
    return _equal_area_bins(edges_fine, B, nbins)

def _equal_S_bins(edges_fine: np.ndarray, S: np.ndarray, nbins: int) -> np.ndarray:
    return _equal_area_bins(edges_fine, S, nbins)

import numpy as np

def _gauss_smooth(y: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return y
    k = int(max(3, 6*sigma)) | 1  # 홀수 커널
    c = np.arange(k) - k//2
    g = np.exp(-0.5*(c/sigma)**2)
    g /= g.sum()
    return np.convolve(y, g, mode='same')

import numpy as np

def _gauss_smooth(y: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return y
    k = int(max(3, 6*sigma)) | 1  # 홀수 커널
    c = np.arange(k) - k//2
    g = np.exp(-0.5*(c/sigma)**2)
    g /= g.sum()
    return np.convolve(y, g, mode='same')

def _t_equal_bins(edges_fine: np.ndarray, S: np.ndarray, B: np.ndarray,
                        nbins: int,
                        b_floor: float = 1e-12,
                        t_clip=(2.0, 98.0),
                        sigma_counts: float = 2.0,   # S,B 스무딩(파인빈 단위)
                        sigma_t: float = 0.0,        # t(x) 추가 스무딩
                        laplace: float = None        # 라플라스 유사 완충
                       ) -> np.ndarray:
    x = edges_fine[:-1]

    # 1) S,B를 먼저 스무딩 → 비율이 훨씬 안정
    S_s = _gauss_smooth(S.astype(float), sigma_counts) if sigma_counts > 0 else S.astype(float)
    B_s = _gauss_smooth(B.astype(float), sigma_counts) if sigma_counts > 0 else B.astype(float)

    # 라플라스 완충(옵션): "전형적인" 파인빈 가중치의 0.1~1배 정도 권장
    if laplace is None:
        pos = np.concatenate([S_s[S_s>0], B_s[B_s>0]])
        typ = np.median(pos) if pos.size else 0.0
        laplace = 0.2*typ  # 필요 시 튜닝: 0.1~1.0*typ

    r = (S_s + laplace) / np.maximum(B_s + laplace, b_floor)
    r = np.maximum(r, 1e-16)
    t = np.log(r)

    # 2) t(x) 스무딩 + 단조 강제
    if sigma_t > 0:
        t = _gauss_smooth(t, sigma_t)
    # Isotonic 대신 간단 버전: 누적 최대(단조↑)
    t = np.maximum.accumulate(t)

    # 극단값 클립
    finite = np.isfinite(t)
    if not np.any(finite):
        return np.linspace(0.0, 1.0, nbins+1)
    lo, hi = np.percentile(t[finite], t_clip)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.linspace(0.0, 1.0, nbins+1)
    t = np.clip(t, lo, hi)

    # 평탄부 압축(동일 t 값 하나로)
    t_u, idx = np.unique(t, return_index=True)
    x_u = x[idx]
    if t_u.size < 2:
        return np.linspace(0.0, 1.0, nbins+1)

    # 3) PCHIP로 x(t) 역보간 (단조·형상 보존 큐빅)
    t_edges = np.linspace(t_u[0], t_u[-1], nbins+1)
    try:
        from scipy.interpolate import PchipInterpolator
        f = PchipInterpolator(t_u, x_u, extrapolate=True)
        u_edges = f(t_edges)
    except Exception:
        raise RuntimeError("scipy.interpolate.PchipInterpolator not available; install scipy.")

    # 경계 정리
    u_edges[0], u_edges[-1] = 0.0, 1.0
    u_edges = np.maximum.accumulate(u_edges)       # 단조 보강
    u_edges = np.clip(u_edges, 0.0, 1.0)
    u_edges = np.unique(u_edges)                   # 중복 제거
    if u_edges.size < 3:
        u_edges = np.linspace(0.0, 1.0, nbins+1)
    return u_edges

# ----------------------------
# Public API
# ----------------------------

def compute_bins(Vcb: Dict[str, np.ndarray],
                 TTLJ: Dict[str, np.ndarray],
                 w: np.ndarray,
                 *,
                 sig_class: int = 0,
                 method: str = "equal_info",
                 nbins: int = 12,
                 fine: int = 1024,
                 smooth: int = 3,
                 b_floor: float = 1e-12,
                 t_clip_low: float = 1.0,
                 t_clip_high: float = 99.0,
                 min_neff_s: float = 0.0,
                 min_neff_b: float = 0.0,
                 u_clip_low: float = 0.0,
                 u_clip_high: float = 1.0,
                 u_rescale: bool = False) -> np.ndarray:
    logp_v = np.vstack([Vcb['classif_0'], Vcb['classif_1'], Vcb['classif_2'], Vcb['classif_3']]).T
    logp_b = np.vstack([TTLJ['classif_0'], TTLJ['classif_1'], TTLJ['classif_2'], TTLJ['classif_3']]).T
    u_v = weighted_sig_prob_from_logp(logp_v, w, sig_class=sig_class)
    u_b = weighted_sig_prob_from_logp(logp_b, w, sig_class=sig_class)

    # ---- TRIM: [u_lo, u_hi] 밖은 전부 버린다 ----
    u_lo = float(np.clip(u_clip_low,  0.0, 1.0))
    u_hi = float(np.clip(u_clip_high, 0.0, 1.0))
    vmax_v = float(np.nanmax(u_v)) if u_v.size else 0.0
    vmax_b = float(np.nanmax(u_b)) if u_b.size else 0.0
    u_hi_data = float(np.clip(max(vmax_v, vmax_b), 0.0, 1.0))
    u_hi = min(u_hi, u_hi_data)
    print(f"binning range: [{u_lo:.6f}, {u_hi:.6f}] (data max {u_hi_data:.6f})")
    if u_hi <= u_lo:
        u_lo, u_hi = 0.0, 1.0  # 무효 지정 시 전체 사용

    m_v = (u_v >= u_lo) & (u_v <= u_hi)
    m_b = (u_b >= u_lo) & (u_b <= u_hi)

    u_v = u_v[m_v]; u_b = u_b[m_b]
    w_v = Vcb['weight_base'].astype(np.float64, copy=False)[m_v]
    w_b = TTLJ['weight_base'].astype(np.float64, copy=False)[m_b]

    if u_v.size == 0 or u_b.size == 0:
        raise RuntimeError("No events remain after u trimming. Relax u_clip or check inputs.")

    # ---- 내부 작업축: 항상 [0,1]에서 비닝 계산 ----
    span = (u_hi - u_lo)
    inv  = 1.0 / span
    u_v_work = (u_v - u_lo) * inv
    u_b_work = (u_b - u_lo) * inv

    # fine S,B는 작업축[0,1]에서 계산
    edges_fine, Sfine, Bfine = _fine_SB(
    u_v_work, u_b_work, w_v, w_b,
    fine=fine, smooth=1,
    # --- NEW: fine-bin 최소 통계 ---
    min_neff_s=float(min_neff_s),
    min_neff_b=float(min_neff_b),
    min_sumw=0.0  # 원하면 예: max(1.0, 0.01*(w_v.sum()+w_b.sum())/fine)
    )

    method = str(method).lower()
    if method == "uniform_score":
        edges_work = np.linspace(0.0, 1.0, nbins + 1)
    elif method == "equal_b":
        edges_work = _equal_B_bins(edges_fine, Bfine, nbins)
    elif method == "equal_s":
        edges_work = _equal_S_bins(edges_fine, Sfine, nbins)
    elif method == "equal_info":
        edges_work = _equal_info_bins(edges_fine, Sfine, Bfine, nbins)
    elif method == "t_equal":
        edges_work = _t_equal_bins(edges_fine, Sfine, Bfine, nbins, b_floor=b_floor, t_clip=(t_clip_low, t_clip_high))
    else:
        raise ValueError(f"Unknown method '{method}'")

    # 통계 가드(Neff)는 작업축에서 수행
    edges_work = _merge_until_valid(edges_work, u_v_work, u_b_work, w_v, w_b,
                                    min_neff_s=float(min_neff_s), min_neff_b=float(min_neff_b))

    # ---- 반환 축 결정: u_rescale면 [0,1], 아니면 원래 u축으로 되돌림 ----
    if u_rescale:
        edges_out = edges_work
    else:
        edges_out = edges_work * span + u_lo

    return edges_out 

# ----------------------------
# Plotting with mplhep
# ----------------------------

def _apply_hep_style(style: str = "CMS"):
    style = (style or "CMS").upper()
    try:
        if style == "ATLAS":
            hep.style.use("ATLAS")
        elif style == "ROOT":
            hep.style.use("ROOT")
        else:
            hep.style.use("CMS")
    except Exception:
        pass  # fallback to mpl defaults if style missing

def _cms_label(label: str = "Preliminary", year: Optional[str] = None, lumi: Optional[str] = None):
    try:
        # Newer mplhep
        hep.cms.label(label, data=True, year=year, lumi=lumi)
    except TypeError:
        try:
            # Older signature
            hep.cms.label(text=label, year=year, lumi=lumi)
        except Exception:
            pass

def _figsize_for_bins(nbins: int,
                      base_w: float = 8.0,
                      per_bin: float = 0.55,
                      h: float = 6.0,
                      wmin: float = 8.0,
                      wmax: float = 40.0) -> Tuple[float, float]:
    """nbins에 따라 가로폭을 선형으로 늘려주는 간단한 규칙."""
    w = base_w + per_bin * nbins
    w = max(wmin, min(w, wmax))
    return (w, h)

def _edge_label_params(nbins: int) -> Tuple[int, int, int]:
    """엣지 라벨 간격(step), 폰트크기, 회전을 nbins에 맞춰 리턴."""
    if   nbins <= 14: return 1, 8, 90
    elif nbins <= 24: return 2, 8, 90
    elif nbins <= 36: return 3, 7, 90
    else:             return max(1, nbins // 15), 7, 90

def _annotate_bin_edges(ax, edges: np.ndarray, nbins: int, y_axis_pos: float = 1.01):
    """
    x=엣지 위치에 값(소수3자리)을 축좌표 y=y_axis_pos(예: 1.01=윗가장자리 살짝 위)에 찍는다.
    """
    step_lbl, fsz, rot = _edge_label_params(nbins)
    for i, xedge in enumerate(edges):
        if i % step_lbl != 0:
            continue
        ax.text(
            xedge, y_axis_pos, f"{xedge:.3f}",
            transform=ax.get_xaxis_transform(),  # x는 데이터, y는 [0,1] 축좌표
            rotation=rot, ha='center', va='bottom',
            fontsize=fsz, clip_on=False,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.4)
        )

def _add_top_edge_axis(ax, edges: np.ndarray, nbins: int):
    step_lbl, fsz, rot = _edge_label_params(nbins)
    ax.set_xlim(0, nbins)  # 막대 인덱스 범위
    ax_top = ax.secondary_xaxis('top')   # 인덱스 공간과 1:1
    ticks = np.arange(nbins + 1)[::step_lbl]
    ax_top.set_xticks(ticks)
    ax_top.set_xticklabels([f"{edges[i]:.3f}" for i in ticks],
                           rotation=rot, ha='center', fontsize=fsz)
    ax_top.set_xlabel("Bin edges (u)")
    ax_top.tick_params(axis='x', pad=2)
        
# 기존 시그니처에 method, fine, b_floor, t_clip_low, t_clip_high 추가
def plot_with_bins(Vcb: Dict[str, np.ndarray], TTLJ: Dict[str, np.ndarray],
                   w: np.ndarray, edges: np.ndarray, outdir: str,
                   tag: str = "",
                   hep_style: str = "CMS",
                   cms_label: str = "Preliminary",
                   year: Optional[str] = None,
                   lumi_text: Optional[str] = None,
                   u_clip_low: float = 0.0,
                   u_clip_high: float = 1.0,
                   u_rescale: bool = False,
                   # --- NEW ---
                   method: str = "equal_info",
                   fine: int = 1024,
                   b_floor: float = 1e-12,
                   t_clip_low: float = 1.0,
                   t_clip_high: float = 99.0
                   ) -> Tuple[str, str]:
    _ensure_dir(outdir)
    _apply_hep_style(hep_style)

    # Build score arrays
    logp_v = np.vstack([Vcb['classif_0'], Vcb['classif_1'], Vcb['classif_2'], Vcb['classif_3']]).T
    logp_b = np.vstack([TTLJ['classif_0'], TTLJ['classif_1'], TTLJ['classif_2'], TTLJ['classif_3']]).T
    u_v = weighted_sig_prob_from_logp(logp_v, w, sig_class=0)
    u_b = weighted_sig_prob_from_logp(logp_b, w, sig_class=0)

    # ---- TRIM ----
    vmax_v = float(np.nanmax(u_v)) if u_v.size else 0.0
    vmax_b = float(np.nanmax(u_b)) if u_b.size else 0.0
    u_hi_data = float(np.clip(max(vmax_v, vmax_b), 0.0, 1.0))
    print(f"[Info] Data max P_w(sig|x) = {u_hi_data:.6f}")
    u_lo = float(np.clip(u_clip_low,  0.0, 1.0))
    u_hi = float(np.clip(u_clip_high, 0.0, 1.0))
    u_hi = min(u_hi, u_hi_data)
    print(f"[Info] Plotting range: [{u_lo:.6f}, {u_hi:.6f}]")
    if u_hi <= u_lo:
        u_lo, u_hi = 0.0, 1.0

    m_v = (u_v >= u_lo) & (u_v <= u_hi)
    m_b = (u_b >= u_lo) & (u_b <= u_hi)
    u_v = u_v[m_v]; u_b = u_b[m_b]

    w_v = Vcb['weight_base'].astype(np.float64, copy=False)[m_v]
    w_b = TTLJ['weight_base'].astype(np.float64, copy=False)[m_b]

    # ---- 플로팅 축 정리 (u_rescale 사용 시 [0,1]로 선형 변환) ----
    span = max(u_hi - u_lo, 1e-12)
    if u_rescale:
        u_v_plot = (u_v - u_lo) / span
        u_b_plot = (u_b - u_lo) / span
        edges_plot = np.asarray(edges, dtype=np.float64)  # compute_bins가 반환한 [0,1] 축과 일치
        x_label = r"Rescaled $P_w(\mathrm{sig}\mid x)$ in [0,1]"
    else:
        u_v_plot = u_v
        u_b_plot = u_b
        edges_plot = np.asarray(edges, dtype=np.float64)  # 원래 u 축
        x_label = r"$P_w(\mathrm{sig}\mid x)$"

    # 1) Density overlay (hep.histplot)  -------------------------
    dens_bins = 50
    hB, eD = np.histogram(u_b_plot, bins=dens_bins, weights=w_b, density=True)
    hS, _  = np.histogram(u_v_plot, bins=eD,        weights=w_v, density=True)

    # --- NEW: density 플롯도 bin 수에 비례해 약간 넓힘 (라벨/엣지 보이도록)
    nb_plot = max(1, len(edges_plot) - 1)
    fig1 = plt.figure(figsize=_figsize_for_bins(nb_plot,
                                            base_w=7.0,
                                            per_bin=0.25,
                                            h=4.8,
                                            wmin=7.5,
                                            wmax=36.0),
                  constrained_layout=True)
    ax1 = plt.gca()
    hep.histplot(hB, eD, label='Bkg TTLJ', histtype='step')
    hep.histplot(hS, eD, label='Sig Vcb', histtype='step')
    for x in edges_plot:
        ax1.axvline(x, linestyle='--', linewidth=0.8)

    _annotate_bin_edges(ax1, edges_plot, nb_plot, y_axis_pos=1.01)
    ttl = (r"$P_w(\mathrm{sig}\mid x)$ density") + (f" [{tag}]" if tag else "")
    #plt.title(ttl)
    plt.xlabel(x_label)
    plt.ylabel("Weighted density")
    plt.yscale('log')
    plt.legend()
    _cms_label(cms_label, year=year, lumi=lumi_text)
    out1 = os.path.join(outdir, f"score_density{('_'+tag) if tag else ''}.png")
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # 2) Per-bin stacked yields (log y)  -------------------------
    uu_v = u_v_plot
    uu_b = u_b_plot
    ee   = edges_plot

    S = np.histogram(uu_v, bins=ee, weights=w_v)[0].astype(np.float64)
    B = np.histogram(uu_b, bins=ee, weights=w_b)[0].astype(np.float64)
    nb = len(ee) - 1
    
    centers = np.arange(nb) + 0.5
    width   = np.ones(nb)
    xticks  = centers
    
    eps    = 0 
    S_plot = np.clip(S, eps, None)
    B_plot = np.clip(B, eps, None)

    # --- NEW: nb에 따라 가로폭/마진/글꼴/회전 자동 조정
    # 회전 각도/폰트
    if nb <= 12:
        rot, fsz = 0, 9
        bottom_pad = 0.16
    elif nb <= 20:
        rot, fsz = 60, 8
        bottom_pad = 0.24
    else:
        rot, fsz = 90, 7
        bottom_pad = 0.32

    fig2 = plt.figure(figsize=_figsize_for_bins(nb,
                                            base_w=8.0,
                                            per_bin=0.55,
                                            h=6.2,
                                            wmin=9.0,
                                            wmax=42.0),
                  constrained_layout=False)
    ax2 = plt.gca()
    ax2.bar(centers, B_plot, width=width, align='center', label='Bkg TTLJ', bottom=eps)
    ax2.bar(centers, S_plot, width=width, align='center', label='Sig Vcb', bottom=B_plot+eps)
    
    _add_top_edge_axis(ax2, edges_plot, nb)
    def _fmt(x): return f"{x:.3g}"
    labels = [f"S={_fmt(s)}\nB={_fmt(b)}\nS/B={_fmt(s/max(b, eps))}" for s, b in zip(S, B)]

    # 라벨 회전/정렬 적용
    ha = 'right' if rot else 'center'
    plt.xticks(xticks, labels, fontsize=fsz, rotation=rot, ha=ha)

    plt.xlabel("Bins")
    plt.ylabel("Weighted yield (log)")
    plt.yscale('log')
    ttl2 = "Per-bin weighted yields" + (f" [{tag}]" if tag else "")
    #plt.title(ttl2)
    plt.legend()

    # --- NEW: 아래쪽 마진을 bin 수에 따라 늘림
    plt.subplots_adjust(bottom=bottom_pad, left=0.10, right=0.98, top=0.90)

    _cms_label(cms_label, year=year, lumi=lumi_text)
    out2 = os.path.join(outdir, f"perbin_yields{('_'+tag) if tag else ''}.png")
    fig2.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # ===== 3) t-curve: method == 't_equal'일 때 log(S/B) (smoothed) 추가 =====
    if str(method).lower() == "t_equal":
        fine_edges, Sfine, Bfine = _fine_SB(
            u_v_plot, u_b_plot, w_v, w_b,
            fine=fine, smooth=1,
            min_neff_s=10, min_neff_b=10, min_sumw=0.0
        )

        # --- (A) S,B 스무딩 + 라플라스 완충 -----------------------------
        sigma_counts = 2.0         # 파인빈 기준 2~4 정도 권장
        Sg = _gauss_smooth(Sfine, sigma_counts)
        Bg = _gauss_smooth(Bfine, sigma_counts)

        # typical bin scale로 라플라스 완충(0-division, plateau 튐 완화)
        pos = np.concatenate([Sg[Sg > 0], Bg[Bg > 0]])
        typ = np.median(pos) if pos.size else 0.0
        laplace = max(1e-16, 0.2 * typ)  # 0.1~1.0*typ 사이에서 취향대로

        r = (Sg + laplace) / np.maximum(Bg + laplace, b_floor)
        r = np.maximum(r, 1e-16)
        t_smooth = np.log(r)

        # --- (B) t(x) 추가 스무딩 + 단조 누적 ---------------------------
        sigma_t = 1.0              # t 곡선 자체를 한 번 더 부드럽게 (0이면 생략)
        if sigma_t > 0:
            t_smooth = _gauss_smooth(t_smooth, sigma_t)

        # 단조 증가 강제(plateau/노이즈 억제)
        t_mono = np.maximum.accumulate(t_smooth)

        # 퍼센타일 클립(꼬리 억제)
        finite = np.isfinite(t_mono)
        if finite.any():
            lo = np.percentile(t_mono[finite], float(t_clip_low))
            hi = np.percentile(t_mono[finite], float(t_clip_high))
            t_plot = np.clip(t_mono, lo, hi)
        else:
            t_plot = t_mono

        # --- (C) 플롯 ---------------------------------------------------
        fig3 = plt.figure(figsize=_figsize_for_bins(nb_plot,
                                                    base_w=7.5, per_bin=0.45,
                                                    h=5.2, wmin=8.0, wmax=38.0),
                        constrained_layout=True)
        ax = plt.gca()

        x = fine_edges[:-1]  # fine bin left edges
        # 스무딩 결과는 연속곡선에 가깝기 때문에 step 대신 line을 추천
        ax.plot(x, t_plot, label=r"$t(u)=\log(S/B)$ (smoothed+mono)")

        for xedge in ee:
            ax.axvline(xedge, linestyle='--', linewidth=0.8)

        _annotate_bin_edges(ax, edges_plot, nb_plot, y_axis_pos=1.01)
        plt.xlabel(x_label)
        plt.ylabel(r"$t(u)=\log(S/B)$")
        plt.legend()
        _cms_label(cms_label, year=year, lumi=lumi_text)
        out3 = os.path.join(outdir, f"t_curve{('_'+tag) if tag else ''}.png")
        fig3.savefig(out3, dpi=220, bbox_inches='tight')
        plt.close(fig3)
        print(" -", out3)

        return out1, out2

# ----------------------------
# CLI
# ----------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build NumPy-based bins for optimized P_w(sig|x) and plot with mplhep (no systematics).")
    ap.add_argument('--mode', choices=['b','c'], required=True)
    ap.add_argument('--ttlj', required=True, help="TTLJ spec: file.root:Tree/Path")
    ap.add_argument('--vcb',  required=True, help="Vcb  spec: file.root:Tree/Path")
    ap.add_argument('--branch-name', type=str, default='template_score_4Class')

    ap.add_argument('--sig-class', type=int, default=0)

    # weights
    ap.add_argument('--w1', type=float, default=None)
    ap.add_argument('--w2', type=float, default=None)
    ap.add_argument('--w3', type=float, default=None)
    ap.add_argument('--weights-json', type=str, default=None,
                    help='Path to optimizer summary JSON (auto-read best weights)')

    # binning method & params
    ap.add_argument('--method', choices=['uniform_score','equal_b','equal_s','equal_info','t_equal'],
                    default='equal_info')
    ap.add_argument('--nbins', type=int, default=12)
    ap.add_argument('--fine', type=int, default=1024)
    ap.add_argument('--smooth', type=int, default=3)
    ap.add_argument('--b-floor', type=float, default=1e-12)
    ap.add_argument('--t-clip-low', type=float, default=1.0)
    ap.add_argument('--t-clip-high', type=float, default=99.0)

    # optional Neff merge
    ap.add_argument('--min-neff-s', type=float, default=0.0)
    ap.add_argument('--min-neff-b', type=float, default=0.0)

    # plotting (mplhep)
    ap.add_argument('--plot-dir', type=str, default=None)
    ap.add_argument('--hep-style', type=str, default='CMS', help='CMS/ATLAS/ROOT')
    ap.add_argument('--label', type=str, default='Preliminary', help='CMS label text')
    ap.add_argument('--year', type=str, default=None)
    ap.add_argument('--lumi', type=str, default=None, help='e.g. "2017 (13 TeV)" or "41.5 fb^{-1} (13 TeV)"')

    # outputs (JSON only)
    ap.add_argument('--save-bins', type=str, default=None,
                    help='Path to write JSON (edges + metadata)')
    ap.add_argument('--u-clip-low',  type=float, default=0.0,
                    help='Clip P_w(sig|x) from below before binning')
    ap.add_argument('--u-clip-high', type=float, default=1.0,
                    help='Clip P_w(sig|x) from above before binning')
    ap.add_argument('--u-rescale', action='store_true',
                    help='After clipping to [low,high], linearly map that interval to [0,1] for binning & plots')

    return ap.parse_args()

def _weights_from_json(path: str) -> Tuple[float,float,float]:
    with open(path, "r") as f:
        data = json.load(f)
    src = data.get("best") or (data.get("topk") or [{}])[0]
    return float(src["w1"]), float(src["w2"]), float(src["w3"])

def _load_samples(ttlj_spec: str, vcb_spec: str, mode: str, branch_name: str) -> Tuple[Dict, Dict]:
    need = ['classif_0','classif_1','classif_2','classif_3','weight_base']
    if USE_MCDATA_IO:
        TTLJ, Vcb = load_signal_and_background(
            ttlj_spec=ttlj_spec, vcb_spec=vcb_spec,
            mode=mode, branch_name=branch_name, num_classes=4
        )
    else:
        if uproot is None:
            raise RuntimeError("Install 'uproot awkward' or provide mcdata_io.")
        TTLJ = _load_arrays(ttlj_spec, need)
        Vcb  = _load_arrays(vcb_spec,  need)
    return TTLJ, Vcb

def main():
    args = _parse_args()

    # Load samples
    TTLJ, Vcb = _load_samples(args.ttlj, args.vcb, args.mode, args.branch_name)

    # Resolve weights
    if args.weights_json:
        w1, w2, w3 = _weights_from_json(args.weights_json)
    else:
        if (args.w1 is None) or (args.w2 is None) or (args.w3 is None):
            raise SystemExit("Provide --w1 --w2 --w3 or --weights-json")
        w1, w2, w3 = float(args.w1), float(args.w2), float(args.w3)
    w = np.array([1.0, w1, w2, w3], dtype=np.float64)

    # Compute bins (NO systematics)
    edges = compute_bins(
        Vcb, TTLJ, w,
        sig_class=args.sig_class,
        method=args.method,
        nbins=args.nbins,
        fine=args.fine,
        smooth=args.smooth,
        b_floor=args.b_floor,
        t_clip_low=args.t_clip_low,
        t_clip_high=args.t_clip_high,
        min_neff_s=args.min_neff_s,
        min_neff_b=args.min_neff_b,
        u_clip_low=args.u_clip_low,
        u_clip_high=args.u_clip_high,
        u_rescale=args.u_rescale)
    

    print("\n=== Bin edges (len={}): ===".format(len(edges)-1))
    print(edges)

    # Save JSON (edges + metadata)
    # Save JSON (edges + metadata)
    if args.save_bins:
        payload = {
            "method": args.method,
            "weights": [1.0, w1, w2, w3],
            "sig_class": int(args.sig_class),
            "nbins": int(len(edges) - 1),
            "notes": "Binning for P_w(sig|x). No systematics used in binning.",
            "u_clip": [float(args.u_clip_low), float(args.u_clip_high)],
            "u_rescale": bool(args.u_rescale),
        }

        # if rescaled, 저장 시 원래 u축으로 환원한 버전도 함께 기록
        if args.u_rescale and (args.u_clip_high - args.u_clip_low) < 1.0:
            u_lo, u_hi = float(args.u_clip_low), float(args.u_clip_high)
            edges_orig = (np.array(edges, dtype=np.float64) * (u_hi - u_lo)) + u_lo
            payload["edges_rescaled"] = np.asarray(edges, dtype=np.float64).tolist()
            edges_orig_list = edges_orig.tolist()
            edges_orig_list[-1] = 1.0  # 끝점은 항상 1.0으로 고정
            payload["edges"] = edges_orig.tolist()   # 기본은 '원래 u축' 기준으로 저장
            
        else:
            edges_list = edges.tolist()
            edges_list[-1] = 1.0  # 끝점은 항상 1.0으로 고정
            payload["edges"] = edges_list

        with open(args.save_bins, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Saved JSON] {args.save_bins}")

    # Plots with mplhep
    if args.plot_dir:
        tag = f"w=[1,{w1:.3g},{w2:.3g},{w3:.3g}]_{args.method}_nb{args.nbins}"
        p1, p2 = plot_with_bins(
            Vcb, TTLJ, w, edges,
            outdir=args.plot_dir, tag=tag,
            hep_style=args.hep_style, cms_label=args.label,
            year=args.year, lumi_text=args.lumi,
            u_clip_low=args.u_clip_low,
            u_clip_high=args.u_clip_high,
            u_rescale=args.u_rescale,
            # --- NEW: t_equal 관련 파라미터 전달 ---
            method=args.method,
            fine=args.fine,
            b_floor=args.b_floor,
            t_clip_low=args.t_clip_low,
            t_clip_high=args.t_clip_high
        )
        
        print("\nWrote plots:")
        print(" -", p1)
        print(" -", p2)
        # t_equal이면 위에서 t_curve 경로도 추가로 한 줄 출력됨

if __name__ == "__main__":
    main()

# ----------------------------
# 예시 실행
# ----------------------------
# JSON=$(ls -t ./SPANeasfsat/summary_*.json | head -n1)
# python BinningFromOptimizedScore_mplhep.py \
#   --mode b \
#   --ttlj "/gv0/.../Vcb_TTLJ_powheg.root:Mu/Central/Result_Tree" \
#   --vcb  "/gv0/.../Vcb_TTLJ_WtoCB_powheg.root:Mu/Central/Result_Tree" \
#   --weights-json "$JSON" \
#   --branch-name template_score_4Class \
#   --method equal_info --nbins 12 --fine 1024 --smooth 3 \
#   --min-neff-b 20 --min-neff-s 5 \
#   --plot-dir ./SPANeasfsat/bins_equal_info \
#   --save-bins ./SPANeasfsat/bins_equal_info.json \
#   --hep-style CMS --label Preliminary --year 2017 --lumi "2017 (13 TeV)"