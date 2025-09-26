import array
from pathlib import Path
from time import perf_counter
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import ROOT
import itertools
import plotly.graph_objs as go
from sklearn.metrics import roc_curve, roc_auc_score
import mplhep as hep
from scipy.stats import ks_2samp

ROOT.gStyle.SetOptStat(0)
ROOT.EnableImplicitMT(16)
# ROOT.gROOT.LoadMacro(os.path.join(os.environ["DIR_PATH"], "tdrStyle.C"))
ROOT.gROOT.LoadMacro("/data6/Users/yeonjoon/VcbMVAStudy/tdrStyle.C")

ROOT.gROOT.ProcessLine("setTDRStyle();")


def shape_test(set1, set2):
    return True if set1.shape[0] == set2.shape[0] else False


def shapeCheck(data_list):
    shape_matched = True
    for p in itertools.combinations(data_list, 2):
        shape_matched &= shape_test(p[0], p[1])
    return shape_matched

def npcoversion(*args):
    """Return numpy arrays for all inputs (no in-place modification)."""
    return tuple(np.array(data) for data in args)

def seperate_sig_bkg(df, branch="", target_Branch="y"):
    sig_value = df[df[target_Branch] == 0][branch].values
    bkg_value = df[df[target_Branch] == 1][branch].values
    sig_idx = df[df[target_Branch] == 0][branch].index
    bkg_idx = df[df[target_Branch] == 1][branch].index

    return sig_value, bkg_value, sig_idx, bkg_idx


# Colorblind-safe (Okabe–Ito)
ROC_COLOR   = "#0072B2"  # blue
BASELINE    = "#7F7F7F"  # gray

def ROC_AUC(score, y, plot_path, weight=None, fname="ROC.png", style="CMS",
            scale="linear", labels=None,
            title=None, subtitle=None,
            extra_text=None, extra_loc="upper left", extra_kwargs=None,
            legend_loc="lower right"):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    try:
        import mplhep as hep
        hep.style.use(style)
    except Exception:
        pass

    # normalize inputs to list-of-series
    is_multi = isinstance(score, (list, tuple))
    if not is_multi:
        score_list  = [score]
        y_list      = [y]
        weight_list = [weight]
    else:
        score_list  = list(score)
        y_list      = list(y) if isinstance(y, (list, tuple)) else [y]*len(score_list)
        weight_list = list(weight) if isinstance(weight, (list, tuple)) else [weight]*len(score_list)

    # fallback colors if user constants are undefined
    try:
        BASELINE
    except NameError:
        BASELINE = "0.5"
    try:
        ROC_COLOR
    except NameError:
        ROC_COLOR = None  # let matplotlib choose

    fig, ax = plt.subplots(figsize=(12.0, 9.0), dpi=150)

    # random baseline
    if scale == "log":
        y = np.logspace(-2, 0, 300)
        ax.plot(y, y, ls="--", lw=1.5, label="Random")
    else:
        ax.plot([0, 1], [0, 1], ls="--", lw=1.5, label="Random")

    aucs = []
    for i, (s, yy, w) in enumerate(zip(score_list, y_list, weight_list)):
        s  = np.asarray(s, dtype=np.float32)
        yy = np.asarray(yy, dtype=np.int8)
        if w is not None:
            w = np.asarray(w, dtype=np.float64)
            mask = np.isfinite(s) & np.isfinite(yy) & np.isfinite(w) & (w > 0)
            s, yy, w = s[mask], yy[mask], w[mask]
        else:
            mask = np.isfinite(s) & np.isfinite(yy)
            s, yy = s[mask], yy[mask]
            w = None

        fpr, tpr, _ = roc_curve(yy, s, sample_weight=w, drop_intermediate=True)
        auc = roc_auc_score(yy, s, sample_weight=w)
        aucs.append(auc)

        lab = labels[i] if (labels and i < len(labels)) else (f"ROC (AUC = {auc:.3f})" if not is_multi else f"Fold {i} (AUC = {auc:.3f})")
        ax.plot(fpr, tpr, lw=2.2, label=lab)

    # axes cosmetics
    if scale == "log":
        ax.set_xscale("log")
        ax.set_xlim(1e-2, 1.0)
    else:
        ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc=legend_loc)

    # CMS-like label (safe to skip if mplhep missing)
    try:
        hep.cms.label(llabel="Preliminary", data=False, com=13, ax=ax)
    except Exception:
        pass

    # Title & Subtitle
    if title:
        ax.set_title(title, loc="left", fontsize=18, pad=10)
        if subtitle:
            # subtitle를 제목 바로 아래에 살짝 작게
            ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)

    # Extra text (multi-line supported)
    if extra_text:
        if isinstance(extra_text, (list, tuple)):
            txt = "\n".join(map(str, extra_text))
        else:
            txt = str(extra_text)

        # 위치 해석
        loc_map = {
            "upper left":  (0.02, 0.98, "left",  "top"),
            "upper right": (0.98, 0.98, "right", "top"),
            "lower left":  (0.02, 0.02, "left",  "bottom"),
            "lower right": (0.98, 0.02, "right", "bottom"),
        }
        x, y, ha, va = loc_map.get(extra_loc, loc_map["upper left"])

        kw = dict(
            transform=ax.transAxes,
            ha=ha, va=va,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.6)
        )
        if extra_kwargs:
            kw.update(extra_kwargs)

        ax.text(x, y, txt, **kw)

    # save
    os.makedirs(plot_path, exist_ok=True)
    out_path = os.path.join(plot_path, fname)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return aucs if is_multi else aucs[0]

def NormalizeHist(sig, bkg):
    if sig.GetSumw2N() == 0:
        sig.Sumw2()
    if bkg.GetSumw2N() == 0:
        bkg.Sumw2()

    if sig.GetSumOfWeights() != 0:
        dx = (sig.GetXaxis().GetXmax() - sig.GetXaxis().GetXmin()) / sig.GetNbinsX()
        sig.Scale(1.0 / sig.GetSumOfWeights() / dx)

    if bkg.GetSumOfWeights() != 0:
        dx = (bkg.GetXaxis().GetXmax() - bkg.GetXaxis().GetXmin()) / bkg.GetNbinsX()
        bkg.Scale(1.0 / bkg.GetSumOfWeights() / dx)

def th1_to_numpy(h):
    ax = h.GetXaxis()
    nb = h.GetNbinsX()
    # bin edges
    edges = np.fromiter((ax.GetBinLowEdge(i) for i in range(1, nb + 1)), dtype=float, count=nb)
    edges = np.append(edges, ax.GetBinUpEdge(nb))
    # bin contents
    vals = np.fromiter((h.GetBinContent(i) for i in range(1, nb + 1)), dtype=float, count=nb)
    errs = np.fromiter((h.GetBinError(i) for i in range(1, nb + 1)), dtype=float, count=nb)
    return edges, vals, errs

def _unit_normalize(vals):
    s = vals.sum()
    return vals / s if s > 0 else vals


def draw_overall_mplhep(
    hist_train_sig=None, hist_val_sig=None,
    hist_train_bkg=None, hist_val_bkg=None,
    plot_dir="plots", fname_base="overall", normalize=True,
    lumi_text="138", sqrts=13, cms_extra="Preliminary",
    kolS=None, kolB=None, dpi=300, postfix=""
):
    """
    전달된 ROOT.TH1*만 그려서 저장(mplhep, CMS 스타일).
      - normalize=True: 1/Events로 정규화(적분 0이면 스킵)
      - kolS, kolB: KS p-value(문자/숫자). 제공 시에만 표기
      - 선형/로그 두 장 저장: {fname_base}_{postfix}.png, ..._log.png
    """
    import os, numpy as np, matplotlib.pyplot as plt, mplhep as hep

    os.makedirs(plot_dir, exist_ok=True)
    hep.style.use(hep.style.CMS)

    # 없으면 아예 종료
    if not any([hist_train_sig, hist_val_sig, hist_train_bkg, hist_val_bkg]):
        print("[draw_overall_mplhep] No histograms provided; nothing to draw.")
        return

    def _clone_and_maybe_norm(h):
        if h is None:
            return None
        h2 = h.Clone()
        h2.SetDirectory(0)
        if normalize:
            integ = float(h2.Integral())
            if integ > 0.0:
                h2.Scale(1.0 / integ)
            else:
                # 적분 0이면 정규화 스킵
                pass
        return h2

    # 정규화/복제
    h_ts = _clone_and_maybe_norm(hist_train_sig)
    h_vs = _clone_and_maybe_norm(hist_val_sig)
    h_tb = _clone_and_maybe_norm(hist_train_bkg)
    h_vb = _clone_and_maybe_norm(hist_val_bkg)

    # numpy 변환 유틸 기대: th1_to_numpy(h) -> (edges, values, errors)
    def _to_np(h):
        if h is None:
            return None
        b, v, e = th1_to_numpy(h)
        return (np.asarray(b), np.asarray(v), np.asarray(e))

    ns = { "tr_sig": _to_np(h_ts), "va_sig": _to_np(h_vs),
           "tr_bkg": _to_np(h_tb), "va_bkg": _to_np(h_vb) }

    # 클래스별(시그널/백그라운드)로만 bin 일치 확인
    def _check_pair(k1, k2, label):
        if ns[k1] is None or ns[k2] is None:
            return
        b1, _, _ = ns[k1]; b2, _, _ = ns[k2]
        if not np.allclose(b1, b2):
            raise ValueError(f"{label}: train/val binning mismatch.")

    _check_pair("tr_sig", "va_sig", "Signal")
    _check_pair("tr_bkg", "va_bkg", "Background")

    # 색상
    c_sig_t = "#009E73"; c_sig_v = "#007656"
    c_bkg_t = "#D55E00"; c_bkg_v = "#A04600"

    def _plot_one(ax, logy=False):
        # 백그라운드: train fill / val step
        if ns["tr_bkg"] is not None:
            b,v,e = ns["tr_bkg"]
            hep.histplot(v, bins=b, yerr=e, histtype="errorbar",
                                               alpha=0.55, label="Train (bkg.)", ax=ax, color=c_bkg_t)

        if ns["va_bkg"] is not None:
            b,v,e = ns["va_bkg"]; hep.histplot(v, bins=b, yerr=e, histtype="step",
                                               lw=2.5, label="Validation (bkg.)", ax=ax, color=c_bkg_v)

        # 시그널: train fill / val step
        if ns["tr_sig"] is not None:
            b,v,e = ns["tr_sig"]; hep.histplot(v, bins=b, yerr=e, histtype="errorbar",
                                               alpha=0.55, label="Train (sig.)", ax=ax, color=c_sig_t)
        if ns["va_sig"] is not None:
            b,v,e = ns["va_sig"]; hep.histplot(v, bins=b, yerr=e, histtype="step",
                                               lw=2.5, label="Validation (sig.)", ax=ax, color=c_sig_v)

        ax.set_xlabel("Score", fontsize=12)
        ax.set_ylabel("1 / Events" if normalize else "Events", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(frameon=False, ncols=2, loc="upper right")

        # CMS 라벨
        hep.cms.label(ax=ax, llabel=cms_extra, data=False, lumi=lumi_text, com=sqrts)

        # KS 텍스트(제공된 것만)
        txts = []
        if kolS is not None and (ns["tr_sig"] is not None and ns["va_sig"] is not None):
            txts.append(f"KS (sig) p = {kolS}")
        if kolB is not None and (ns["tr_bkg"] is not None and ns["va_bkg"] is not None):
            txts.append(f"KS (bkg) p = {kolB}")
        if txts:
            ax.text(0.015, 0.94, "  ".join(txts), transform=ax.transAxes, fontsize=11, va="top")

        if logy:
            ax.set_yscale("log")
            # 현재 플롯된 모든 양수 값 중 최소치를 찾아서 ymin 설정
            positives = []
            for key in ("tr_sig","va_sig","tr_bkg","va_bkg"):
                if ns[key] is None: continue
                _, v, _ = ns[key]
                pv = v[v > 0]
                if pv.size: positives.append(pv.min())
            ymin = min(positives) if positives else 1e-7
            ax.set_ylim(max(ymin * 0.5, 1e-7), None)
        else:
            positives = []
            for key in ("tr_sig","va_sig","tr_bkg","va_bkg"):
                if ns[key] is None: continue
                _, v, _ = ns[key]
                pv = v[v > 0]
                if pv.size: positives.append(pv.max())
            ymax = max(positives) if positives else 1.0
            ax.set_ylim(0.0, ymax * 1.2)

    # 저장 파일명 구성
    base = fname_base if not postfix else f"{fname_base}_{postfix}"

    # 선형
    fig1, ax1 = plt.subplots(figsize=(10, 8), dpi=dpi)
    _plot_one(ax1, logy=False)
    fig1.tight_layout()
    fig1.savefig(os.path.join(plot_dir, f"{base}.png"), bbox_inches="tight")
    plt.close(fig1)

    # 로그
    fig2, ax2 = plt.subplots(figsize=(10, 8), dpi=dpi)
    _plot_one(ax2, logy=True)
    fig2.tight_layout()
    fig2.savefig(os.path.join(plot_dir, f"{base}_log.png"), bbox_inches="tight")
    plt.close(fig2)


def draw_feature_importance_mplhep(
    varlist,
    feature_importance,
    plot_dir="plots",
    fname_base="feature_importance",
    *,
    top_k=None,                # 상위 k개만 표기 (None이면 전체)
    normalize=True,            # 중요도 합=1로 정규화
    absolute=True,             # 중요도에 절댓값 적용 (음수 SHAP 등 대비)
    sort_desc=True,            # 내림차순 정렬
    dpi=160,
    cms_extra="Preliminary",
    lumi_text="138",
    sqrts=13,
    show_values=True           # 막대 끝에 값 라벨
):
    """
    varlist: List[str] — 특징 이름들
    feature_importance: array-like — 각 특징의 중요도 (varlist와 길이 동일)

    저장: {plot_dir}/{fname_base}.png (+ pdf)
    반환: 저장 경로 dict
    """
    os.makedirs(plot_dir, exist_ok=True)
    hep.style.use(hep.style.CMS)

    # --- 입력 정리 ---
    names = list(varlist)
    imp = np.asarray(feature_importance, dtype=float).copy()

    if imp.ndim != 1:
        raise ValueError(f"feature_importance must be 1-D, got shape {imp.shape}")
    if len(names) != imp.size:
        raise ValueError(f"len(varlist)={len(names)} != len(feature_importance)={imp.size}")

    # NaN/Inf 방지
    imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)

    if absolute:
        imp = np.abs(imp)
    if normalize:
        s = imp.sum()
        if s > 0:
            imp = imp / s

    # 정렬 및 top-k
    order = np.argsort(imp)
    if sort_desc:
        order = order[::-1]
    if top_k is not None:
        order = order[:top_k]

    names_sorted = [names[i] for i in order]
    imp_sorted   = imp[order]

    # --- 그림 크기 동적 설정 (가로 막대: 변수 많을수록 세로 키움) ---
    n = len(names_sorted)
    height = max(2.8, min(0.45 * n + 1.2, 16.0))   # 합리적 범위
    fig, ax = plt.subplots(figsize=(10, height), dpi=dpi)

    # --- 플롯 ---
    y = np.arange(n)
    ax.barh(y, imp_sorted, align="center", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted)
    ax.invert_yaxis()  # 가장 중요한 게 위로 오도록
    ax.set_xlabel("Feature Importance" + (" (normalized)" if normalize else ""))
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.margins(x=0.02)

    # 값 표시(우측 끝)
    if show_values:
        for yi, val in zip(y, imp_sorted):
            if normalize:
                txt = f"{val*100:.1f}%"
            else:
                txt = f"{val:.3g}"
            ax.text(val + (0.01 if normalize else 0.0), yi, txt,
                    va="center", ha="left", fontsize=10)

    # CMS 라벨
    hep.cms.label(ax=ax, llabel=cms_extra, data=False,
                  lumi=lumi_text, com=sqrts)

    fig.tight_layout()
    out_png = os.path.join(plot_dir, f"{fname_base}.png")
    out_pdf = os.path.join(plot_dir, f"{fname_base}.pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return {"png": out_png, "pdf": out_pdf}

def _neff(th1):
    s = sum(th1.GetBinContent(i) for i in range(1, th1.GetNbinsX()+1))
    w2 = sum(th1.GetBinError(i)**2 for i in range(1, th1.GetNbinsX()+1))
    return (s*s)/max(w2,1e-12)
    

def numpy_hist_to_root(counts, sumW2, bins, name="histo", title=""):
    # sanity
    bins  = np.asarray(bins,  dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    sumW2  = np.asarray(sumW2,  dtype=np.float64)
    nbins = len(bins) - 1
    assert len(counts) == nbins and len(sumW2) == nbins, "bin size mismatch"

    # variable-bin constructor (한 번만!)
    edges = array.array("d", bins.tolist())
    h = ROOT.TH1D(name, title, nbins, edges)
    h.SetDirectory(0)     # gDirectory에서 분리
    h.Sumw2()             # error storage 확보

    # fill contents/errors
    for i in range(nbins):
        c   = float(counts[i])
        e2  = float(sumW2[i]) if np.isfinite(sumW2[i]) and sumW2[i] >= 0.0 else 0.0
        h.SetBinContent(i+1, c)
        h.SetBinError(i+1, e2**0.5)

    return h

import numpy as np

def quantile_edges(*arrays, nbins=50, data_range=(0.0, 1.0)):
    """
    Make variable-width bin edges by (unweighted) quantiles.
    Pass the two samples being compared (e.g., train and val for one class).
    Returns strictly non-decreasing edges suitable for np.histogram / TH1D.
    """
    xs = [np.asarray(a, dtype=np.float64).ravel() for a in arrays if a is not None]
    x = np.concatenate(xs) if xs else np.empty(0)
    x = x[np.isfinite(x)]

    lo, hi = data_range if data_range is not None else (np.min(x), np.max(x))
    if x.size == 0 or lo == hi:
        return np.array([lo, hi], dtype=np.float64)

    # keep within range (scores가 [0,1]이면 (0,1) 그대로 두세요)
    x = np.clip(x, lo, hi)

    # nbins+1 quantiles → edges
    q = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(x, q, method="linear")

    # 중복 edge 정리 (동일 값이 많을 때)
    edges = np.unique(edges)
    # 양끝 고정
    if edges[0] > lo:  edges = np.r_[lo, edges]
    if edges[-1] < hi: edges = np.r_[edges, hi]

    # 최소 2개 보장
    if edges.size < 2:
        edges = np.array([lo, hi], dtype=np.float64)

    return edges

def KS_test(
    train_score=None,
    val_score=None,
    train_w=None,
    val_w=None,
    train_y=None,
    val_y=None,
    plotPath="",
    postfix="",
    use_weight=False
):
    # --- types to cut memory bandwidth ---
    score_t = np.asarray(train_score, dtype=np.float32)
    w_t     = np.asarray(train_w,     dtype=np.float32)
    y_t     = np.asarray(train_y,     dtype=np.int8)
    score_v = np.asarray(val_score,   dtype=np.float32)
    w_v     = np.asarray(val_w,       dtype=np.float32)
    y_v     = np.asarray(val_y,       dtype=np.int8)
    
    # --- is use_weight is set to false, weight is just unity ---
    if not use_weight:
        print(f"Ignore weights for KS test, all weight is treated as 1")
        w_t = np.ones_like(score_t)
        w_v = np.ones_like(score_v)

    # --- sanity checks ---
    n_eq = (score_t.shape[0] == w_t.shape[0] == y_t.shape[0]) and \
           (score_v.shape[0] == w_v.shape[0] == y_v.shape[0])
    if not n_eq:
        print("Shape mismatching")
        return

    # --- multithreading (use all cores unless already enabled) ---
    if not ROOT.ROOT.IsImplicitMTEnabled():
        ROOT.ROOT.EnableImplicitMT()


    # --- one RDF per split; filter by class inside RDF ---
    bins = np.linspace(0,1,50)
    bins_quantile_sig = quantile_edges(score_v[y_v==0], nbins=40)
    bins_quantile_bkg = quantile_edges(score_v[y_v==1],nbins=40)
    h_tr_sig, _ = np.histogram(score_t[(y_t == 0)], bins=bins_quantile_sig, weights=w_t[(y_t == 0)])
    h_tr_bkg, _ = np.histogram(score_t[(y_t == 1)], bins=bins_quantile_bkg, weights=w_t[(y_t == 1)])
    h_vl_sig, _ = np.histogram(score_v[(y_v == 0)], bins=bins_quantile_sig, weights=w_v[(y_v == 0)])
    h_vl_bkg, _ = np.histogram(score_v[(y_v == 1)], bins=bins_quantile_bkg, weights=w_v[(y_v == 1)])
    
    h_tr_sig_w2, _ = np.histogram(score_t[(y_t == 0)], bins=bins_quantile_sig, weights=w_t[(y_t == 0)]**2)
    h_tr_bkg_w2, _ = np.histogram(score_t[(y_t == 1)], bins=bins_quantile_bkg, weights=w_t[(y_t == 1)]**2)
    h_vl_sig_w2, _ = np.histogram(score_v[(y_v == 0)], bins=bins_quantile_sig, weights=w_v[(y_v == 0)]**2)
    h_vl_bkg_w2, _ = np.histogram(score_v[(y_v == 1)], bins=bins_quantile_bkg, weights=w_v[(y_v == 1)]**2)

    th1_tr_sig = numpy_hist_to_root(h_tr_sig, h_tr_sig_w2, bins_quantile_sig, name="h_tr_sig", title="Train signal")
    th1_tr_bkg = numpy_hist_to_root(h_tr_bkg, h_tr_bkg_w2, bins_quantile_bkg, name="h_tr_bkg", title="Train background")
    th1_vl_sig = numpy_hist_to_root(h_vl_sig, h_vl_sig_w2, bins_quantile_sig, name="h_vl_sig", title="Validation signal")
    th1_vl_bkg = numpy_hist_to_root(h_vl_bkg, h_vl_bkg_w2, bins_quantile_bkg, name="h_vl_bkg", title="Validation background")
    
    h_tr_sig_draw, _ = np.histogram(score_t[(y_t == 0)], bins=bins, weights=w_t[(y_t == 0)])
    h_tr_bkg_draw, _ = np.histogram(score_t[(y_t == 1)], bins=bins, weights=w_t[(y_t == 1)])
    h_vl_sig_draw, _ = np.histogram(score_v[(y_v == 0)], bins=bins, weights=w_v[(y_v == 0)])
    h_vl_bkg_draw, _ = np.histogram(score_v[(y_v == 1)], bins=bins, weights=w_v[(y_v == 1)])
    
    h_tr_sig_w2_draw, _ = np.histogram(score_t[(y_t == 0)], bins=bins, weights=w_t[(y_t == 0)]**2)
    h_tr_bkg_w2_draw, _ = np.histogram(score_t[(y_t == 1)], bins=bins, weights=w_t[(y_t == 1)]**2)
    h_vl_sig_w2_draw, _ = np.histogram(score_v[(y_v == 0)], bins=bins, weights=w_v[(y_v == 0)]**2)
    h_vl_bkg_w2_draw, _ = np.histogram(score_v[(y_v == 1)], bins=bins, weights=w_v[(y_v == 1)]**2)
    
    th1_tr_sig_draw = numpy_hist_to_root(h_tr_sig_draw, h_tr_sig_w2_draw, bins, name="h_tr_sig_draw", title="Train signal")
    th1_tr_bkg_draw = numpy_hist_to_root(h_tr_bkg_draw, h_tr_bkg_w2_draw, bins, name="h_tr_bkg_draw", title="Train background")
    th1_vl_sig_draw = numpy_hist_to_root(h_vl_sig_draw, h_vl_sig_w2_draw, bins, name="h_vl_sig_draw", title="Validation signal")
    th1_vl_bkg_draw = numpy_hist_to_root(h_vl_bkg_draw, h_vl_bkg_w2_draw, bins, name="h_vl_bkg_draw", title="Validation background")
    
    print("Neff(train sig)=", _neff(th1_tr_sig), "Neff(val sig)=", _neff(th1_vl_sig))
    print("Neff(train bkg)=", _neff(th1_tr_bkg), "Neff(val bkg)=", _neff(th1_vl_bkg))

    # --- KS (use matching classes!) ---
    # "M" = (ROOT option for more accurate distance); use as you prefer
    kolS_max = th1_vl_sig.KolmogorovTest(th1_tr_sig, "M")
    kolB_max = th1_vl_bkg.KolmogorovTest(th1_tr_bkg, "M")

    # symmetric p-values (their "X" opt returns the p-value)
    kolS = th1_tr_sig.KolmogorovTest(th1_vl_sig, "X")
    kolB = th1_tr_bkg.KolmogorovTest(th1_vl_bkg, "X")
    #kolS = th1_vl_sig.KolmogorovTest(th1_tr_sig, "X")
    #kolB = th1_vl_bkg.KolmogorovTest(th1_tr_bkg, "X")
    print(f"Signal, dist = {kolS_max}, p-value={kolS}")
    print(f"Background, dist = {kolB_max}, p-value={kolB}")
    

    draw_overall_mplhep(th1_tr_sig_draw, th1_vl_sig_draw, th1_tr_bkg_draw, th1_vl_bkg_draw, plot_dir=plotPath, kolS=kolS, kolB=kolB, postfix=postfix)
    #draw_overall_mplhep(th1_tr_sig, th1_vl_sig, None, None, plot_dir=plotPath, kolS=kolS, kolB=None, postfix=postfix+"_sig_quantile")
    #draw_overall_mplhep(None, None, th1_tr_bkg, th1_vl_bkg, plot_dir=plotPath, kolS=None, kolB=kolB, postfix=postfix+"_bkg_quantile")
    return kolS, kolB
   
# def getECDF(data1, data2):
#     y1 = []
#     y2 = []
#     length1 = len(data1)
#     length2 = len(data2)
#     a = min(min(data1), min(data2))
#     b = max(max(data1), max(data2))
#     width = b - a
#     a = a - width * 0.3
#     b = b + width * 0.3
#     x = np.linspace(a, b, 100000)
#     for i in x:
#         smaller1 = (data1 < i).sum()
#         y1.append(smaller1 / length1)
#         smaller2 = (data2 < i).sum()
#         y2.append(smaller2 / length2)
#     return np.array(x), np.array(y1), np.array(y2)

def getECDF(data1, data2):
    """Return a common x-grid and ECDFs for two 1D arrays, vectorized and fast.
    x is the sorted union of unique values from both arrays (with guard rails)."""
    a1 = np.asarray(data1).ravel()
    a2 = np.asarray(data2).ravel()
    if a1.size == 0 or a2.size == 0:
        raise ValueError("getECDF requires non-empty arrays")
    # Build x grid from union plus small guard rails
    x_core = np.union1d(np.unique(a1), np.unique(a2))
    # add guard rails at ends to see full vertical KS line
    width = x_core[-1] - x_core[0]
    if width == 0:
        width = 1.0
    x = np.concatenate([[x_core[0] - 0.3*width], x_core, [x_core[-1] + 0.3*width]])
    # ECDF via searchsorted
    y1 = np.searchsorted(np.sort(a1), x, side='right') / a1.size
    y2 = np.searchsorted(np.sort(a2), x, side='right') / a2.size
    return x, y1, y2


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def Draw_KS_test(train_score, val_score, plotPath="img.png", isSig=True, kol=0,
                 cms_llabel="Preliminary", lumi=None, year=None):
    """
    Plot a KS test in CMS style (mplhep) with ECDFs of train vs. validation.
    - train_score, val_score: 1D arrays
    - plotPath: save path (.png, .pdf, …)
    - isSig: True -> 'Signal' title, False -> 'Bkg.'
    - kol: extra number to annotate (user field)
    - cms_llabel: 'Preliminary'/'Internal'/etc.
    - lumi (e.g., '138 fb$^{-1}$'), year (e.g., 2018) optional
    """
    # --- sanitize & numpyfy ---
    tr = np.asarray(train_score).ravel()
    va = np.asarray(val_score).ravel()
    if tr.size == 0 or va.size == 0:
        raise ValueError("Draw_KS_test: empty input")
    if not np.isfinite(tr).all() or not np.isfinite(va).all():
        raise ValueError("Draw_KS_test: non-finite values in input")

    # --- ECDFs on a shared x-grid ---
    x, y_tr, y_va = getECDF(tr, va)

    # --- KS test (scipy) ---
    ks = ks_2samp(tr, va, alternative="two-sided", mode="auto")
    D = float(ks.statistic)

    # 위치: scipy가 제공하면 사용, 아니면 ECDF 차이의 argmax로 추정
    try:
        x_star = float(ks.statistic_location)
    except Exception:
        x_star = float(x[np.argmax(np.abs(y_tr - y_va))])

    # 수직선 y값(정확한 ECDF값) — 위치에서의 누적분포를 직접 계산
    tr_sorted = np.sort(tr)
    va_sorted = np.sort(va)
    y_tr_star = np.searchsorted(tr_sorted, x_star, side="right") / tr.size
    y_va_star = np.searchsorted(va_sorted, x_star, side="right") / va.size

    # --- CMS style ---
    hep.style.use(hep.style.CMS)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=160)

    # ECDF는 계단(step)으로
    ax.step(x, y_tr, where="post", linewidth=1.8, label="Train")
    ax.step(x, y_va, where="post", linewidth=1.8, label="Validation")

    # KS 수직선
    ax.plot([x_star, x_star], [y_tr_star, y_va_star],
            linestyle="--", linewidth=2, label=f"KS $D$ = {D:.3f}")

    # 발표용 레이블
    title = f"{'Signal' if isSig else 'Bkg.'}  |  KS D={D:.3f}  |  kol={kol}"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Score")
    ax.set_ylabel("ECDF")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=False, loc="lower right")

    # CMS 라벨 (가능하면 활용)
    try:
        # 예: lumi="138 fb$^{-1}$", year=2018
        hep.cms.label(ax=ax, llabel=cms_llabel, data=True, lumi=lumi, year=year)
    except Exception:
        # mplhep 버전에 따라 label 인자가 다를 수 있어 안전장치
        hep.cms.text(cms_llabel)

    # 저장
    Path(plotPath).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plotPath, bbox_inches="tight")
    plt.close(fig)

    # 유용하게 KS 결과를 리턴
    return {"D": D, "x_star": x_star, "y_train": y_tr_star, "y_val": y_va_star}
