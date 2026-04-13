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
_TDR_STYLE = "/data6/Users/yeonjoon/VcbMVAStudy/tdrStyle.C"
if os.path.exists(_TDR_STYLE):
    ROOT.gROOT.LoadMacro(_TDR_STYLE)
    ROOT.gROOT.ProcessLine("setTDRStyle();")

try:
    hep.style.use(hep.style.CMS)
except Exception:
    pass


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


# Colorblind-safe (Okabe-Ito) palette for CMS-like plots
COLORS = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
BASELINE = "#7F7F7F"


def _apply_mplhep_style(style="CMS"):
    try:
        if style == "CMS":
            hep.style.use(hep.style.CMS)
        else:
            hep.style.use(getattr(hep.style, style, style))
    except Exception:
        pass


def _format_floatish(value, ndigits=3):
    try:
        return f"{float(value):.{ndigits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_roc_legend_label(label, auc, index, is_multi):
    if label is not None:
        label = str(label).strip()
        if label:
            return label if "auc" in label.lower() else f"{label} (AUC: {auc:.4f})"
    if is_multi:
        return f"Fold {index + 1} (AUC: {auc:.4f})"
    return f"AUC: {auc:.4f}"


def ROC_AUC(score, y, plot_path, weight=None, fname="ROC.png", style="CMS",
            scale="linear", labels=None, log_xmin=1e-2,
            title=None, subtitle=None,
            extra_text=None, extra_loc="upper left", extra_kwargs=None,
            legend_loc="lower right", legend_bbox_to_anchor=None,
            legend_ncols=1, legend_fontsize=16):
    _apply_mplhep_style(style)

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

    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=300)

    # random baseline
    if scale == "log":
        x_base = np.logspace(np.log10(log_xmin), 0, 300)
        ax.plot(x_base, x_base, ls="--", lw=2.0, color=BASELINE, label="Random")
    else:
        ax.plot([0, 1], [0, 1], ls="--", lw=2.0, color=BASELINE, label="Random")

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

        label = labels[i] if (labels and i < len(labels)) else None
        color = COLORS[i % len(COLORS)]
        lab = _format_roc_legend_label(label, auc, i, is_multi)
        ax.plot(fpr, tpr, lw=2.5, color=color, label=lab)

    # axes cosmetics
    if scale == "log":
        ax.set_xscale("log")
        ax.set_xlim(log_xmin, 1.0)
    else:
        ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Background Efficiency", fontsize=20)
    ax.set_ylabel("Signal Efficiency", fontsize=20)
    ax.grid(False)
    ax.tick_params(which="both", direction="in", top=True, right=True, length=6, width=1.2, labelsize=16)
    ax.tick_params(which="minor", length=3)
    legend_kwargs = dict(frameon=False, loc=legend_loc, fontsize=legend_fontsize, ncols=legend_ncols)
    if legend_bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
    ax.legend(**legend_kwargs)

    # CMS-like label (safe to skip if mplhep missing)
    try:
        hep.cms.label(llabel="Simulation Preliminary", data=False, com=13, ax=ax, fontsize=22)
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
            "upper left":  (0.04, 0.96, "left",  "top"),
            "upper right": (0.96, 0.96, "right", "top"),
            "lower left":  (0.04, 0.04, "left",  "bottom"),
            "lower right": (0.96, 0.04, "right", "bottom"),
        }
        x, y, ha, va = loc_map.get(extra_loc, loc_map["upper left"])

        kw = dict(
            transform=ax.transAxes,
            ha=ha, va=va,
            fontsize=16,
            bbox=dict(boxstyle="square,pad=0", fc="none", ec="none")
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
    os.makedirs(plot_dir, exist_ok=True)
    _apply_mplhep_style("CMS")

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
    c_sig_t = "#5790fc"; c_sig_v = "#002b80"
    c_bkg_t = "#e42536"; c_bkg_v = "#800000"

    def _plot_one(ax, logy=False):
        # 백그라운드: train fill / val step
        if ns["tr_bkg"] is not None:
            b,v,e = ns["tr_bkg"]
            hep.histplot(v, bins=b, yerr=e, histtype="errorbar",
                                               alpha=0.55, label="Train Bkg.", ax=ax, color=c_bkg_t)

        if ns["va_bkg"] is not None:
            b,v,e = ns["va_bkg"]; hep.histplot(v, bins=b, yerr=e, histtype="step",
                                               lw=2.5, label="Validation Bkg.", ax=ax, color=c_bkg_v)

        # 시그널: train fill / val step
        if ns["tr_sig"] is not None:
            b,v,e = ns["tr_sig"]; hep.histplot(v, bins=b, yerr=e, histtype="errorbar",
                                               alpha=0.55, label="Train Sig.", ax=ax, color=c_sig_t)
        if ns["va_sig"] is not None:
            b,v,e = ns["va_sig"]; hep.histplot(v, bins=b, yerr=e, histtype="step",
                                               lw=2.5, label="Validation Sig.", ax=ax, color=c_sig_v)

        ax.set_xlabel("MVA Score", fontsize=20)
        ax.set_ylabel("Arbitrary Units" if normalize else "Events", fontsize=20)
        ax.grid(False)
        ax.tick_params(which="both", direction="in", top=True, right=True, length=6, width=1.2, labelsize=16)
        ax.tick_params(which="minor", length=3)
        ax.legend(frameon=False, ncols=2, loc="upper center", fontsize=16)

        # CMS 라벨
        hep.cms.label(ax=ax, llabel=cms_extra, data=False, lumi=lumi_text, com=sqrts)

        # KS 텍스트(제공된 것만)
        txts = []
        if kolS is not None and (ns["tr_sig"] is not None and ns["va_sig"] is not None):
            txts.append(f"KS Sig. p-val = {_format_floatish(kolS)}")
        if kolB is not None and (ns["tr_bkg"] is not None and ns["va_bkg"] is not None):
            txts.append(f"KS Bkg. p-val = {_format_floatish(kolB)}")
        if txts:
            ax.text(0.95, 0.75, "\n".join(txts), transform=ax.transAxes,
                    fontsize=16, va="top", ha="right")

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
    fig1, ax1 = plt.subplots(figsize=(8, 8), dpi=dpi)
    _plot_one(ax1, logy=False)
    fig1.tight_layout()
    fig1.savefig(os.path.join(plot_dir, f"{base}.png"), bbox_inches="tight")
    plt.close(fig1)

    # 로그
    fig2, ax2 = plt.subplots(figsize=(8, 8), dpi=dpi)
    _plot_one(ax2, logy=True)
    fig2.tight_layout()
    fig2.savefig(os.path.join(plot_dir, f"{base}_log.png"), bbox_inches="tight")
    plt.close(fig2)
    
_JET_LATEX_LABELS = {
    # 복잡한 첨자 대신 간결한 표준 기호 사용
    "had_t_b": r"b_h",
    "w_u": r"q_U",
    "w_d": r"q_D",
    "lep_t_b": r"b_l",
}

_JET_OBSERVABLE_LATEX = {
    "pt": r"p_{\mathrm{T}}",
    "eta": r"\eta",
    "phi": r"\phi",
}

_JET_TAGGER_LATEX = {
    "bvsc": r"\mathrm{BvsC}",
    "cvsb": r"\mathrm{CvsB}",
    "cvsl": r"\mathrm{CvsL}",
}

def _wrap_math(expr):
    return rf"${expr}$"

def _feature_name_to_latex(name):
    prefix, sep, suffix = name.partition("_")
    if sep and suffix in _JET_LATEX_LABELS:
        jet_expr = _JET_LATEX_LABELS[suffix]
        if prefix in _JET_OBSERVABLE_LATEX:
            return _wrap_math(rf"{_JET_OBSERVABLE_LATEX[prefix]}({jet_expr})")
        if prefix in _JET_TAGGER_LATEX:
            return _wrap_math(rf"{_JET_TAGGER_LATEX[prefix]}({jet_expr})")

    special_labels = {
        # 질량 변수들은 개별 제트의 나열이 아닌, 재구성된 입자 이름으로 직관적으로 표현
        "m_had_w": _wrap_math(r"m(W_h)"),
        "m_had_t": _wrap_math(r"m(t_h)"),
        "n_bjets": _wrap_math(r"N_{\mathrm{b\text{-}jet}}"),
        "n_cjets": _wrap_math(r"N_{\mathrm{c\text{-}jet}}"),
        "n_jets": _wrap_math(r"N_{\mathrm{jet}}"),
        "best_mva_score": _wrap_math(r"\mathcal{D}_{\mathrm{Reco}}"),
        "least_dr_bb": _wrap_math(r"\min \Delta R(b,b)"),
        "least_m_bb": _wrap_math(r"\min m(b,b)"),
        "pt_tt": _wrap_math(r"p_{\mathrm{T}}(t\bar{t})"),
        "ht": _wrap_math(r"H_{\mathrm{T}}"),
        "year_index": "Data taking year", # 단순 Year index보다 논문에 적합
    }
    return special_labels.get(name, name)

def draw_feature_importance_mplhep(
    varlist,
    feature_importance,
    plot_dir="plots",
    fname_base="feature_importance_compact",
    *,
    top_k=None,
    normalize=True,
    absolute=True,
    sort_desc=True,
    dpi=300,
    cms_extra="Preliminary",
    lumi_text="138",
    sqrts=13,
    show_values=True
):
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. CMS 스타일 적용
    hep.style.use(hep.style.CMS)
    
    # 2. [핵심] 수식(MathText) 폰트만 Computer Modern (Serif)으로 덮어쓰기
    # 일반 텍스트(CMS 라벨 등)는 Helvetica(Sans-serif)로 유지됩니다.
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
    })

    # --- 입력 배열 정리 ---
    names = list(varlist)
    imp = np.asarray(feature_importance, dtype=float).copy()
    imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)

    if absolute:
        imp = np.abs(imp)
    if normalize:
        s = imp.sum()
        if s > 0:
            imp = imp / s

    order = np.argsort(imp)
    if sort_desc:
        order = order[::-1]
    if top_k is not None:
        order = order[:top_k]

    names_sorted = [_feature_name_to_latex(names[i]) for i in order]
    imp_sorted   = imp[order]

    n = len(names_sorted)
    
    # 3. [핵심] 레이아웃 컴팩트화
    # 그림의 세로 길이를 줄여서 라벨과 바(Bar)가 더 밀도 있게 모이도록 합니다.
    height = max(3.0, n * 0.45) 
    fig, ax = plt.subplots(figsize=(9, height), dpi=dpi)

    # 바 높이(두께)를 키우고(0.75), 간격을 좁힙니다.
    y = np.arange(n)
    bars = ax.barh(y, imp_sorted, align="center", height=0.75, alpha=0.9, color="#5790fc")
    
    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted, fontsize=16) # 라벨 폰트 크기 최적화
    ax.invert_yaxis()  
    
    xlabel_text = "Feature Importance" + (" (normalized)" if normalize else "")
    ax.set_xlabel(xlabel_text, fontsize=18)
    
    # 그리드 스타일 정리 (점선으로 약하게)
    ax.xaxis.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax.set_axisbelow(True)
    
    # 상하 여백을 확 줄여서 낭비되는 공간 최소화
    ax.margins(x=0.05, y=0.01)

    # 4. 값 텍스트 표시
    if show_values:
        max_val = np.max(imp_sorted)
        padding = max_val * 0.02 # 바 끝에서 텍스트까지의 간격
        
        for yi, val in zip(y, imp_sorted):
            if normalize:
                # % 기호 포맷팅 (레이텍 충돌 방지를 위해 이스케이프)
                txt = f"{val*100:.1f}\%" 
            else:
                txt = f"{val:.3g}"
            
            ax.text(val + padding, yi, txt,
                    va="center", ha="left", fontsize=13)

    # 5. CMS 헤더
    hep.cms.label(ax=ax, llabel=cms_extra, data=False,
                  lumi=lumi_text, com=sqrts, fontsize=18)

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
    _apply_mplhep_style("CMS")

    # --- figure ---
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    color_train = "#5790fc" if isSig else "#e42536"
    color_val = "#002b80" if isSig else "#800000"

    # ECDF는 계단(step)으로
    ax.step(x, y_tr, where="post", linewidth=2.5, label="Train", color=color_train)
    ax.step(x, y_va, where="post", linewidth=2.5, label="Validation", color=color_val)

    # KS 수직선
    ax.plot([x_star, x_star], [y_tr_star, y_va_star],
            linestyle="--", linewidth=2.5, color="black", label=f"KS $D$ = {D:.3f}")

    ax.set_xlabel("MVA Score", fontsize=20)
    ax.set_ylabel("Cumulative Probability", fontsize=20)
    ax.grid(False)
    ax.tick_params(which="both", direction="in", top=True, right=True, length=6, width=1.2, labelsize=16)
    ax.tick_params(which="minor", length=3)
    ax.text(0.05, 0.90, "Signal" if isSig else "Background",
            transform=ax.transAxes, fontsize=22, fontweight="bold")
    ax.legend(frameon=False, loc="center right", fontsize=16)

    # CMS 라벨 (가능하면 활용)
    try:
        hep.cms.label(ax=ax, llabel=cms_llabel, data=False, lumi=lumi, year=year, com=13)
    except Exception:
        try:
            hep.cms.label(ax=ax, llabel=cms_llabel, data=False, lumi=lumi, com=13)
        except Exception:
            hep.cms.text(cms_llabel)

    # 저장
    Path(plotPath).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plotPath, bbox_inches="tight")
    plt.close(fig)

    # 유용하게 KS 결과를 리턴
    return {"D": D, "x_star": x_star, "y_train": y_tr_star, "y_val": y_va_star}
