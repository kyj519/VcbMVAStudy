import uuid
import ROOT
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
import re
from collections import defaultdict
import os
import shutil
import tempfile


map_short_name = {}
#map_short_name["VJets"] = ["WJets", "DYJets"]
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

def getShortName(file):

    for short, keys in map_short_name.items():
        for key in keys:
            if key in file:
                return short
    raise ValueError(f"getShortName: No short name matched for file {file}. Check the file name.")


def addRFcolumn(df, file):
    e = ""
    if "2016preVFP" in file:
        e = "2016preVFP"
    elif "2016postVFP" in file:
        e = "2016postVFP"
    elif "2017" in file:
        e = "2017"
    elif "2018" in file:
        e = "2018"
    else:
        print("root_data_loader.py: addRFcolumn: Error: No year is matched. Check the file name.")
        exit()
    print(f"Calculate RF from {file}")
    short_mc = getShortName(file)
    print(f"{file} is matched to {short_mc}")

    if short_mc == "DATA":
        df = df.Define("weight_RFPatch", "1.f")
        return df

    if short_mc == "TTLJ":
        # whatMode만 필요 (isBB/isCC는 이제 안 씀. 남기고 싶으면 그대로 두셔도 무방)
        df = addHisto_idx(df)  # 여기서 whatMode 계산 (2/4/45)
        df = df.Define(
            "weight_RFPatch",
            f"""(whatMode == 2 ?  RFPatch_TTLJ_2_{e}->GetEventSF(Jet_Pt, Jet_Eta, Jet_Flavor) :
                (whatMode == 4 ?  RFPatch_TTLJ_4_{e}->GetEventSF(Jet_Pt, Jet_Eta, Jet_Flavor) :
                                  RFPatch_TTLJ_45_{e}->GetEventSF(Jet_Pt, Jet_Eta, Jet_Flavor)))"""
        )
    else:
        # whatMode 없음
        df = df.Define(
            "weight_RFPatch",
            f"RFPatch_{short_mc}_{e}->GetEventSF(Jet_Pt, Jet_Eta, Jet_Flavor)"
        )

    return df


def addHisto_idx(df):
    df = df.Define("isBB", "((genTtbarId % 100) >= 51 && (genTtbarId % 100) <= 55)")
    df = df.Define("isCC", "((genTtbarId % 100) >= 41 && (genTtbarId % 100) <= 45)")

    df = df.Define(
        "whatMode",
        "(decay_mode == 21 || decay_mode == 23) ? 2 : "
        "(decay_mode == 41 || decay_mode == 43) ? 4 : "
        "(decay_mode == 45 ? 45 : -1)"  # default -1로 보정
    )
    return df



YEAR_MAP = {
    "2016preVFP": 0,
    "2016postVFP": 1,
    "2017": 2,
    "2018": 3,
}
IDX_TO_YEAR = {v: k for k, v in YEAR_MAP.items()}

LUMI_MAP = {
    "2016preVFP": 16.81,
    "2016postVFP": 19.52,
    "2017": 41.53,
    "2018": 59.74,
}

def _detect_year_index(path: str) -> int:
    for k, v in YEAR_MAP.items():
        if k in path:
            return v
    raise ValueError(f"[year_index] No matching year in filename: {path}")

def _detect_syst(path: str) -> str:
    # 경로상 syst 힌트(폴더명) 추출, 필요시 규칙 보강
    parts = os.path.normpath(path).split(os.sep)
    for p in reversed(parts):
        pl = p.lower()
        if ("syst" in p) or ("up" in pl) or ("down" in pl) or ("central" in pl):
            return p
    return "Central"

_SAMPLE_RE = re.compile(r"Vcb_([^/]+?)\.root$")

def _extract_sample(path: str) -> str:
    m = _SAMPLE_RE.search(path)
    if m:
        return m.group(1)
    return os.path.basename(path).replace(".root", "")

def _year_fractions(present_year_idxs: np.ndarray):
    lumis = {yi: LUMI_MAP.get(IDX_TO_YEAR.get(int(yi), ""), 0.0) for yi in present_year_idxs}
    total = float(sum(lumis.values()))
    if total <= 0:
        # fallback to uniform if not found
        return {int(yi): 1.0 / len(present_year_idxs) for yi in present_year_idxs}
    return {int(yi): lumis[int(yi)] / total for yi in present_year_idxs}

def _stable_perm(ids: np.ndarray, seed: int, salt: int) -> np.ndarray:
    """Return a deterministic permutation index for `ids` using an LCG hash.
    We avoid RNG state so results are fully stable across platforms.
    """
    ids64 = ids.astype(np.uint64, copy=False)
    # 64-bit LCG parameters
    mixed = (ids64 * np.uint64(6364136223846793005) + np.uint64(1442695040888963407 + (seed + 1315423911 * salt)))
    # use bytes as pseudo-random key
    order = np.argsort(mixed)
    return order


def _downsample_to_lumi_ratio(eids: np.ndarray, years: np.ndarray, seed: int, salt: int) -> np.ndarray:
    """Return boolean mask over entries **within one class** so that the kept samples
    are downsampled to match LUMI_MAP ratios across present years.

    Strategy: choose the **largest feasible total** T = min_y floor(n_avail[y] / frac[y]) and
    set target[y] ≈ round(frac[y] * T) while not exceeding availability, using a stable order.
    """
    n = years.size
    if n == 0:
        return np.zeros(0, dtype=bool)

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
        return np.ones(n, dtype=bool)
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
        order = _stable_perm(eids[idx], seed=seed, salt=salt + yi)
        k = min(target.get(yi, 0), idx.size)
        keep[idx[order[:k]]] = True

    return keep

def _clean_weight_expr(df: ROOT.RDataFrame, use_btag: bool) -> str:
    # 존재하는 항만 곱해 안전하게 정의 (중복 제거)
    tokens = [
        ("weight_b_tag" if use_btag else "weight_c_tag"),
        "weight_el_id",
        "weight_mc",
        "weight_mu_id",
        "weight_mu_iso",
        "weight_el_reco",
        "weight_pileup",
        "weight_prefire",
        "weight_sl_trig",
        "weight_top_pt",
        "weight_RFPatch",
    ]
    present = [t for t in tokens if df.HasColumn(t)]
    return " * ".join(present) if present else "1."

def _ensure_columns(df: ROOT.RDataFrame, cols):
    # varlist에 있는데 스키마가 달라 빠진 컬럼은 0.0으로 보정 (대부분 float 가정)
    for c in cols:
        if not df.HasColumn(c):
            df = df.Define(c, "0.0")
    return df

def getRDataFrame(tree_path_filter_str, varlist, training_mode, add_year_index, loadBTag=True):
    """
    tree_path_filter_str: list of tuples (filepath, treename, filterstr)
    varlist: list[str] or []
    training_mode: bool
    add_year_index: bool
    loadBTag: True→weight_b_tag, False→weight_c_tag
    """
    if training_mode:
        print("Root2numpy loader on training mode")
        ROOT.EnableImplicitMT()
        print("ROOT.EnableImplicitMT().....OK.")
    else:
        print("Root2numpy loader on infering mode")
        print("Important! Multicore processing should not be used when infering.")
        print("It will shuffle the order of events.")
        ROOT.DisableImplicitMT()
        print("ROOT.DisableImplicitMT().....OK.")

    # (tree, filter, year, syst, sample) 별 그룹핑
    groups = defaultdict(list)
    for fp, tree, fstr in tree_path_filter_str:
        yidx = _detect_year_index(fp)
        syst = _detect_syst(fp)
        sample = _extract_sample(fp)
        key = (tree, fstr, yidx, syst, sample)
        groups[key].append(fp)

    #그룹 체크 위해 출력
    for (tree, fstr, yidx, syst, sample), files in groups.items():
        print(f"{tree}, Filter: {fstr}, Year: {yidx}, Syst: {syst}, Sample: {sample}, Files: {files}")
        
    results = []

    # ---- 훈련 모드: 파일별 addRFcolumn → 스냅샷 → 그룹 합쳐서 1회 AsNumpy ----
    if training_mode:
        for (tree, fstr, yidx, syst, sample), files in groups.items():
            # 그룹별 임시 디렉토리
            group_tmp_dir = tempfile.mkdtemp(prefix=f"rdf_{sample}_{yidx}_")
            tmp_files = []
            try:
                # 파일별 개별 전처리 & 스냅샷
                for fp in files:
                    df = ROOT.RDataFrame(tree, fp)
                    if fstr:
                        df = df.Filter(fstr)
                    # year_index per-file
                    if add_year_index:
                        if "year_index" not in df.GetColumnNames():
                            df = df.Define("year_index", str(_detect_year_index(fp)))

                    # 파일별 RF 컬럼 (여기가 핵심)
                    df = addRFcolumn(df, fp)

                    # weight: varlist에 필요하고 아직 없으면 안전 정의
                    # (파일마다 weight 구성 컬럼이 다를 수 있으니 per-file로 처리)
                    vl = list(varlist) if varlist else list(df.GetColumnNames())
                    if ("weight" in vl) and ("weight" not in df.GetColumnNames()):
                        wexpr = _clean_weight_expr(df, use_btag=loadBTag)
                        df = df.Define("weight", wexpr)



                    # 스냅샷용 최종 varlist(년도/가중치 포함 보장)
                    vlf = vl.copy()
                    if add_year_index and "year_index" not in vlf:
                        vlf.append("year_index")
                    if ("weight" in vl) and ("weight" not in vlf):
                        vlf.append("weight")

                    # 스키마 불일치 보정(없으면 0.0으로)
                    df = _ensure_columns(df, vlf)

                    outpath = os.path.join(group_tmp_dir, f"tmp_{uuid.uuid4().hex}.root")
                    df.Snapshot("Events", outpath, vlf)
                    tmp_files.append(outpath)

                # 그룹 스냅샷 병합 후 단 한 번만 변환
                df_merged = ROOT.RDataFrame("Events", tmp_files)

                # 최종 varlist 재확인(스냅샷에서 컬럼 보장됨)
                final_vl = list(varlist) if varlist else list(df_merged.GetColumnNames())
                if add_year_index and "year_index" not in final_vl:
                    final_vl.append("year_index")
                if df_merged.HasColumn("weight") and ("weight" in (varlist or [])) and ("weight" not in final_vl):
                    final_vl.append("weight")

                results.append(df_merged.AsNumpy(final_vl))

            finally:
                # 임시파일 정리
                try:
                    for p in tmp_files:
                        if os.path.exists(p):
                            os.remove(p)
                    if os.path.isdir(group_tmp_dir):
                        shutil.rmtree(group_tmp_dir, ignore_errors=True)
                except Exception as e:
                    print(f"[WARN] tmp cleanup failed: {e}")

    # ---- 추론 모드: 순서 보존을 위해 파일 단위로 바로 AsNumpy ----
    else:
        for fp, tree, fstr in tree_path_filter_str:
            df = ROOT.RDataFrame(tree, fp)

            if add_year_index:
                if "year_index" not in df.GetColumnNames():
                    df = df.Define("year_index", str(_detect_year_index(fp)))

            # 추론 모드에서는 addRFcolumn 호출 X (원 코드와 동일한 동작)
            vl = list(varlist) if varlist else list(df.GetColumnNames())

            # weight는 추론 모드에선 기본 1.0 (원 코드와 동일)
            if ("weight" in vl) and ("weight" not in df.GetColumnNames()):
                df = df.Define("weight", "1.0")

            if fstr:
                df = df.Filter(fstr)

            # 없는 컬럼 0.0 보정(필요시)
            #df = _ensure_columns(df, vl)

            results.append(df.AsNumpy(vl))

    return results

def summarize_by_year_cls(train_y, train_weight, train_features, features, year_col="year_index"):
    assert year_col in features, f"{year_col} 가 features에 없습니다."
    col = features.index(year_col)

    # year_index: float32일 수 있으니 NaN 방어 + 정수화
    year_raw = train_features[:, col]
    year_idx = np.where(np.isfinite(year_raw), year_raw, -1).astype(np.int64, copy=False)

    df_stat = pd.DataFrame({
        "cls":  train_y.astype(np.int64, copy=False),
        "year": year_idx,
        "w":    train_weight.astype(np.float64, copy=False),
    })

    # 0..3만 사용
    df_stat = df_stat[df_stat["year"].isin([0,1,2,3])]

    # 집계
    agg = (df_stat
           .groupby(["cls","year"], as_index=False)
           .agg(nMC = ("w","size"),
                sumW = ("w","sum"),
                w_min=("w","min"),
                w_max=("w","max"),
                w_mean=("w","mean"),
                w_std=("w","std"))
          )

    # 2018(=3) 대비 비율
    ref = (agg[agg["year"]==3][["cls","nMC","sumW"]]
           .rename(columns={"nMC":"nMC_2018","sumW":"sumW_2018"}))
    agg = agg.merge(ref, on="cls", how="left")
    agg["nMC_ratio_2018"]  = np.where(agg["nMC_2018"] > 0,  agg["nMC"]/agg["nMC_2018"],  np.nan)
    agg["sumW_ratio_2018"] = np.where(agg["sumW_2018"] != 0, agg["sumW"]/agg["sumW_2018"], np.nan)
    agg = agg.drop(columns=["nMC_2018","sumW_2018"]).sort_values(["cls","year"]).reset_index(drop=True)

    # 보기 좋게 라벨/반올림
    year_labels = {0:"2016a", 1:"2016b", 2:"2017", 3:"2018"}
    pretty = agg.copy()
    pretty["year"] = pretty["year"].map(year_labels)
    for c in ["sumW","w_min","w_max","w_mean","w_std","nMC_ratio_2018","sumW_ratio_2018"]:
        pretty[c] = pretty[c].astype(float).round(6)
    return pretty

def load_data(
    tree_path_filter_str=([], []),
    varlist=[],
    test_ratio=0.1,
    val_ratio=0.2,
    cat_vars=["year_index"],
    makeStandard=False,
    useLabelEncoder=True,
    add_year_index=False,
    sample_bkg=False
):
    if add_year_index:
        varlist.append("year_index")
    print(varlist)
    num_classes = len(tree_path_filter_str)
    class_dicts = []  # A dictionary list for each class
    training_mode = True if not (val_ratio == 0 and test_ratio == 0) else False

    loadBTag = not any("cvsb" in s for s in varlist)

    if training_mode:
        ROOT.gROOT.LoadMacro(
            "/data6/Users/yeonjoon/VcbMVAStudy/Corrections/SFProducer.h"
        )
        for e in ["2016preVFP", "2016postVFP", "2017", "2018"]:
            if loadBTag:
                path = f"/data6/Users/yeonjoon/VcbMVAStudy/Corrections/bTag/Vcb_Tagging_RF_Flavor_{e}.root"
            else:
                path = f"/data6/Users/yeonjoon/VcbMVAStudy/Corrections/cTag/Vcb_Tagging_RF_Flavor_{e}.root"
            print(f"Load SFProducer for {e} from {path}")
            
            for short, long_list in map_short_name.items():

                if "TTLJ" in short:
                    for mode in [2, 4, 45]:
                        ROOT.gInterpreter.Declare(f"""
                        std::unique_ptr<JetSFProducer> RFPatch_{short}_{mode}_{e};
                        """)
                        ROOT.gInterpreter.ProcessLine(f"""
                        RFPatch_{short}_{mode}_{e}.reset(new JetSFProducer("{e}", "{path}", "{short}", "{"B_Tag" if loadBTag else "C_Tag"}", {mode}));
                        RFPatch_{short}_{mode}_{e}->Load();
                        """)
                else:
                    ROOT.gInterpreter.Declare(f"""
                    std::unique_ptr<JetSFProducer> RFPatch_{short}_{e};
                    """)
                    ROOT.gInterpreter.ProcessLine(f"""
                    RFPatch_{short}_{e}.reset(new JetSFProducer("{e}", "{path}", "{short}", "{"B_Tag" if loadBTag else "C_Tag"}"));
                    RFPatch_{short}_{e}->Load();
                    """)

    data_dicts = []
    for i in range(num_classes):
        this_df = getRDataFrame(tree_path_filter_str[i], varlist, training_mode, add_year_index, loadBTag)
        class_dicts.append(this_df)
        data_dicts.append({})
        
    
    for i in range(num_classes):
        
        if(len(class_dicts[i]) != 0):
            for key in class_dicts[i][0]:
                data_dicts[i][key] = np.concatenate([arr[key] for arr in class_dicts[i]])

    for i in range(num_classes):
        pd_df = pd.DataFrame({k: v for k, v in data_dicts[i].items()})
        print(f"set y to {i}.....")
        pd_df["y"] = np.ones(pd_df.shape[0]) * i
    
        if i == 0:
            df = pd_df
        else:
            df = pd.concat([df, pd_df], ignore_index=True)

    df = df.reset_index(drop=True)
    np.random.seed(42)
    
    print(df)
    if val_ratio == 0 and test_ratio == 0:
        df["Set"] = np.random.choice(["train"], p=[1.0], size=(df.shape[0],))
        df["Set"] = "train"
    elif test_ratio == 0 and val_ratio != 0:
        df["Set"] = np.random.choice(
            ["train", "val"],
            p=[1.0 - (val_ratio + test_ratio), val_ratio],
            size=(df.shape[0],),
        )
    else:
        df["Set"] = np.random.choice(
            ["train", "val", "test"],
            p=[1.0 - (val_ratio + test_ratio), val_ratio, test_ratio],
            size=(df.shape[0],),
        )

    if not (val_ratio == 0 and test_ratio == 0):
        print("Training mode, eliminate negative weight sample.......Passed!!")
        # df = df[df['weight'] >= 0]


    dict_idx = {}
    for i in range(num_classes):
        dict_idx[i] = df[df["y"] == i].index
    df["y"] = df["y"].astype(int)
    if "weight" not in df.columns:
        df["weight"] = 1.0
    total_samples = df["weight"].sum()
    # sumW_temp = []
    sumW = []
    count = []
    
    # for i in range(num_classes):
    #     sumW_temp.append(df["weight"].values[dict_idx[i]].sum())
    #     
    #     df.loc[dict_idx[i], "weight"] *= count[i] / sumW_temp[i]
    
    for i in range(num_classes):
        count.append(len(dict_idx[i]))
        sumW.append(df["weight"].values[dict_idx[i]].sum())
    #     print(f"sumW of class {i} is {sumW_temp[i]}->{sumW[i]}")
        
    class_weight = (total_samples / np.asarray(sumW, dtype=np.float64)).astype(np.float32)

        


    # df["sample_and_class_weight"] = df.apply(
    #     lambda row: row["weight"] * class_weight[row["y"]], axis=1
    # )

    y_arr = df["y"].to_numpy(dtype=np.int64, copy=False)
    w_arr = df["weight"].to_numpy(dtype=np.float32, copy=False)
    df["sample_and_class_weight"] = w_arr * np.take(class_weight, y_arr)

    # train_indices = df[df.Set == "train"].index
    # val_indices = df[df.Set == "val"].index
    # test_indices = df[df.Set == "test"].index
    
    set_vals   = df["Set"].to_numpy()
    mask_train = (set_vals == "train")
    mask_val   = (set_vals == "val")
    mask_test  = (set_vals == "test")
    
    unused_feat = ["Set", "weight", "sample_and_class_weight"]
    target = "y"

    categorical_columns = []
    categorical_dims = {}
    categorical_labelencoder = {}
    if useLabelEncoder:
        if cat_vars:  # 이미 넘겨줄 수 있으면 제일 좋음
            categorical_columns = [c for c in cat_vars if c not in ["y","Set","weight","sample_and_class_weight"]]
        else:
            # 정수형 컬럼만 후보(연속형은 스킵)
            int_cols = df.select_dtypes(include=["int8","int16","int32","int64","uint8","uint16","uint32","uint64"]).columns
            categorical_columns = [c for c in int_cols if c not in ["y","Set","weight","sample_and_class_weight"]]

        # 0-base 보장 + 차원 계산
        for c in categorical_columns:
            #nunique 계산해서 200 미만이면 categorical로 간주
            if df[c].nunique() >= 200:
                print(f"Variable {c} is not categorical, but continuous with {df[c].nunique()} unique values.")
                continue
            col = df[c].to_numpy(copy=False)
            minv = int(col.min())
            categorical_dims[c] = int(df[c].max()) - minv + 1
            print(f"Variable {c} is categorical with dimension of {categorical_dims[c]} ")


    
    features = [c for c in df.columns if c not in ["Set","weight","sample_and_class_weight","y"]]
    ##print in red
    print(f"\033[91m Use Following Features: {features}\033[0m")
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [
        categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns
    ]
    X = df[features].to_numpy(dtype=np.float32, copy=False)
    y = df["y"].to_numpy(dtype=np.int64,  copy=False)
    w = df["weight"].to_numpy(dtype=np.float32, copy=False)
    


    train_features = X[mask_train]; val_features  = X[mask_val];  test_features = X[mask_test]
    train_y        = y[mask_train]; val_y         = y[mask_val];   test_y        = y[mask_test]
    train_weight   = w[mask_train]; val_weight    = w[mask_val];   test_weight   = w[mask_test]
    


    # ---- 0) 준비: 공통 RNG ----
    rng = np.random.default_rng(42)

    # ---- 1) Train 샘플링: 신호 전부 + 배경은 sample_bkg*n_sig 만큼(클래스별) ----
    classes = np.unique(train_y)
    signal_cls = 0

    mask_train_sample = np.zeros_like(train_y, dtype=bool)
    sig_idx = np.flatnonzero(train_y == signal_cls)
    mask_train_sample[sig_idx] = True
    n_sig = sig_idx.size

    if sample_bkg and n_sig > 0:
        n_bkg_target_per_cls = int(round(sample_bkg * n_sig))
        for cls in classes:
            if cls == signal_cls:
                continue
            bkg_idx_cls = np.flatnonzero(train_y == cls)
            t = min(n_bkg_target_per_cls, bkg_idx_cls.size)
            if t > 0:
                chosen = rng.choice(bkg_idx_cls, size=t, replace=False)
                mask_train_sample[chosen] = True
            print(f"class {cls}: picked {t:,} / {bkg_idx_cls.size:,} (target={n_bkg_target_per_cls:,})")
    else:
        # sample_bkg가 없으면(train 축소 X) 전부 사용
        mask_train_sample[:] = True

    print(f"[TRAIN] selected {mask_train_sample.sum():,} / {len(train_y):,}")

    # (선택) 실제로 train 데이터를 줄이고 싶다면 아래 주석 해제
    # train_features = train_features[mask_train_sample]
    # train_y        = train_y[mask_train_sample]
    # train_weight   = train_weight[mask_train_sample]

    # ---- 2) Validation을 Train 비율로 '층화 샘플링' (총 개수는 기존 len(val_y) 유지) ----
    cnt_train = np.array([np.sum(train_y[mask_train_sample] == c) for c in classes], dtype=int)
    if cnt_train.sum() == 0:
        raise ValueError("No training samples selected; check sample_bkg or labels.")

    p_train = cnt_train / cnt_train.sum()

    def distribute_with_caps(probs, total, caps):
        """비율 probs로 total개를 배분하되, caps(각 클래스 최대치)를 넘지 않도록.
        total은 그대로 유지하려고 노력하며, cap 때문에 모자라면 다른 클래스에 재분배."""
        probs = np.asarray(probs, float)
        if probs.sum() <= 0:
            # 전부 0이면 균등 분배
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs.sum()

        total = int(total)
        caps  = np.asarray(caps, int)

        # 1차 배분: 바닥(floor) 후 cap 적용
        alloc = np.floor(probs * total).astype(int)
        alloc = np.minimum(alloc, caps)

        # 남은 몫 재분배: cap 여유 있는 클래스에 우선순위(probs 큰 순)로 1개씩
        rem = total - int(alloc.sum())
        order = np.argsort(-probs)
        while rem > 0:
            progressed = False
            for i in order:
                if rem == 0:
                    break
                if alloc[i] < caps[i]:
                    alloc[i] += 1
                    rem -= 1
                    progressed = True
            if not progressed:
                # 더 넣을 데가 없는 경우(모든 클래스가 cap 만빵)
                break
        return alloc

    def pick_indices_by_class(y, classes, per_cls_target, rng):
        out = []
        for c, t in zip(classes, per_cls_target):
            if t <= 0:
                continue
            cand = np.flatnonzero(y == c)
            t = min(int(t), cand.size)
            if t > 0:
                sel = rng.choice(cand, size=t, replace=False)
                out.append(sel)
        return np.concatenate(out) if out else np.array([], dtype=int)

   # train 비율
    classes = np.unique(train_y)
    cnt_train = np.array([np.sum(train_y[mask_train_sample] == c) for c in classes], int)
    p_train  = cnt_train / cnt_train.sum()

    # 각 클래스가 VAL에 가진 최대치(cap)
    caps_val = np.array([np.sum(val_y == c) for c in classes], int)

    # 가능한 총량 T: 모든 c에 대해 p_train[c]*T <= caps_val[c]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(p_train > 0, caps_val / p_train, np.inf)
    T = int(np.floor(ratios.min()))
    T = max(T, 0)

    # 목표 per-class 수량
    raw = np.floor(p_train * T).astype(int)
    frac = p_train * T - raw
    # 반올림 보정(+1씩), cap 넘지 않게
    order = np.argsort(-frac)
    rem = T - raw.sum()
    for i in order:
        if rem == 0: break
        add = min(rem, caps_val[i] - raw[i])
        if add > 0:
            raw[i] += add
            rem -= add

    # 클래스별로 추출(비복원)
    rng = np.random.default_rng(42)
    val_sel = []
    for c, t in zip(classes, raw):
        if t <= 0: continue
        cand = np.flatnonzero(val_y == c)
        if t > cand.size: t = cand.size
        if t > 0:
            val_sel.append(rng.choice(cand, size=t, replace=False))
    val_sel = np.concatenate(val_sel) if len(val_sel) else np.array([], dtype=int)

    mask_val_downsample = np.zeros_like(val_y, dtype=bool)
    mask_val_downsample[val_sel] = True

    # 적용
    val_features = val_features[mask_val_downsample]
    val_y        = val_y[mask_val_downsample]
    val_weight   = val_weight[mask_val_downsample]

    print(f"[VAL] resized to {mask_val_downsample.sum():,} (exactly matched train ratio)")
    for c in classes:
        tr = np.sum(train_y[mask_train_sample]==c)
        vl = np.sum(val_y==c)
        te = np.sum(test_y==c)  # test는 그대로
        print(f"class {c}: train {tr:,}, val {vl:,}, test {te:,}") 
            
        

    else:
        mask_train_sample = np.ones_like(train_y, dtype=bool)

    train_features = train_features[mask_train_sample]
    train_y = train_y[mask_train_sample]
    train_weight = train_weight[mask_train_sample]
    
    #Inspect nMc*sumW ratio is set as lumi W in wach class
    summary = summarize_by_year_cls(train_y, train_weight, train_features, features, year_col="year_index")
    print(summary.to_string(index=False))
    data = {
        "train_features": train_features,
        "test_features": test_features,
        "val_features": val_features,
        "train_y": train_y,
        "test_y": test_y,
        "val_y": val_y,
        "class_weight": class_weight,
        "train_weight": train_weight,
        "val_weight": val_weight,
        "test_weight": test_weight,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
        "cat_columns": categorical_columns,
        "cat_labelencoder": categorical_labelencoder,
        "sumW": sumW,
        "count": count,
    }


  

    return data


def classWtoSampleW(dataset, class_weights):
    weight = []
    for class_data in dataset:
        weight.append(class_weights[class_data])
    return np.array(weight)
