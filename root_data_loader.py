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

def _clean_weight_expr(df: ROOT.RDataFrame, use_btag: bool) -> str:
    # 존재하는 항만 곱해 안전하게 정의 (중복 제거)
    tokens = [
        ("weight_b_tag" if use_btag else "weight_c_tag"),
        "weight_el_id",
        "weight_mc",
        "weight_lumi",
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
    

    if sample_bkg:
        sig_idx = np.flatnonzero(train_y == 0)
        n_sig = sig_idx.size
        n_bkg_target = int(round(sample_bkg * n_sig))

        rng = np.random.default_rng(42)

        mask_train_sample = np.zeros_like(train_y, dtype=bool)
        mask_train_sample[sig_idx] = True  # 신호는 전부 사용

   
        for cls in np.unique(train_y):
            if cls == 0:
                continue  
            bkg_idx_cls = np.flatnonzero(train_y == cls)

            n_bkg_target_cls = min(n_bkg_target, bkg_idx_cls.size)
            print(
                f"class {cls}: sample {n_bkg_target_cls} "
                f"from {bkg_idx_cls.size} (target={sample_bkg} * {n_sig})"
            )

            if n_bkg_target_cls > 0:
                chosen_bkg_cls = rng.choice(
                    bkg_idx_cls, size=n_bkg_target_cls, replace=False
                )
                mask_train_sample[chosen_bkg_cls] = True

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
