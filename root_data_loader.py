import ROOT
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
ROOT.EnableImplicitMT(1)

def addRFcolumn(df,file):
    print(f'RFProducer created for file {file}, as sample TTLJ')
    df = df.Define("weight_RFPatch",    f"RFPatch->GetSF(n_vertex,ht,isBB,isCC,whatMode)")
    return df

# @ROOT.Numba.Declare(['RVecI', 'RVecI'], 'int')
# def isBB(origs,flavs):
#     for i in range(len(origs)):
#         orig = origs[i]
#         flav = flavs[i]
#         if flav == 5 and abs(orig) != 6 and abs(orig) != 24:
#             return 1
#     return 0

# @ROOT.Numba.Declare(['RVecI', 'RVecI'], 'int')
# def isCC(origs,flavs):
#     for i in range(len(origs)):
#         orig = origs[i]
#         flav = flavs[i]
#         if flav == 4 and abs(orig) != 6 and abs(orig) != 24:
#             return 1
#     return 0
        

def addHisto_idx(df):
    #df = df.Define('isBB','std::any_of(Sel_Gen_HF_Origin.begin(), Sel_Gen_HF_Origin.end(), [&](int orig) { return Sel_Gen_HF_Flavour[&orig - &Sel_Gen_HF_Origin[0]] == 5 && abs(orig) != 6 && abs(orig) != 24; })')
    #df = df.Define('isCC','std::any_of(Sel_Gen_HF_Origin.begin(), Sel_Gen_HF_Origin.end(), [&](int orig) { return Sel_Gen_HF_Flavour[&orig - &Sel_Gen_HF_Origin[0]] == 4 && abs(orig) != 6 && abs(orig) != 24; })')
    df = df.Define('isBB','for(size_t i = 0; i < Sel_Gen_HF_Origin.size(); ++i) if (Sel_Gen_HF_Flavour[i] == 5 && abs(Sel_Gen_HF_Origin[i]) != 6 && abs(Sel_Gen_HF_Origin[i]) != 24) return 1; return 0;')
    df = df.Define('isCC_temp','for(size_t i = 0; i < Sel_Gen_HF_Origin.size(); ++i) if (Sel_Gen_HF_Flavour[i] == 5 && abs(Sel_Gen_HF_Origin[i]) != 6 && abs(Sel_Gen_HF_Origin[i]) != 24) return 1; return 0;')
    #since both BB and CC cannot be 1 in same time
    df = df.Define('isCC', '!isBB && isCC_temp')

    df = df.Define('whatMode','(decay_mode == 21 || decay_mode == 23) ?2 : (decay_mode == 41 || decay_mode == 43) ? 4 : (decay_mode == 45) ? 45 : 0')
    return df
    
def load_data(tree_path_filter_str=([], []), varlist=[], test_ratio=0.1, val_ratio=0.2, cat_vars=[], makeStandard=False, useLabelEncoder=True):
    print(varlist)
    sig_dict = []
    bkg_dict = []
    training_mode = False
    if not(val_ratio == 0 and test_ratio == 0):
        training_mode = True
        print('Root2numpy loader on training mode')
    else:
        print('Root2numpy loader on infering mode')
        
    if training_mode:
        path='/data6/Users/yeonjoon/VcbMVAStudy/Vcb_Tagging_RF_2017.root'
        ROOT.gROOT.LoadMacro("/data6/Users/yeonjoon/VcbMVAStudy/SFProducer.h")
        ROOT.gInterpreter.Declare(f"std::unique_ptr<SFProducer> RFPatch(new SFProducer(\"2017\",\"{path}\",\"TTLJ\"));")
        ROOT.gInterpreter.ProcessLine('RFPatch->LoadSF();')
    
    for tup in tree_path_filter_str[0]:
        print('load Data for')
        print(tup)
        print('\n')
        
        f = ROOT.TFile(tup[0], "READ")
        tree = tup[1]
        filterstr = tup[2]
        tr = f.Get(tree)
        df = ROOT.RDataFrame(tr)

        if varlist == []:
            varlist = [col for col in df.GetColumnNames()]
        if training_mode:
            #df = addHisto_idx(df)
            #df = addRFcolumn(df,tup[0]) 
            if 'weight' in varlist and 'weight' not in df.GetColumnNames():
                df = df.Define('weight', '''weight_c_tag*
                                        weight_el_id*
                                        weight_lumi*
                                        weight_mc*
                                        weight_mu_id*
                                        weight_mu_iso*
                                        weight_pileup*
                                        weight_prefire*
                                        weight_sl_trig*
                                        weight_top_pt''')
                                        #weight_RFPatch

        df = df if filterstr == '' else df.Filter(filterstr)
        sig_dict.append(df.AsNumpy(varlist))

    for tup in tree_path_filter_str[1]:
        print('load Data for')
        print(tup)
        print('\n')
        
        f = ROOT.TFile(tup[0], "READ")
        tree = tup[1]
        filterstr = tup[2]
        tr = f.Get(tree)
        df = ROOT.RDataFrame(tr)
        if training_mode:
            #df = addHisto_idx(df)
            #f = addRFcolumn(df,tup[0]) 
            if 'weight' in varlist and 'weight' not in df.GetColumnNames():
            #continue
                df = df.Define('weight', '''weight_c_tag*
                                        weight_el_id*
                                        weight_lumi*
                                        weight_mc*
                                        weight_mu_id*
                                        weight_mu_iso*
                                        weight_pileup*
                                        weight_prefire*
                                        weight_sl_trig*
                                        weight_top_pt''')
                                        #weight_RFPatch
        
        df = df if filterstr == '' else df.Filter(filterstr)
        bkg_dict.append(df.AsNumpy(varlist))

    data_sig = {}
    data_bkg = {}
    for key in sig_dict[0]:
        data_sig[key] = np.concatenate([arr[key] for arr in sig_dict])
    if len(bkg_dict) != 0:
        for key in bkg_dict[0]:
            data_bkg[key] = np.concatenate([arr[key] for arr in bkg_dict])

    sig = pd.DataFrame({k: v for k, v in data_sig.items()})
    sig['y'] = np.ones(sig.shape[0])
    class_weight = {}
    if len(bkg_dict) != 0:
        bkg = pd.DataFrame({k: v for k, v in data_bkg.items()})
        bkg['y'] = np.zeros(bkg.shape[0])
        df = pd.concat([sig, bkg], ignore_index=True)
    else:
        df = pd.DataFrame(sig)

    df = df.reset_index(drop=True)
    np.random.seed(42)
    if val_ratio == 0 and test_ratio == 0:
        df["Set"] = np.random.choice(["train"], p=[1.], size=(df.shape[0],))
        df['Set'] = 'train'
    elif test_ratio == 0 and val_ratio != 0:
        df["Set"] = np.random.choice(["train", "val"], p=[
                                     1. - (val_ratio+test_ratio), val_ratio], size=(df.shape[0],))
    else:
        df["Set"] = np.random.choice(["train", "val", "test"], p=[
                                     1. - (val_ratio+test_ratio), val_ratio, test_ratio], size=(df.shape[0],))

    if not(val_ratio == 0 and test_ratio == 0):
        print("Training mode, eliminate negative weight sample")
        df = df[df['weight'] >= 0]

    df = df.reset_index(drop=True)
    # calculate class weights
    sig_idx = df[df['y'] == 1].index
    bkg_idx = df[df['y'] == 0].index
    if 'weight' not in df.columns:
        df['weight'] = 1.
    total_samples = df['weight'].sum()
    
    class_weight =  {0: 1., 1: 1.} if tree_path_filter_str[1] == [] else {
        0: total_samples / df['weight'].values[bkg_idx].sum(),
        1: total_samples / df['weight'].values[sig_idx].sum()
    }
    
    df['sample_and_class_weight'] = df.apply(
        lambda row: row['weight'] * class_weight[row['y']], axis=1)

    train_indices = df[df.Set == "train"].index
    val_indices = df[df.Set == "val"].index
    test_indices = df[df.Set == "test"].index
    unused_feat = ['Set', 'weight', 'sample_and_class_weight']
    target = 'y'

    nunique = df.nunique()
    types = df.dtypes
    categorical_columns = []
    categorical_dims = {}
    categorical_labelencoder = {}
    if useLabelEncoder:
        for col in df.columns:
            if col == target or col in unused_feat:
                continue
            if cat_vars == []:
                if nunique[col] < 200:

                    categorical_columns.append(col)
                    categorical_dims[col] = nunique[col]+5
                    #categorical_labelencoder[col] = l_enc
            else:
                if col in cat_vars:
                    #l_enc = LabelEncoder()
                    #df[col] = l_enc.fit_transform(df[col].values)
                    categorical_columns.append(col)
                    categorical_dims[col] = nunique[col]+5
                    #categorical_labelencoder[col] = l_enc

    #######################
    # TEMPORARY HARDCODED!!!
    ########################
    #df.loc[df['n_bjets'] > 7, 'n_bjets'] = 7
    #df.loc[df['n_cjets'] > 11, 'n_cjets'] = 11

    features = [col for col in df.columns if col not in unused_feat+[target]]
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [categorical_dims[f]
                for i, f in enumerate(features) if f in categorical_columns]
    train_features = np.array(df[features].values[train_indices])
    train_y = np.array(df[target].values[train_indices])

    val_features = np.array(df[features].values[val_indices])
    val_y = np.array(df[target].values[val_indices])

    test_features = np.array(df[features].values[test_indices])
    test_y = np.array(df[target].values[test_indices])

    train_weight = np.array(df['weight'].values[train_indices])
    val_weight = np.array(df['weight'].values[val_indices])
    test_weight = np.array(df['weight'].values[test_indices])

    train_sample_and_class_weight = np.array(
        df['sample_and_class_weight'].values[train_indices])
    val_sample_and_class_weight = np.array(
        df['sample_and_class_weight'].values[val_indices])
    test_sample_and_class_weight = np.array(
        df['sample_and_class_weight'].values[test_indices])
    scaler = StandardScaler()

    if makeStandard:
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)

    #train_features = np.clip(train_features, -5, 5)
    #val_features = np.clip(val_features, -5, 5)
    #test_features = np.clip(test_features, -5, 5)

    data = {'train_features': train_features, 'test_features': test_features, 'val_features': val_features, 'train_y': train_y, 'test_y': test_y, 'val_y': val_y,
            'class_weight': class_weight, 'train_weight': train_weight, 'val_weight': val_weight, 'test_weight': test_weight, 'cat_idxs': cat_idxs, 'cat_dims': cat_dims, 'cat_columns': categorical_columns, 'cat_labelencoder': categorical_labelencoder,
            'train_sample_and_class_weight': train_sample_and_class_weight, 'val_sample_and_class_weight': val_sample_and_class_weight,
            'test_sample_and_class_weight': test_sample_and_class_weight}

    return data


def classWtoSampleW(dataset, class_weights):
    weight = []
    for class_data in dataset:
        weight.append(class_weights[class_data])
    return np.array(weight)
