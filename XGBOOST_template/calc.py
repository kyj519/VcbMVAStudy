import ROOT
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
ROOT.EnableImplicitMT(16)
model = XGBClassifier() 
model.load_model('/data6/Users/yeonjoon/VcbMVAStudy/XGBOOST_template/XGBOOST_model.json')
modelist = ['45','43','41','23','21']
for mode in modelist:
    varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets']
    df = ROOT.RDataFrame(f'Reco_{mode}','/gv0/Users/yeonjoon/Vcb_2018_Mu_Reco_Tree.root')
    arr = df.AsNumpy(varlist)
    sig = pd.DataFrame({k:v for k,v in arr.items()})
    sig = sig.reset_index(drop=True)
    arr = np.array(sig)
    
    y = [p[1] for p in model.predict_proba(arr)]
    plt.xlim([0, 1])
    plt.hist(y,bins=30)
    plt.savefig(f'/data6/Users/yeonjoon/VcbMVAStudy/XGBOOST_template/result_{mode}.png')
    plt.clf()
    del df
    del arr
    del sig