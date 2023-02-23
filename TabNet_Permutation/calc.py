import ROOT
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
ROOT.EnableImplicitMT(16)
model = load_model('/data6/Users/yeonjoon/VcbMVAStudy/keras_template/model_fromknu.h5')
modelist = ['45','43','41','23','21']
for mode in modelist:
    varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets']
    df = ROOT.RDataFrame(f'Reco_{mode}','/gv0/Users/yeonjoon/Vcb_2018_Mu_Reco_Tree.root')
    arr = df.AsNumpy(varlist)
    sig = pd.DataFrame({k:v for k,v in arr.items()})
    sig = sig.reset_index(drop=True)
    arr = np.array(sig)
    
    y = model.predict(arr)
    plt.hist(y,bins=40)
    plt.savefig(f'/data6/Users/yeonjoon/VcbMVAStudy/keras_template/result_{mode}.png')
    plt.clf()
    del df
    del arr
    del sig