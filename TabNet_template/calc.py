import ROOT
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys,os
sys.path.append(os.environ["DIR_PATH"])
ROOT.EnableImplicitMT(16)
from root_data_loader import load_data
model = TabNetClassifier()
model.load_model('/cms/ldap_home/yeonjoon/working_dir/VcbMVAStudy/TabNet_template/model_largebatch_nosmote.pt.zip')
modelist = ['45','43','41','23','21']
#varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','pt_w_u','pt_w_d','weight']
# varlist.extend(['n_jets',
#                 'pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b',
#                 'eta_had_t_b','eta_w_u','eta_w_d','eta_lep_t_b',
#                 'bvsc_lep_t_b','bvsc_had_t_b',
#                 'm_w_u','m_w_d'])
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
result = []
for mode in modelist:
    data =  load_data(file_path='root://cluster142.knu.ac.kr//store/user/yeonjoon/Vcb_2018_Mu_Reco_Tree.root',varlist=varlist,test_ratio=0,val_ratio=0,sigTree=[f'Reco_{mode}'],bkgTree=[])
    arr = data['train_features']
    pred = model.predict_proba(arr)[:,1]
    result.append(pred)
    plt.hist(pred, bins=40, weights=data['train_weight'])
    plt.savefig(f'/cms/ldap_home/yeonjoon/working_dir/VcbMVAStudy/TabNet_template/result_{mode}.png')
    plt.clf()

    
    del data
    del arr 
    
df = pd.concat([pd.DataFrame(a, columns=[modelist[i]]) for i, a in enumerate(result)], axis=1)
fig = df.plot.hist(stacked=True, bins=30, figsize=(10, 6), grid=True, weights=data['weight'])
fig.figure.savefig('stack.png',dpi=600)
    