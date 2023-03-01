import ROOT
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys,os
from scipy import interpolate
import plotly

sys.path.append(os.environ["DIR_PATH"])
ROOT.EnableImplicitMT(16)
from root_data_loader import load_data
from sklearn.metrics import roc_curve, auc

model = TabNetClassifier()
model.load_model('/cms/ldap_home/yeonjoon/working_dir/VcbMVAStudy/TabNet_Permutation/model.pt.zip')
modelist = ['Correct','Wrong']
varlist = ['pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b','bvsc_had_t_b','bvsc_lep_t_b',
   'theta_w_u_w_d','theta_lep_neu','theta_lep_w_lep_t_b', 'del_phi_had_t_lep_t',
   'had_t_mass','had_w_mass','lep_t_mass','lep_t_partial_mass','pt_ratio','chi2','weight']
    
# for mode in modelist:
    
#     data =  load_data(file_path='root://cluster142.knu.ac.kr//store/user/yeonjoon/Vcb_Mu_TTLJ_WtoCB_powheg.root',varlist=varlist,test_ratio=0,val_ratio=0,sigTree=[f'Permutation_{mode}'],bkgTree=[])
#     arr = data['train_features']
#     plt.hist(model.predict_proba(arr)[:,1],bins=40)
#     plt.savefig(f'/cms/ldap_home/yeonjoon/working_dir/VcbMVAStudy/TabNet_template/result_{mode}.png')
#     plt.clf()

#     del arr
#     del data
    
data =  load_data(file_path='root://cluster142.knu.ac.kr//store/user/yeonjoon/Vcb_Mu_TTLJ_WtoCB_powheg.root',varlist=varlist,test_ratio=0,val_ratio=0,sigTree=['Permutation_Correct'],bkgTree=['Permutation_Wrong'])
y= model.predict_proba(data['train_features'])[:,1]
fpr, tpr, _ =roc_curve(data['train_y'], y)
auc_val = auc(fpr,tpr)
plt.title(f'AUC is {auc_val}')
plt.plot(fpr,tpr)
plt.savefig(f'ROC.png')


    