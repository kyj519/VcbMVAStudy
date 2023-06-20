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
from postTrainingToolkit import seperate_sig_bkg, KS_test
from sklearn.metrics import roc_curve, auc

model = TabNetClassifier()
model.load_model('/data6/Users/yeonjoon/VcbMVAStudy/TabNet_Permutation/model.pt.zip')
modelist = ['Correct','Wrong']
varlist = ['pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b','bvsc_had_t_b','bvsc_lep_t_b',
   'theta_w_u_w_d','theta_lep_neu','theta_lep_w_lep_t_b', 'del_phi_had_t_lep_t',
   'had_t_mass','had_w_mass','lep_t_mass','lep_t_partial_mass','pt_ratio','chi2','weight']

  
path_sample = os.environ["WtoCB_PATH"]
filename = 'Vcb_Mu_TTLJ_WtoCB_powheg.root'
#,'Reco_41','Reco_23','Reco_21'
data = load_data(file_path=os.path.join(path_sample,filename),filterstr='n_jets==4',varlist=varlist,val_ratio=0.2,test_ratio=0.1,sigTree=['Permutation_Correct'],bkgTree=['Permutation_Wrong'])  
y_score = model.predict_proba(data['test_features'])[:,1]

_,_,test_sig_idx,test_bkg_idx = seperate_sig_bkg(pd.DataFrame(data['test_y'],columns=['y']),branch = 'y')

plt.hist(y_score[test_sig_idx], bins=40)
plt.savefig(os.path.join(os.environ["DIR_PATH"],'TabNet_Permutation/result_sig.png'))
plt.clf()
    
plt.hist(y_score[test_bkg_idx], bins=40)
plt.savefig(os.path.join(os.environ["DIR_PATH"],'TabNet_Permutation/result_bkg.png'))
plt.clf()

train_score = model.predict_proba(data['train_features'])[:,1]
val_score = model.predict_proba(data['val_features'])[:,1]
kolS, kolB = KS_test(train_score,val_score,data['train_weight'],data['val_weight'],data['train_y'],data['val_y'],plotPath='/data6/Users/yeonjoon/VcbMVAStudy/TabNet_Permutation')
print(f'{kolS}, {kolB}')



    