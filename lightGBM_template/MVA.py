import os, sys
import numpy as np
import lightgbm

import sys
sys.path.append(os.environ["DIR_PATH"])
from root_data_loader import load_data, classWtoSampleW
from matplotlib import pyplot as plt



varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','pt_w_u','pt_w_d','weight']
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
#varlist = ['cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
# varlist.extend(['n_jets',
#                 'pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b',
#                 'eta_had_t_b','eta_w_u','eta_w_d','eta_lep_t_b',
#                 'bvsc_lep_t_b','bvsc_had_t_b'])
  
if __name__ == '__main__':

  
  
  path_sample = os.environ["WtoCB_PATH"]
  #filename = 'Vcb_Mu_TTLJ_WtoCB_powheg_25.root'
  filename = 'Vcb_2018_Mu_Reco_Tree.root' 
  #,'Reco_41','Reco_23','Reco_21'
  data =  load_data(file_path=os.path.join(path_sample,filename),varlist=varlist,test_ratio=0,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43','Reco_41','Reco_23','Reco_21'])
  lgbm = lightgbm.LGBMClassifier(n_estimators=1000,is_unbalance=True, max_depth = 15)
  evals = [(data['val_features'],data['val_y'])]
  lgbm.fit(data['train_features'],data['train_y'],early_stopping_rounds = 100, eval_metric='auc',eval_set = evals, verbose = True)
  data2 =  load_data(file_path=os.path.join(path_sample,filename),varlist=varlist,test_ratio=0,val_ratio=0.0,sigTree=['Reco_45'],bkgTree=[])
  preds = lgbm.predict_proba(data2['train_features'])[:,1]
  plt.hist(preds)
  plt.savefig('45.png')
  plt.clf()
  
  data2 =  load_data(file_path=os.path.join(path_sample,filename),varlist=varlist,test_ratio=0,val_ratio=0.0,sigTree=['Reco_43'],bkgTree=[])
  preds = lgbm.predict_proba(data2['train_features'])[:,1]
  plt.hist(preds)
  plt.savefig('43.png')
  plt.clf()

