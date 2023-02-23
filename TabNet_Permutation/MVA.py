import ROOT
import os, sys
import numpy as np
#######
#BELOW CODE IS OPTIMIZED FOR TENSORFLOW-2.4.1!
#######
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import sys
sys.path.append(os.environ["DIR_PATH"])
from root_data_loader import load_data, classWtoSampleW
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())

ROOT.EnableImplicitMT()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


varlist_prekin = ['had_t_b_pt','w_u_pt','w_d_pt','lep_t_b_pt','had_t_b_bscore','lep_t_b_bscore',
   'theta_w_u_w_d','theta_lep_neu','theta_lep_w_lep_t_b', 'del_phi_had_t_lep_t',
   'had_t_mass','had_w_mass','lep_t_mass','lep_t_partial_mass']
varlist = varlist_prekin + ['chi2']
  
if __name__ == '__main__':

  
  
  path_sample = os.environ["WtoCB_PATH"]
  #filename = 'Vcb_Mu_TTLJ_WtoCB_powheg_25.root'
  filename = 'Vcb_Mu_TTLJ_WtoCB_powheg.root'
 
  data =  load_data(os.path.join(path_sample,filename), 'n_jets==4',varlist,0.1,0.2,['Permutation_Correct'],['Permutation_Wrong'])
  clf = TabNetClassifier()  #TabNetRegressor()
  clf.fit(
    data['train_features'],data['train_y'],
    eval_set=[(data['val_features'], data['val_y'])],
    eval_metric=['auc'],
    weights=data['class_weight']
  )
  preds = clf.predict(data['test_features'])
  clf.save_model('/data6/Users/yeonjoon/VcbMVAStudy/TabNet_Permutation/')
  
  

