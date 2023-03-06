import os, sys
import numpy as np
print(np.__file__)
from pytorch_tabnet.tab_model import TabNetClassifier
from copy import deepcopy
import torch
print(torch.__file__)

import sys
sys.path.append(os.environ["DIR_PATH"])
from root_data_loader import load_data, classWtoSampleW
print(torch.cuda.is_available())
#print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt


varlist = ['pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b','bvsc_had_t_b','bvsc_lep_t_b',
   'theta_w_u_w_d','theta_lep_neu','theta_lep_w_lep_t_b', 'del_phi_had_t_lep_t',
   'had_t_mass','had_w_mass','lep_t_mass','lep_t_partial_mass','pt_ratio','chi2','weight']
  
if __name__ == '__main__':

  
  
  path_sample = os.environ["WtoCB_PATH"]
  filename = 'Vcb_Mu_TTLJ_WtoCB_powheg.root'
  #,'Reco_41','Reco_23','Reco_21'
  data = load_data(file_path=os.path.join(path_sample,filename),filterstr='n_jets==4',varlist=varlist,val_ratio=0.2,test_ratio=0.1,sigTree=['Permutation_Correct'],bkgTree=['Permutation_Wrong'])
  print("file loaded")
  from pytorch_tabnet.augmentations import ClassificationSMOTE
  aug = ClassificationSMOTE(p=0.5)
  clf = TabNetClassifier(
    verbose=1,
    cat_idxs=data['cat_idxs'],
    cat_dims=data['cat_dims'],
    cat_emb_dim=2
    )  
  clf.fit(
    X_train=data['train_features'],y_train=data['train_y'],
    eval_set=[(data['val_features'], data['val_y'])],
    eval_metric=['auc'],
    #weights=data['class_weight']
    batch_size=4096,
    virtual_batch_size=256,
    num_workers=4,
    augmentations=aug,
    patience=10
    #callbacks=[pytorch_tabnet.callbacks.History(clf,verbose=1)]
  )
  clf.save_model('/cms/ldap_home/yeonjoon/working_dir/VcbMVAStudy/TabNet_Permutation/model.pt')
  y= clf.predict_proba(data['test_features'])[:,1]
  fpr, tpr, _ = roc_curve(data['test_y'], y)
  auc_val = auc(fpr,tpr)
  plt.title(f'AUC is {auc_val}')
  plt.plot(fpr,tpr)
  plt.savefig(f'ROC_test.png')

  

