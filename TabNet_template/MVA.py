import os, sys
import numpy as np
print(np.__file__)
from pytorch_tabnet.tab_model import TabNetClassifier

import torch
print(torch.__file__)

import sys
sys.path.append(os.environ["DIR_PATH"])
from root_data_loader import load_data, classWtoSampleW
print(torch.cuda.is_available())
#print(torch.cuda.get_device_name(0))
#print(torch.cuda.device_count())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
from copy import deepcopy



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
  from pytorch_tabnet.augmentations import ClassificationSMOTE
  aug = ClassificationSMOTE(p=0.5)




  
  print("file loaded")
  print(data['cat_idxs'],data['cat_dims']) 
  clf = TabNetClassifier(
    n_d=32,
    n_a=32,
    verbose=1,
    cat_idxs=deepcopy(data['cat_idxs']),
    cat_dims=deepcopy(data['cat_dims']),
    cat_emb_dim=3,
    n_steps=5
    )  


  data =  load_data(file_path=os.path.join(path_sample,filename),varlist=varlist,test_ratio=0.1,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43'])

  from sklearn.metrics import mean_squared_error
  from pytorch_tabnet.metrics import Metric
  class WeightedMSE(Metric):
    def __init__(self):
        self._name = "WeightedMSE"
        self._maximize = False

    def __call__(self, y_true, y_score):
        weight = []
        for y in y_true:
          if y == 1:
            weight.append(data['class_weight'][1])
          else:
            weight.append(data['class_weight'][0])
        mse = mean_squared_error(y_true, y_score[:, 1],sample_weight=weight)
        return mse
  
  clf.fit(
    X_train=data['train_features'],y_train=data['train_y'],
    eval_set=[(data['val_features'], data['val_y'])],
    eval_metric=['auc','balanced_accuracy',WeightedMSE],
    max_epochs=1000,
    num_workers=12,
    #weights=data['class_weight']
    weights=1,
    batch_size=int(2097152/8),
    virtual_batch_size=512,
    #augmentations=aug,
    patience=50
    #callbacks=[pytorch_tabnet.callbacks.History(clf,verbose=1)]
  )
  
  clf.save_model('/cms/ldap_home/yeonjoon/working_dir/VcbMVAStudy/TabNet_template/model_largebatch_nosmote.pt')
  
  

