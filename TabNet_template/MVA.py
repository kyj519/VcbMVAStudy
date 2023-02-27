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



varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']

  
if __name__ == '__main__':

  
  
  path_sample = os.environ["WtoCB_PATH"]
  #filename = 'Vcb_Mu_TTLJ_WtoCB_powheg_25.root'
  filename = 'Vcb_2018_Mu_Reco_Tree.root' 
  #,'Reco_41','Reco_23','Reco_21'
  data =  load_data(os.path.join(path_sample,filename), '-10.<bvsc_w_u',varlist,0.1,0.2,['Reco_45'],['Reco_43','Reco_41'])
  print("file loaded")
  print(data['cat_idxs'],data['cat_dims']) 
  clf = TabNetClassifier(
    verbose=1,
    cat_idxs=data['cat_idxs'],
    cat_dims=data['cat_dims'],
    cat_emb_dim=3
    )  
  clf.fit(
    X_train=data['train_features'],y_train=data['train_y'],
    eval_set=[(data['val_features'], data['val_y'])],
    eval_metric=['auc'],
    patience=20,
    #weights=data['class_weight']
    weights=1,
    batch_size=1024,
    #callbacks=[pytorch_tabnet.callbacks.History(clf,verbose=1)]
  )
  #preds = clf.predict(data['test_features'])
  clf.save_model('/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/model.pt'))
  
  

