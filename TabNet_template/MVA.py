import ROOT
import os, sys
import numpy as np
#######
#BELOW CODE IS OPTIMIZED FOR TENSORFLOW-2.4.1!
#######
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import sys
sys.path.append('/data6/Users/yeonjoon/VcbMVAStudy')
from root_data_loader import load_data, classWtoSampleW
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())

ROOT.EnableImplicitMT()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']

  
if __name__ == '__main__':

  
  
  path_sample = os.environ["WtoCB_PATH"]
  #filename = 'Vcb_Mu_TTLJ_WtoCB_powheg_25.root'
  filename = 'Vcb_2018_Mu_Reco_Tree.root' 
 
  data =  load_data(os.path.join(path_sample,filename), '-10.<bvsc_w_u',varlist,0.1,0.2,['Reco_45'],['Reco_43','Reco_41','Reco_23','Reco_21'])
  clf = TabNetClassifier()  #TabNetRegressor()
  clf.fit(
    data['train_features'],data['train_y'],
    eval_set=[(data['val_features'], data['val_y'])],
    eval_metric=['auc'],
    weights=data['class_weight']
  )
  preds = clf.predict(data['test_features'])
  clf.save_model('/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/')
  
  

