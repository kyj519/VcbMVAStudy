import ROOT
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys,os
sys.path.append(os.environ["DIR_PATH"])
ROOT.EnableImplicitMT(4)
from root_data_loader import load_data
model = TabNetClassifier()
folder_path = '/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/model_2017_mu+el_sample_calssweight_RFCompleted_RECOMVAincluded_swap_nstep2_smallBN_only45'
folder_path = '/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/model_2017_mu+el_sample_calssweight_RFCompleted_RECOMVAincluded_swap_nstep2_smallBN_addhadTWmass_model_size_bbang_bbang_onlySR'
folder_path = '/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/test'
folder_path = '/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/model_2017_mu+el_sample_calssweight_RFCompleted_RECOMVAincluded_swap_nstep2_smallBN_addhadTWmass_model_size_bbang_bbang_focal_loss_lib_class_weighted_largebatch_weight0_largemodel_completeWeight_NOCvsB'
folder_path = '/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/model_Mu_RunResult_twostep'
files = os.listdir(folder_path)
# Filter for files with a .pt.zip extension
pt_zip_files = [f for f in files if f.endswith('.zip')]

model.load_model(os.path.join(folder_path,pt_zip_files[0]))         
import postTrainingToolkit
data = np.load(os.path.join(folder_path,'data.npz'),allow_pickle=True)
data = data['arr_0'][()]

#postTrainingToolkit.ROC_AUC(score=model.predict_proba(data['test_features'])[:,1],y=data['test_y'],weight=data['test_weight'],plot_path=folder_path)
postTrainingToolkit.ROC_AUC(score=model.predict_proba(data['test_features'])[:,1],y=data['test_y'],plot_path=folder_path)


train_score = model.predict_proba(data['train_features'])[:,1]
val_score = model.predict_proba(data['val_features'])[:,1]

kolS, kolB = postTrainingToolkit.KS_test(train_score=train_score,val_score=val_score,train_w=data['train_weight'],val_w=data['val_weight'],train_y=data['train_y'],val_y=data['val_y'],plotPath=folder_path)
print(f'{kolS}, {kolB}')


res_explain, res_masks = model.explain(data['test_features'])
np.save(os.path.join(folder_path,'explain.npy'), res_explain)
np.save(os.path.join(folder_path,'mask.npy'),res_masks)
np.save(os.path.join(folder_path,'y.npy'),data['train_y'])
feature_importances_ = model._compute_feature_importances(data['test_features'])
print(feature_importances_)
