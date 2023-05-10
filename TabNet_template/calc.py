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
folder_path = '/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/model_KPS_balancedAccu'
files = os.listdir(folder_path)
# Filter for files with a .pt.zip extension
pt_zip_files = [f for f in files if f.endswith('.pt.zip')]

model.load_model(os.path.join(folder_path,pt_zip_files[0]))
modelist = ['45','43','41','23','21']
varlist = ['bvsc_w_d','cvsl_w_u','cvsb_w_u','cvsb_w_d','n_bjets','pt_had_t_b','pt_w_d','bvsc_had_t_b','weight']
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','pt_w_u','pt_w_d','weight']
varlist.extend(['n_jets',
                'pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b',
                'eta_had_t_b','eta_w_u','eta_w_d','eta_lep_t_b',
                'bvsc_lep_t_b','bvsc_had_t_b',
                'm_w_u','m_w_d'])


#varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
#varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
#varlist = ['cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
#varlist.extend(['n_jets',
                # 'pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b',
                # 'eta_had_t_b','eta_w_u','eta_w_d','eta_lep_t_b',
                # 'bvsc_lep_t_b','bvsc_had_t_b'])
######fullinput
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
varlist.extend(['n_jets',
                'pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b',
                'eta_had_t_b','eta_w_u','eta_w_d','eta_lep_t_b',
                'bvsc_lep_t_b','bvsc_had_t_b'])


  
#KPS modification
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d',
           'cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight'
           ,'pt_w_u','pt_w_d','eta_w_u','eta_w_d','best_mva_score']
result = []
outfile = ROOT.TFile(os.path.join(folder_path,'predictions.root'), "RECREATE")
for mode in modelist:
    continue
    for reco_status in ['Correct','Fail_00','Fail_10','Fail_01','Fail_11']:
        data =  load_data(file_path='/gv0/Users/yeonjoon/Vcb_2018_Mu_Reco_Tree_byRecoStatus.root',varlist=varlist,test_ratio=0,val_ratio=0,sigTree=[f'Reco_{mode}_{reco_status}'],bkgTree=[],filterstr="n_bjets>=3")
        arr = data['train_features']
        weights = data['train_weight']
        pred = model.predict_proba(arr)[:,1]
        hist = ROOT.TH1F(f"pred_{mode}_{reco_status}", f"Predictions for Reco_{mode}_{reco_status}", 40, 0., 1.)
        for i in range(len(pred)):
            hist.Fill(pred[i], weights[i])
        hist.Write() 
        del data
        del arr
        del weights
        del pred
outfile.Close()
# df = pd.concat([pd.DataFrame(a, columns=[modelist[i]]) for i, a in enumerate(result)], axis=1)
# fig = df.plot.hist(stacked=True, bins=30, figsize=(10, 6), grid=True)
# fig.figure.savefig('stack.png',dpi=600)
import postTrainingToolkit

data = load_data(file_path='/gv0/Users/yeonjoon/Vcb_2017_Mu_Reco_Tree.root',varlist=varlist,test_ratio=0,val_ratio=0,sigTree=['Reco_45'],bkgTree=['Reco_43','Reco_41','Reco_21','Reco_23'])
arr = data['train_features']
weights = data['train_weight']
y=data['train_y']
pred = model.predict_proba(arr)[:,1]
postTrainingToolkit.ROC_AUC(score=pred,y=y,weight=weights,plot_path=folder_path)

data =  load_data(file_path='/gv0/Users/yeonjoon/Vcb_2017_Mu_Reco_Tree.root',varlist=varlist,test_ratio=0.1,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43'])

train_score = model.predict_proba(data['train_features'])[:,1]
val_score = model.predict_proba(data['val_features'])[:,1]
kolS, kolB = postTrainingToolkit.KS_test(train_score,val_score,data['train_weight'],data['val_weight'],data['train_y'],data['val_y'],plotPath=folder_path)
print(f'{kolS}, {kolB}')

sys.exit()

res_explain, res_masks = model.explain(data['train_features'])
np.save(os.path.join(folder_path,'explain.npy'), res_explain)
np.save(os.path.join(folder_path,'mask.npy'),res_masks)
np.save(os.path.join(folder_path,'y.npy'),data['train_y'])
#feature_importances_ = model._compute_feature_importances(data['train_features'])
#print(feature_importances_)