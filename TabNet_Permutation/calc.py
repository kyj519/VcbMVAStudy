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
folder_path = '/data6/Users/yeonjoon/VcbMVAStudy/TabNet_Permutation/model_4jets'
files = os.listdir(folder_path)
# Filter for files with a .pt.zip extension
pt_zip_files = [f for f in files if f.endswith('.zip')]

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


  
varlist = ["met_pt", "neutrino_p", "lepton_pt", "pt_ratio",
    "pt_had_t_b", "pt_w_u", "pt_w_d", "pt_lep_t_b",
    "bvsc_had_t_b", "cvsb_had_t_b", "cvsl_had_t_b",     "bvsc_lep_t_b", "cvsb_lep_t_b", "cvsl_lep_t_b",
    "theta_w_u_w_d", "theta_lep_neu", "theta_lep_w_lep_t_b", "del_phi_had_t_lep_t", 
    "had_t_mass", "had_w_mass", "lep_t_mass","lep_t_partial_mass",
    "chi2_jet_had_t_b","chi2_jet_w_u","chi2_jet_w_d","chi2_jet_lep_t_b",
    "chi2_constraint_had_t", "chi2_constraint_had_w",
    "chi2_constraint_lep_t", "chi2_constraint_lep_w"]
# varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d',
#            'cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight'
#            ,'m_had_t','m_had_w','best_mva_score']           
result = []
# outfile = ROOT.TFile(os.path.join(folder_path,'predictions.root'), "RECREATE")
# for mode in modelist:
#     for reco_status in ['Correct','Fail_00','Fail_10','Fail_01','Fail_11']:
#         filter_str = f'decay_mode=={mode}'
    
#         file = '/gv0/Users/yeonjoon/Vcb/Sample/2018/Mu/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root' if mode =='45' else '/gv0/Users/yeonjoon/Vcb/Sample/2018/Mu/RunResult/Central_Syst/Vcb_TTLJ_powheg.root'
#         input_tuple=( #first element of tuple = signal tree, second =bkg tree.
#     [('/gv0/Users/yeonjoon/Vcb/Sample/2018/Mu/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','POGTightWithTightIso_Central/Result_Tree',f'chk_reco_=={reco_status}&&decay_mode=={mode}')] ##TTLJ_WtoCB Reco 1, (file_path, tree_path, filterstr)
#     ,[] ##TTLJ_WtoCB cs decay
#     )
#         data =  load_data(tree_path_filter_str=input_tuple,varlist=varlist,test_ratio=0.1,val_ratio=0.2)
#         arr = data['train_features']
#         weights = data['train_weight']
#         pred = model.predict_proba(arr)[:,1]
#         hist = ROOT.TH1F(f"pred_{mode}_{reco_status}", f"Predictions for Reco_{mode}_{reco_status}", 40, 0., 1.)
#         for i in range(len(pred)):
#             hist.Fill(pred[i], weights[i])
#         hist.Write() 
#         del data
#         del arr
#         del weights
#         del pred
# outfile.Close()
# df = pd.concat([pd.DataFrame(a, columns=[modelist[i]]) for i, a in enumerate(result)], axis=1)
# fig = df.plot.hist(stacked=True, bins=30, figsize=(10, 6), grid=True)
# fig.figure.savefig('stack.png',dpi=600)
import postTrainingToolkit
n_jets = 4
input_tuple=( #first element of tuple = signal tree, second =bkg tree.
    [('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/Mu/RunPermutationTree/Vcb_TTLJ_WtoCB_powheg.root','Permutation_Correct',f'n_jets=={n_jets}'),
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/El/RunPermutationTree/Vcb_TTLJ_WtoCB_powheg.root','Permutation_Correct',f'n_jets=={n_jets}')
     ], ##TTLJ_WtoCB Reco 1, (file_path, tree_path, filterstr)
    
    [('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/Mu/RunPermutationTree/Vcb_TTLJ_WtoCB_powheg.root','Permutation_Wrong',f'n_jets=={n_jets}'),
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/El/RunPermutationTree/Vcb_TTLJ_WtoCB_powheg.root','Permutation_Wrong',f'n_jets=={n_jets}')
     ] ##TTLJ_WtoCB cs decay
  )
data =  load_data(tree_path_filter_str=input_tuple,varlist=varlist,test_ratio=0,val_ratio=0)
arr = data['train_features']
weights = data['train_weight']
y=data['train_y']
pred = model.predict_proba(arr)[:,1]
postTrainingToolkit.ROC_AUC(score=pred,y=y,weight=weights,plot_path=folder_path)
del data

data =  load_data(tree_path_filter_str=input_tuple,varlist=varlist,test_ratio=0.1,val_ratio=0.2)
arr = data['train_features']
weights = data['train_weight']
y=data['train_y']

train_score = model.predict_proba(data['train_features'])[:,1]
val_score = model.predict_proba(data['val_features'])[:,1]
kolS, kolB = postTrainingToolkit.KS_test(train_score,val_score,data['train_weight'],data['val_weight'],data['train_y'],data['val_y'],plotPath=folder_path)
print(f'{kolS}, {kolB}')


res_explain, res_masks = model.explain(data['train_features'])
np.save(os.path.join(folder_path,'explain.npy'), res_explain)
np.save(os.path.join(folder_path,'mask.npy'),res_masks)
np.save(os.path.join(folder_path,'y.npy'),data['train_y'])
#feature_importances_ = model._compute_feature_importances(data['train_features'])
#print(feature_importances_)