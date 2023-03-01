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
model.load_model('/cms/ldap_home/yeonjoon/working_dir/VcbMVAStudy/TabNet_template/model.pt.zip')
modelist = ['45','43','41','23','21']
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
# varlist.extend(['n_jets',
#                 'pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b',
#                 'eta_had_t_b','eta_w_u','eta_w_d','eta_lep_t_b',
#                 'bvsc_lep_t_b','bvsc_had_t_b',
#                 'm_w_u','m_w_d'])
for mode in modelist:
    data =  load_data(file_path='root://cluster142.knu.ac.kr//store/user/yeonjoon/Vcb_2018_Mu_Reco_Tree.root',varlist=varlist,test_ratio=0,val_ratio=0,sigTree=[f'Reco_{mode}'],bkgTree=[])
    arr = data['train_features']
    plt.hist(model.predict_proba(arr)[:,1],bins=40)
    plt.savefig(f'/cms/ldap_home/yeonjoon/working_dir/VcbMVAStudy/TabNet_template/result_{mode}.png')
    plt.clf()

    del arr
    