import ROOT
import os 
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from root_data_loader import load_data

if __name__ == '__main__':
    filepath = 'root://cluster142.knu.ac.kr//store/user/yeonjoon/Vcb_2018_Mu_Reco_Tree.root'
    f = ROOT.TFile(filepath, 'READ')
    model = TabNetClassifier()
    model.load_model('/cms/ldap_home/yeonjoon/working_dir/VcbMVAStudy/TabNet_template/model_largebatch_nosmote.pt.zip')
    varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
	data = load_data(file_path=filepath,varlist=varlist,test_ratio=0.1,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43','Reco_41','Reco_21','Reco_23'])
 