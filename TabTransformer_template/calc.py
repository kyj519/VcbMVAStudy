import ROOT
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys,os
sys.path.append(os.environ["DIR_PATH"])
ROOT.EnableImplicitMT(16)
from root_data_loader import load_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from copy import deepcopy
from tab_transformer_pytorch import TabTransformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyData(Dataset):
  def __init__(self,this_x,this_y):
    self.x = this_x
    self.y = this_y
  def __getitem__(self, index):
    this_x = self.x[index,:]
    this_y = self.y[index]
    return this_x, this_y
  def __len__(self):
    return len(self.x)
  

    
modelist = ['45','43','41','23','21']
#varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','pt_w_u','pt_w_d','weight']
# varlist.extend(['n_jets',
#                 'pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b',
#                 'eta_had_t_b','eta_w_u','eta_w_d','eta_lep_t_b',
#                 'bvsc_lep_t_b','bvsc_had_t_b',
#                 'm_w_u','m_w_d'])
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
#varlist = ['cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
# varlist.extend(['n_jets',
#                 'pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b',
#                 'eta_had_t_b','eta_w_u','eta_w_d','eta_lep_t_b',
#                 'bvsc_lep_t_b','bvsc_had_t_b'])
result = []

path_sample = os.environ["WtoCB_PATH"]
#filename = 'Vcb_Mu_TTLJ_WtoCB_powheg_25.root'
filename = 'Vcb_2018_Mu_Reco_Tree.root' 
data =  load_data(file_path=os.path.join(path_sample,filename),varlist=varlist,test_ratio=0,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43','Reco_41','Reco_23','Reco_21'])
cat_idxs = deepcopy(data['cat_idxs'])
temp_cat_dims = deepcopy(data['cat_dims'])
cont_idxs = [i for i in range(data['train_features'].shape[1]) if i not in cat_idxs]

model = TabTransformer(
    categories = tuple(temp_cat_dims),      # tuple containing the number of unique values within each category
    num_continuous = data['train_features'].shape[1]-len(cat_idxs),                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = 1,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1,                   # feed forward dropout
    mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    #continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
  )

def predict_proba(x_smp):
        model.load_state_dict(
          torch.load('/data6/Users/yeonjoon/VcbMVAStudy/TabTransformer_template/trans_bestmodel.pth',map_location=torch.device('cpu'))
        )
        fun=nn.Softmax(dim=1)
        y_temp = torch.tensor(np.zeros((len(x_smp),1)))
        smp_data = MyData(x_smp,y_temp)
        smploader = DataLoader(smp_data, batch_size=512, shuffle=False,num_workers=0)
        with torch.no_grad():
            model.eval()
            y_pred = torch.empty(0).to(device)
            for i, data in enumerate(smploader, 0):

                x_categ = data[0][:,cat_idxs].int().to(device)     # category values, from 0 - max number of categories, in the order as passed into the constructor above
                x_cont = data[0][:,cont_idxs].float().to(device)    # assume continuous values are already normalized individually
                y_outs = model(x_categ,x_cont)
                y_pred = torch.cat([y_pred,y_outs],dim=0)
            #y_pred = fun(y_pred)
            y_pred = y_pred.cpu().numpy()
        return y_pred
    
for mode in modelist:
    
    data =  load_data(file_path='/gv0/Users/yeonjoon/Vcb_2018_Mu_Reco_Tree.root',varlist=varlist,test_ratio=0,val_ratio=0.2,sigTree=[f'Reco_{mode}'],bkgTree=[])
    arr = data['train_features']
    
    pred = predict_proba(arr)
    print(np.max(pred))
    print(np.min(pred))
    result.append(pred)
    plt.hist(pred, bins=40, weights=data['train_weight'])
    plt.savefig(f'/data6/Users/yeonjoon/VcbMVAStudy/TabTransformer_template/result_{mode}.png')
    plt.clf()

    
    del data
    del arr 
    
df = pd.concat([pd.DataFrame(a, columns=[modelist[i]]) for i, a in enumerate(result)], axis=1)
fig = df.plot.hist(stacked=True, bins=30, figsize=(10, 6), grid=True)
fig.figure.savefig('stack.png',dpi=600)

data =  load_data(file_path='/gv0/Users/yeonjoon/Vcb_2018_Mu_Reco_Tree.root',varlist=varlist,test_ratio=0.1,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43'])
import postTrainingToolkit
train_score = predict_proba(data['train_features'])
val_score = predict_proba(data['val_features'])
kolS, kolB = postTrainingToolkit.KS_test(train_score,val_score,data['train_weight'],data['val_weight'],data['train_y'],data['val_y'])
print(f'{kolS}, {kolB}')
feature_importances_ = model._compute_feature_importances(data['train_features'])
print(feature_importances_)