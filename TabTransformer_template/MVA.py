import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

from tqdm import tqdm
sys.path.append(os.environ["DIR_PATH"])
from root_data_loader import load_data, classWtoSampleW
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
from copy import deepcopy
from tab_transformer_pytorch import TabTransformer
from sklearn.metrics import mean_squared_error
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','pt_w_u','pt_w_d','weight']
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
#varlist = ['cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
# varlist.extend(['n_jets',
#                 'pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b',
#                 'eta_had_t_b','eta_w_u','eta_w_d','eta_lep_t_b',
#                 'bvsc_lep_t_b','bvsc_had_t_b'])
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
  
  
def mean_sq_error(model, dloader, device,cat_idxs,con_idxs):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ = data[0][:,cat_idxs].int().to(device)    
            x_cont = data[0][:,con_idxs].float().to(device)   
            y_t = data[1].to(device)
            y_outs = model(x_categ,x_cont)
            y_test = torch.cat([y_test,y_t.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,y_outs],dim=0)
        rmse = mean_squared_error(y_test.cpu(), y_pred.cpu(), squared=False)
        return rmse
  
if __name__ == '__main__':

  MAX_EPOCH = 1000
  
  path_sample = os.environ["WtoCB_PATH"]
  #filename = 'Vcb_Mu_TTLJ_WtoCB_powheg_25.root'
  filename = 'Vcb_2018_Mu_Reco_Tree.root' 
  #,'Reco_41','Reco_23','Reco_21'
  data =  load_data(file_path=os.path.join(path_sample,filename),varlist=varlist,test_ratio=0,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43','Reco_41','Reco_23','Reco_21'])

  cat_idxs = deepcopy(data['cat_idxs'])
  temp_cat_dims = deepcopy(data['cat_dims'])
  
  data =  load_data(file_path=os.path.join(path_sample,filename),varlist=varlist,test_ratio=0,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43'])

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
  train_data = MyData(data['train_features'],data['train_y'])
  trainloader = DataLoader(train_data, batch_size=1024, shuffle=True,num_workers=8)
  val_data = MyData(data['val_features'],data['val_y']) 
  valloader = DataLoader(val_data, batch_size=256, shuffle=False,num_workers=8)
  criterion = nn.MSELoss().to(device)
  model.to(device)
  optimizer = optim.AdamW(model.parameters(),lr=0.0001)
  
  cont_idxs = [i for i in range(data['train_features'].shape[1]) if i not in cat_idxs]
  #cat_features = data['train_features'][:,[i for i in temp_cat_idxs]]
  #cont_features = data['train_features'][:,[cont_idxs]] 
  train_running_loss = 0.0
  best_valid_auroc = 0
  best_valid_accuracy = 0
  best_valid_rmse = 100000
  modelsave_path = '/data6/Users/yeonjoon/VcbMVAStudy/TabTransformer_template'
  for epoch in range(MAX_EPOCH):
    model.train()
    for i,data in enumerate(tqdm(trainloader)):
      optimizer.zero_grad()
      x_cat = data[0][:,cat_idxs].int().to(device)
      x_cont = data[0][:,cont_idxs].float().to(device)
      y_outs = model(x_cat,x_cont)
      y_dim = y_outs.shape[1]
      print(f'y_dim is {y_dim}')  
      print(f'y_out shape {y_outs.shape}')  
      y_t = data[1].float().to(device)
      print(f'y_t is {y_t.shape}') 
      loss_list = []
      for loss_idx in range(y_dim):
        loss_list.append(criterion(y_outs[:,loss_idx].reshape(-1,1),y_t.reshape(-1,1)))
      loss = sum(loss_list)/y_dim

      loss.backward()
      optimizer.step()
      train_running_loss += loss.item()
    if epoch%1 == 0:
      model.eval()
      with torch.no_grad():
        val_rmse = mean_sq_error(model, valloader, device,cat_idxs,cont_idxs) 
        train_rmse = mean_sq_error(model, trainloader, device,cat_idxs,cont_idxs) 
        print('[EPOCH %d] train RMSE: %.5f VALID RMSE: %.5f' %
        (epoch + 1, train_rmse, val_rmse))
      if val_rmse < best_valid_rmse:
        best_valid_rmse = val_rmse
        torch.save(model.state_dict(),'%s/trans_bestmodel.pth' % (modelsave_path))
        
      model.train()
      
      


