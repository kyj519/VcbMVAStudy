import os, sys
import numpy as np
print(np.__file__)
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.augmentations import ClassificationSMOTE
import torch
print(torch.__file__)

import sys
sys.path.append(os.environ["DIR_PATH"])
from root_data_loader import load_data, classWtoSampleW
print(torch.cuda.is_available())


os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"


print(torch.cuda.device_count())

import optuna
n_trial = 50
gpu_queue = [0 if i % 2 == 0 else 1 for i in range(50)]

varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
# varlist.extend(['n_jets',
#                 'pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b',
#                 'eta_had_t_b','eta_w_u','eta_w_d','eta_lep_t_b',
#                 'bvsc_lep_t_b','bvsc_had_t_b',
#                 'm_w_u','m_w_d'])
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.metrics import Metric

    
def objective(trial: optuna.Trial, data):
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
  param_model = {
    'n_d':trial.suggest_categorical('n_d',[pow(2,i) for i in [3,4,5,6]]),
    'n_steps':trial.suggest_int('n_steps',3,10),
    'gamma':trial.suggest_float('gamma',1.,2.),
    'cat_emb_dim':trial.suggest_int('cat_emb_dim',1,5),
    'n_independent':trial.suggest_int('n_independent',1,5),
    'n_shared':trial.suggest_int('n_independent',1,5),
    'device_name':f'cuda:{gpu_queue[trial.number]}'
  }
  param_model['n_a'] = param_model['n_d']
  param_fit = {
    'batch_size':trial.suggest_categorical('batch_size',[pow(2,i) for i in [16,17,18,19]]),
    'virtual_batch_size':trial.suggest_categorical('virtual_batch_size',[pow(2,i) for i in [7,8,9,10]])
  }
  print(param_model)
  print(param_fit)
  trial_num = trial.number
  print(f'trial number is {trial_num}')
  aug = ClassificationSMOTE(p=0.5)
  clf = TabNetClassifier(
    **param_model
    )  
  clf.fit(
    X_train=data['train_features'],y_train=data['train_y'],
    eval_set=[(data['val_features'], data['val_y'])],
    eval_name=['val'],
    eval_metric=['auc',WeightedMSE],
    weights=0,
    batch_size=param_fit['batch_size'],
    #augmentations=aug,
    drop_last=False,
    num_workers=12,
    max_epochs=50
  )
  
  return clf.history('wmse')[-1]


  
if __name__ == '__main__':

  
  from copy import deepcopy
  path_sample = os.environ["WtoCB_PATH"]
  #filename = 'Vcb_Mu_TTLJ_WtoCB_powheg_25.root'
  filename = 'Vcb_2018_Mu_Reco_Tree.root' 
  #,'Reco_41','Reco_23','Reco_21'
  data =  load_data(file_path=os.path.join(path_sample,filename),varlist=varlist,test_ratio=0,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43','Reco_41','Reco_23','Reco_21'])
  idx_c = deepcopy(data['cat_idxs'])
  dim_c = deepcopy(data['cat_dims'])
  data =  load_data(file_path=os.path.join(path_sample,filename),varlist=varlist,test_ratio=0.1,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43'])
  data['cat_idxs'] = idx_c
  data['cat_dims'] = dim_c
  optuna.logging.set_verbosity(optuna.logging.DEBUG)
  study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
  study.optimize(lambda trial: objective(trial, data) ,n_trials=n_trial,n_jobs=2)
  
  print(f"here is the result of hyperparameter tuning, best score = {study.best_value}:\n")
  print(study.best_trial.params)


  

