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
#print(torch.cuda.get_device_name(0))
#print(torch.cuda.device_count())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
import optuna


varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
# varlist.extend(['n_jets',
#                 'pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b',
#                 'eta_had_t_b','eta_w_u','eta_w_d','eta_lep_t_b',
#                 'bvsc_lep_t_b','bvsc_had_t_b',
#                 'm_w_u','m_w_d'])

def objective(trial: optuna.Trial, data):
  param_model = {
    'n_d':trial.suggest_categorical('n_d',[pow(2,i) for i in [3,4,5,6]]),
    'n_steps':trial.suggest_int('n_steps',3,10),
    'gamma':trial.suggest_float('gamma',1.,2.),
    'cat_emb_dim':trial.suggest_int('cat_emb_dim',1,5),
    'n_independent':trial.suggest_int('n_independent',1,5),
    'n_shared':trial.suggest_int('n_independent',1,5),
  }
  param_model['n_a'] = param_model['n_d']
  param_fit = {
    'batch_size':trial.suggest_categorical('batch_size',[pow(2,i) for i in [17,18,19,20]])
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
    eval_metric=['auc'],
    weights=0,
    batch_size=param_fit['batch_size'],
    augmentations=aug,
    drop_last=False,
    num_workers=12,
    max_epochs=1
  )
  
  return clf.history('val_auc')[0]


  
if __name__ == '__main__':

  
  
  path_sample = os.environ["WtoCB_PATH"]
  #filename = 'Vcb_Mu_TTLJ_WtoCB_powheg_25.root'
  filename = 'Vcb_2018_Mu_Reco_Tree.root' 
  #,'Reco_41','Reco_23','Reco_21'
  data =  load_data(file_path=os.path.join(path_sample,filename),varlist=varlist,test_ratio=0,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43','Reco_41','Reco_23','Reco_21'])
  optuna.logging.set_verbosity(optuna.logging.DEBUG)
  study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), study_name="dist", storage='/cms/ldap_home/yeonjoon/working_dir/VcbMVAStudy/TabNet_template/study')
  study.optimize(lambda trial: objective(trial, data) ,n_trials=100,n_jobs=1)
  print(f"here is the result of hyperparameter tuning, best score = {study.best_value}:\n")
  print(study.best_trial.params)
  print("file loaded")

  

