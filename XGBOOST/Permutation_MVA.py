import argparse
from re import T, X
import ROOT
import os
import numpy as np
import optuna
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from scipy import interpolate
import xgboost
from matplotlib import pyplot as plt
plt.ioff()

varlist_prekin = ['had_t_b_pt','w_u_pt','w_d_pt','lep_t_b_pt','had_t_b_bscore','lep_t_b_bscore',
   'theta_w_u_w_d','theta_lep_neu','theta_lep_w_lep_t_b', 'del_phi_had_t_lep_t',
   'had_t_mass','had_w_mass','lep_t_mass','lep_t_partial_mass']
varlist = varlist_prekin + ['chi2']
def save_roc_plot(prob, answer, save_path):
  thresholds = np.linspace(0.01,0.99,20)
  x = []
  y = []
  for threshold in thresholds:
    pred = np.array([1 if p[1] > threshold else 0 for p in prob])
    n_true_positive = 0
    n_false_positive = 0
    n_true_negative = 0
    n_false_negative = 0
    for idx in range(len(answer)):
      if pred[idx] == 1:
        if answer[idx] == 1: n_true_positive = n_true_positive + 1
        elif answer[idx] == 0: n_false_positive = n_false_positive + 1
      if pred[idx] == 0:
        if answer[idx] == 1: n_false_negative = n_false_negative + 1
        elif answer[idx] == 0: n_true_negative = n_true_negative + 1

    sensitivity = n_true_positive/(n_true_positive+n_false_negative)
    specificity = n_true_negative/(n_true_negative+n_false_positive)
    x.append(1-specificity)
    y.append(sensitivity)
  from matplotlib import pyplot as plt
  plt.xlim([0,1])
  plt.ylim([0,1])
  print(x)
  print(y)
  plt.plot(x,y)
  plt.savefig(save_path)
  
    
  
      
    
  
  
      
def load_data(file_path, n_jet, pre_kin):
  print(file_path)
  data_sig = ROOT.RDataFrame('Permutation_Correct',file_path).Filter(f'n_jets=={n_jet:.0f}').AsNumpy(columns=varlist_prekin if pre_kin else varlist)
  data_bkg = ROOT.RDataFrame('Permutation_Wrong', file_path).Filter(f'n_jets=={n_jet:.0f}').AsNumpy(columns=varlist_prekin if pre_kin else varlist)
  if pre_kin:
    x_sig = np.vstack([data_sig[var] for var in varlist_prekin]).T
    x_bkg = np.vstack([data_bkg[var] for var in varlist_prekin]).T
  else:
    x_sig = np.vstack([data_sig[var] for var in varlist]).T
    x_bkg = np.vstack([data_bkg[var] for var in varlist]).T 
  np.random.seed(555)
  np.random.shuffle(x_sig)
  np.random.shuffle(x_bkg)
  x_sig_train = x_sig[:int(x_sig.shape[0]/10*9)][:]
  x_bkg_train = x_bkg[:int(x_bkg.shape[0]/10*9)][:]
  x_sig_test = x_sig[int(x_sig.shape[0]/10*9):][:]
  x_bkg_test = x_bkg[int(x_bkg.shape[0]/10*9):][:]

  x_train = np.vstack([x_sig_train, x_bkg_train])
  x_test = np.vstack([x_sig_test, x_bkg_test])
  num_sig_train = x_sig_train.shape[0]
  num_sig_test = x_sig_test.shape[0]
  num_bkg_train= x_bkg_train.shape[0]
  num_bkg_test = x_bkg_test.shape[0]
  y_train = np.hstack([np.ones(num_sig_train), np.zeros(num_bkg_train)])
  y_test = np.hstack([np.ones(num_sig_test), np.zeros(num_bkg_test)]) 
  num_all_train = num_bkg_train + num_sig_train
  num_all_test = num_bkg_test + num_sig_test
  
  w_train = np.hstack([np.ones(num_sig_train) * num_all_train / num_sig_train, np.ones(num_bkg_train) * num_all_train / num_bkg_train])
  w_test = np.hstack([np.ones(num_sig_test) * num_all_test / num_sig_test, np.ones(num_bkg_test) * num_all_test / num_bkg_test])
  
  return x_train, y_train, w_train, x_test, y_test, w_test


  
if __name__ == '__main__':
  ROOT.EnableImplicitMT(32)
  parser = argparse.ArgumentParser()
  parser.add_argument('--n_jet', type=int)
  parser.add_argument('--pre_kin', type=int)
  args = parser.parse_args()
  n_jet = args.n_jet
  pre_kin = True if args.pre_kin else False
  
  path_dir = os.environ["DIR_PATH"]+"/result/XGBOOST/"
  path_sample = os.environ["WtoCB_PATH"]
  filename = 'Vcb_Mu_TTLJ_WtoCB_powheg_25.root'
  
  #prepare_variable(ROOT.RDataFrame("Permutation_Correct",path_sample+filename+'.root'),path_sample+filename+'_signal.root')
  #prepare_variable(ROOT.RDataFrame("Permutation_Wrong",path_sample+filename+'.root'),path_sample+filename+'_bkg.root')
  x_train, y_train, w_train, x_test, y_test, w_test = load_data(os.path.join(path_sample,filename),n_jet,pre_kin)
  
  

  
  if not os.path.isfile(os.path.join(os.environ['DIR_PATH'],'XGBOOST',f'XGBOOST_{n_jet}_{pre_kin}_model.json')):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial:objectiveXGB(trial, x_train,y_train,x_test,y_test,w_train), n_trials=50)
    print(f"here is the result of hyperparameter tuning, best score = {study.best_value}:\n")
    print(study.best_trial.params)
    bdt = XGBClassifier(**study.best_trial.params)
    bdt.fit(x_train,y_train,sample_weight=w_train)
  else:
    bdt = XGBClassifier()
    bdt.load_model(os.path.join(os.environ['DIR_PATH'],'XGBOOST',f'XGBOOST_{n_jet}_{pre_kin}_model.json'))

  y_pred = bdt.predict_proba(x_test)
  fpr, tpr, _ =roc_curve(y_test, [y[1] for y in y_pred])
  auc_val = auc(fpr,tpr)
  roc_interpld = interpolate.interp1d(fpr,tpr,kind='linear')
  x_for_tmva_comparison = [0.01,0.1,0.3]
  y_for_tmva_comparison = roc_interpld(x_for_tmva_comparison)
  ROOT.TMVA.Experimental.SaveXGBoost(bdt, "myBDT", f'XGBOOST_{n_jet}_{pre_kin}.root',num_inputs = len(varlist_prekin) if pre_kin else len(varlist) )
  plt.title(f'AUC is {auc_val}')
  plt.plot(fpr,tpr)
  plt.savefig(f'XGBOOST_{n_jet}_{pre_kin}.png')
  for tup in zip(x_for_tmva_comparison,y_for_tmva_comparison):
    print(f'effciency @B{tup[0]} is {tup[1]}\n')
  
  
def objectiveXGB(trial: optuna.Trial, x_train, y_train, x_test, y_test, w_train):
  param = {
    'eval_metrics' : 'logloss',
    'n_estimators': trial.suggest_int('n_estimators',500,4000),
    'max_depth': trial.suggest_int('max_depth', 8, 16),
    'min_child_weight': trial.suggest_int('min_child_weight',1,300),
    'gamma': trial.suggest_int('gamma',1,3),
    'learning_rate': trial.suggest_float('learning_rate',0.01,0.05),
    'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree',0.5,1,0.1),
    'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
    'alpha': trial.suggest_loguniform('alpha',1e-3,10.0),
  }
  model = XGBClassifier(**param)
  xgb_model=model.fit(x_train,y_train,sample_weight=w_train)
  y_pred = xgb_model.predict_proba(x_test)
  fpr, tpr, _ =roc_curve(y_test, [y[1] for y in y_pred])
  auc_val = auc(fpr,tpr)
  return auc_val
  