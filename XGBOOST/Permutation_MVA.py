import argparse
from re import T, X
import ROOT
import os, sys
import numpy as np
import optuna
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from scipy import interpolate
import xgboost
from matplotlib import pyplot as plt
import plotly
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from root_data_loader import load_data, classWtoSampleW

plt.ioff()

varlist_prekin = ['had_t_b_pt','w_u_pt','w_d_pt','lep_t_b_pt','had_t_b_bscore','lep_t_b_bscore',
   'theta_w_u_w_d','theta_lep_neu','theta_lep_w_lep_t_b', 'del_phi_had_t_lep_t',
   'had_t_mass','had_w_mass','lep_t_mass','lep_t_partial_mass']
varlist = varlist_prekin + ['chi2']
  
    
  
    
# def load_data(file_path, n_jet, pre_kin):
#   print(file_path)
#   data_sig = ROOT.RDataFrame('Permutation_Correct',file_path).Filter(f'n_jets=={n_jet:.0f}').AsNumpy(columns=varlist_prekin if pre_kin else varlist)
#   data_bkg = ROOT.RDataFrame('Permutation_Wrong', file_path).Filter(f'n_jets=={n_jet:.0f}').AsNumpy(columns=varlist_prekin if pre_kin else varlist)
#   if pre_kin:
#     x_sig = np.vstack([data_sig[var] for var in varlist_prekin]).T
#     x_bkg = np.vstack([data_bkg[var] for var in varlist_prekin]).T
#   else:
#     x_sig = np.vstack([data_sig[var] for var in varlist]).T
#     x_bkg = np.vstack([data_bkg[var] for var in varlist]).T 
#   np.random.seed(555)
#   np.random.shuffle(x_sig)
#   np.random.shuffle(x_bkg)
#   x_sig_train = x_sig[:int(x_sig.shape[0]/10*9)][:]
#   x_bkg_train = x_bkg[:int(x_bkg.shape[0]/10*9)][:]
#   x_sig_test = x_sig[int(x_sig.shape[0]/10*9):][:]
#   x_bkg_test = x_bkg[int(x_bkg.shape[0]/10*9):][:]

#   x_train = np.vstack([x_sig_train, x_bkg_train])
#   x_test = np.vstack([x_sig_test, x_bkg_test])
#   num_sig_train = x_sig_train.shape[0]
#   num_sig_test = x_sig_test.shape[0]
#   num_bkg_train= x_bkg_train.shape[0]
#   num_bkg_test = x_bkg_test.shape[0]
#   y_train = np.hstack([np.ones(num_sig_train), np.zeros(num_bkg_train)])
#   y_test = np.hstack([np.ones(num_sig_test), np.zeros(num_bkg_test)]) 
#   num_all_train = num_bkg_train + num_sig_train
#   num_all_test = num_bkg_test + num_sig_test
  
#   w_train = np.hstack([np.ones(num_sig_train) * num_all_train / num_sig_train, np.ones(num_bkg_train) * num_all_train / num_bkg_train])
#   w_test = np.hstack([np.ones(num_sig_test) * num_all_test / num_sig_test, np.ones(num_bkg_test) * num_all_test / num_bkg_test])
  
#   return x_train, y_train, w_train, x_test, y_test, w_test

def objectiveXGB(trial: optuna.Trial, x_train, y_train, x_test, y_test ,x_val,y_val, w_train):
  param = {
    'n_estimators': trial.suggest_int('n_estimators',500,3000),
    'max_depth': trial.suggest_int('max_depth', 2, 10),
    'min_child_weight': trial.suggest_int('min_child_weight',1,300),
    'gamma': trial.suggest_float('gamma',1,3),
    'learning_rate': trial.suggest_float('learning_rate',0.01,0.05),
    'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree',0.5,1,0.1),
    'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
    'alpha': trial.suggest_loguniform('alpha',1e-3,10.0)
    
  }
  model = XGBClassifier(**param, show_progress_bar=True, use_label_encoder=False, early_stopping_rounds=1000,tree_method='gpu_hist',eval_metric = 'logloss')
  xgb_model=model.fit(x_train,y_train,sample_weight=w_train,verbose=True,eval_set=[(x_val,y_val)])
  y_pred = xgb_model.predict_proba(x_test)
  fpr, tpr, _ =roc_curve(y_test, [y[1] for y in y_pred])
  auc_val = auc(fpr,tpr)
  return auc_val

  
if __name__ == '__main__':
  ROOT.EnableImplicitMT()
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

  data_dict = load_data(os.path.join(path_sample,filename),n_jet,varlist_prekin if pre_kin else varlist,0.2,0.2)
  

  
  if not os.path.isfile(os.path.join(os.environ['DIR_PATH'],'XGBOOST',f'XGBOOST_{n_jet}_{pre_kin}_model.json')):
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    trainsetSampleW = classWtoSampleW(data_dict['train_y'],data_dict['class_weights'])
    study.optimize(lambda trial: objectiveXGB(trial, data_dict['train_features'],data_dict['train_y'],data_dict['test_features'],data_dict['test_y'],data_dict['val_features'],data_dict['val_y'],trainsetSampleW),n_trials=70)
    print(f"here is the result of hyperparameter tuning, best score = {study.best_value}:\n")
    print(study.best_trial.params)
    bdt = XGBClassifier(**study.best_trial.params, tree_method='gpu_hist', show_progress_bar=True, use_label_encoder=False)
    bdt.fit(data_dict['train_features'],data_dict['train_y'],sample_weight=trainsetSampleW)
    fig_contour = optuna.visualization.plot_contour(study)
    fig_importance = optuna.visualization.plot_param_importances(study)
    fig_contour.write_html(f'XGBOOST_{n_jet}_{pre_kin}_opt_contour.html')
    fig_importance.write_html(f'XGBOOST_{n_jet}_{pre_kin}_opt_importance.html')
  else:
    bdt = XGBClassifier()
    bdt.load_model(os.path.join(os.environ['DIR_PATH'],'XGBOOST',f'XGBOOST_{n_jet}_{pre_kin}_model.json'))
  bdt.save_model(os.path.join(os.environ['DIR_PATH'],'XGBOOST',f'XGBOOST_{n_jet}_{pre_kin}_model.json'))
  y_pred = bdt.predict_proba(data_dict['test_features'])
  fpr, tpr, _ =roc_curve(data_dict['test_y'], [y[1] for y in y_pred])
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
  
  

  