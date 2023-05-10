import os, sys, argparse
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from imblearn.over_sampling import SMOTENC
import torch
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.metrics import Metric
import tqdm

sys.path.append(os.environ["DIR_PATH"])
from root_data_loader import load_data, classWtoSampleW



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is {device}')



varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','pt_w_u','pt_w_d','weight']
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
#varlist = ['cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
# varlist.extend(['n_jets',
#                 'pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b',
#                 'eta_had_t_b','eta_w_u','eta_w_d','eta_lep_t_b',
#                 'bvsc_lep_t_b','bvsc_had_t_b'])
varlist = ['bvsc_w_d','cvsl_w_u','cvsb_w_u','cvsb_w_d','n_bjets','pt_had_t_b','pt_w_d','bvsc_had_t_b','weight']
  
#KPS modification
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d',
           'cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight'
           ,'pt_w_u','pt_w_d','eta_w_u','eta_w_d','best_mva_score']
#


def train(input_root_file, model_save_path, doSmote=False):
  
  data =  load_data(file_path=input_root_file,varlist=varlist,test_ratio=0.1,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43','Reco_41','Reco_23','Reco_21'])
  sm = SMOTENC(random_state=42, categorical_features=deepcopy(data["cat_idxs"]))
  clf = TabNetClassifier(
    n_d=32,
    n_a=32,
    verbose=1,
    cat_idxs=deepcopy(data['cat_idxs']),
    cat_dims=deepcopy(data['cat_dims']),
    cat_emb_dim=3,
    n_steps=5
    )  

  del data
  data =  load_data(file_path=input_root_file,varlist=varlist,test_ratio=0.1,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43'])


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
  if doSmote:
    data['train_features'], data['train_y'] = sm.fit_resample(data['train_features'], data['train_y'])
    
  clf.fit(
    X_train=data['train_features'],y_train=data['train_y'],
    eval_set=[(data['val_features'], data['val_y'])],
    eval_metric=['auc','balanced_accuracy'],
    max_epochs=1000,
    num_workers=12,
    #weights=data['class_weight']
    #weights=0 if doSmote else 1,
    weights=data['train_sample_and_class_weight'],
    batch_size=int(2097152/4),
    virtual_batch_size=512,
    #augmentations=aug,
    patience=100
    #callbacks=[pytorch_tabnet.callbacks.History(clf,verbose=1)]
  )
  
  clf.save_model(model_save_path)
  
def plot(input_root_file,input_model_path,out_path):
  import ROOT
  import postTrainingToolkit
    
  model = TabNetClassifier()
  model.load_model(input_model_path)
  result = []
  modelist = ['45','43','41','23','21']
  outfile = ROOT.TFile(os.path.join(out_path,'predictions.root'), "RECREATE")
  for mode in modelist:
      for reco_status in ['Correct','Fail_00','Fail_10','Fail_01','Fail_11']:
          data =  load_data(file_path=input_root_file,varlist=varlist,test_ratio=0,val_ratio=0,sigTree=[f'Reco_{mode}_{reco_status}'],bkgTree=[],filterstr="n_bjets>=3")
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


  data = load_data(file_path=input_root_file,varlist=varlist,test_ratio=0,val_ratio=0,sigTree=['Reco_45'],bkgTree=['Reco_43','Reco_41','Reco_21','Reco_23'])
  arr = data['train_features']
  weights = data['train_weight']
  y=data['train_y']
  pred = model.predict_proba(arr)[:,1]
  postTrainingToolkit.ROC_AUC(score=pred,y=y,weight=weights,plot_path=out_path)

  data =  load_data(file_path=input_root_file,varlist=varlist,test_ratio=0.1,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43'])

  train_score = model.predict_proba(data['train_features'])[:,1]
  val_score = model.predict_proba(data['val_features'])[:,1]
  kolS, kolB = postTrainingToolkit.KS_test(train_score,val_score,data['train_weight'],data['val_weight'],data['train_y'],data['val_y'],plotPath=out_path)
  print(f'{kolS}, {kolB}')

  res_explain, res_masks = model.explain(data['train_features'])
  np.save(os.path.join(out_path,'explain.npy'), res_explain)
  np.save(os.path.join(out_path,'mask.npy'),res_masks)
  np.save(os.path.join(out_path,'y.npy'),data['train_y'])

def infer(input_root_file,input_model_path):
  import array, ROOT
  print(f'infering started for file {input_root_file}')
  model = TabNetClassifier()
  model.load_model(input_model_path)
  f = ROOT.TFile(input_root_file, "READ")
  dirName = f.GetListOfKeys()[0].ReadObj().GetName()
  trName = dirName + '/Result_Tree'
  print(trName)
  f.Close()
  if 'weight' in varlist:
    varlist.remove('weight')

  data =  load_data(file_path=input_root_file,varlist=varlist,test_ratio=0,val_ratio=0,sigTree=["Result_Tree"],bkgTree=[],dirName=dirName)
  arr = data['train_features']
  weights = data['train_weight']
  print(arr)
  pred = model.predict_proba(arr)[:,1]


  # Open the ROOT file
  file = ROOT.TFile(input_root_file, "UPDATE")
  file.cd(dirName)
  # Get the TTree from the file
  tree = file.Get(dirName+"/Result_Tree")

  # Create a new branch in the tree
  new_branch_name = "template_MVA_score"
  new_branch_value = array.array('d', [0.])
  new_branch = tree.Branch(new_branch_name, new_branch_value, new_branch_name+"/D")
  
  # Create a numpy array to be written to the branch
  array_to_write = pred

  # Loop over the entries in the tree and write the array to the branch
  print('infering done. start writing.....')
  for i in tqdm.tqdm(range(tree.GetEntries())):
      tree.GetEntry(i)
      new_branch_value[0] = array_to_write[i]
      new_branch.Fill()

  # Write the updated TTree to the file and close it
  tree.Write("", ROOT.TObject.kOverwrite)
  file.Close()
  


def infer_with_iter(input_folder,input_model_path):
  eras = ['2018','2017','2016preVFP','2016postVFP']
  chs = ['Mu','El','EE','MM','ME']
  for era in eras:
    for ch in chs:
      if not os.path.isdir(os.path.join(input_folder,era,ch,'RunResult')):
        continue
      systs=os.listdir(os.path.join(input_folder,era,ch,'RunResult'))
      #to select directory only
      systs=[f for f in systs if not '.' in f]
      for syst in systs:
       files = [os.path.join(input_folder,era,ch,'RunResult',syst,f) for f in os.listdir(os.path.join(input_folder,era,ch,'RunResult',syst))]
       for file in files:
         infer(file,input_model_path)
  
  
if __name__ == '__main__':

  


  # Define the available working modes
  MODES = ['train', 'plot', 'infer']

  # Create an argument parser
  parser = argparse.ArgumentParser(description='Select working mode')

  # Add an argument to select the working mode
  parser.add_argument('--working_mode', choices=MODES, help='select working mode')
  parser.add_argument('--input_root_file' ,dest='input_root_file', type=str,  default="")
  parser.add_argument('--input_model' ,dest='input_model', type=str,  default="")
  parser.add_argument('--out_path' ,dest='out_path', type=str,  default="")
  # Parse the arguments from the command line
  args = parser.parse_args()

  # Handle the selected working mode
  if args.working_mode == 'train':
      print('Training Mode')
      train(args.input_root_file,args.out_path)

  elif args.working_mode== 'plot':
      print('Plotting Mode')
      plot(args.input_root_file,args.input_model,args.out_path)
      # Add code for mode 2 here
  elif args.working_mode == 'infer':
      print('Inffering Mode')
      infer_with_iter(args.input_root_file,args.input_model)
      #infer(args.input_root_file,args.input_model)
      # Add code for mode 3 here
  else:
    print('Wrong working mode')

  
 
  

