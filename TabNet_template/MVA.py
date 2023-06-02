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
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight','pt_w_u','pt_w_d','eta_w_u','eta_w_d','best_mva_score']

#add mva input
varlist.extend([])
def train( model_save_path, doSmote=False):
  if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)
  input_tuple=( #first element of tuple = signal tree, second =bkg tree.
    [('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/Mu/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','POGTightWithTightIso_Central/Result_Tree','chk_reco_correct==1'),
     
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/El/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','passTightID_Central/Result_Tree','chk_reco_correct==1')
     ], ##TTLJ_WtoCB Reco 1, (file_path, tree_path, filterstr)
    
    [
      ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/Mu/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','POGTightWithTightIso_Central/Result_Tree','chk_reco_correct==0'), ##TTLJ_WtoCB Reco 0
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/Mu/RunResult/Central_Syst/Vcb_TTLJ_powheg.root','POGTightWithTightIso_Central/Result_Tree','decay_mode==43'),
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/El/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','passTightID_Central/Result_Tree','chk_reco_correct==0'), ##TTLJ_WtoCB Reco 0
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/El/RunResult/Central_Syst/Vcb_TTLJ_powheg.root','passTightID_Central/Result_Tree','decay_mode==43')
     ] ##TTLJ_WtoCB cs decay
    
  )
  data =  load_data(tree_path_filter_str=input_tuple,varlist=varlist,test_ratio=0.1,val_ratio=0.2)
  sm = SMOTENC(random_state=42, categorical_features=data["cat_idxs"])
  clf = TabNetClassifier(
    n_d=32,
    n_a=32,
    verbose=1,
    cat_idxs=data['cat_idxs'],
    cat_dims=data['cat_dims'],
    cat_emb_dim=3,
    n_steps=5
    )  


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
    weights=0,
    weights_for_loss=data['train_sample_and_class_weight']/np.mean(data['train_sample_and_class_weight']),
    batch_size=int(2097152/4),
    virtual_batch_size=512,
    #augmentations=aug,
    patience=50
    #callbacks=[pytorch_tabnet.callbacks.History(clf,verbose=1)]
  )
  
  clf.save_model(os.path.join(model_save_path,'model.zip'))
  
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
  import array, ROOT, uproot
  ROOT.EnableImplicitMT()
  print(f'infering started for file {input_root_file}')
  model = TabNetClassifier()
  model.load_model(input_model_path)
  outname = input_root_file.split('/')
  outname[-1] = outname[-1].replace('.root','')
  outname = '_'.join(outname[-5:])
  try:
    f = ROOT.TFile(input_root_file, "UPDATE")
    dirName = f.GetListOfKeys()[0].ReadObj().GetName()
    trName = dirName + '/Result_Tree'
    print(trName)
    f.Close() 

    input_tuple=([(input_root_file,trName,'')],[])
    data =  load_data(tree_path_filter_str=input_tuple,varlist=varlist,test_ratio=0,val_ratio=0)
    print("Data loaded")
    arr = data['train_features']
    weights = data['train_weight']
    pred = deepcopy(model.predict_proba(arr)[:,1])
    # print(pred.shape,"pred shape")
    # print(pred,"pred")
    # print(pred.dtype())
    
    # from matplotlib import pyplot as plt
    # #plt.hist(pred,bins=40)
    # #plt.savefig(input_root_file.replace('.root','.png'))
    # opts = ROOT.RDF.RSnapshotOptions()
    # opts.fMode = "update"
    # df = ROOT.RDF.MakeNumpyDataFrame({"template_MVA_score": pred})
    # df = df.Redefine("template_MVA_score","(double)template_MVA_score")
    #df.Snapshot(trName,input_root_file,"",opts)
    
    # #Open the ROOT file
    # file = ROOT.TFile(input_root_file, "UPDATE")
    # file.cd(dirName)
    # #Get the TTree from the file
    # tree = file.Get(dirName+"/Result_Tree")

    
    # #Create a new branch in the tree
    # new_branch_name = "template_MVA_score"
    # new_branch_value = array.array('f', [0.])
    
    # if tree.GetListOfBranches().Contains(new_branch_name):
    #   new_branch = tree.GetBranch(new_branch_name)
    # else:
    #   new_branch = tree.Branch(new_branch_name, new_branch_value, new_branch_name+"/F")
    
    # #Create a numpy array to be written to the branch
    # array_to_write = pred

    # #Loop over the entries in the tree and write the array to the branch
    # print('infering done. start writing.....')

    # for i in tqdm.tqdm(range(tree.GetEntries())):
    #   tree.GetEntry(i)
    #   new_branch_value[0] = float(array_to_write[i])
    #   new_branch.Fill()

    # #Write the updated TTree to the file and close it
    # tree.Write("", ROOT.TObject.kOverwrite)

    
    # file.Close()
    import time
    start = time.time()
    file = uproot.update(input_root_file)
    file[trName].extend({"template_MVA_score_uproot": pred})
    end = time.time()
    print(f"{end - start:.5f} sec")
    
  except Exception as e:
    import fcntl
    print(e)
    file_path = "/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_log/error_list"
    with open(file_path, 'a') as file:
      fcntl.flock(file, fcntl.LOCK_EX)
      file.write(outname+"\n")
      fcntl.flock(file, fcntl.LOCK_UN)
    


def infer_with_iter(input_folder,input_model_path):
  import htcondor, shutil
  if os.path.isdir("/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_log/"):
    shutil.rmtree("/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_log/")
  os.makedirs("/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_log/")
  eras = ['2017'] #'2016preVFP','2016postVFP']
  chs = ['Mu'] #,'El']
  for era in eras:
    for ch in chs:
      if not os.path.isdir(os.path.join(input_folder,era,ch,'RunResult')):
        continue
      systs=os.listdir(os.path.join(input_folder,era,ch,'RunResult'))
      #to select directory only
      #systs=[f for f in systs if not '.' in f]o
      systs=['Central_Syst']
      for syst in systs:
       files = [os.path.join(input_folder,era,ch,'RunResult',syst,f) for f in os.listdir(os.path.join(input_folder,era,ch,'RunResult',syst))]
       for file in files:
        #infer(file,input_model_path)
        outname = file.split('/')
        outname[-1] = outname[-1].replace('.root','')
        outname = '_'.join(outname[-5:])
        job = htcondor.Submit({
            "universe":"vanilla",
            "getenv": True,
            "jobbatchname": f"Vcb_infer",
            "executable": "/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_write.sh",
            "arguments": f"{input_model_path} {file}",
            "output": f"/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_log/{outname}.out",
            "error": f"/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_log/{outname}.err",
            "request_memory": "32GB" if 'TTLJ' in file else "8GB",
            "request_gpus": 1 if 'TTLJ' in file else 0,
            "should_transfer_files":"YES",
            "when_to_transfer_output" : "ON_EXIT",
        })
        
        schedd = htcondor.Schedd()
        with schedd.transaction() as txn:
          cluster_id = job.queue(txn)
        print("Job submitted with cluster ID:", cluster_id)
        
  
  
if __name__ == '__main__':

  


  # Define the available working modes
  MODES = ['train', 'plot','infer_iter','infer']

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
    train(args.out_path)

  elif args.working_mode== 'plot':
    print('Plotting Mode')
    plot(args.input_root_file,args.input_model,args.out_path)
    # Add code for mode 2 here
  elif args.working_mode == 'infer_iter':
    print('Inffering Mode (all file iteration)')
    infer_with_iter(args.input_root_file,args.input_model)
    #infer(args.input_root_file,args.input_model)
    # Add code for mode 3 here
  elif args.working_mode == 'infer':
    print('infering and writing')
    infer(args.input_root_file,args.input_model)
  else:
    print('Wrong working mode')

  
 
  

