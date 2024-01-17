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



varlist = ["met_pt", "neutrino_p", "lepton_pt", "pt_ratio",
    "pt_had_t_b", "pt_w_u", "pt_w_d", "pt_lep_t_b",
    "bvsc_had_t_b", "cvsb_had_t_b", "cvsl_had_t_b",     "bvsc_lep_t_b", "cvsb_lep_t_b", "cvsl_lep_t_b",
    "theta_w_u_w_d", "theta_lep_neu", "theta_lep_w_lep_t_b", "del_phi_had_t_lep_t", 
    "had_t_mass", "had_w_mass", "lep_t_mass","lep_t_partial_mass",
    "chi2_jet_had_t_b","chi2_jet_w_u","chi2_jet_w_d","chi2_jet_lep_t_b",
    "chi2_constraint_had_t", "chi2_constraint_had_w",
    "chi2_constraint_lep_t", "chi2_constraint_lep_w"]


def train(model_save_path, n_jets):
  n_jets = int(n_jets) + 4
  if n_jets > 4:
    varlist.extend(['chi2_jet_extra'])
  model_save_path = model_save_path+f'_{n_jets}jets'
  if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)
  input_tuple=( #first element of tuple = signal tree, second =bkg tree.
    [('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/Mu/RunPermutationTree/Vcb_TTLJ_WtoCB_powheg.root','Permutation_Correct',f'n_jets=={n_jets}'),
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/El/RunPermutationTree/Vcb_TTLJ_WtoCB_powheg.root','Permutation_Correct',f'n_jets=={n_jets}')
     ], ##TTLJ_WtoCB Reco 1, (file_path, tree_path, filterstr)
    
    [('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/Mu/RunPermutationTree/Vcb_TTLJ_WtoCB_powheg.root','Permutation_Wrong',f'n_jets=={n_jets}'),
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/El/RunPermutationTree/Vcb_TTLJ_WtoCB_powheg.root','Permutation_Wrong',f'n_jets=={n_jets}')
     ] ##TTLJ_WtoCB cs decay
  )
  data_info = {'tree_path_filter_str':input_tuple, 'varlist':varlist, 'test_ratio':0.1,'val_ratio':0.2}
  data =  load_data(**data_info)

  model_info = {
    'n_d':16,
    'n_a':16,
    'verbose':1,
    'cat_idxs':data['cat_idxs'],
    'cat_dims':data['cat_dims'],
    'cat_emb_dim':3,
    'n_steps':5
    }
  clf = TabNetClassifier(**model_info)  


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

  train_info={
    'X_train':data['train_features'],
    'y_train':data['train_y'],
    'eval_set':[(data['val_features'], data['val_y'])],
    'eval_metric':['balanced_accuracy','WeightedMSE','auc'],
    'max_epochs':1000,
    'num_workers':12,
    'weights':1,
    'batch_size':int(2097152/8),
    'virtual_batch_size':512,
    #augmentations=aug,
    'patience':20
    #callbacks=[pytorch_tabnet.callbacks.History(clf,verbose=1)]
  }
  clf.fit(
    **train_info
  )

  
  
  clf.save_model(os.path.join(model_save_path,'model.zip'))
  del train_info['X_train']
  del train_info['y_train']
  del train_info['eval_set']
  
  info_arr={'train_info':train_info, 'model_info':model_info, 'data_info':data_info}
  info_arr=np.array(info_arr)
  np.save(os.path.join(model_save_path,'info.npy'),info_arr)
  
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
    pred = model.predict_proba(arr)[:,1]
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
    
    #Open the ROOT file
    file = ROOT.TFile(input_root_file, "UPDATE")
    
    #Get the TTree from the file
    tree = file.Get(dirName+"/Result_Tree")

    
    #Create a new branch in the tree
    new_branch_name = "template_score"
    new_branch_value = array.array('f', [0.])
    
    # if tree.GetListOfBranches().Contains(new_branch_name):
    #   b1 = tree.GetBranch(new_branch_name)
    #   tree.GetListOfBranches().Remove(b1)

    # new_branch = tree.Branch(new_branch_name, new_branch_value, new_branch_name+"/F")
    
    # #Create a numpy array to be written to the branch
    # array_to_write = pred
    # from matplotlib import pyplot as plt
    # outname = input_root_file.split('/')
    # outname[-1] = outname[-1].replace('.root','')
    # outname = '_'.join(outname[-5:])
    # plt.hist(pred)
    # plt.savefig('/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_log'+'/'+outname+'.png')

    # #Loop over the entries in the tree and write the array to the branch
    # print('infering done. start writing.....')

    # for i in tqdm.tqdm(range(tree.GetEntries())):
    #   tree.GetEntry(i)
    #   new_branch_value[0] = float(array_to_write[i])
    #   new_branch.Fill()

    # #Write the updated TTree to the file and close it
    # tree.Write("", ROOT.TObject.kOverwrite)

    
    file.Close()
    del data
    data = ROOT.RDataFrame(trName,input_root_file)
    data = data.AsNumpy()
    data['Template_Score'] = pred
    #for key, item in data.items():
    #  if data[key].dtype == 'object':
    #    data[key] = data[key].astype(bool)
        


    
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
  chs = ['Mu' ,'El']
  for era in eras:
    for ch in chs:
      if not os.path.isdir(os.path.join(input_folder,era,ch,'RunResult')):
        continue
      systs=os.listdir(os.path.join(input_folder,era,ch,'RunResult'))
      #to select directory only
      systs=[f for f in systs if not '.' in f]
      #systs=['Central_Syst']
      for syst in systs:
       files = [os.path.join(input_folder,era,ch,'RunResult',syst,f) for f in os.listdir(os.path.join(input_folder,era,ch,'RunResult',syst))]
       for file in files:
        #infer(file,input_model_path)
        outname = file.split('/')
        outname[-1] = outname[-1].replace('.root','')
        outname = '_'.join(outname[-5:])
        if not (syst == 'Central_Syst' and 'WtoCB' in file):
          continue
        job = htcondor.Submit({
            "universe":"vanilla",
            "getenv": True,
            "jobbatchname": f"Vcb_infer_{outname}",
            "executable": "/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_write.sh",
            "arguments": f"{input_model_path} {file}",
            "output": f"/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_log/{outname}.out",
            "error": f"/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_log/{outname}.err",
            "request_memory": "64GB" if 'TTLJ' in file else "16GB",
            "request_gpus": 0 if 'TTLJ' in file else 0,
            "request_cpus": 16 if 'TTLJ' in file else 4,
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
  parser.add_argument('--n_jets', dest='n_jets', type=str, default=4)
  # Parse the arguments from the command line
  args = parser.parse_args()

  # Handle the selected working mode
  if args.working_mode == 'train':
    print('Training Mode')
    train(args.out_path, args.n_jets)

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

  
 
  

