import os, sys, argparse, shutil
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from imblearn.over_sampling import SMOTENC
import torch
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.metrics import Metric
import tqdm
import hashlib
import ROOT
sys.path.append(os.environ["DIR_PATH"])
sys.path.append(os.environ["DIR_PATH"]+"/Class-balanced-loss-pytorch")
from root_data_loader import load_data, classWtoSampleW
from class_balanced_loss import CB_loss
killist = [] #cleanup list for temp files
def cleanup_temp_files():
  for file in killist:
    if os.path.exists(file):
      os.remove(file)

def generate_random_file_name(input_string):
    # Generate a hash value from the input string
    hash_value = hashlib.sha256(input_string.encode()).hexdigest()

    # Take the first 8 characters from the hash value
    random_string = hash_value

    # Combine the random string and number to form the file name
    file_name = random_string

    return file_name

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device is {device}, count is {torch.cuda.device_count()}')



varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','pt_w_u','pt_w_d','weight']
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
#varlist = ['cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
# varlist.extend(['n_jets',
#                 'pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b',
#                 'eta_had_t_b','eta_w_u','eta_w_d','eta_lep_t_b',
#                 'bvsc_lep_t_b','bvsc_had_t_b'])
varlist = ['bvsc_w_d','cvsl_w_u','cvsb_w_u','cvsb_w_d','n_bjets','pt_had_t_b','pt_w_d','bvsc_had_t_b','weight']
  
#KPS modification
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight','pt_w_u','pt_w_d','eta_w_u','eta_w_d','best_mva_score','m_had_t','m_had_w']
#varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','n_bjets','n_cjets','weight','pt_w_u','pt_w_d','eta_w_u','eta_w_d','best_mva_score','m_had_t','m_had_w']
#
# varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d',
#            'cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight'
#            ,'m_had_t','m_had_w','best_mva_score']   
# #add mva input
# varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d',
#            'cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight'
#            ,'m_had_t','m_had_w','best_mva_score',
#            'met_pt','lepton_pt','pt_had_t_b','pt_w_u','pt_w_d','pt_lep_t_b',
#            'bvsc_had_t_b','cvsb_had_t_b','cvsl_had_t_b',
#            'bvsc_lep_t_b','cvsb_lep_t_b','cvsl_lep_t_b',]  
#varlist.extend([])



  
def train(model_save_path, floss_gamma, doSmote=False):
  if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)
  input_tuple=( #first element of tuple = signal tree, second =bkg tree.
    [('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2016preVFP/Mu/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==1', '(0.3041 / 8850014.0)'),
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2016preVFP/El/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==1', '(0.3041 / 8850014.0)'),
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2016postVFP/Mu/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==1', '(0.3041 / 8850014.0)'),
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2016postVFP/El/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==1', '(0.3041 / 8850014.0)'),
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/Mu/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==1', '(0.3041 / 8850014.0)'),
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/El/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==1', '(0.3041 / 8850014.0)'),
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2018/Mu/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==1', '(0.3041 / 8850014.0)'),
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2018/El/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==1', '(0.3041 / 8850014.0)'),
     ], ##TTLJ_WtoCB Reco 1, (file_path, tree_path, filterstr)
    
    [
      ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2016preVFP/El/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==0', '(0.3041 / 8850014.0)'), ##TTLJ_WtoCB Reco 0
      ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2016preVFP/El/RunResult/Central_Syst/Vcb_TTLJ_powheg.root','Central/Result_Tree','decay_mode==43', '(365.34 / 352462632.0)'),
      ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2016preVFP/Mu/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==0', '(0.3041 / 8850014.0)'), ##TTLJ_WtoCB Reco 0
      ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2016preVFP/Mu/RunResult/Central_Syst/Vcb_TTLJ_powheg.root','Central/Result_Tree','decay_mode==43', '(365.34 / 352462632.0)'),
       ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2016postVFP/El/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==0', '(0.3041 / 8850014.0)'), ##TTLJ_WtoCB Reco 0
      ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2016postVFP/El/RunResult/Central_Syst/Vcb_TTLJ_powheg.root','Central/Result_Tree','decay_mode==43', '(365.34 / 352462632.0)'),
      ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2016postVFP/Mu/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==0', '(0.3041 / 8850014.0)'), ##TTLJ_WtoCB Reco 0
      ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2016postVFP/Mu/RunResult/Central_Syst/Vcb_TTLJ_powheg.root','Central/Result_Tree','decay_mode==43', '(365.34 / 352462632.0)'),
           ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/El/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==0', '(0.3041 / 8850014.0)'), ##TTLJ_WtoCB Reco 0
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/El/RunResult/Central_Syst/Vcb_TTLJ_powheg.root','Central/Result_Tree','decay_mode==43', '(365.34 / 352462632.0)'),
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/Mu/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==0', '(0.3041 / 8850014.0)'), ##TTLJ_WtoCB Reco 0
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/Mu/RunResult/Central_Syst/Vcb_TTLJ_powheg.root','Central/Result_Tree','decay_mode==43', '(365.34 / 352462632.0)'),
           ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2018/El/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==0', '(0.3041 / 8850014.0)'), ##TTLJ_WtoCB Reco 0
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2018/El/RunResult/Central_Syst/Vcb_TTLJ_powheg.root','Central/Result_Tree','decay_mode==43', '(365.34 / 352462632.0)'),
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2018/Mu/RunResult/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root','Central/Result_Tree','chk_reco_correct==0', '(0.3041 / 8850014.0)'), ##TTLJ_WtoCB Reco 0
     ('/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2018/Mu/RunResult/Central_Syst/Vcb_TTLJ_powheg.root','Central/Result_Tree','decay_mode==43', '(365.34 / 352462632.0)'),
     #('/data6/Users/yeonjoon/Vcb_Post_Analysis/Sample/2017/Mu/RunResult/Central_Syst/Vcb_ttHTobb.root','Central/Result_Tree',''),
     #('/data6/Users/yeonjoon/Vcb_Post_Analysis/Sample/2017/El/RunResult/Central_Syst/Vcb_ttHTobb.root','Central/Result_Tree',''),
     #('/data6/Users/yeonjoon/Vcb_Post_Analysis/Sample/2017/Mu/RunResult/Central_Syst/Vcb_ttZToQQ.root','Central/Result_Tree',''),
     #('/data6/Users/yeonjoon/Vcb_Post_Analysis/Sample/2017/El/RunResult/Central_Syst/Vcb_ttZToQQ.root','Central/Result_Tree',''),
     ] ##TTLJ_WtoCB cs decay
    
  )
  data_info = {'tree_path_filter_str':input_tuple, 'varlist':varlist, 'test_ratio':0.1,'val_ratio':0.2}
  data =  load_data(**data_info)
  np.savez(os.path.join(model_save_path,'data.npz'),data)

  sm = SMOTENC(random_state=42, categorical_features=data["cat_idxs"])
  model_info = {
    'n_d':32,
    'n_a':32,
    'verbose':1,
    'cat_idxs':data['cat_idxs'],
    'cat_dims':data['cat_dims'],
    'cat_emb_dim':1,
    'n_steps':3
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
      
  gamma = floss_gamma
  def focal_loss(y_pred,y_true,y_weight=None, force_use_cpu=False):
    #y_pred = torch.nn.Softmax(dim=-1)(y_pred) 
    cb_loss = CB_loss(y_true, y_pred, [count_0,count_1] ,no_of_classes=2, gamma=gamma, evt_weight=y_weight, force_use_cpu=force_use_cpu)
    return cb_loss
  
  class focal_loss_metric(Metric):
    def __init__(self):
      self._name = "focal_loss_metric"
      self._maximize = False

    def __call__(self, y_true, y_score):
      weight = []
      mse = focal_loss(torch.tensor(y_score).float(),torch.tensor(y_true).to(device), force_use_cpu=True)
      
      return mse.detach().numpy().tolist()

      
  
  if doSmote:
    data['train_features'], data['train_y'] = sm.fit_resample(data['train_features'], data['train_y'])
  print(f"Total number of training data = {data['train_y'].shape}")
  
  print(f"sumW of each class is {data['sumW']}, count is {data['count']}")
  count_0 = data['sumW'][0]
  count_1 = data['sumW'][1]
  print(f'{count_0} of bkg. sample, {count_1} of sig.sample')
  
  beta_0 = (count_0-1)/(count_0)
  beta_1 = (count_1-1)/(count_1)
  print(f'value of beta_0 is {beta_0}, beta_1 is {beta_1}')
  w_b = (1-beta_0)/(1-pow(beta_0,count_0))
  w_s = (1-beta_1)/(1-pow(beta_1,count_1))
  
  print(f'w_b is {w_b/(w_b+w_s)*2} w_s is {w_s/(w_b+w_s)*2}')
  

    
    
  train_info={
    'X_train':data['train_features'],
    'y_train':data['train_y'],
    'w_train':data['train_weight'],
    'eval_set':[(data['val_features'], data['val_y'])],
    'eval_metric':['balanced_accuracy','WeightedMSE','auc','focal_loss_metric'],
    'max_epochs':1000,
    'num_workers':4,
    #'weights':1,
    #'weights':data['train_weight'],#data['train_sample_and_class_weight'],
    'weights':0,
    'batch_size':8192*8,#int(2097152/16),#1024,,#8192,#int(2097152/16),
    'virtual_batch_size':256,
    #augmentations=aug,
    'patience':100,
    'loss_fn':focal_loss
    #callbacks=[pytorch_tabnet.callbacks.History(clf,verbose=1)]
  }
  clf.fit(
    **train_info
  )

  



  clf.save_model(os.path.join(model_save_path,'model.zip'))
  del train_info['X_train']
  del train_info['y_train']
  del train_info['eval_set']
  train_info.pop('w_train', None)
  train_info.pop('loss_fn', None)
  
  
  file_name = os.path.join(model_save_path,'info.txt')
  # Open the file for writing
  with open(file_name, 'w') as file:
    file.write(f"Training info\n")
    for key, value in train_info.items():
        file.write(f"{key}: {value}\n")
    file.write(f"gamma: {gamma}\n")
    file.write(f"###########################################################\n")
    file.write(f"model info\n")
    for key, value in model_info.items():
        file.write(f"{key}: {value}\n")
        
    file.write(f"###########################################################\n")
    file.write(f"data info\n")
    for key, value in data_info.items():
        file.write(f"{key}: {value}\n")

  # Close the file
  file.close()
 
  info_arr={'train_info':train_info, 'model_info':model_info, 'data_info':data_info}
  info_arr=np.array(info_arr)
  np.save(os.path.join(model_save_path,'info.npy'),info_arr)
  
  plot(model_save_path)
  
  
def plot(model_save_path):
  import postTrainingToolkit
  files = os.listdir(model_save_path)
  # Filter for files with a .pt.zip extension
  pt_zip_files = [f for f in files if f.endswith('.zip')]
  model = TabNetClassifier()
  model.load_model(os.path.join(model_save_path,pt_zip_files[0]))         
  import postTrainingToolkit
  data = np.load(os.path.join(model_save_path,'data.npz'),allow_pickle=True)
  data = data['arr_0'][()]

  postTrainingToolkit.ROC_AUC(score=model.predict_proba(data['test_features'])[:,1],y=data['test_y'],plot_path=model_save_path)


  train_score = model.predict_proba(data['train_features'])[:,1]
  val_score = model.predict_proba(data['val_features'])[:,1]

  kolS, kolB = postTrainingToolkit.KS_test(train_score=train_score,val_score=val_score,train_w=data['train_weight'],val_w=data['val_weight'],train_y=data['train_y'],val_y=data['val_y'],plotPath=model_save_path)
  print(f'{kolS}, {kolB}')


  res_explain, res_masks = model.explain(data['test_features'])
  np.save(os.path.join(model_save_path,'explain.npy'), res_explain)
  np.save(os.path.join(model_save_path,'mask.npy'),res_masks)
  np.save(os.path.join(model_save_path,'y.npy'),data['train_y'])
  feature_importances_ = model._compute_feature_importances(data['test_features'])
  print('Feature Importaces:\n')
  print(feature_importances_)
  
def infer_and_write(root_file,model,varlist,new_branch_name,model_folder):
    import array, tqdm
    
    data_info = np.load(os.path.join(model_folder,'info.npy'),allow_pickle=True)
    data_info = data_info[()]
    data_info = data_info['data_info']
    varlist = data_info['varlist']
    
    input_tuple=([(root_file,'Result_Tree','')],[])
    if 'weight' in varlist:
      varlist.remove('weight')

    data =  load_data(tree_path_filter_str=input_tuple,varlist=varlist,test_ratio=0,val_ratio=0)
    print("Data loaded")
    arr = data['train_features']

    pred = model.predict_proba(arr)[:,1]
    print('infer is done. start writing...')
    
    new_branch_name = "template_score"
    root_file = ROOT.TFile.Open(root_file,'UPDATE')


        
    ##########################################################
    ############## classical iterator
    ########################################################## 
    tree = root_file.Get("Result_Tree")
    


    
    new_branch_value = array.array('f', [0.])
    new_branch = tree.Branch(new_branch_name, new_branch_value, new_branch_name+"/F")
    for i in range(tree.GetEntries()):
        new_branch_value[0] = float(pred[i])
        new_branch.Fill()
        
    tree.Write("",ROOT.TObject.kOverwrite)
    root_file.Close()
    
def infer(input_root_file,input_model_path,branch_name='template_score'):
  import array,shutil,time
  start_time = time.time()

  print(input_model_path)
  

  
  model_folder = '/'.join(input_model_path.split('/')[:-1])
  #ROOT.EnableImplicitMT() ImplicitMT should not be used
  model = TabNetClassifier()
  model.load_model(input_model_path)
  outname = input_root_file.split('/')
  outname[-1] = outname[-1].replace('.root','')
  outname = '_'.join(outname[-5:])
  
  try: 
    
    
    new_branch_name=branch_name
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    
    input_file = ROOT.TFile.Open(input_root_file,'READ')
    output_files = []
    
    for key in input_file.GetListOfKeys():
      obj = key.ReadObj()
      if isinstance(obj, ROOT.TDirectory):
        # Create an output file for each TDirectory
        output_file_name = input_root_file.replace('.root',f'_{obj.GetName()}.root')
        output_file = ROOT.TFile.Open(output_file_name, "RECREATE")
        output_files.append(output_file_name)

        # Enter the TDirectory and copy its contents to the new file
        
        input_file.cd(obj.GetName())
        for inner_key in ROOT.gDirectory.GetListOfKeys():
          inner_obj = inner_key.ReadObj()
          output_file.cd()
          if isinstance(inner_obj, ROOT.TTree):
              # Clone the tree and write it to the output file
              if hasattr(inner_obj,new_branch_name):
                inner_obj.SetBranchStatus(new_branch_name,0)
              cloned_tree = inner_obj.CloneTree(-1, "fast")
              cloned_tree.Write() 

          else:
              # For other objects (e.g., histograms), simply write them
              inner_obj.SetName(inner_obj.GetName().split('/')[-1]) 
              inner_obj.Write()
              
        output_file.Close()
            
    
    ####now start multiprocess to perform a infer for each seprated files
    procs = []

    for i, file in enumerate(output_files):
      p = mp.Process(target=infer_and_write, args=(file,model,varlist,new_branch_name,model_folder))
      p.start()
      procs.append(p)
    
        
    for p in procs:
      p.join()
    
    ####start to merge into one file again
    merged_file = ROOT.TFile.Open(input_root_file.replace('.root','_merged.root'), "RECREATE")

    for file in output_files:
      dirname = file.split('_')[-1].replace('.root','')
      this_dir = merged_file.mkdir(dirname)
      file = ROOT.TFile.Open(file,'READ')
      

      file.cd()
      for inner_key in ROOT.gDirectory.GetListOfKeys():
          inner_obj = inner_key.ReadObj()
          this_dir.cd()
          if isinstance(inner_obj, ROOT.TTree):
      # Clone the tree and write it to the output file
              inner_obj.SetBranchStatus("*",1)
              cloned_tree = inner_obj.CloneTree(-1, "fast")
              cloned_tree.Write()
          else:
              # For other objects (e.g., histograms), simply write them
              inner_obj.Write()
      file.Close()
  
    merged_file.Close()
    
    
    ##clean the residuals created during task
    for file in output_files:
        os.remove(file)
    os.remove(input_root_file)
    shutil.move(input_root_file.replace('.root','_merged.root'),input_root_file)
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Elapsed Time: {elapsed_time} seconds")


    
  except Exception as e:
    import fcntl
    print(e)
    file_path = "/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_log_2/error_list"
    with open(file_path, 'a') as file:
      fcntl.flock(file, fcntl.LOCK_EX)
      file.write(outname+"\n")
      fcntl.flock(file, fcntl.LOCK_UN)
    


def infer_with_iter(input_folder,input_model_path,branch_name):
  import htcondor, shutil, ROOT
  log_path = os.path.join(os.environ['DIR_PATH'],'TabNet_template','infer_log')
  if os.path.isdir(log_path):
    shutil.rmtree(log_path)
  os.makedirs(log_path)
  eras = ['2016preVFP'] #'2016preVFP','2016postVFP']
  chs = ['Mu','El']
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

         
          # ##########
          # ######clear residuals
          # ##########
          
          if 'temp' in file or 'update' in file:
            os.remove(file)
            continue
        
        #infer(file,input_model_path)
    
          outname = file.split('/')
          outname[-1] = outname[-1].replace('.root','')
          outname = '_'.join(outname[-5:])
          job = htcondor.Submit({
              "universe":"vanilla",
              "getenv": True,
              "jobbatchname": f"Vcb_infer_{input_model_path}",
              "executable": "/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_write.sh",
              "arguments": f"{input_model_path} {file} {branch_name}",
              "output": os.path.join(log_path,f"{outname}.out"),
              "error": os.path.join(log_path,f"{outname}.err"),
              "log": os.path.join(log_path,f"{outname}.log"),
              "request_memory": "128GB" if ('TTLJ_powheg' in outname or 'TTLL_powheg' in outname) and 'Central' in outname else "16GB",
              "request_gpus": 0 if ('TTLJ_powheg' in outname and 'Central' in outname) else 0,
              "request_cpus": 8 if 'TT' in file else 1,
              "should_transfer_files":"YES",
              "when_to_transfer_output" : "ON_EXIT",
          })
          
          schedd = htcondor.Schedd()
          with schedd.transaction() as txn:
            cluster_id = job.queue(txn)
          print("Job submitted with cluster ID:", cluster_id)

def train_submit(model_folder,floss_gamma): 
  import htcondor
  if os.path.isdir(model_folder):
    user_input = input(f'Folder {model_folder} already exist. do you want to proceed? [y/n]')

    if user_input.lower() == 'y':
      shutil.rmtree(model_folder)
    elif user_input.lower() == 'n':
      print('Job aborted by user')
      exit()
    else:
      print('Type y or n')
      
    
  os.makedirs(model_folder)

  job = htcondor.Submit({
      "universe":"vanilla",
      "getenv": True,
      "jobbatchname": f"Vcb_Train",
      "executable": "MVA.sh",
      "arguments": f"{model_folder} {floss_gamma}",
      "output": f"{model_folder}/job.out",
      "error": f"{model_folder}/job.err",
      "log": f"{model_folder}/job.log",
      "request_memory": "200GB",
      "request_gpus": 1,
      "request_cpus": 8,
      "should_transfer_files":"YES",
      "when_to_transfer_output" : "ON_EXIT",
  })
          
  schedd = htcondor.Schedd()
  with schedd.transaction() as txn:
    cluster_id = job.queue(txn)
  print("Job submitted with cluster ID:", cluster_id)
        
  
  
if __name__ == '__main__':

  


  # Define the available working modes
  MODES = ['train', 'train_submit' ,'plot','infer_iter','infer']

  # Create an argument parser
  parser = argparse.ArgumentParser(description='Select working mode')

  # Add an argument to select the working mode
  parser.add_argument('--working_mode', choices=MODES, help='select working mode')
  parser.add_argument('--input_root_file' ,dest='input_root_file', type=str,  default="", help="target inferring folder")
  parser.add_argument('--input_model' ,dest='input_model', type=str,  default="", help = "input model to infer")
  parser.add_argument('--out_path' ,dest='out_path', type=str,  help="training model output path")
  parser.add_argument('--branch_name',dest='branch_name', type=str,  help="inferring branch name")
  parser.add_argument('--floss_gamma',dest='floss_gamma', type=float,  default=2.0 ,help="focal loss gamm")
  
  # Parse the arguments from the command line
  args = parser.parse_args()

  # Handle the selected working mode
  if args.working_mode == 'train':
    print('Training Mode')
    train(args.out_path, args.floss_gamma)

  elif args.working_mode == 'train_submit':
    train_submit(args.out_path,args.floss_gamma)

  elif args.working_mode== 'plot':
    print('Plotting Mode')
    plot(args.input_root_file,args.input_model,args.out_path)
    # Add code for mode 2 here
  elif args.working_mode == 'infer_iter':
    print('Inffering Mode (all file iteration)')
    infer_with_iter(args.input_root_file,args.input_model,args.branch_name)
    #infer(args.input_root_file,args.input_model)
    # Add code for mode 3 here
  elif args.working_mode == 'infer':
    print('infering and writing')
    infer(args.input_root_file,args.input_model,args.branch_name)
  else:
    print('Wrong working mode')

  
 
  

