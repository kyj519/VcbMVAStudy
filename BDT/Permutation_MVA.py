import argparse
from ROOT import TMVA, TString, TFile, TTree, TCut
import os
from parso import parse

from requests import delete

def Permutation_MVA(n_jet = 4, chk_pre_kin = False):
  print(chk_pre_kin, n_jet)
  TMVA.gConfig().GetVariablePlotting().fNbins1D = 500
  
  path_sample = os.environ["WtoCB_PATH"]
  path_dir = os.environ["DIR_PATH"]+"/result/BDT/"
  
  fin = TFile(path_sample+"Vcb_Mu_TTLJ_WtoCB_powheg_25.root", "READ")
  tree_correct = fin.Get("Permutation_Correct")
  tree_wrong = fin.Get("Permutation_Wrong")
  
  fout_name = ''
  if not chk_pre_kin: fout_name = f'{path_dir}Vcb_Permutation_TTLJ_WtoCB_{n_jet}.root'
  else: fout_name = f'{path_dir}Vcb_PreKin_TTLJ_WtoCB_{n_jet}.root'
  
  fout = TFile(fout_name, "RECREATE")
  factory = TMVA.Factory("TMVAClassification", fout,"!V:!Silent:Color:DrawProgressBar:Transformations=I:AnalysisType=Classification")
  data_loader = TMVA.DataLoader('dataset')
  
  data_loader.AddSpectator("n_jets","n_jets","units")
  data_loader.AddVariable("had_t_b_pt",   "had_t_b_pt",   "units")
  data_loader.AddVariable("w_u_pt",       "w_u_pt",       "units")
  data_loader.AddVariable("w_d_pt",       "w_d_pt",       "units")
  data_loader.AddVariable("lep_t_b_pt",   "lep_t_b_pt",   "units")
  data_loader.AddVariable("had_t_b_bscore",   "had_t_b_bscore",   "units")
  data_loader.AddVariable("lep_t_b_bscore",   "lep_t_b_bscore",   "units")
  data_loader.AddVariable("theta_w_u_w_d", "theta_w_u_w_d", "units")
  data_loader.AddVariable("theta_lep_neu", "theta_lep_neu", "units")
    
  data_loader.AddVariable("theta_lep_w_lep_t_b", "theta_lep_w_lep_t_b", "units")
  data_loader.AddVariable("del_phi_had_t_lep_t", "del_phi_had_t_lep_t", "units")

  data_loader.AddVariable("had_t_mass",   "had_t_mass",   "units")
  data_loader.AddVariable("had_w_mass",   "had_w_mass",   "units")
  data_loader.AddVariable("lep_t_mass",   "lep_t_mass",   "units")
  data_loader.AddVariable("lep_t_partial_mass",   "lep_t_partial_mass",   "units")
  
  if not chk_pre_kin:  data_loader.AddVariable("chi2",         "chi2",   "units")
  
  data_loader.AddSignalTree(tree_correct, 1.0)
  data_loader.AddBackgroundTree(tree_wrong, 1.0)
  
  cut_s = TCut(f'n_jets=={n_jet:.0f}')
  cut_b = TCut(f'n_jets=={n_jet:.0f}')
  n_train_signal = tree_correct.GetEntries(f'n_jets=={n_jet:.0f}')/2
  n_train_back = tree_wrong.GetEntries(f'n_jets=={n_jet:.0f}')/10
  
  data_loader.PrepareTrainingAndTestTree(cut_s, cut_b, f'nTrain_Signal={n_train_signal}:nTrain_Background={n_train_back}:nTest_Signal={n_train_signal}:nTest_Background={n_train_back}:SplitMode=Random:NormMode=NumEvents:V')
  
  factory.BookMethod(data_loader, TMVA.Types.kBDT, "BDTG",
		      "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=100:MaxDepth=2" )
  
  factory.TrainAllMethods()
  factory.TestAllMethods()
  factory.EvaluateAllMethods()
  fout.Close()
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--n_jet', type=int)
  parser.add_argument('--pre_kin', type=int)
  args = parser.parse_args()
  n_jet = args.n_jet
  pre_kin = True if args.pre_kin else False
  Permutation_MVA(n_jet=n_jet,chk_pre_kin=pre_kin)
