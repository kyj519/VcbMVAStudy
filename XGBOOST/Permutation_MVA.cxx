void Permutation_MVA(int n_jet=4, bool chk_pre_kin=false)
{
  cout << chk_pre_kin << " " << n_jet << endl;
  
  //TMVA::gConfig().GetVariablePlotting().fNbinsMVAoutput = 200;
  TMVA::gConfig().GetVariablePlotting().fNbins1D = 500; 
  
  const TString path = getenv("WtoCB_PATH");
  
  TFile* fin;
  fin = TFile::Open(path+"Vcb_Mu_TTLJ_WtoCB_powheg_25.root");
  
  TTree* tree_correct = (TTree*)fin->Get("Permutation_Correct");
  TTree* tree_wrong = (TTree*)fin->Get("Permutation_Wrong");

  TString fout_name;
  if(!chk_pre_kin) fout_name = Form("Vcb_Permutation_TTLJ_WtoCB_%d.root", n_jet);
  else fout_name = Form("Vcb_PreKin_TTLJ_WtoCB_%d.root", n_jet);

  TFile* fout = TFile::Open(fout_name, "RECREATE");

  TMVA::Factory* factory = new TMVA::Factory("TMVAClassification", fout,
                                             "!V:!Silent:Color:DrawProgressBar:Transformations=I:AnalysisType=Classification");
  
  //TString name_data_loader;
  //if(!chk_pre_kin) name_data_loader = Form("%dJets", n_jet);
  //else name_data_loader = Form("Pre_Kin_%dJets", n_jet);
  
  TMVA::DataLoader* data_loader = new TMVA::DataLoader("dataset");
  //data_loader->AddVariable( "weight",       "weight",       "units", 'F');
  //data_loader->AddVariable( "n_jets",       "n_jets",       "units", 'I');
  data_loader->AddSpectator("n_jets",       "n_jets",       "units", 'I');

  data_loader->AddVariable("had_t_b_pt",   "had_t_b_pt",   "units", 'F');
  data_loader->AddVariable("w_u_pt",       "w_u_pt",       "units", 'F');
  data_loader->AddVariable("w_d_pt",       "w_d_pt",       "units", 'F');
  data_loader->AddVariable("lep_t_b_pt",   "lep_t_b_pt",   "units", 'F');
  
  data_loader->AddVariable("had_t_b_bscore",   "had_t_b_bscore",   "units", 'F');
  data_loader->AddVariable("lep_t_b_bscore",   "lep_t_b_bscore",   "units", 'F');
  
  //data_loader->AddVariable("del_phi_w_u_w_d", "del_phi_w_u_w_d", "units", 'F');
  //data_loader->AddVariable("del_eta_w_u_w_d", "del_eta_w_u_w_d", "units", 'F');
  //data_loader->AddVariable("del_r_w_u_w_d", "del_r_w_u_w_d", "units", 'F');
  data_loader->AddVariable("theta_w_u_w_d", "theta_w_u_w_d", "units", 'F');

  //data_loader->AddVariable("del_phi_had_w_had_t_b", "del_phi_had_w_had_t_b", "units", 'F');
  //data_loader->AddVariable("del_eta_had_w_had_t_b", "del_eta_had_w_had_t_b", "units", 'F');
  //data_loader->AddVariable("del_r_had_w_had_t_b", "del_r_had_w_had_t_b", "units", 'F');
  data_loader->AddVariable("theta_had_w_had_t_b", "theta_had_w_had_t_b", "units", 'F');
  
  //data_loader->AddVariable("del_phi_lep_neu", "del_phi_lep_neu", "units", 'F');
  //data_loader->AddVariable("del_eta_lep_neu", "del_eta_lep_neu", "units", 'F');
  //data_loader->AddVariable("del_r_lep_neu", "del_r_lep_neu", "units", 'F');
  data_loader->AddVariable("theta_lep_neu", "theta_lep_neu", "units", 'F');

  //data_loader->AddVariable("del_phi_lep_w_lep_t_b", "del_phi_lep_w_lep_t_b", "units", 'F');
  //data_loader->AddVariable("del_eta_lep_w_lep_t_b", "del_eta_lep_w_lep_t_b", "units", 'F');
  //data_loader->AddVariable("del_r_lep_w_lep_t_b", "del_r_lep_w_lep_t_b", "units", 'F');
  data_loader->AddVariable("theta_lep_w_lep_t_b", "theta_lep_w_lep_t_b", "units", 'F');
  
  data_loader->AddVariable("del_phi_had_t_lep_t", "del_phi_had_t_lep_t", "units", 'F');
  //data_loader->AddVariable("del_eta_had_t_lep_t", "del_eta_had_t_lep_t", "units", 'F');
  //data_loader->AddVariable("del_r_had_t_lep_t", "del_r_had_t_lep_t", "units", 'F');
  //data_loader->AddVariable("theta_had_t_lep_t", "theta_had_t_lep_t", "units", 'F');
  
  data_loader->AddVariable("had_t_mass",   "had_t_mass",   "units", 'F');
  data_loader->AddVariable("had_w_mass",   "had_w_mass",   "units", 'F');
  data_loader->AddVariable("lep_t_mass",   "lep_t_mass",   "units", 'F');
  data_loader->AddVariable("lep_t_partial_mass",   "lep_t_partial_mass",   "units", 'F');
 
  // data_loader->AddVariable("chi2_jet_had_t_b",   "chi2_jet_had_t_b",   "units", 'F');
  // data_loader->AddVariable("chi2_jet_w_u",   "chi2_jet_w_u",   "units", 'F');
  // data_loader->AddVariable("chi2_jet_w_d",   "chi2_jet_w_d",   "units", 'F');
  // data_loader->AddVariable("chi2_jet_lep_t_b",   "chi2_jet_lep_t_b",   "units", 'F');
  // if(n_jet==4) data_loader->AddSpectator("chi2_jet_extra",   "chi2_jet_extra",   "units", 'F');
  // else data_loader->AddVariable("chi2_jet_extra",   "chi2_jet_extra",   "units", 'F');
  // data_loader->AddVariable("chi2_constraint_had_t",   "chi2_constraint_had_t",   "units", 'F');
  // data_loader->AddVariable("chi2_constraint_had_w",   "chi2_constraint_had_w",   "units", 'F');
  // data_loader->AddVariable("chi2_constraint_lep_t",   "chi2_constraint_lep_t",   "units", 'F');
  // data_loader->AddVariable("chi2_constraint_lep_w",   "chi2_constraint_lep_w",   "units", 'F');
  
  if(!chk_pre_kin) data_loader->AddVariable("chi2",         "chi2",   "units", 'F');

  data_loader->AddSignalTree(tree_correct, 1.0);
  data_loader->AddBackgroundTree(tree_wrong, 1.0);

  TCut cut_s = Form("n_jets==%d", n_jet);
  TCut cut_b = Form("n_jets==%d", n_jet);

  int n_train_signal = tree_correct->GetEntries(cut_s)/2;
  int n_train_back = tree_wrong->GetEntries(cut_b)/10;

  data_loader->PrepareTrainingAndTestTree(cut_s, cut_b,
                                          Form("nTrain_Signal=%d:nTrain_Background=%d:nTest_Signal=%d:nTest_Background=%d:SplitMode=Random:NormMode=NumEvents:V", n_train_signal, n_train_back, n_train_signal, n_train_back));

  // //Adaptive Boost
  // factory->BookMethod(data_loader, TMVA::Types::kBDT, "BDT",
  // 		      "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=100");

  //Gradient Boost
  factory->BookMethod(data_loader, TMVA::Types::kBDT, "BDTG",
		      "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=100:MaxDepth=2" );

  //Fisher
  //factory->BookMethod(data_loader, TMVA::Types::kFisher, "Fisher", "H:!V:Fisher:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=100:NsmoothMVAPdf=10" );

  //DNN_CPU
  //General layout
  // TString layoutString("Layout=RELU|128,RELU|128,RELU|128,LINEAR");
  
  // //Define Training strategy
  // TString trainingStrategyString = ("TrainingStrategy=LearningRate=1e-4,Momentum=0.9,"
  // 				    "ConvergenceSteps=20,BatchSize=100,TestRepetitions=1,"
  // 				    "WeightDecay=1e-4,Regularization=None,"
  // 				    "DropConfig=0.0+0.5+0.5+0.5");
  
  // //General options
  // TString dnnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:"
  // 		      "WeightInitialization=XAVIERUNIFORM");
  // dnnOptions.Append (":"); dnnOptions.Append (layoutString);
  // dnnOptions.Append (":"); dnnOptions.Append (trainingStrategyString);

  // TString cpuOptions = dnnOptions + ":Architecture=CPU";

  // factory->BookMethod(data_loader, TMVA::Types::kDL, "DNN_CPU", cpuOptions);

  factory->TrainAllMethods();
  factory->TestAllMethods();
  factory->EvaluateAllMethods();

  fout->Close();

  delete factory;
  delete data_loader;

  //TMVA::TMVAGui(fout_name);

  return;
}
