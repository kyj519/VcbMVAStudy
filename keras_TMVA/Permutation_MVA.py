import argparse
from ROOT import TMVA, TString, TFile, TTree, TCut
from subprocess import call
import ROOT
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
class mva_variable():
    def __init__(self, name, title, units, type="F", isSpectator=False):
        self.name = name
        self.title = title
        self.units = units
        self.type = type
        self.isSpectator = isSpectator


varlist = [#mva_variable("n_jets", "n_jets", "#", isSpectator=True),
           mva_variable("had_t_b_pt",   "had_t_b_pt", "GeV"),
           mva_variable("w_u_pt",       "w_u_pt", "GeV"),
           mva_variable("w_d_pt",       "w_d_pt",       "GeV"),
           mva_variable("lep_t_b_pt",   "lep_t_b_pt",   "GeV"),
           mva_variable("had_t_b_bscore",   "had_t_b_bscore",   "points"),
           mva_variable("lep_t_b_bscore",   "lep_t_b_bscore",   "points"),
           mva_variable("theta_w_u_w_d", "theta_w_u_w_d", "Rad."),
           mva_variable("theta_lep_neu", "theta_lep_neu", "Rad."),
           mva_variable("theta_lep_w_lep_t_b", "theta_lep_w_lep_t_b", "Rad."),
           mva_variable("del_phi_had_t_lep_t", "del_phi_had_t_lep_t", "Rad."),
           mva_variable("had_t_mass",   "had_t_mass",   "GeV"),
           mva_variable("had_w_mass",   "had_w_mass",   "GeV"),
           mva_variable("lep_t_mass",   "lep_t_mass",   "GeV"),
           mva_variable("lep_t_partial_mass",   "lep_t_partial_mass",   "GeV")
           ]




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jet', type=int)
    parser.add_argument('--pre_kin', type=int)
    args = parser.parse_args()
    n_jet = args.n_jet
    chk_pre_kin = True if args.pre_kin else False

    print(chk_pre_kin, n_jet)

    reader = TMVA.Reader("Color:!Silent")

    path_sample = os.environ["WtoCB_PATH"]
    path_dir = os.environ["DIR_PATH"]+"/result/BDT/"

    fin = TFile(path_sample+"Vcb_Mu_TTLJ_WtoCB_powheg_25.root", "READ")
    tree_correct = fin.Get("Permutation_Correct")
    tree_wrong = fin.Get("Permutation_Wrong")

    fout_name = ''
    if not chk_pre_kin:
        fout_name = '%sVcb_Permutation_TTLJ_WtoCB_%s.root' % (path_dir, n_jet)
    else:
        fout_name = '%sVcb_PreKin_TTLJ_WtoCB_%s.root' % (path_dir, n_jet)

    fout = TFile(fout_name, "RECREATE")
    factory = TMVA.Factory('TMVAClassification', fout,
                           '!V:!Silent:Color:DrawProgressBar:Transformations=D,G:AnalysisType=Classification')
    data_loader = TMVA.DataLoader('dataset')

    if not chk_pre_kin:
        varlist.append(mva_variable("chi2",         "chi2",   "points"))
    n_variables = len(varlist)
    for var in varlist:
        if var.isSpectator:
            data_loader.AddSpectator(var.name, var.title, var.units)
        else:
            data_loader.AddVariable(var.name, var.title, var.units, var.type)
            n_variables = n_variables + 1

    data_loader.AddSignalTree(tree_correct, 1.0)
    data_loader.AddBackgroundTree(tree_wrong, 1.0)

    cut_s = TCut('n_jets==%s' % n_jet)
    cut_b = TCut('n_jets==%s' % n_jet)
    n_train_signal = tree_correct.GetEntries('n_jets==%s' % n_jet)/2
    n_train_back = tree_wrong.GetEntries('n_jets==%s' % n_jet)/10

    data_loader.PrepareTrainingAndTestTree(
        cut_s, cut_b, 'nTrain_Signal=%s:nTrain_Background=%s:SplitMode=Random:NormMode=NumEvents:V' % (n_train_signal,n_train_back))

    # Define model
# Define model
    model = Sequential()
    model.add(Dense(64, activation='relu',  input_dim=n_variables))
    model.add(Dense(2, activation='softmax'))
    
    # Set loss and optimizer
    model.compile(loss='categorical_crossentropy',
                optimizer=SGD(lr=0.01), metrics=['accuracy', ])
    # Store model to file
    model.save('/data6/Users/yeonjoon/VcbMVAStudy/keras_TMVA/model_%s_%s.h5'%(n_jet,chk_pre_kin))
    model.summary()
    
    
    dnnOption = "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:WeightInitialization=XAVIERUNIFORM:Layout=TANH|100,TANH|128,TANH|128,LINEAR:TrainingStrategy=LearningRate=1e-2,Momentum=0.9,ConvergenceSteps=20,BatchSize=100,TestRepetitions=1,WeightDecay=1e-4,Regularization=None,DropConfig=0.0+0.5+0.5+0.5:Architecture=GPU" 
                    
    parstr = 'H:V:VarTransform=D,G:FilenameModel=/data6/Users/yeonjoon/VcbMVAStudy/keras_TMVA/model_%s_%s.h5:NumEpochs=20:BatchSize=32' % (n_jet,chk_pre_kin)
    #factory.BookMethod(data_loader, TMVA.Types.kPyKeras, 'PyKeras', parstr)
    factory.BookMethod(data_loader, TMVA.Types.kBDT, "BDTG",
                      "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=100:MaxDepth=2")
    factory.BookMethod(data_loader, TMVA.Types.kDL, "DNN_CPU", dnnOption)
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()
    fout.Close()
