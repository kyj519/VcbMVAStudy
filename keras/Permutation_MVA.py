import argparse
from ROOT import TMVA, TString, TFile, TTree, TCut
from subprocess import call
import ROOT
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from tensorflow.keras.layers import Add, Lambda
from tensorflow.keras.constraints import max_norm
#from tf.keras.layers.noise import GaussianNoise
from keras.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import plot_model
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from root_data_loader import load_data, classWtoSampleW

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import optuna
import plotly


print('training using keras')
varlist_prekin = ['had_t_b_pt','w_u_pt','w_d_pt','lep_t_b_pt','had_t_b_bscore','lep_t_b_bscore',
   'theta_w_u_w_d','theta_lep_neu','theta_lep_w_lep_t_b', 'del_phi_had_t_lep_t',
   'had_t_mass','had_w_mass','lep_t_mass','lep_t_partial_mass']
varlist = varlist_prekin + ['chi2']

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

varlist_prekin = ['had_t_b_pt','w_u_pt','w_d_pt','lep_t_b_pt','had_t_b_bscore','lep_t_b_bscore',
   'theta_w_u_w_d','theta_lep_neu','theta_lep_w_lep_t_b', 'del_phi_had_t_lep_t',
   'had_t_mass','had_w_mass','lep_t_mass','lep_t_partial_mass']
varlist = varlist_prekin + ['chi2']



# Define initialization
def normal(shape, name=None):
  return initializers.normal(shape, scale=0.05, name=name)

# Generate model
class KerasModel():

  def __init__(self):
    self.model = Sequential()


  def defineModel_3layer(self,input_dim_,depth,neuron_exponent,maxnorm):
    K.clear_session()
    # Define model

    #
    # we can think of this chunk as the input layer
    self.model.add(Lambda(lambda X : X, input_shape=(input_dim_,))) #dummy Lamda layer for test
    for i in range(depth):
      self.model.add(Dense(pow(2,neuron_exponent+1), kernel_initializer=initializers.he_normal(seed=1232), kernel_constraint=max_norm(maxnorm)))
      self.model.add(BatchNormalization())
      self.model.add(Activation('elu'))
      self.model.add(Dropout(0.50))

      self.model.add(Dense(pow(2,neuron_exponent), kernel_initializer=initializers.he_normal(seed=1232), kernel_constraint=max_norm(maxnorm)))
      self.model.add(BatchNormalization())
      self.model.add(Activation('elu'))
      self.model.add(Dropout(0.50))


    # we can think of this chunk as the output layer
    self.model.add(Dense(1, kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=1234)))
    self.model.add(Activation('sigmoid'))

    #self.model.add(Dense(64, kernel_initializer=initializers.he_normal(seed=None), activation='relu', input_dim=input_dim_))
    #self.model.add(Dense(32, kernel_initializer=initializers.he_normal(seed=None), activation='relu'))
    #self.model.add(Dense(2, kernel_initializer=initializers.he_normal(seed=None), activation='softmax'))

  def compile(self,lossftn=BinaryCrossentropy(),
           #optimizer_=SGD(lr=0.1,decay=1e-5),
           # default lr=0.001
           #optimizer_=Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1),
           #optimizer_=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
           optimizer_=Adadelta(learning_rate=1.0, rho=0.95, epsilon=None, decay=0.0, clipnorm=0.1),
           metrics_=['AUC']
         ):
    # Set loss and optimizer
    self.model.compile(loss=lossftn, optimizer=optimizer_, metrics=metrics_)

  def save(self, modelName="model.h5"):
    self.model.save(modelName)

  def summary(self):
    self.model.summary()

  def plot_mymodel(self,outFile='model.png'):
    print('plot model............')
    try:
      plot_model(self.model,  to_file=outFile, show_shapes = False)
    except:
      print('[INFO] Failed to make model plot')
      



def objective_keras(trial: optuna.Trial, data):
  param = {
    'depth':trial.suggest_int('depth',3,5),
    'neuron_exponent':trial.suggest_int('neuron_exponent',3,5),
    'rho':trial.suggest_uniform('rho',0.3,0.99),
    'epsilon':trial.suggest_uniform('epsilon',1e-10,1e-7),
    'batch_size':trial.suggest_categorical('batch_size',[pow(2,i) for i in range(9,15)]),
    'max_norm':trial.suggest_uniform('max_norm',1,10)
  }
  modelDNN = KerasModel()
  modelDNN.defineModel_3layer(len(varlist_prekin) if chk_pre_kin else len(varlist),param['depth'],param['neuron_exponent'],param['max_norm'])
  modelDNN.compile(optimizer_=Adadelta(learning_rate=1.0, rho=param['rho'], epsilon=param['epsilon'], decay=0.0, clipnorm=0.1))
  EPOCHS_SIZE = 1000
  BATCH_SIZE = param['batch_size']
  early_stopping = EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)
  modelDNN.model.fit(data['train_features'],data['train_y'],epochs=EPOCHS_SIZE,batch_size=BATCH_SIZE,callbacks=[early_stopping],validation_data=(data['val_features'],data['val_y']))
  modelDNN.plot_mymodel(outFile=f'model_trial_{trial.number}.png')
  test_result = modelDNN.model.evaluate(data['test_features'],data['test_y'],batch_size=1, verbose = 0)
  print(modelDNN.model.metrics_names)
  print(test_result)
  return test_result[1]
  
if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--n_jet', type=int)
  parser.add_argument('--pre_kin', type=int)
  args = parser.parse_args()
  n_jet = args.n_jet
  chk_pre_kin = True if args.pre_kin else False

  print(chk_pre_kin, n_jet)
  
  path_sample = os.environ["WtoCB_PATH"]
  filename = 'Vcb_Mu_TTLJ_WtoCB_powheg_25.root'
  
  data =   data_dict = load_data(os.path.join(path_sample,filename),n_jet,varlist_prekin if chk_pre_kin else varlist,0.2,0.2)
  optuna.logging.set_verbosity(optuna.logging.DEBUG)
  study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
  study.optimize(lambda trial: objective_keras(trial, data),n_trials=15)
  fig_contour = optuna.visualization.plot_contour(study)
  fig_importance = optuna.visualization.plot_param_importances(study)
  fig_contour.write_html(f'opt_contour.html')
  fig_importance.write_html(f'opt_importance.html')
  print(f"here is the result of hyperparameter tuning, best score = {study.best_value}:\n")
  print(study.best_trial.params)
  param = study.best_params
  modelDNN = KerasModel()
  modelDNN.defineModel_3layer(len(varlist_prekin) if chk_pre_kin else len(varlist),param['depth'],param['neuron_exponent'])
  modelDNN.compile(optimizer_=Adadelta(learning_rate=1.0, rho=param['rho'], epsilon=param['epsilon'], decay=0.0, clipnorm=0.1))
  EPOCHS_SIZE = 1000
  BATCH_SIZE = param['batch_size']
  early_stopping = EarlyStopping(
    monitor='val_loss', 
    verbose=1,
    patience=1,
    mode='auto',
    restore_best_weights=True)
  modelDNN.model.fit(data['train_features'],data['train_y'],epochs=EPOCHS_SIZE,batch_size=BATCH_SIZE,callbacks=[early_stopping],validation_data=(data['val_features'],data['val_y']))
  modelDNN.save()
  
  

