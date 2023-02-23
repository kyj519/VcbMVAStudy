import argparse
from subprocess import call
import ROOT
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#######
#BELOW CODE IS OPTIMIZED FOR TENSORFLOW-2.4.1!
#######
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Add, Lambda
from tensorflow.keras.constraints import max_norm
#from tf.keras.layers.noise import GaussianNoise
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import plot_model
import sys
sys.path.append('/data6/Users/yeonjoon/VcbMVAStudy')
from root_data_loader import load_data, classWtoSampleW

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import optuna
import plotly

ROOT.EnableImplicitMT()


print('training using keras')


varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']



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

  def compile(self,optimizer_,lossftn=BinaryCrossentropy(),
           #optimizer_=SGD(lr=0.1,decay=1e-5),
           # default lr=0.001
           #optimizer_=Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1),
           #optimizer_=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
           metrics_=['AUC']         ):
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
    'neuron_exponent':trial.suggest_int('neuron_exponent',3,4),
    #'rho':trial.suggest_uniform('rho',0.3,0.99),
    #'epsilon':trial.suggest_uniform('epsilon',1e-10,1e-7),
    'batch_size':trial.suggest_categorical('batch_size',[pow(2,i) for i in range(9,15)]),
    'max_norm':trial.suggest_float('max_norm',1,10)
  }
  modelDNN = KerasModel()
  modelDNN.defineModel_3layer(varlist,param['depth'],param['neuron_exponent'],param['max_norm'])
  optimizer=Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-7, clipnorm=0.1)
  #optimizer.build(modelDNN.model.trainable_variables)
  modelDNN.compile(optimizer_=optimizer)
  
  #optimizer_=Adadelta(learning_rate=1.0, rho=param['rho'], epsilon=param['epsilon'], decay=0.0, clipnorm=0.1)
  
  EPOCHS_SIZE = 1000
  BATCH_SIZE = param['batch_size']
  early_stopping = EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)
  modelDNN.model.fit(data['train_features'],data['train_y'],epochs=EPOCHS_SIZE,batch_size=BATCH_SIZE,callbacks=[early_stopping],validation_data=(data['val_features'],data['val_y']),class_weights=data['class_weights'])
  modelDNN.plot_mymodel(outFile=f'model_trial_{trial.number}.png')
  test_result = modelDNN.model.evaluate(data['test_features'],data['test_y'],batch_size=1, verbose = 0)
  print(modelDNN.model.metrics_names)
  print(test_result)
  del modelDNN
  return test_result[1]
  
if __name__ == '__main__':

  
  
  path_sample = os.environ["WtoCB_PATH"]
  #filename = 'Vcb_Mu_TTLJ_WtoCB_powheg_25.root'
  filename = 'Vcb_2018_Mu_Reco_Tree.root' 
 
  data =  load_data(os.path.join(path_sample,filename), '-10.<bvsc_w_u',varlist,0.1,0.2,['Reco_45'],['Reco_43','Reco_41','Reco_23','Reco_21'])
  optuna.logging.set_verbosity(optuna.logging.DEBUG)
  #study = optuna.create_study(direction='maximize', s2ampler=optuna.samplers.TPESampler())
  #study.optimize(lambda trial: objective_keras(trial, data),n_trials=20)
  #fig_contour = optuna.visualization.plot_contour(study)
  #fig_importance = optuna.visualization.plot_param_importances(study)
  #fig_contour.write_html(f'opt_contour.html')
  #fig_importance.write_html(f'opt_importance.html')
  #print(f"here is the result of hyperparameter tuning, best score = {study.best_value}:\n")
  #print(study.best_trial.params)
  #param = study.best_params
  
  param = {}
  param['activation'] = 'elu'
  param['optimizer'] = False
  param['batch_size'] = 512
  param['neuron_exponent'] = 4
  param['depth'] = 16
  param['max_norm'] = 2

  modelDNN = KerasModel()
  modelDNN.defineModel_3layer(len(varlist)-1,param['depth'],param['neuron_exponent'],param['max_norm'])
  optimizer=Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-7, clipnorm=0.1)
  #optimizer.build(modelDNN.model.trainable_variables)
  modelDNN.compile(optimizer_=optimizer)

  EPOCHS_SIZE = 1000
  BATCH_SIZE = param['batch_size']
  early_stopping = EarlyStopping(
    monitor='val_loss', 
    verbose=1,
    patience=20,
    mode='auto',
    restore_best_weights=True)
  modelDNN.model.fit(data['train_features'],data['train_y'],
                     epochs=EPOCHS_SIZE,batch_size=BATCH_SIZE,
                     callbacks=[early_stopping],
                     validation_data=(data['val_features'],data['val_y']),
                     class_weight=data['class_weight'])
  modelDNN.save('/data6/Users/yeonjoon/VcbMVAStudy/keras_template/model.h5')
  modelDNN.plot_mymodel(outFile='plot.png')
  
  

