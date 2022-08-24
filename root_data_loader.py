import ROOT
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(file_path, n_jet, varlist,test_ratio, val_ratio):
  print(file_path)
  data_sig = ROOT.RDataFrame('Permutation_Correct',file_path).Filter(f'n_jets=={n_jet:.0f}').AsNumpy(varlist)
  data_bkg = ROOT.RDataFrame('Permutation_Wrong', file_path).Filter(f'n_jets=={n_jet:.0f}').AsNumpy(varlist)

  sig = pd.DataFrame({k:v for k,v in data_sig.items()})
  sig['y'] = np.ones(sig.shape[0])
  bkg = pd.DataFrame({k:v for k,v in data_bkg.items()})
  bkg['y'] = np.zeros(bkg.shape[0])
  class_weights = {0:(sig.shape[0]+bkg.shape[0])/bkg.shape[0],1:(sig.shape[0]+bkg.shape[0])/sig.shape[0]}
  df = pd.concat([sig,bkg],ignore_index=True)
  df = df.sample(frac=1, random_state=999)
  df = df.reset_index(drop=True)
  train_df, test_df = train_test_split(df, test_size=test_ratio)
  train_df, val_df = train_test_split(train_df, test_size=val_ratio)
  
  train_y = np.array(train_df.pop('y').reset_index(drop=True))
  test_y = np.array(test_df.pop('y').reset_index(drop=True))
  val_y = np.array(val_df.pop('y').reset_index(drop=True))
  
  
  
  train_features = np.array(train_df)
  test_features = np.array(test_df)
  val_features = np.array(val_df)

  scaler = StandardScaler()

  train_features = scaler.fit_transform(train_features)
  val_features = scaler.transform(val_features)
  test_features = scaler.transform(test_features)
  
  train_features = np.clip(train_features, -5, 5)
  val_features = np.clip(val_features, -5, 5)
  test_features = np.clip(test_features, -5, 5)
 
  
  return {'train_features': train_features, 'test_features': test_features, 'val_features': val_features
        ,'train_y': train_y, 'test_y': test_y, 'val_y': val_y, 'class_weights':class_weights}
  
def classWtoSampleW(dataset,class_weights):
    weight = []
    print(dataset)
    for class_data in dataset:
        weight.append(class_weights[class_data])
    return np.array(weight)