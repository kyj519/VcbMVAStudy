import ROOT
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder

ROOT.EnableImplicitMT()

def load_data(file_path, filterstr, varlist,test_ratio, val_ratio,sigTree,bkgTree,makeStandard=False, useLabelEncoder = True):
  sig_dict = []
  bkg_dict = []
 
  for tree in sigTree:
    df = ROOT.RDataFrame(tree,file_path)
    df = df.Filter(filterstr)
    sig_dict.append(df.AsNumpy(varlist))
  for tree in bkgTree:
    df = ROOT.RDataFrame(tree,file_path)
    df = df.Filter(filterstr)
    bkg_dict.append(df.AsNumpy(varlist))
  data_sig = {}
  data_bkg = {}
  for key in sig_dict[0]:
    data_sig[key] = np.concatenate([arr[key] for arr in sig_dict])
  for key in bkg_dict[0]:
    data_bkg[key] = np.concatenate([arr[key] for arr in bkg_dict])

  
  sig = pd.DataFrame({k:v for k,v in data_sig.items()})
  sig['y'] = np.ones(sig.shape[0])
  bkg = pd.DataFrame({k:v for k,v in data_bkg.items()})
  bkg['y'] = np.zeros(bkg.shape[0])
  class_weight = {0:(sig.shape[0]+bkg.shape[0])/bkg.shape[0],1:(sig.shape[0]+bkg.shape[0])/sig.shape[0]}
  df = pd.concat([sig,bkg],ignore_index=True)
  df = df.sample(frac=1, random_state=999)
  df = df.reset_index(drop=True)
  train_df, test_df = train_test_split(df, test_size=test_ratio)
  train_df, val_df = train_test_split(train_df, test_size=val_ratio)
  
  train_y = np.array(train_df.pop('y').reset_index(drop=True))
  test_y = np.array(test_df.pop('y').reset_index(drop=True))
  val_y = np.array(val_df.pop('y').reset_index(drop=True))
  
  nunique = train_df.nunique()
  types = train_df.dtypes
  print(types)
  categorical_columns = []
  categorical_dims =  {}
  if useLabelEncoder:
    for col in train_df.columns:
        if nunique[col] < 200:
            print(col, train_df[col].nunique())
            l_enc = LabelEncoder()
            train_df[col] = l_enc.fit_transform(train_df[col].values)
            val_df[col] = l_enc.fit_transform(val_df[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        

  features = [ col for col in train_df.columns] 
  cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
  cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
  
  train_weight = np.array(train_df.pop('weight'))
  test_df.pop('weight')
  val_weight = np.array(val_df.pop('weight'))
  
  train_features = np.array(train_df.values)
  test_features = np.array(test_df.values)
  val_features = np.array(val_df.values)
  print(train_features[0])
  scaler = StandardScaler()
  if makeStandard:
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

  
  #train_features = np.clip(train_features, -5, 5)
  #val_features = np.clip(val_features, -5, 5)
  #test_features = np.clip(test_features, -5, 5)
 
  
  return {'train_features': train_features, 'test_features': test_features, 'val_features': val_features
        ,'train_y': train_y, 'test_y': test_y, 'val_y': val_y,
        'class_weight':class_weight,'train_weight':train_weight, 'val_weight':val_weight
        ,'cat_idxs':cat_idxs,'cat_dims':cat_dims}
  
def classWtoSampleW(dataset,class_weights):
    weight = []
    for class_data in dataset:
        weight.append(class_weights[class_data])
    return np.array(weight)
