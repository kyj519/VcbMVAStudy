import ROOT
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
ROOT.EnableImplicitMT(16)
import sys,os
sys.path.append(os.environ["DIR_PATH"])
ROOT.EnableImplicitMT(16)
from root_data_loader import load_data

model = load_model('/data6/Users/yeonjoon/VcbMVAStudy/keras_template/model_best.h5')
modelist = ['45','43','41','23','21']

varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d','cvsb_w_u','cvsb_w_d','n_bjets','n_cjets','weight']
result = []
for mode in modelist:
    continue
    data =  load_data(file_path='/gv0/Users/yeonjoon/Vcb_2018_Mu_Reco_Tree.root',varlist=varlist,test_ratio=0.1,val_ratio=0.2,sigTree=[f'Reco_{mode}'],bkgTree=[])
    arr = data['test_features']
    pred = model.predict(arr)
    result.append(pred)
    plt.hist(pred, bins=40, weights=data['test_weight'])
    plt.savefig(f'/data6/Users/yeonjoon/VcbMVAStudy/keras_template/result_{mode}.png')
    plt.clf()

    
    del data
    del arr 
    
#df = pd.concat([pd.DataFrame(a, columns=[modelist[i]]) for i, a in enumerate(result)], axis=1)
#fig = df.plot.hist(stacked=True, bins=30, figsize=(10, 6), grid=True)
#fig.figure.savefig('stack.png',dpi=600)

data =  load_data(file_path='/gv0/Users/yeonjoon/Vcb_2018_Mu_Reco_Tree.root',varlist=varlist,test_ratio=0.1,val_ratio=0.2,sigTree=['Reco_45'],bkgTree=['Reco_43'])
import postTrainingToolkit
train_score = np.array(model.predict(data['train_features'])).T[0]
val_score = np.array(model.predict(data['val_features'])).T[0]
print(val_score)
print(val_score.shape)
kolS, kolB = postTrainingToolkit.KS_test(train_score,val_score,data['train_weight'],data['val_weight'],data['train_y'],data['val_y'])
print(f'{kolS}, {kolB}')
#feature_importances_ = model._compute_feature_importances(data['train_features'])
#print(feature_importances_)