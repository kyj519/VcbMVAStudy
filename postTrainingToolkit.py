import numpy as np
import pandas as pd
import os
import ROOT
import itertools
import plotly.graph_objs as go
from sklearn.metrics import roc_curve, auc

ROOT.gStyle.SetOptStat(0)
ROOT.EnableImplicitMT(16)
ROOT.gROOT.LoadMacro(os.path.join(os.environ["DIR_PATH"],"tdrStyle.C"))
ROOT.gROOT.ProcessLine("setTDRStyle();")
def shape_test(set1,set2):
    return True if set1.shape[0] == set2.shape[0] else False

def shapeCheck(data_list):
    shape_matched = True
    for p in itertools.combinations(data_list,2):
            shape_matched &= shape_test(p[0],p[1])
    return shape_matched

def npcoversion(*args):
    for data in args:
        data = np.array(data)

def seperate_sig_bkg(df, branch = '', target_Branch = 'y'):
    sig_value = df[df['y']==1][branch].values
    bkg_value = df[df['y']==0][branch].values
    sig_idx = df[df['y']==1][branch].index
    bkg_idx = df[df['y']==0][branch].index
    
    return sig_value, bkg_value, sig_idx, bkg_idx
    
def ROC_AUC(score, y, weight, plot_path):
    # Compute the ROC curve
    fpr, tpr, _ = roc_curve(y, score, sample_weight=weight)
    unique_fpr, unique_indices = np.unique(fpr, return_index=True)
    tpr = tpr[unique_indices]
    # Compute the AUC
    roc_auc = auc(unique_fpr, tpr)
    # Create a Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random guessing',
                            line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name='ROC curve (AUC = %0.2f)' % roc_auc))
    fig.update_layout(xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    title='Receiver Operating Characteristic (ROC) Curve')

    # Save the figure as a PNG file
    fig.write_image(os.path.join(plot_path,'ROC.png'), width=800, height=600)



def KS_test(train_score,val_score,train_w,val_w,train_y,val_y,plotPath=""):
    npcoversion(train_score,val_score,train_w,val_w,train_y,val_y) 
    if not shapeCheck([train_score,train_w,train_y]):
        print(f"Shape mismatching")
        return
    if not shapeCheck([val_score,val_w,val_y]):
        print(f"Shape mismatching")
        return

    train_df = pd.DataFrame({'score':train_score,'weight':train_w,'y':train_y})
    val_df = pd.DataFrame({'score':val_score,'weight':val_w,'y':val_y})
    print(train_df)
    print(val_df)
    ### split pandas dataframe to bkg and sig, store as ndarray.
    
    val_score_sig, val_score_bkg, _ , _ = seperate_sig_bkg(val_df,'score')
    train_score_sig, train_score_bkg, _ , _ = seperate_sig_bkg(train_df,'score')
    

    val_w_sig, val_w_bkg, _ , _ = seperate_sig_bkg(val_df,'weight')
    train_w_sig, train_w_bkg, _ , _ = seperate_sig_bkg(train_df,'weight')


    val_y_sig, val_y_bkg, _ , _ = seperate_sig_bkg(val_df,'score')
    train_y_sig, train_y_bkg, _ , _ = seperate_sig_bkg(train_df,'score')
    

    ##convert ndarray to ROOT RDataFrame
    
    train_df_sig = ROOT.RDF.MakeNumpyDataFrame({'score':train_score_sig,'weight':train_w_sig,'y':train_y_sig})
    train_df_bkg = ROOT.RDF.MakeNumpyDataFrame({'score':train_score_bkg,'weight':train_w_bkg,'y':train_y_bkg})
    val_df_sig = ROOT.RDF.MakeNumpyDataFrame({'score':val_score_sig,'weight':val_w_sig,'y':val_y_sig})
    val_df_bkg = ROOT.RDF.MakeNumpyDataFrame({'score':val_score_bkg,'weight':val_w_bkg,'y':val_y_bkg})
    
    ##define histogram model. TMVA default bin N seems 40. modify below to use different value.
    hist_model = ROOT.RDF.TH1DModel("score", "", 40, 0, 1)
    
    ##get hist. from RDataFrame.
    hist_train_sig = train_df_sig.Histo1D(hist_model,"score","weight")
    hist_val_sig = val_df_sig.Histo1D(hist_model,"score","weight")
    hist_train_bkg = train_df_bkg.Histo1D(hist_model,"score","weight")
    hist_val_bkg = val_df_bkg.Histo1D(hist_model,"score","weight")
    
    ##do Clone() because Histo1D method returns pointer of histogram. now T1HD stored in hist* variables. 

    hist_train_sig = hist_train_sig.Clone()
    hist_val_sig = hist_val_sig.Clone()
    hist_train_bkg = hist_train_bkg.Clone()
    hist_val_bkg = hist_val_bkg.Clone()
    
    hist_train_sig.Sumw2()
    hist_val_sig.Sumw2()
    hist_train_bkg.Sumw2()
    hist_val_bkg.Sumw2()
    ##draw overall plot
    den_train_sig = hist_train_sig.Clone()
    den_val_sig = hist_val_sig.Clone()
    den_train_bkg = hist_train_bkg.Clone()
    den_val_bkg = hist_val_bkg.Clone()
    
    factor = 1.
    den_train_sig.Scale(factor/hist_train_sig.Integral("width"))
    den_val_sig.Scale(factor/hist_val_sig.Integral("width"))
    den_train_bkg.Scale(factor/hist_train_bkg.Integral("width"))
    den_val_bkg.Scale(factor/hist_val_bkg.Integral("width"))
    
    c1 = ROOT.TCanvas("","",1600,1600)
    c1.cd()
    den_train_sig.SetFillColorAlpha(ROOT.kGreen+1,0.8)
    den_train_sig.SetLineColorAlpha(ROOT.kGreen+1,1)
    den_train_sig.SetLineWidth(3)
     
    den_val_sig.SetFillColorAlpha(ROOT.kGreen+2,0)
    den_val_sig.SetLineColorAlpha(ROOT.kGreen+3,1)
    den_val_sig.SetLineWidth(4)

    den_train_bkg.SetFillColorAlpha(ROOT.kOrange+1,0.8)
    den_train_bkg.SetLineColorAlpha(ROOT.kOrange+1,1) 
    den_train_bkg.SetLineWidth(3)
    
    den_val_bkg.SetFillColorAlpha(ROOT.kOrange+2,0)
    den_val_bkg.SetLineColorAlpha(ROOT.kOrange+3,1)   
    den_val_bkg.SetLineWidth(4)
    
    xLat, yLat = 0.25, 0.91
    xLeg, yLeg = xLat + 0.30, yLat
    leg_h = 0.03 * 4
    leg = ROOT.TLegend(xLeg, yLeg - leg_h, xLeg + 0.35, yLeg)
    leg.SetNColumns(2)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(43)
    leg.SetTextSize(18)
    leg.AddEntry(den_train_bkg, "Train(bkg.)", "f")
    leg.AddEntry(den_train_sig, "Train(sig.)", "f")
    leg.AddEntry(den_val_bkg, "Validation(bkg.)", "f")
    leg.AddEntry(den_val_sig, "Validataion(sig.)", "f")
    maximum = max(den_train_bkg.GetMaximum(),den_train_sig.GetMaximum(),den_val_bkg.GetMaximum(),den_val_sig.GetMaximum())
    maximum = maximum *1.2
    den_train_bkg.SetMaximum(maximum)
    den_train_bkg.Draw('hist e1') 
    den_val_bkg.Draw('hist e1 same') 
    den_train_sig.Draw('hist e1 same') 
    den_val_sig.Draw('hist e1 same')
    leg.Draw('same')

    
    c1.Draw()
    c1.SaveAs(os.path.join(plotPath,'overall.png'))
    
    ##doing KS test
    repeat = 1000
    kolS_history = []
    kolB_history = []
    kolS_max = hist_val_sig.KolmogorovTest(hist_train_sig,"M") 
    kolB_max = hist_val_sig.KolmogorovTest(hist_train_sig,"M") 

    for i in range(repeat):
        kolS_history.append(hist_val_sig.KolmogorovTest(hist_train_sig,"X"))
        kolB_history.append(hist_val_bkg.KolmogorovTest(hist_train_bkg,"X"))
        print(f'S={kolS_history[i]}, B={kolB_history[i]}')
    kolB_history = np.array(kolB_history)
    kolS_history = np.array(kolS_history)
    
    print(f'kolS mean = {np.mean(kolS_history)}, std = {np.std(kolS_history)}')
    print(f'kolB mean = {np.mean(kolB_history)}, std = {np.std(kolB_history)}')
    
    kolS = np.mean(kolS_history) 
    kolB = np.mean(kolB_history)

    print(f'Signal, dist = {kolS_max}, pseudo_prob={kolS}')
    print(f'Bkg, dist = {kolB_max}, pseudo_prob={kolB}')
    
    
    ##draw sample plot of KS_test
    Draw_KS_test(train_score_sig,val_score_sig,plotPath=os.path.join(plotPath,'img_sig_full.png'),isSig=True, kol=kolS)
    Draw_KS_test(train_score_bkg,val_score_bkg,plotPath=os.path.join(plotPath,'img_bkg_full.png'),isSig=False,kol=kolB)
    
    return kolS, kolB



def getECDF(data1, data2):
    y1 = []
    y2 = []
    length1 = len(data1)
    length2 = len(data2)
    a=min(min(data1),min(data2))
    b=max(max(data1),max(data2))
    width = b-a
    a = a - width*0.3
    b=b + width*0.3
    x = np.linspace(a,b,100000)
    for i in x:
        smaller1 = (data1<i).sum()
        y1.append(smaller1/length1)
        smaller2 = (data2<i).sum()
        y2.append(smaller2/length2)
    return np.array(x), np.array(y1), np.array(y2)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def Draw_KS_test(train_score,val_score,plotPath="img.png",isSig=True,kol=0):
    from matplotlib import pyplot as plt
    from scipy.stats import ks_2samp
    # x,y_1,y_2 = getECDF(train_score,val_score)
    # plt.clf() 
    # plt.plot(x,y_1)
    # plt.plot(x,y_2)
    # result = ks_2samp(train_score,val_score)
    # location = result.statistic_location
    # dist = result.statistic
    # plt.title(f"Signal, dist={dist}" if isSig else f"Bkg., dist={dist}")
    # x1, x1_index = find_nearest(x,location)
    # x2, x2_index = find_nearest(x,location)
    # y1 = y_1[x1_index]
    # y2 = y_2[x2_index]
    # plt.vlines(x1, y1, y2,color='black', linestyle='--',linewidth=2)
    # plt.savefig(plotPath)
    # plt.clf()
    
    x, y_1, y_2 = getECDF(train_score, val_score)

    trace1 = go.Scatter(x=x, y=y_1, name="Train")
    trace2 = go.Scatter(x=x, y=y_2, name="Validation")

    result = ks_2samp(train_score, val_score)
    location = result.statistic_location
    dist = result.statistic

    x1, x1_index = find_nearest(x, location)
    x2, x2_index = find_nearest(x, location)
    y1 = y_1[x1_index]
    y2 = y_2[x2_index]

    trace3 = go.Scatter(x=[x1, x2], y=[y1, y2], mode="lines", name="KS test")

    title = f"Signal, dist={dist}, kol={kol}" if isSig else f"Bkg., dist={dist}, kol={kol}"
    layout = go.Layout(
        title=title,
        xaxis=dict(title="Score"),
        yaxis=dict(title="ECDF")
    )

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    fig.write_image(plotPath)
