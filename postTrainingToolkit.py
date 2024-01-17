import numpy as np
import pandas as pd
import os
import ROOT
import itertools
import plotly.graph_objs as go
from sklearn.metrics import roc_curve, roc_auc_score

ROOT.gStyle.SetOptStat(0)
ROOT.EnableImplicitMT(16)
# ROOT.gROOT.LoadMacro(os.path.join(os.environ["DIR_PATH"], "tdrStyle.C"))
ROOT.gROOT.LoadMacro("/data6/Users/yeonjoon/VcbMVAStudy/tdrStyle.C")

ROOT.gROOT.ProcessLine("setTDRStyle();")


def shape_test(set1, set2):
    return True if set1.shape[0] == set2.shape[0] else False


def shapeCheck(data_list):
    shape_matched = True
    for p in itertools.combinations(data_list, 2):
        shape_matched &= shape_test(p[0], p[1])
    return shape_matched


def npcoversion(*args):
    for data in args:
        data = np.array(data)


def seperate_sig_bkg(df, branch="", target_Branch="y"):
    sig_value = df[df[target_Branch] == 1][branch].values
    bkg_value = df[df[target_Branch] == 0][branch].values
    sig_idx = df[df[target_Branch] == 1][branch].index
    bkg_idx = df[df[target_Branch] == 0][branch].index

    return sig_value, bkg_value, sig_idx, bkg_idx


def ROC_AUC(score, y, plot_path, weight=None):
    # Compute the ROC curve
    fpr, tpr, _ = roc_curve(y, score)  # , sample_weight=weight)
    unique_fpr, unique_indices = np.unique(fpr, return_index=True)
    tpr = tpr[unique_indices]
    np.append(tpr, 1)
    np.append(unique_fpr, 1)

    # Compute the AUC
    y = np.array(y)
    score = np.array(score)
    weight = np.array(weight)

    sorted_index = score.argsort()

    score = score[sorted_index]
    y = y[sorted_index]
    weight = weight[sorted_index]

    score = score[weight > 0]
    y = y[weight > 0]
    weight = weight[weight > 0]

    unique_score, unique_score_indices = np.unique(score, return_index=True)
    y = y[unique_score_indices]
    weight = weight[unique_score_indices]

    roc_auc = roc_auc_score(y, unique_score, sample_weight=weight)
    # Create a Plotly figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], name="Random guessing", line=dict(dash="dash"))
    )
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC curve (AUC = %0.2f)" % roc_auc))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="Receiver Operating Characteristic (ROC) Curve",
    )

    # Save the figure as a PNG file
    fig.write_image(os.path.join(plot_path, "ROC.png"), width=800, height=600)


def NormalizeHist(sig, bkg):
    if sig.GetSumw2N() == 0:
        sig.Sumw2()
    if bkg.GetSumw2N() == 0:
        bkg.Sumw2()

    if sig.GetSumOfWeights() != 0:
        dx = (sig.GetXaxis().GetXmax() - sig.GetXaxis().GetXmin()) / sig.GetNbinsX()
        sig.Scale(1.0 / sig.GetSumOfWeights() / dx)

    if bkg.GetSumOfWeights() != 0:
        dx = (bkg.GetXaxis().GetXmax() - bkg.GetXaxis().GetXmin()) / bkg.GetNbinsX()
        bkg.Scale(1.0 / bkg.GetSumOfWeights() / dx)


def KS_test(
    train_score=[],
    val_score=[],
    train_w=[],
    val_w=[],
    train_y=[],
    val_y=[],
    plotPath="",
):
    npcoversion(train_score, val_score, train_w, val_w, train_y, val_y)
    if not shapeCheck([train_score, train_w, train_y]):
        print(f"Shape mismatching")
        return
    if not shapeCheck([val_score, val_w, val_y]):
        print(f"Shape mismatching")
        return

    train_df = pd.DataFrame({"score": train_score, "weight": train_w, "y": train_y})
    val_df = pd.DataFrame({"score": val_score, "weight": val_w, "y": val_y})
    print(train_df)
    print(val_df)
    # split pandas dataframe to bkg and sig, store as ndarray.

    val_score_sig, val_score_bkg, _, _ = seperate_sig_bkg(val_df, branch="score")
    train_score_sig, train_score_bkg, _, _ = seperate_sig_bkg(train_df, branch="score")

    val_w_sig, val_w_bkg, _, _ = seperate_sig_bkg(val_df, branch="weight")
    train_w_sig, train_w_bkg, _, _ = seperate_sig_bkg(train_df, branch="weight")

    val_y_sig, val_y_bkg, _, _ = seperate_sig_bkg(val_df, branch="y")
    train_y_sig, train_y_bkg, _, _ = seperate_sig_bkg(train_df, branch="y")

    # convert ndarray to ROOT RDataFrame

    train_df_sig = ROOT.RDF.FromNumpy(
        {"score": train_score_sig, "weight": train_w_sig, "y": train_y_sig}
    )
    train_df_bkg = ROOT.RDF.FromNumpy(
        {"score": train_score_bkg, "weight": train_w_bkg, "y": train_y_bkg}
    )
    val_df_sig = ROOT.RDF.FromNumpy(
        {"score": val_score_sig, "weight": val_w_sig, "y": val_y_sig}
    )
    val_df_bkg = ROOT.RDF.FromNumpy(
        {"score": val_score_bkg, "weight": val_w_bkg, "y": val_y_bkg}
    )

    # define histogram model. TMVA default bin N seems 40. modify below to use different value.
    hist_model = ROOT.RDF.TH1DModel("score", "", 100, 0, 1)

    # get hist. from RDataFrame.
    hist_train_sig = train_df_sig.Histo1D(hist_model, "score", "weight")
    hist_val_sig = val_df_sig.Histo1D(hist_model, "score", "weight")
    hist_train_bkg = train_df_bkg.Histo1D(hist_model, "score", "weight")
    hist_val_bkg = val_df_bkg.Histo1D(hist_model, "score", "weight")

    # do Clone() because Histo1D method returns pointer of histogram. now T1HD stored in hist* variables.

    hist_train_sig = hist_train_sig.Clone()
    hist_val_sig = hist_val_sig.Clone()
    hist_train_bkg = hist_train_bkg.Clone()
    hist_val_bkg = hist_val_bkg.Clone()

    # doing KS test
    repeat = 1
    kolS_history = []
    kolB_history = []
    kolS_max = hist_val_sig.KolmogorovTest(hist_train_sig, "M")
    kolB_max = hist_val_sig.KolmogorovTest(hist_train_sig, "M")

    for i in range(repeat):
        kolS_history.append(hist_val_sig.KolmogorovTest(hist_train_sig, "X"))
        kolB_history.append(hist_val_bkg.KolmogorovTest(hist_train_bkg, "X"))
        print(f"S={kolS_history[i]}, B={kolB_history[i]}")
    kolB_history = np.array(kolB_history)
    kolS_history = np.array(kolS_history)

    NormalizeHist(hist_val_sig, hist_val_bkg)

    print(f"kolS mean = {np.mean(kolS_history)}, std = {np.std(kolS_history)}")
    print(f"kolB mean = {np.mean(kolB_history)}, std = {np.std(kolB_history)}")

    kolS = np.mean(kolS_history)
    kolB = np.mean(kolB_history)

    print(f"Signal, dist = {kolS_max}, pseudo_prob={kolS}")
    print(f"Bkg, dist = {kolB_max}, pseudo_prob={kolB}")

    NormalizeHist(hist_train_sig, hist_train_bkg)
    # draw plot
    c1 = ROOT.TCanvas("", "", 2400, 1600)
    c1.cd()
    hist_train_sig.SetFillColorAlpha(ROOT.kGreen + 1, 0.8)
    hist_train_sig.SetLineColorAlpha(ROOT.kGreen + 1, 1)
    hist_train_sig.SetLineWidth(3)

    hist_val_sig.SetFillColorAlpha(ROOT.kGreen + 2, 0)
    hist_val_sig.SetLineColorAlpha(ROOT.kGreen + 3, 1)
    hist_val_sig.SetLineWidth(4)

    hist_train_bkg.SetFillColorAlpha(ROOT.kOrange + 1, 0.8)
    hist_train_bkg.SetLineColorAlpha(ROOT.kOrange + 1, 1)
    hist_train_bkg.SetLineWidth(3)

    hist_val_bkg.SetFillColorAlpha(ROOT.kOrange + 2, 0)
    hist_val_bkg.SetLineColorAlpha(ROOT.kOrange + 3, 1)
    hist_val_bkg.SetLineWidth(4)

    stack = ROOT.THStack("stack", "stack")
    stack.Add(hist_train_bkg)
    stack.Add(hist_train_sig)
    stack.Add(hist_val_bkg)
    stack.Add(hist_val_sig)

    xLat, yLat = 0.25, 0.91
    xLeg, yLeg = xLat + 0.30, yLat
    leg_h = 0.03 * 4
    leg = ROOT.TLegend(xLeg, yLeg - leg_h, xLeg + 0.35, yLeg)
    leg.SetNColumns(2)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(43)
    leg.SetTextSize(24)
    leg.AddEntry(hist_train_bkg, "Train(bkg.)", "f")
    leg.AddEntry(hist_train_sig, "Train(sig.)", "f")
    leg.AddEntry(hist_val_bkg, "Validation(bkg.)", "f")
    leg.AddEntry(hist_val_sig, "Validataion(sig.)", "f")

    maximum = max(
        hist_train_bkg.GetMaximum(),
        hist_train_sig.GetMaximum(),
        hist_val_bkg.GetMaximum(),
        hist_val_sig.GetMaximum(),
    )
    maximum = maximum * 1.2
    stack.SetMaximum(maximum)

    stack.Draw("hist e1 nostack")
    # It should draw once before set label

    # Adjust the position of the text if needed
    text = ROOT.TText(0.2, 0.88, f"KS Test: sig.(bkg.) probabillity = {kolS}({kolB})")
    text.SetTextFont(62)
    text.SetTextSize(40)
    text.SetNDC()

    latex = ROOT.TLatex()
    latex.SetTextSize(0.07)
    latex.SetNDC()
    latex.SetTextAlign(13)
    latex.DrawLatex(0.16, 0.95, "CMS #bf{#it{Preliminary}}")
    latex.SetTextAlign(31)
    latex.SetTextColor(2)
    latex.SetTextAlign(11)

    stack.GetXaxis().SetTitle("Score")
    stack.GetXaxis().SetLabelSize(0.05)
    stack.GetYaxis().SetTitle("1/Events")
    stack.GetYaxis().SetLabelSize(0.05)
    stack.Draw("hist e1 nostack")

    leg.Draw("same")
    latex.Draw("same")
    text.Draw("same")

    c1.Update()
    c1.Draw()
    c1.SaveAs(os.path.join(plotPath, "overall.png"))

    # draw sample plot of KS_test
    Draw_KS_test(
        train_score_sig,
        val_score_sig,
        plotPath=os.path.join(plotPath, "img_sig_full.png"),
        isSig=True,
        kol=kolS,
    )
    Draw_KS_test(
        train_score_bkg,
        val_score_bkg,
        plotPath=os.path.join(plotPath, "img_bkg_full.png"),
        isSig=False,
        kol=kolB,
    )

    return kolS, kolB


def getECDF(data1, data2):
    y1 = []
    y2 = []
    length1 = len(data1)
    length2 = len(data2)
    a = min(min(data1), min(data2))
    b = max(max(data1), max(data2))
    width = b - a
    a = a - width * 0.3
    b = b + width * 0.3
    x = np.linspace(a, b, 100000)
    for i in x:
        smaller1 = (data1 < i).sum()
        y1.append(smaller1 / length1)
        smaller2 = (data2 < i).sum()
        y2.append(smaller2 / length2)
    return np.array(x), np.array(y1), np.array(y2)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def Draw_KS_test(train_score, val_score, plotPath="img.png", isSig=True, kol=0):
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

    title = (
        f"Signal, dist={dist}, kol={kol}" if isSig else f"Bkg., dist={dist}, kol={kol}"
    )
    layout = go.Layout(title=title, xaxis=dict(title="Score"), yaxis=dict(title="ECDF"))

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    fig.write_image(plotPath)
