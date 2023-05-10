#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-a', dest="Analyser", default='')
parser.add_argument('-e', dest='Era', default='2018')
parser.add_argument('-ch', dest='Channel', default='Mu')
parser.add_argument('-data', action='store_true', default="")
parser.add_argument('-mc', action='store_true', default="")
parser.add_argument('-target', dest='target', default='All')
args = parser.parse_args()

if args.Era=="2016a": args.Era="2016preVFP"
if args.Era=="2016b": args.Era="2016postVFP"

target_path_dict = {'Syst':'Central_Syst',
                  'JecDown':'JetEnDown', 'JecUp':'JetEnUp', 
                  'JerDown':'JetResDown', 'JerUp':'JetResUp'}
if args.Channel == 'El' or args.Channel == 'ME' or args.Channel == "EE":
    target_path_dict['EecDown'] = 'ElectronEnDown'
    target_path_dict['EecUp'] = 'ElectronEnUp'
    target_path_dict['EerDown'] = 'ElectronResDown'
    target_path_dict['EerUp'] = 'ElectronResUp'



mc_list = ['TTLJ_powheg',
           'TTLL_powheg',
           'SingleTop_tW_top_NoFullyHad',
           'SingleTop_tW_antitop_NoFullyHad',
           'DYJets_MG',
           'WJets_MG',
           'QCD_bEnriched_HT100to200',
           'QCD_bEnriched_HT200to300',
           'QCD_bEnriched_HT300to500',
           'QCD_bEnriched_HT500to700',
           'QCD_bEnriched_HT700to1000',
           'QCD_bEnriched_HT1000to1500',
           'QCD_bEnriched_HT1500to2000',
           'QCD_bEnriched_HT2000toInf',
           'TTLJ_WtoCB_powheg']

period = []
if args.Era == "2016preVFP": period = ['B_ver2', 'C', 'D', 'E', 'F']
elif args.Era == "2016postVFP": period = ['F', 'G', 'H']
elif args.Era == "2017": period = ['B', 'C', 'D', 'E', 'F']
elif args.Era == "2018": period = ['A', 'B', 'C', 'D']
 
single_muon_list = ['SingleMuon_' + p for p in period]
single_electron_list = ['EGamma_' + p for p in period]

data_list = []
if args.Channel == "Mu": data_list = single_muon_list
if args.Channel == "El": data_list = single_electron_list
if args.Channel == "MM" or args.Channel == "ME" or args.Channel == "EE" : data_list = single_muon_list + single_electron_list
 

out_base = ""
if args.Analyser == "Vcb": out_base = f"/data6/Users/isyoon/SKFlatOutput/Run2UltraLegacy_v3/Vcb/{args.Era}"
elif args.Analyser ==  "Vcb_DL": out_base = f"/data6/Users/isyoon/SKFlatOutput/Run2UltraLegacy_v3/Vcb_DL/{args.Era}" 
target_base = f"/data6/Users/isyoon/Vcb_Post_Analysis/Sample/{args.Era}/{args.Channel}"

import shutil
import os.path

if args.mc == True:
    for target in target_path_dict.keys():
        if args.target == target or args.target == 'All':
            out_path = f"{out_base}/Run{args.Channel}__RunResult__Run{target}__"
            
            for mc in mc_list:
                des = ""
                if args.Analyser == "Vcb": des = f"{out_path}/Vcb_{mc}.root"  
                elif args.Analyser == "Vcb_DL": des =  f"{out_path}/Vcb_DL_{mc}.root"
                
                #goal = f"{target_base}/RunResult/{target_path_dict[target]}/Vcb_{mc}.root"
                goal = f"{target_base}/RunResult/{target_path_dict[target]}/"
                
                print(des, goal, "\n")
    
                if os.path.isfile(des):
                    os.makedirs(os.path.dirname(goal), exist_ok=True)
                    shutil.copy(des, goal)
                else: print("Can't find the file") 

                    
    #Top syst      
    if args.target == 'Top_Syst' or args.target == 'All':
        out_path = f"{out_base}/Run{args.Channel}__RunResult__"
       
        f_list = ['CP5Down', 'CP5Up', 'hdampDown', 'hdampUp', 'mtop171p5', 'mtop173p5']
                        
        for r_file in f_list:
            des = ""
            if args.Analyser == "Vcb": des = f"{out_path}/Vcb_TTLJ_powheg_{r_file}.root"
            elif args.Analyser == "Vcb_DL": des = f"{out_path}/Vcb_DL_TTLJ_powheg_{r_file}.root"

            goal = f"{target_base}/RunResult/Top_Syst/"
            print(des, goal, "\n")
            
            if os.path.isfile(des):
                os.makedirs(os.path.dirname(goal), exist_ok=True)
                shutil.copy(des, goal)
            else: print("Can't find the file")


if args.data == True:
    out_path = f"{out_base}/Run{args.Channel}__RunResult__/DATA"

    for data in data_list:
        des = ""
        if args.Analyser == "Vcb": des = f"{out_path}/Vcb_{data}.root"
        elif args.Analyser == "Vcb_DL": des = f"{out_path}/Vcb_DL_{data}.root"
        
        goal = f"{target_base}/RunResult/DATA/"
        print(des, goal, "\n")

        if os.path.isfile(des):
            os.makedirs(os.path.dirname(goal), exist_ok=True)
            shutil.copy(des, goal)
        else: print("Can't find the file")

        
