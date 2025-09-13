from contextlib import contextmanager, nullcontext
import re
import os, sys, argparse, shutil
from time import perf_counter

from pytorch_tabnet.callbacks import Callback

import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import scipy
import scipy.sparse as sp
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.utils import SparsePredictDataset, PredictDataset
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingWarmRestarts

# from pytorch_tabnet
#from imblearn.over_sampling import SMOTENC
import torch
from tqdm.rich import tqdm
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.metrics import Metric
import tqdm
import hashlib
import ROOT
from sklearn.metrics import roc_auc_score
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from typing import Iterable, Optional, Dict, Any
era = ''
sys.path.append(os.environ["DIR_PATH"])
sys.path.append(os.environ["DIR_PATH"] + "/Class-balanced-loss-pytorch")
from root_data_loader import load_data, classWtoSampleW
from class_balanced_loss import CB_loss

count_0 = -1
count_1 = -1
counts = []
gamma = -999.0
no_of_classes = 4
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def focal_loss(y_pred, y_true, y_weight=None, force_use_cpu=False):
    # y_pred = torch.nn.Softmax(dim=-1)(y_pred)
    cb_loss = CB_loss(
        y_true,
        y_pred,
        #[count_0, count_1],
        counts,
        #no_of_classes=2,
        no_of_classes=no_of_classes,
        device=device,
        gamma=gamma,
        evt_weight=y_weight,
        force_use_cpu=force_use_cpu,
    )
    return cb_loss


class focal_loss_metric(Metric):
    def __init__(self):
        self._name = "focal_loss_metric"
        self._maximize = False

    def __call__(self, y_true, y_score, y_w):
        # in this, y_score is final output of model, which is already softmaxed.
        # but focal_loss has softmax inside it.
        # so we have to inverse it...

        y_score[y_score <= 1e-9] = 1e-9
        y_score = np.log(y_score)

        mse = focal_loss(
            torch.tensor(y_score).float(),
            torch.tensor(y_true).to(device),
            torch.tensor(y_w).to(device),
            force_use_cpu=True,
        )

        return mse.detach().numpy().tolist()


class WeightedAUC(Metric):
    def __init__(self):
        self._name = "WeightedAUC"
        self._maximize = True

    def __call__(self, y_true, y_score, y_w):
        # Compute the AUC
        y_score = y_score[:, 1]
        y = np.array(y_true)
        score = np.array(y_score)
        weight = np.array(y_w)

        score = score[weight > 0]
        y = y[weight > 0]
        weight = weight[weight > 0]

        sorted_index = score.argsort()

        score = score[sorted_index]
        y = y[sorted_index]
        weight = weight[sorted_index]

        unique_score, unique_score_indices = np.unique(score, return_index=True)
        y = y[unique_score_indices]
        weight = weight[unique_score_indices]

        roc_auc = roc_auc_score(y, unique_score, sample_weight=weight)
        return roc_auc



class MaxSignificance(Metric):
    """
    Metric: score∈[0,1]에서 상위 컷(threshold)을 스캔하여 최대 유의도 반환.
      - bins: 100 -> np.linspace(0,1,101)
      - mode: 'asimov' | 's_sqrt_sb' | 's_sqrt_b'
      - clamp_nonneg: 누적 s,b<0를 0으로 클램프(음수 가중치 대응)
    반환값: 최대 Z (float)
    추가 정보: self.best_threshold, self.best_s, self.best_b 에 저장
    """
    def __init__(self, bins=100, mode='asimov', clamp_nonneg=False, clip01=False):
        self._name = f"MaxSignificance[{mode}]"
        self._maximize = True
        self.bins = int(bins)
        self.mode = mode
        self.clamp_nonneg = clamp_nonneg
        self.clip01 = clip01

        # extras
        self.best_threshold = None
        self.best_s = None
        self.best_b = None
        self.curve_ = None    # Z(threshold) 배열
        self.edges_ = None    # bin edges

    def _asimov_z(self, s, b):
        z = np.zeros_like(s, dtype=np.float64)
        m1 = (s > 0) & (b > 0)
        z[m1] = np.sqrt(2.0 * ((s[m1] + b[m1]) * np.log(1.0 + s[m1] / b[m1]) - s[m1]))
        m2 = (s > 0) & (b == 0)
        z[m2] = np.sqrt(2.0 * s[m2])  # B≈0 극한
        return z

    def _z_from_sb(self, s, b):
        if self.mode == 's_sqrt_sb':
            denom = np.sqrt(s + b)
        else:  # 's_sqrt_b'
            denom = np.sqrt(b)
        z = np.divide(s, denom, out=np.zeros_like(s, dtype=np.float64), where=denom > 0)
        return z

    def __call__(self, y_true, y_score, y_w):
        # y_score: (N,2) → 양성 확률 칼럼 사용, or (N,)
        ys = np.asarray(y_score)
        if ys.ndim == 2 and ys.shape[1] >= 2:
            score = ys[:, 0]
            score = score.astype(np.float32, copy=False)
        else:
            raise ValueError("Invalid shape for y_score")
            score = ys.astype(np.float32, copy=False)
        y = np.asarray(y_true, dtype=np.int8)
        w = np.asarray(y_w, dtype=np.float64)

        # 안전 필터 + score 클리핑
        m = np.isfinite(score) & np.isfinite(y) & np.isfinite(w)
        score, y, w = score[m], y[m], w[m]
        if self.clip01:
            score = np.clip(score, 0.0, 1.0)

        # --- binning: [0,1]을 100등분 ---
        edges = np.linspace(0.0, 1.0, self.bins + 1)
        s_mask = (y == 1) | (y == 0)
        s_hist, _ = np.histogram(score[s_mask], bins=edges, weights=w[s_mask])
        b_hist, _ = np.histogram(score[~s_mask], bins=edges, weights=w[~s_mask])

        # 위에서부터 누적(고점수 → 저점수)
        s_cum = np.cumsum(s_hist[::-1])[::-1].astype(np.float64, copy=False)
        b_cum = np.cumsum(b_hist[::-1])[::-1].astype(np.float64, copy=False)

        if self.clamp_nonneg:
            # 음수 가중치로 인해 누적이 음수가 되는 비물리 상황 방지
            s_cum = np.maximum(s_cum, 0.0)
            b_cum = np.maximum(b_cum, 0.0)

        # --- Z 계산 ---
        if self.mode == 'asimov':
            z = self._asimov_z(s_cum, b_cum)
        elif self.mode in ('s_sqrt_sb', 's_sqrt_b'):
            z = self._z_from_sb(s_cum, b_cum)
        else:
            raise ValueError("mode must be 'asimov', 's_sqrt_sb', or 's_sqrt_b'")

        if z.size == 0:
            return np.nan

        # 최대점 선택
        idx = int(np.nanargmax(z))
        best_z = float(z[idx])

        # 리포트용 속성 보관(컷은 'score >= threshold')
        self.best_threshold = float(edges[idx])
        self.best_s = float(s_cum[idx])
        self.best_b = float(b_cum[idx])
        self.curve_ = z
        self.edges_ = edges
        print(f"MaxSignificance[{self.mode}] = best_z : {best_z}, best_threshold : {self.best_threshold}")
        return best_z
    
killist = []  # cleanup list for temp files


def cleanup_temp_files():
    for file in killist:
        if os.path.exists(file):
            os.remove(file)


def generate_random_file_name(input_string):
    # Generate a hash value from the input string
    hash_value = hashlib.sha256(input_string.encode()).hexdigest()

    # Take the first 8 characters from the hash value
    random_string = hash_value

    # Combine the random string and number to form the file name
    file_name = random_string

    return file_name


print(f"device is {device}, count is {torch.cuda.device_count()}")




def get_betas(train_y, train_weight):
    print(f"Total number of training data = {train_y.shape}")
    labels, inv = np.unique(train_y, return_inverse=True)
    sums = np.bincount(inv, weights=train_weight, minlength=labels.size)
    print(f"sumW of each class is {sums}")
    global count_0
    count_0 = sums[0]
    global count_1
    count_1 = sums[1]
    print(f"{count_0} of bkg. sample, {count_1} of sig.sample")
    beta_0 = (count_0 - 1) / (count_0)
    beta_1 = (count_1 - 1) / (count_1)
    print(f"value of beta_0 is {beta_0}, beta_1 is {beta_1}")
    w_b = (1 - beta_0) / (1 - pow(beta_0, count_0))
    w_s = (1 - beta_1) / (1 - pow(beta_1, count_1))

    print(f"w_b is {w_b/(w_b+w_s)*2} w_s is {w_s/(w_b+w_s)*2}")
    global counts
    for i in range(len(sums)):
        counts.append(sums[i])
    counts = np.array(counts,np.float64)
    betas = (counts - 1) / counts
    ws = (1 - betas) / (1 - np.power(betas, counts))
    print(f"counst are {counts}, betas are {betas}, ws are {ws}")
    print(f"ws are {ws/(ws.sum())*len(ws)}")

def train(
    model_save_path,
    floss_gamma,
    result_folder_name,
    sample_folder_loc,
    pretrained_model=None,
    checkpoint=None,
    add_year_index = 0
):
    if checkpoint is not None and pretrained_model is not None:
        print(f"하나만 해라")
    fineTune = True if pretrained_model is not None else False
    if fineTune:
        model_save_path = os.path.join(model_save_path, "FineTune")
    varlist = [
        "pt_w_u",
        "pt_w_d",
        "pt_had_t_b",
        "pt_lep_t_b",
        #
        "m_had_t",
        "m_had_w",
        "bvsc_had_t_b",
        "bvsc_lep_t_b",
        "bvsc_w_u",
        "bvsc_w_d",
        "n_bjets",
    
        "n_jets",

        ##
        #   "n_cjets",
        #   "cvsb_had_t_b",
        #   "cvsb_lep_t_b",
        #   "cvsl_had_t_b",
        #   "cvsl_lep_t_b",
        #   "cvsl_w_u",
        #   "cvsl_w_d",
        #   "cvsb_w_u",
        #   "cvsb_w_d",
        
        "best_mva_score",
        ##
        "least_dr_bb",
        "least_m_bb",
        #"pt_tt",
        "ht",
        #SPANET ONLY
        # "mva_hf_score",
        # "mva_bb_score",
        # "mva_cc_score",
        ##
        "weight",

        
    ]
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
        
    REL_ISO_MUON = 0.15
    MET_PT = 25
    D_COND_MU = f"(met_pt>{MET_PT})&&(lepton_rel_iso<{REL_ISO_MUON})"
    D_COND_EL = D_COND_EL = f"(met_pt>{MET_PT})&&((electron_id_bit & (1 << 5)) != 0)"  
    HF_BB_MASK = "(((genTtbarId%100) >= 51) && ((genTtbarId%100) <= 55))"# || (((genTtbarId%100) >= 41) && ((genTtbarId%100) <= 45))";
    HF_CC_MASK = "(((genTtbarId%100) >= 41) && ((genTtbarId%100) <= 45))"# || (((genTtbarId%100) >= 41) && ((genTtbarId%100) <= 45))";
    HF_MASK = HF_BB_MASK + f"||{HF_CC_MASK}"
    
    eras = [era] if era != "All" else ["2016preVFP","2016postVFP", "2017", "2018"]
    input_tuple = ([],[],[],[],[])
    
    #input_tuple = ([], [])
    global no_of_classes
    no_of_classes = len(input_tuple)
    for e in eras:
        print(eras, era, e)
        #class 0. signal.
        input_tuple[0].append(  # first element of tuple = signal tree, second =bkg tree.
                    (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root",
                    "Mu/Central/Result_Tree",
                    f"(chk_reco_correct==0)&&{D_COND_MU}"
                    )
                )
        input_tuple[0].append(  # first element of tuple = signal tree, second =bkg tree.
                    (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root",
                    "El/Central/Result_Tree",
                    f"(chk_reco_correct==0)&&{D_COND_EL}"
                    )
                )
        
        input_tuple[1].append(  # first element of tuple = signal tree, second =bkg tree.
                    (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root",
                    "Mu/Central/Result_Tree",
                    f"(chk_reco_correct==0)&&{D_COND_MU}"
                    )
                )
        input_tuple[1].append(  # first element of tuple = signal tree, second =bkg tree.
                    (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root",
                    "El/Central/Result_Tree",
                    f"(chk_reco_correct==0)&&{D_COND_EL}"
                    )
                )

               
        input_tuple[2].append(
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "Mu/Central/Result_Tree",
                    D_COND_MU +f"&&({HF_BB_MASK})",
                )
        )

        input_tuple[2].append(
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "El/Central/Result_Tree",
                    D_COND_EL +f"&&({HF_BB_MASK})",
                )
        )
        


        input_tuple[3].append(
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "Mu/Central/Result_Tree",
                    D_COND_MU + f"&&({HF_CC_MASK})",
                )
        )

        input_tuple[3].append(
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "El/Central/Result_Tree",
                    D_COND_EL+ f"&&({HF_CC_MASK})",
                )
        )
        input_tuple[4].append(
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "Mu/Central/Result_Tree",
                    D_COND_MU + f"&&!({HF_MASK})",
                )
        )

        input_tuple[4].append(
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "El/Central/Result_Tree",
                    D_COND_EL+ f"&&!({HF_MASK})",
                )
        )

        #class 3. ttjj & non-tt bkg.

        # input_tuple[-1].append(
        #         (
        #             f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
        #             "Mu/Central/Result_Tree",
        #             #f"(!isCC && !isBB )&&(n_bjets>=3)&&{D_COND_MU}",
        #             #f"(n_bjets>=3)&&{D_COND_MU}",
        #             D_COND_MU if not fineTune else f"({D_COND_MU})&&({HF_MASK})",
        #         )
        # )

        # input_tuple[-1].append(
        #         (
        #             f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
        #             "El/Central/Result_Tree",
        #             #f"(!isCC && !isBB )&&(n_bjets>=3)&&{D_COND_EL}",
        #             #f"(n_bjets>=3)&&{D_COND_EL}",
        #             D_COND_EL if not fineTune else f"({D_COND_EL})&&({HF_MASK})",
        #         )
        # )
        

        for file in [
            "Vcb_DYJets_MG.root",
            "Vcb_QCD_bEnriched_HT1000to1500.root",
            #"Vcb_QCD_bEnriched_HT100to200.root",
            "Vcb_QCD_bEnriched_HT1500to2000.root",
            "Vcb_QCD_bEnriched_HT2000toInf.root",
            "Vcb_QCD_bEnriched_HT200to300.root",
            "Vcb_QCD_bEnriched_HT300to500.root",
            "Vcb_QCD_bEnriched_HT500to700.root",
            "Vcb_QCD_bEnriched_HT700to1000.root",
            "Vcb_SingleTop_sch_Lep.root",
            "Vcb_SingleTop_tW_antitop_NoFullyHad.root",
            "Vcb_SingleTop_tW_top_NoFullyHad.root",
            "Vcb_SingleTop_tch_antitop_Incl.root",
            "Vcb_SingleTop_tch_top_Incl.root",
            #"Vcb_TTJJ_powheg.root",
            "Vcb_TTLL_powheg.root",
            "Vcb_WJets_HT100to200.root",
            "Vcb_WJets_HT1200to2500.root",
            "Vcb_WJets_HT200to400.root",
            "Vcb_WJets_HT2500toInf.root",
            "Vcb_WJets_HT400to600.root",
            "Vcb_WJets_HT600to800.root",
            "Vcb_WJets_HT800to1200.root",
            "Vcb_WW_pythia.root",
            "Vcb_WZ_pythia.root",
            "Vcb_ZZ_pythia.root",
            "Vcb_ttHToNonbb.root",
            "Vcb_ttHTobb.root",
            "Vcb_ttWToLNu.root",
            "Vcb_ttWToQQ.root",
            "Vcb_ttZToLLNuNu.root",
            "Vcb_ttZToQQ.root",
            "Vcb_ttZToQQ_ll.root",
        ]: 
            break#not add others
            if pretrained_model is not None:
                break
            if not "TTLJ" in file:
                mu_tuple = (
                    os.path.join(
                        f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst",
                        file,
                    ),
                    "Mu/Central/Result_Tree",
                    #f"(n_bjets>=3)&&{D_COND_MU}",
                    D_COND_MU,
                )
                el_tuple = (
                    os.path.join(
                        f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst",
                        file,
                    ),
                    "El/Central/Result_Tree",
                    #f"(n_bjets>=3)&&{D_COND_EL}",
                    D_COND_EL,
                )
                input_tuple[-1].append(mu_tuple)
                input_tuple[-1].append(el_tuple)
    
    data_info = {
        "tree_path_filter_str": input_tuple,
        "varlist": varlist,
        "test_ratio": 0.1,
        "val_ratio": 0.2,
        "add_year_index": add_year_index,
        #"sample_bkg": 2
    }
    if os.path.exists(os.path.join(model_save_path, "data.npz")):
        while True:
            prompt = f"{os.path.join(model_save_path, 'data.npz')} already exists. Do you want to overwrite it? (y/n): "
            answer = input(prompt).strip().lower()
            if answer == "y":
                data = load_data(**data_info)
                np.savez_compressed(os.path.join(model_save_path, "data.npz"),
                    **{k: np.asarray(v) for k, v in data.items()})
                break
            elif answer == "n":
                
                data = np.load(os.path.join(model_save_path, "data.npz"), allow_pickle=True)
                #data = data["arr_0"][()]
                print(f"Using existing data {os.path.join(model_save_path, 'data.npz')}")
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    else:
        data = load_data(**data_info)
        np.savez_compressed(os.path.join(model_save_path, "data.npz"),
            **{k: np.asarray(v) for k, v in data.items()})



    T0=10
    if not fineTune:
        optimizer_fn = torch.optim.AdamW
        optimizer_params = dict(
            lr=2e-4,            
            weight_decay=1e-2, 
            betas=(0.9, 0.999)
        )
        scheduler_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        scheduler_params = dict(T_0=T0, T_mult=2, eta_min=1e-5, last_epoch=-1, verbose=False)
        model_info = {
            "n_d": 32,
            "n_a": 32,
            "verbose": 1,
            "cat_idxs": list(data["cat_idxs"]),
            "cat_dims": list(data["cat_dims"]),
            "cat_emb_dim": 1,
            "n_steps": 6,
            "lambda_sparse": 1e-4,
            "gamma": 1.5,
            "mask_type": "entmax",
            "device_name": str(device),
            "optimizer_fn": optimizer_fn,
            "optimizer_params": optimizer_params,
            "scheduler_fn": scheduler_fn,
            "scheduler_params": scheduler_params
        }
        clf = TabNetClassifier(**model_info)
        if checkpoint is not None:
            print(f"Loading checkpoint from {checkpoint}")
            clf.load_model(checkpoint)
            clf.network.to(device)
            clf.device_name = str(device)                
            clf.device = torch.device(clf.device_name)
    else:
        # ---- Fine-tuning ----
        # 1) AdamW: decoupled weight-decay (일반 Adam보다 regularization 안정적)
        optimizer_fn   = torch.optim.AdamW
        optimizer_params = dict(
            lr=5e-5,            # 기준 lr의 1/20 정도부터 시도
            weight_decay=1e-2,  # BN·bias는 내부에서 자동 제외 가능(AdamW 구현 따라 다름)
            betas=(0.9, 0.999)  # 기본값
        )

        # 2) CosineAnnealing *단발* 스케줄
        #    - Warm-restart 없이 T_max = 전체 fine-tune epoch
        #    - eta_min 은 정지 직전 lr
        scheduler_fn   = torch.optim.lr_scheduler.CosineAnnealingLR
        scheduler_params = dict(
            T_max=30,           # fine-tune epoch 수와 동일하게 맞추기
            eta_min=1e-6,
            last_epoch=-1,
            verbose=False
        )

        clf = TabNetClassifier()
        clf.load_model(pretrained_model)
        clf.device_name = str(device)                
        clf.device = torch.device(clf.device_name)   
        clf.network.to(clf.device)    
        clf.network.to(device)
        # replace optimizer and scheduler
        clf.optimizer_fn = optimizer_fn
        clf.scheduler_fn = scheduler_fn
        clf.optimizer_params = optimizer_params
        clf.scheduler_params = scheduler_params
        model_info = {
            "n_d": clf.n_d,
            "n_a": clf.n_a,
            "verbose": clf.verbose,
            "cat_idxs": clf.cat_idxs,
            "cat_dims": clf.cat_dims,
            "cat_emb_dim": clf.cat_emb_dim,
            "n_steps": clf.n_steps,
            "lambda_sparse": clf.lambda_sparse,
            "gamma": clf.gamma,
            "mask_type": clf.mask_type,
            "device_name": clf.device_name,
            "optimizer_fn": clf.optimizer_fn,
            "optimizer_params": clf.optimizer_params,
            "scheduler_fn": clf.scheduler_fn,
            "scheduler_params": clf.scheduler_params
        }

    global gamma
    gamma = floss_gamma

    
    get_betas(train_y=data["train_y"], train_weight=data["train_weight"])
    batch_size = 8192 * 16
    num_min_batch = 32
    class SaveEachEpochCallback(Callback):
        def __init__(self, save_dir="checkpoints"):
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)

        def on_epoch_end(self, epoch, logs=None):
            opt = getattr(self.trainer, "optimizer", None) or getattr(self.trainer, "_optimizer", None)
            sch = getattr(self.trainer, "scheduler", None) or getattr(self.trainer, "_scheduler", None)

            if sch is not None and hasattr(sch, "get_last_lr"):
                lrs = sch.get_last_lr()            # 스케줄러 기준(권장)
            elif opt is not None:
                lrs = [g["lr"] for g in opt.param_groups]  # 옵티마이저에서 직접
            else:
                lrs = ["(unknown)"]
            print(f"[epoch {epoch}] lr: {lrs}")
            filename = os.path.join(self.save_dir, f"model_epoch{epoch:03d}")
            # TabNet은 zip archive로 저장됨
            self.trainer.save_model(filename)
            print(f"[Checkpoint] Saved model at {filename}.zip")
    train_info = {
        "X_train": data["train_features"],
        "y_train": data["train_y"],
        "w_train": data["train_weight"],
        "eval_set": [(data["val_features"], data["val_y"], data["val_weight"])],
        # "eval_metric": ["balanced_accuracy", "WeightedMSE", "auc"],
        "eval_metric": ["MaxSignificance[asimov]"],
        #"eval_metric": [WeightedAUC, focal_loss_metric],
        "max_epochs": 1000,
        "num_workers": 16,
        "pin_memory": True,
        ### weights parameter is depricated. use w_train instead.
        #'weights':1,
        # "weights": data["train_weight"],  # data['train_sample_and_class_weight'],
        # "weights": 0,
        "batch_size": batch_size,  # int(2097152/16),#1024,,#8192,#int(2097152/16),
        "virtual_batch_size": batch_size // num_min_batch,
        # augmentations=aug,
        "patience": 2*T0,
        "loss_fn": focal_loss,
        # callbacks=[pytorch_tabnet.callbacks.History(clf,verbose=1)]
        "callbacks" :[SaveEachEpochCallback(save_dir=f"{model_save_path}/checkpoints")],
        "compute_importance" : False
    }

    train_info_save = {} 
    for k,v in train_info.items():
        if k in ["X_train", "y_train", "w_train","eval_set", "loss_fn", "eval_metric", "callbacks"]:
            continue
        train_info_save[k] = v
        
    

    file_name = os.path.join(model_save_path, "info.txt")
    # Open the file for writing
    with open(file_name, "w") as file:
        file.write(f"Training info\n")
        for key, value in train_info.items():
            file.write(f"{key}: {value}\n")
        file.write(f"gamma: {gamma}\n")
        file.write(f"###########################################################\n")
        file.write(f"model info\n")
        for key, value in model_info.items():
            file.write(f"{key}: {value}\n")

        file.write(f"###########################################################\n")
        file.write(f"data info\n")
        for key, value in data_info.items():
            file.write(f"{key}: {value}\n")

    # Close the file
    file.close()

    info_arr = {
        "train_info": train_info_save,
        "model_info": model_info,
        "data_info": data_info,
    }
    info_arr = np.array(info_arr)
    np.save(os.path.join(model_save_path, "info.npy"), info_arr)


    clf.fit(**train_info)
    clf.save_model(os.path.join(model_save_path, "model"))

    plot(model_save_path)

@contextmanager
def task(name: str):
    t0 = perf_counter()
    print(f"▶ Starting {name} ...")
    try:
        yield
    finally:
        dt = perf_counter() - t0
        print(f"✓ Finished {name} in {dt:.2f} seconds\n")


def predict_proba_fast(model, X,
                       batch_size=None,
                       num_workers=16,
                       amp=True,
                       amp_dtype=torch.float16,
                       pin_memory=True,
                       prefetch_factor=16,
                       persistent_workers=False,
                       return_numpy=True):
    
    # --- match original behavior ---
    model.network.eval()

    # DataLoader construction (reuse the same datasets as original)
    if scipy.sparse.issparse(X):
        dataset = SparsePredictDataset(X)
    else:
        dataset = PredictDataset(X)

    bs = int(batch_size) if batch_size is not None else int(model.batch_size)

    use_cuda = isinstance(model.device, torch.device) and model.device.type == "cuda" \
               or (isinstance(model.device, str) and model.device.startswith("cuda"))

    dl = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        pin_memory=(pin_memory and use_cuda),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(persistent_workers and num_workers > 0),
        drop_last=False,
    )

    out_batches = []
    torch.set_grad_enabled(False)
    # inference-only, no grad / no BN stat updates
    with torch.inference_mode():
        amp_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                   if (amp and use_cuda) else nullcontext())
        with amp_ctx:
            for data in tqdm.tqdm(dl):
                # dataset yields a Tensor directly (same as your reference)
                xb = data.to(model.device, non_blocking=use_cuda).float()
                # TabNet forward returns (logits, M_loss); we only need logits
                logits, _ = model.network(xb)
                probs = torch.softmax(logits, dim=1)
                # move to CPU now to keep GPU mem low; convert to numpy at the end
                out_batches.append(probs.detach().cpu())

    out_cpu = torch.cat(out_batches, dim=0)
    return out_cpu.numpy() if return_numpy else out_cpu
def _load_one_npz_key(path: str, key: str, allow_pickle: bool) -> tuple[str, Any]:
    # 각 워커가 파일을 새로 열어 해당 key만 압축 해제하여 메모리에 복사
    with np.load(path, allow_pickle=allow_pickle) as f:
        if key not in f.files:
            raise KeyError(f"Key '{key}' not found in {path}")
        return key, f[key].copy()

def _decide_workers(workers: Optional[int], n_items: int) -> int:
    if workers is None:
        # 너무 크지 않게 기본값 설정
        return max(1, min(8, n_items))
    return max(1, workers)


def _csr_to_torch_sparse_coo(csr: sp.csr_matrix, dtype=np.float32):
    coo = csr.tocoo()
    indices = np.vstack([coo.row, coo.col]).astype(np.int64)
    values = coo.data.astype(dtype, copy=False)
    i = torch.from_numpy(indices)
    v = torch.from_numpy(values)
    return torch.sparse_coo_tensor(i, v, size=coo.shape, dtype=torch.float32, device="cpu")

def _ensure_reducing_t(model, device):
    if hasattr(model, "_reducing_t") and model._reducing_t is not None:
        Rt = model._reducing_t
        if Rt.device != device:
            model._reducing_t = Rt.to(device)
        return model._reducing_t

    R = getattr(model, "reducing_matrix", None)
    if R is None:
        raise ValueError("model.reducing_matrix가 필요합니다.")

    if sp.issparse(R):
        R_csr = R.tocsr()
        Rt = _csr_to_torch_sparse_coo(R_csr)
    else:
        R_csr = sp.csr_matrix(np.asarray(R, dtype=np.float32))
        Rt = _csr_to_torch_sparse_coo(R_csr)

    model._reducing_t = Rt.to(device)
    return model._reducing_t

def explain_fast(model, X, normalize: bool = False, num_workers: int | None = None, desc="Explaining", batch_size=8192):
    device = torch.device(getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    if sp.issparse(X):
        dataset = SparsePredictDataset(X)
    else:
        dataset = PredictDataset(X)

    if num_workers is None:
        try:
            import os
            nw = max(2, (os.cpu_count() or 4) // 2)
        except Exception:
            nw = 2
    else:
        nw = num_workers

    pin = device.type == "cuda"
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=nw, pin_memory=pin)

    model.network.eval()
    R = _ensure_reducing_t(model, device)

    res_explain_parts = []
    res_masks_lists = None

    eps = 1e-12

    with torch.inference_mode():
        for data in tqdm.tqdm(loader, desc=desc):
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(device, non_blocking=True).float()

            M_explain, masks = model.network.forward_masks(data)

            M_red = torch.sparse.mm(R, M_explain.transpose(0, 1)).transpose(0, 1)

            if normalize:
                s = M_red.sum(dim=1, keepdim=True).clamp_min(eps)
                M_red = M_red / s

            res_explain_parts.append(M_red.cpu().numpy())

            proc_masks = {}
            for k, v in masks.items():
                v_red = torch.sparse.mm(R, v.transpose(0, 1)).transpose(0, 1)
                if normalize:
                    sv = v_red.sum(dim=1, keepdim=True).clamp_min(eps)
                    v_red = v_red / sv
                proc_masks[k] = v_red.cpu().numpy()

            if res_masks_lists is None:
                res_masks_lists = {k: [proc_masks[k]] for k in proc_masks.keys()}
            else:
                for k in proc_masks.keys():
                    res_masks_lists[k].append(proc_masks[k])

    res_explain = np.concatenate(res_explain_parts, axis=0)
    res_masks = {k: np.concatenate(vlist, axis=0) for k, vlist in res_masks_lists.items()}

    return res_explain, res_masks

def materialize_npz_parallel_auto(
    path: str | Path,
    *,
    include: Optional[Iterable[str]] = None,   # 로딩할 키를 제한하고 싶으면 지정
    exclude: Optional[Iterable[str]] = None,   # 빼고 싶은 키
    prefix: Optional[str] = None,              # 접두어로 필터링 (ex. 'train_')
    use_process: bool = True,                  # CPU 압축해제 가속용 (멀티프로세스 권장)
    workers: Optional[int] = None,             # 워커 수 (미지정 시 키 개수 기반)
    allow_pickle: bool = False,                # 객체 배열이면 True로
) -> Dict[str, Any]:
    """
    `.npz` 파일의 키를 자동으로 수집해 병렬 materialize하여 {key: ndarray} dict 반환.
    """
    path = str(path)

    # 1) 전체 키 수집
    with np.load(path, allow_pickle=allow_pickle) as f:
        keys = list(f.files)

    # 2) 필터 적용 (include/exclude/prefix)
    keyset = set(keys)
    if include is not None:
        inc = set(include)
        keys = [k for k in keys if k in inc]
    if exclude is not None:
        exc = set(exclude)
        keys = [k for k in keys if k not in exc]
    if prefix is not None:
        keys = [k for k in keys if k.startswith(prefix)]

    if not keys:
        raise ValueError("No keys selected to load (check include/exclude/prefix filters).")

    # 3) 병렬 로딩
    n_workers = _decide_workers(workers, len(keys))
    Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor

    results: Dict[str, Any] = {}
    with Executor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(_load_one_npz_key, path, k, allow_pickle): k
            for k in keys
        }
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                key, arr = fut.result()
                results[key] = arr
            except Exception as e:
                # 필요하면 여기서 로깅/재시도 전략 추가 가능
                raise RuntimeError(f"Failed to load key '{k}': {e}") from e

    return results

def plot(model_save_path, checkpoint_path=None):
    import postTrainingToolkit

    BATCH_LOAD = 8192 * 64 if checkpoint_path is None else 8192 * 64

    with task("Loading dataset (data.npz)"):
        #data = materialize_npz_parallel_auto(os.path.join(model_save_path, "data.npz"), allow_pickle=True)
        data = np.load(os.path.join(model_save_path, "data.npz"), allow_pickle=True)
    with task("Model loading"):
        if checkpoint_path is None:
            files = os.listdir(model_save_path)
            pt_zip_files = [f for f in files if f.endswith(".zip")]
            if not pt_zip_files:
                raise FileNotFoundError("No model .zip file found.")
            model = TabNetClassifier()
            model.load_model(os.path.join(model_save_path, pt_zip_files[0]))
            print(f"  - Loaded from: {pt_zip_files[0]}")
        else:
            print(f"  - Loaded from: {checkpoint_path}")
            model = TabNetClassifier()
            model.load_model(checkpoint_path)
            m = re.search(r"model_epoch(\d+)\.zip$", checkpoint_path)
            epoch_str = m.group(1)   # '000'
            epoch_int = int(epoch_str)  
            model_save_path = os.path.join(model_save_path, 'plot' ,f"ckpt{epoch_str}")
            os.makedirs(model_save_path, exist_ok=True)
            
    with task("ROC AUC evaluation and plotting"):
        y_inv = np.logical_not(data["test_y"]).astype(int)
        proba0 = predict_proba_fast(model, data["test_features"], batch_size=BATCH_LOAD)[:, 0]
        #proba0 = model.predict_proba(data["test_features"])[:, 0]
        postTrainingToolkit.ROC_AUC(
            score=proba0,
            y=y_inv,
            plot_path=model_save_path,
            #weight=data["test_weight"],
        )

    with task("Train/Validation predictions"):
        train_score = predict_proba_fast(model,data["train_features"], batch_size=BATCH_LOAD)
        val_score   = predict_proba_fast(model,data["val_features"], batch_size=BATCH_LOAD)
        #train_score = model.predict_proba(data["train_features"])
        #val_score   = model.predict_proba(data["val_features"])

    from itertools import combinations
    EPS = 1e-9
    # with task("KS test and plotting"):
    #     num_class = train_score.shape[1]
    #     print(f"Number of classes: {num_class}")
    #     for sig_idx, bkg_idx in list(combinations(range(num_class), 2)):
    #         train_sig_mask = data["train_y"] == sig_idx
    #         train_bkg_mask = data["train_y"] == bkg_idx
    #         val_sig_mask = data["val_y"] == sig_idx
    #         val_bkg_mask = data["val_y"] == bkg_idx
            
          

    #         kolS, kolB = postTrainingToolkit.KS_test(
    #             train_score=np.concatenate((train_score[train_sig_mask, sig_idx], train_score[train_bkg_mask, bkg_idx]), axis=0),
    #             val_score=np.concatenate((val_score[val_sig_mask, sig_idx], val_score[val_bkg_mask, bkg_idx]), axis=0),
    #             train_w=np.concatenate((data["train_weight"][train_sig_mask], data["train_weight"][train_bkg_mask]), axis=0),
    #             val_w=np.concatenate((data["val_weight"][val_sig_mask], data["val_weight"][val_bkg_mask]), axis=0),
    #             train_y=np.concatenate((np.zeros(train_score[train_sig_mask].shape[0]), np.ones(train_score[train_bkg_mask].shape[0])), axis=0),
    #             val_y=np.concatenate((np.zeros(val_score[val_sig_mask].shape[0]), np.ones(val_score[val_bkg_mask].shape[0])), axis=0),
    #             plotPath=model_save_path,
    #             postfix=f"score_{sig_idx}_sig_{sig_idx}_bkg_{bkg_idx}"
    #         )
    #         print(f"  - KS statistic (Signal): {kolS}, (Background): {kolB}")
    with task("KS test and plotting"):
        num_class = train_score.shape[1]
        print(f"Number of classes: {num_class}")

        
        for sig_idx, bkg_idx in combinations(range(num_class), 2):
            # 두 클래스만 추려서 pair-wise 스코어 만들기
            tr_mask = (data["train_y"] == sig_idx) | (data["train_y"] == bkg_idx)
            vl_mask = (data["val_y"]   == sig_idx) | (data["val_y"]   == bkg_idx)
            sig_mask = data["train_y"] == sig_idx
            bkg_mask = data["train_y"] == bkg_idx

            # pair-wise score: p_sig / (p_sig + p_bkg)
            tr_p_sig = train_score[tr_mask, sig_idx]
            tr_p_bkg = train_score[tr_mask, bkg_idx]
            vl_p_sig =  val_score[vl_mask,  sig_idx]
            vl_p_bkg =  val_score[vl_mask,  bkg_idx]

            tr_s = tr_p_sig / (tr_p_sig + tr_p_bkg + EPS)
            vl_s = vl_p_sig / (vl_p_sig + vl_p_bkg + EPS)

            # 라벨: sig=0, bkg=1 (KS_test가 내부에서 분리)
            tr_y = np.where(data["train_y"][tr_mask] == sig_idx, 0, 1).astype(np.int8)
            vl_y = np.where(data["val_y"][vl_mask]   == sig_idx, 0, 1).astype(np.int8)

            tr_w = data["train_weight"][tr_mask]
            vl_w = data["val_weight"][vl_mask]

            kolS, kolB = postTrainingToolkit.KS_test(
                train_score=tr_s,
                val_score=vl_s,
                train_w=tr_w,
                val_w=vl_w,
                train_y=tr_y,
                val_y=vl_y,
                plotPath=model_save_path,
                postfix=f"pair_sig{sig_idx}_bkg{bkg_idx}",
                use_weight=False
            )
            print(f"  - p-value (Signal): {kolS:.3g}, (Background): {kolB:.3g}")

    with task("Explainability (masks) and saving"):
        res_explain, res_masks = explain_fast(model, data["test_features"], normalize=False, batch_size = BATCH_LOAD)
        np.save(os.path.join(model_save_path, "explain.npy"), res_explain)
        np.save(os.path.join(model_save_path, "mask.npy"),    res_masks)
        np.save(os.path.join(model_save_path, "y.npy"),       data["train_y"])
        M_explain = res_explain
        sum_explain = M_explain.sum(axis=0)
        feature_importances_ = sum_explain / np.sum(sum_explain)
        print(feature_importances_)


    with task("Variable list loading and cleaning"):
        info_arr = np.load(os.path.join(model_save_path, "info.npy"), allow_pickle=True)
        info_arr = info_arr[()]
        varlist = list(info_arr['data_info']['varlist'])
        if "weight" in varlist:
            varlist.remove("weight")

    with task("Feature importance plotting (mplhep)"):
        if len(varlist) != len(feature_importances_):
            varlist.append("year_index")
        postTrainingToolkit.draw_feature_importance_mplhep(
            varlist=varlist,
            feature_importance=feature_importances_,
            plot_dir=model_save_path,
            fname_base="feature_importance",
        )

def predict_log_proba(model, X):
        """
        Make predictions for classification on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        res : np.ndarray

        """
        model.network.eval()

        if scipy.sparse.issparse(X):
            dataloader = DataLoader(
                SparsePredictDataset(X),
                batch_size=model.batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=model.batch_size,
                shuffle=False,
            )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(model.device).float()

            output, M_loss = model.network(data)
            predictions = torch.nn.LogSoftmax(dim=1)(output).cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return res
    
def infer_and_write(root_file, input_model_path, new_branch_name, model_folder):
    try:
        import array, tqdm
        model = TabNetClassifier()
        model.load_model(input_model_path)
        data_info = np.load(os.path.join(model_folder, "info.npy"), allow_pickle=True)
        data_info = data_info[()]
        data_info = data_info["data_info"]
        varlist = data_info["varlist"]
        add_year_index = data_info["add_year_index"]

        input_tuple = ([(root_file, "Result_Tree", "")],[])
        if "weight" in varlist:
            varlist.remove("weight")
        if add_year_index:
            data = load_data(
                tree_path_filter_str=input_tuple, varlist=varlist, test_ratio=0, val_ratio=0,useLabelEncoder=False,add_year_index=1
            )        
        else:
            data = load_data(
                tree_path_filter_str=input_tuple, varlist=varlist, test_ratio=0, val_ratio=0,useLabelEncoder=False
            )
        print("Data loaded")
        arr = data["train_features"]

        pred = predict_log_proba(model, arr)
        num_class = pred.shape[1]
        print("infer is done. start writing...")

        root_file = ROOT.TFile.Open(root_file, "UPDATE")

        ##########################################################
        ############## classical iterator
        ##########################################################
        tree = root_file.Get("Result_Tree")
        bufs = []
        branchs = []
        for cls in range(num_class):
            bufs.append(array.array("f", [0.0]))
            branchs.append(tree.Branch(new_branch_name + f"_log_prob_{cls}", bufs[cls], new_branch_name + f"_logit_{cls}" + "/F"))

        
        for i in range(tree.GetEntries()):
            for cls in range(num_class):
                bufs[cls][0] = float(pred[i, cls])
                branchs[cls].Fill()
        tree.Write("", ROOT.TObject.kOverwrite)
        root_file.Close()
    except Exception as e:
        # Log the exception
        return f"Error {str(e)} occurred while processing {root_file}"
    return f"Successfully processed {root_file}"

def infer(input_root_file, input_model_path, branch_name="template_score"):
    import array, shutil, time

    start_time = time.time()
    # Absolute path to the input file
    input_root_file = os.path.abspath(input_root_file)
    print(input_model_path)

    model_folder = "/".join(input_model_path.split("/")[:-1])
    # ROOT.EnableImplicitMT() ImplicitMT should not be used
    #model = TabNetClassifier()
    #model.load_model(input_model_path)
    outname = input_root_file.split("/")
    outname[-1] = outname[-1].replace(".root", "")
    outname = "_".join(outname[-5:])

    try:
        new_branch_name = branch_name
        import torch.multiprocessing as mp
        #import multiprocessing as mp
        # set to log all info 
        #import logging
        #mp.log_to_stderr(logging.DEBUG)
        
        mp.set_start_method("spawn")
        print(f"Start to process {input_root_file}")
        input_file = ROOT.TFile.Open(input_root_file, "READ")
        print(f"Opened {input_root_file}")
        output_files = []
        for ch in input_file.GetListOfKeys():
            if isinstance(ch.ReadObj(), ROOT.TDirectory):
                chdirname = input_root_file.replace(".root", f"/{ch.GetName()}")
                print(f"Will create a new directory: {chdirname}")
                os.makedirs(chdirname, exist_ok=True)
                for key in ch.ReadObj().GetListOfKeys():
                    obj = key.ReadObj()
                    if isinstance(obj, ROOT.TDirectory):
                        # Create an output file for each TDirectory
                        output_file_name = chdirname + '/' + f"{obj.GetName()}.root"
                        output_file = ROOT.TFile.Open(output_file_name, "RECREATE")
                        output_files.append(output_file_name)

                        # Enter the TDirectory and copy its contents to the new file

                        input_file.cd(chdirname.split('/')[-1] + '/' + obj.GetName())
                        for inner_key in ROOT.gDirectory.GetListOfKeys():
                            inner_obj = inner_key.ReadObj()

                            output_file.cd()
                            prefix = str(new_branch_name)

                            # (선택) 실제로 어떤 브랜치가 대상인지 로그로 확인
                            if inner_obj.InheritsFrom("TTree"):
                                tree = inner_obj
                                # 어떤 브랜치가 지워질지 미리 로깅
                                branches = tree.GetListOfBranches()
                                names = [branches.At(i).GetName() for i in range(branches.GetSize())]
                                matches = [n for n in names if n.startswith(prefix)]

                                if matches:
                                    print(f"[{ch.GetName()}/{obj.GetName()}] "
                                        f"Tree '{tree.GetName()}': delete {len(matches)} branches with prefix '{prefix}': {matches}")
                                    # 1) 모두 활성화
                                    tree.SetBranchStatus("*", 1)
                                    # 2) prefix* 비활성화
                                    tree.SetBranchStatus(prefix + "*", 0)
                                    # 3) 활성 브랜치만 복제
                                    cloned = tree.CloneTree(-1, "fast")
                                    cloned.SetName(tree.GetName())
                                    cloned.Write("", ROOT.TObject.kOverwrite)
                                    # 4) 상태 복구
                                    tree.SetBranchStatus("*", 1)
                                else:
                                    # 지울 것 없으면 전체 복제해서 복사
                                    cloned = tree.CloneTree(-1, "fast")
                                    cloned.SetName(tree.GetName())
                                    cloned.Write("", ROOT.TObject.kOverwrite)

                            else:
                                # --- 히스토그램/그래프/기타 객체는 그대로 복사 ---
                                # 필요시 이름 유지해서 덮어쓰기
                                inner_obj.Write(inner_obj.GetName(), ROOT.TObject.kOverwrite)


                        output_file.Close()

        ###now start multiprocess to perform a infer for each seprated files
        
        procs = []
        errors = []
        success = []
        with mp.Pool(processes=7) as pool:

                # Use pool.apply_async to asynchronously apply the function to each file
                results = [pool.apply_async(infer_and_write, (file, input_model_path, new_branch_name, model_folder))
                        for file in output_files]
                for result in results:
                    try:
                        output = result.get()
                        if "Error" in output:
                            errors.append(output)
                        else:
                            success.append(output)
                    except Exception as e:
                        errors.append(f"Error {str(e)} occurred while processing {file}")

                # Close the pool and wait for the work to finish
                pool.close()
                pool.join()
        
        if errors:
            print("Errors encountered during processing:")
            for error in errors:
                print(error)
        if success:
            print("Successfully processed:")
            for s in success:
                print(s)
    

        # for i, file in enumerate(output_files):
        #     p = mp.Process(
        #         target=infer_and_write,
        #         args=(file, input_model_path, new_branch_name, model_folder),
        #     )
        #     p.start()
        #     procs.append(p)

        #for p in procs:
        #    p.join()
        # for i, file in enumerate(output_files):
        #     infer_and_write(file, model, new_branch_name, model_folder)

        ####start to merge into one file again
        merged_file = ROOT.TFile.Open(
            input_root_file.replace(".root", "_merged.root"), "RECREATE"
        )

        for file in output_files:
            chdir = file.split("/")[-2]
            filedir = file.split("/")[-1].replace(".root", "")
            file = ROOT.TFile.Open(file, "READ")
            file.cd()
            for inner_key in ROOT.gDirectory.GetListOfKeys():
                inner_obj = inner_key.ReadObj()
                merged_file.cd()
                #check dir is exist
                if not merged_file.GetDirectory(chdir + '/' + filedir):
                    this_dir = merged_file.mkdir(chdir + '/' + filedir)
                else:
                    this_dir = merged_file.GetDirectory(chdir + '/' + filedir)
                this_dir.cd()
                if isinstance(inner_obj, ROOT.TTree):
                    # Clone the tree and write it to the output file
                    inner_obj.SetBranchStatus("*", 1)
                    cloned_tree = inner_obj.CloneTree(-1, "fast")
                    cloned_tree.Write()
                else:
                    # For other objects (e.g., histograms), simply write them
                    inner_obj.Write()
            file.Close()

        # for file in output_files:
        #     dirname = file.split("_")[-1].replace(".root", "")
        #     this_dir = merged_file.mkdir(dirname)
        #     file = ROOT.TFile.Open(file, "READ")

        #     file.cd()
        #     for inner_key in ROOT.gDirectory.GetListOfKeys():
        #         inner_obj = inner_key.ReadObj()
        #         this_dir.cd()
        #         if isinstance(inner_obj, ROOT.TTree):
        #             # Clone the tree and write it to the output file
        #             inner_obj.SetBranchStatus("*", 1)
        #             cloned_tree = inner_obj.CloneTree(-1, "fast")
        #             cloned_tree.Write()
        #         else:
        #             # For other objects (e.g., histograms), simply write them
        #             inner_obj.Write()
        #     file.Close()

        merged_file.Close()

        ##clean the residuals created during task
        #for file in output_files:
        #    os.remove(file)
        #os.remove(input_root_file)
        
        shutil.move(input_root_file.replace(".root", "_merged.root"), input_root_file)
        shutil.rmtree(input_root_file.replace(".root", ""))
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        print(f"Elapsed Time: {elapsed_time} seconds")

    except Exception as e:
        import fcntl

        print(e)
        file_path = (
            "/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_log/error_list"
        )
        with open(file_path, "a") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.write(outname + "\n")
            fcntl.flock(file, fcntl.LOCK_UN)


def infer_with_iter(input_folder, input_model_path, branch_name, result_folder_name):
    import htcondor, shutil, ROOT, pathlib

    log_path = os.path.join(
        os.environ["DIR_PATH"],
        "TabNet_template",
        pathlib.Path(input_model_path).parent.absolute(),
        "infer_log",
    )

    #if os.path.isdir(log_path):
    #    shutil.rmtree(log_path)
    #os.makedirs(log_path)
    eras = [era] if era != "All" else ["2016preVFP","2016postVFP", "2017", "2018"]
    chs = ["Mu", "El"]
    for e in eras:
        #for ch in chs:
        print(os.path.join(input_folder, e, result_folder_name))
        if not os.path.isdir(
            os.path.join(input_folder, e, result_folder_name)
        ):
            continue
        systs = os.listdir(os.path.join(input_folder, e, result_folder_name))

        # to select directory only
        systs = [f for f in systs if not "." in f]
        # systs=['Central_Syst']
        for syst in systs:
            print(syst)
            files = [
                os.path.join(
                    input_folder,
                    e,
                    result_folder_name,
                    syst,
                    f,
                )
                for f in os.listdir(
                    os.path.join(
                        input_folder,
                        e,
                        result_folder_name,
                        syst,
                    )
                )
            ]
            for file in files:
                # ##########
                # ######clear residuals
                # ##########

                if not "QCD" in file:
                    continue
                if "temp" in file or "update" in file:
                    os.remove(file)
                    continue
 

                print(file)
                outname = file.split("/")
                outname[-1] = outname[-1].replace(".root", "")
                outname = "_".join(outname[-5:])
                job = htcondor.Submit(
                    {
                        "universe": "vanilla",
                        "getenv": True,
                        "jobbatchname": f"Vcb_infer_{e}_{syst}_{outname}",
                        "executable": "/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_write.sh",
                        "arguments": f"{input_model_path} {file} {branch_name} {e}",
                        "output": os.path.join(log_path, f"{outname}.out"),
                        "error": os.path.join(log_path, f"{outname}.err"),
                        "log": os.path.join(log_path, f"{outname}.log"),
                        "request_memory": (
                            "220GB"
                            if (
                                "TTLJ_powheg" in outname or "TTLL_powheg" in outname
                            )
                            and "Central" in outname
                            else "32GB"
                        ),
                        "request_gpus": (
                            0
                            if ("TTLJ_powheg" in outname and "Central" in outname)
                            else 0
                        ),
                        "request_cpus": 32,
                        "should_transfer_files": "YES",
                        "on_exit_hold" : "(ExitBySignal == True) || (ExitCode != 0)"
                    }
                )

                schedd = htcondor.Schedd()
                with schedd.transaction() as txn:
                    cluster_id = job.queue(txn)
                print("Job submitted with cluster ID:", cluster_id)


def train_submit(
    model_folder,
    floss_gamma,
    result_folder_name,
    sample_folder_loc,
    pretrained_model=None,
    add_year_index = 0
):
    import htcondor

    if os.path.isdir(model_folder):
        user_input = input(
            f"Folder {model_folder} already exist. do you want to proceed? [y/n]"
        )

        if user_input.lower() == "y":
            shutil.rmtree(model_folder)
        elif user_input.lower() == "n":
            print("Job aborted by user")
            exit()
        else:
            print("Type y or n")

    os.makedirs(model_folder)

    job = htcondor.Submit(
        {
            "universe": "vanilla",
            "getenv": True,
            "jobbatchname": f"Vcb_Train",
            "executable": "MVA.sh",
            "arguments": (
                f"{model_folder} {floss_gamma} {result_folder_name} {sample_folder_loc} {era} {add_year_index}"
                if pretrained_model is None
                else f"{model_folder} {floss_gamma} {result_folder_name} {sample_folder_loc} {pretrained_model} {era} {add_year_index}"
            ),
            "output": f"{model_folder}/job.out",
            "error": f"{model_folder}/job.err",
            "log": f"{model_folder}/job.log",
            "request_memory": "64GB",
            "request_gpus": 1,
            "request_cpus": 8,
        }
    )

    schedd = htcondor.Schedd()
    with schedd.transaction() as txn:
        cluster_id = job.queue(txn)
    print("Job submitted with cluster ID:", cluster_id)


if __name__ == "__main__":
    # Define the available working modes
    MODES = ["train", "train_submit", "plot", "infer_iter", "infer"]

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Select working mode")

    # Add an argument to select the working mode
    parser.add_argument("--working_mode", choices=MODES, help="select working mode")
    parser.add_argument(
        "--sample_folder_loc",
        dest="sample_folder_loc",
        type=str,
        default="",
        help="target sample folder",
    )
    parser.add_argument(
        "--input_model",
        dest="input_model",
        type=str,
        default="",
        help="input model to infer",
    )
    parser.add_argument(
        "--model_out_path", dest="out_path", type=str, help="training model output path"
    )
    parser.add_argument(
        "--branch_name",
        dest="branch_name",
        type=str,
        help="inferring branch name",
        default="template_score",
    )
    parser.add_argument(
        "--result_folder_name",
        dest="result_folder_name",
        type=str,
        help="name of RunResult folder",
    ) 
    parser.add_argument(
        "--floss_gamma",
        dest="floss_gamma",
        type=float,
        default=0.0,
        help="focal loss gamm",
    )
    parser.add_argument(
        "--input_root_file",
        dest="input_root_file",
        type=str,
        help="file to infer",
    )
    parser.add_argument(
        "--pretrained_model",
        dest="pretrained_model",
        type=str,
        default=None,
        help="path of pretrained model",
    )
    parser.add_argument(
        "--era",
        dest="era",
        type=str,
        default=None,
        help="era",
    )
    parser.add_argument(
        "--add_year_index",
        dest="add_year_index",
        type=int,
        default=0,
        help="add year index"
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        type=str,
        default=None,
        help="path to checkpoint model",
    )

    # Parse the arguments from the command line
    args = parser.parse_args()
    era = args.era
   
            
    # Handle the selected working mode
    if args.add_year_index and (args.working_mode != "train" and args.working_mode != "train_submit"):
        print("add_year_index is only available for train mode. It will be ignored.")
    

    if args.working_mode == "train":
        print("Training Mode")
        train(
            model_save_path=args.out_path,
            floss_gamma=args.floss_gamma,
            result_folder_name=args.result_folder_name,
            sample_folder_loc=args.sample_folder_loc,
            pretrained_model=args.pretrained_model,
            checkpoint=args.checkpoint,
            add_year_index=args.add_year_index
        )

    elif args.working_mode == "train_submit":
        train_submit(
            model_folder=args.out_path,
            floss_gamma=args.floss_gamma,
            result_folder_name=args.result_folder_name,
            sample_folder_loc=args.sample_folder_loc,
            pretrained_model=args.pretrained_model,
            add_year_index=args.add_year_index
        )

    elif args.working_mode == "plot":
        print("Plotting Mode")
        plot(args.input_model, args.checkpoint)
        # Add code for mode 2 here
    elif args.working_mode == "infer_iter":
        print("Inffering Mode (all file iteration)")
        infer_with_iter(
            branch_name=args.branch_name,
            input_folder=args.sample_folder_loc,
            input_model_path=args.input_model,
            result_folder_name=args.result_folder_name,
        )
        # infer(args.input_root_file,args.input_model)
        # Add code for mode 3 here
    elif args.working_mode == "infer":
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        ROOT.DisableImplicitMT()
        print("infering and writing")
        infer(args.input_root_file, args.input_model, args.branch_name)
    else:
        print("Wrong working mode")
