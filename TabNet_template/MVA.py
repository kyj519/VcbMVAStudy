# Standard library
from contextlib import contextmanager, nullcontext
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import traceback
from typing import Any, Dict, Iterable, Optional
import argparse
import hashlib
import os
import re
import shutil
import sys
import array
import importlib.util

# Third-party
import numpy as np
import ROOT
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
import torch.multiprocessing as mp
from tqdm.auto import tqdm
import TrainingConfig

# Local
from helpers import (
    compute_class_counts,
    pick_best_device,
    task,
    predict_logit_fast,
    explain_fast,
    predict_proba_fast,
    predict_log_proba,
    _discover_folds,
    _pick_zip,
    _load_training_config_module,
    view_fold
)
from eval_functions import ClassBalancedFocalLoss, CBFocalLossMetric

era = ""

from root_data_loader import load_data, classWtoSampleW
from root_data_loader_awk import load_root_as_dataset_kfold as load_data_kfold
from root_data_loader_awk import save_dataset_npz_json, load_dataset_npz_json

mp.set_sharing_strategy("file_system")


# 사용 예시
device = pick_best_device(min_free_gb=6)  # 6GB 이상 비어있는 GPU 중 최적 선택
print("Using device:", device)


def train(
    model_save_path,
    floss_gamma,
    result_folder_name,
    sample_folder_loc,
    pretrained_model=None,
    checkpoint=None,
    add_year_index=0,
    fold=0,
    TC=None,
):
    if TC is None:
        raise ValueError("TC (TrainingConfig module) must be provided.")

    TabNetTrainConfig = TC.TabNetTrainConfig

    # checkpoint와 pretrained를 동시에 쓰지 않기
    if checkpoint is not None and pretrained_model is not None:
        print("하나만 해라")

    fine_tune = pretrained_model is not None
    out_dir = (
        os.path.join(model_save_path, "FineTune") if fine_tune else model_save_path
    )
    os.makedirs(out_dir, exist_ok=True)

    cfg = TabNetTrainConfig(
        floss_gamma=floss_gamma,
        fine_tune=fine_tune,
        pretrained_model=pretrained_model,
    )
    # 기존 코드와 동일하게 patience = 2*T0
    cfg.patience = 2 * cfg.T0

    # ----- data_info 생성 -----
    # era는 기존 전역 변수를 그대로 사용
    data_info = cfg.make_data_info(
        sample_folder_loc=sample_folder_loc,
        result_folder_name=result_folder_name,
        era=era,
        add_year_index=add_year_index,
        sample_bkg=1,
        include_extra_bkgs=False,  # 예전 코드의 break 동작을 그대로 유지
    )

    # ----- 데이터 로딩(+ 캐시) -----
    data_npz = os.path.join(out_dir, "data.npz")
    if os.path.exists(data_npz):
        while True:
            ans = (
                input(
                    f"{data_npz} already exists. Do you want to overwrite it? (y/n): "
                )
                .strip()
                .lower()
            )
            if ans == "y":
                data = load_data_kfold(**data_info)

                save_dataset_npz_json(model_save_path, data)
                break
            elif ans == "n":
                data = load_dataset_npz_json(model_save_path)
                print(f"Using existing data {data_npz}")
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    else:
        data = load_data_kfold(**data_info)
        save_dataset_npz_json(model_save_path, data)

    # ----- 모델/로스/메트릭/트레이닝 인포 -----
    device = pick_best_device(6.0)
    data_k = view_fold(data, fold)
    out_dir = os.path.join(out_dir, f"fold{fold}")
    clf, model_info = cfg.build_model(data_k, device=device, checkpoint=checkpoint)
    loss_fn, eval_metrics, counts_train, _ = cfg.build_loss_and_metrics(
        data_k, device=device
    )
    train_info = cfg.build_train_info(data_k, loss_fn, eval_metrics, out_dir)

    # ----- 정보 파일 저장(+ 설정 py/json 보존) -----
    cfg.write_info_files(
        model_save_path=out_dir,
        data_info=data_info,
        model_info=model_info,
        train_info=train_info,
        floss_gamma=cfg.floss_gamma,
        counts_train=counts_train,
        cfg=cfg,  # 현재 설정 인스턴스
        save_config_py=True,  # 이 설정이 정의된 .py 그대로 복사
        save_config_json=True,  # 설정 JSON도 함께 저장
    )

    # ----- 학습/저장/플롯 -----
    clf.fit(**train_info)
    clf.save_model(os.path.join(out_dir, "model"))


# def plot(model_save_path, checkpoint_path=None):
#     import postTrainingToolkit

#     BATCH_LOAD = 8192 * 64 if checkpoint_path is None else 8192 * 64

#     with task("Loading dataset (data.npz)"):
#         data = np.load(os.path.join(model_save_path, "data.npz"), allow_pickle=True)
#     with task("Model loading"):
#         if checkpoint_path is None:
#             files = os.listdir(model_save_path)
#             pt_zip_files = [f for f in files if f.endswith(".zip")]
#             if not pt_zip_files:
#                 raise FileNotFoundError("No model .zip file found.")
#             model = TabNetClassifier()
#             model.load_model(os.path.join(model_save_path, pt_zip_files[0]))
#             print(f"  - Loaded from: {pt_zip_files[0]}")
#         else:
#             print(f"  - Loaded from: {checkpoint_path}")
#             model = TabNetClassifier()
#             model.load_model(checkpoint_path)
#             m = re.search(r"model_epoch(\d+)\.zip$", checkpoint_path)
#             epoch_str = m.group(1)   # '000'
#             epoch_int = int(epoch_str)
#             model_save_path = os.path.join(model_save_path, 'plot' ,f"ckpt{epoch_str}")
#             os.makedirs(model_save_path, exist_ok=True)

#     with task("ROC AUC evaluation and plotting"):
#         y_inv = np.logical_not(data["test_y"]).astype(int)
#         proba0 = predict_proba_fast(model, data["test_features"], batch_size=BATCH_LOAD)[:, 0]
#         #proba0 = model.predict_proba(data["test_features"])[:, 0]
#         postTrainingToolkit.ROC_AUC(
#             score=proba0,
#             y=y_inv,
#             plot_path=model_save_path,
#             #weight=data["test_weight"],
#         )

#     with task("Train/Validation predictions"):
#         train_score = predict_logit_fast(model,data["train_features"], batch_size=BATCH_LOAD)
#         val_score   = predict_logit_fast(model,data["val_features"], batch_size=BATCH_LOAD)
#         #train_score = model.predict_proba(data["train_features"])
#         #val_score   = model.predict_proba(data["val_features"])

#     from itertools import combinations
#     EPS = 1e-9
#     # with task("KS test and plotting"):
#     #     num_class = train_score.shape[1]
#     #     print(f"Number of classes: {num_class}")
#     #     for sig_idx, bkg_idx in list(combinations(range(num_class), 2)):
#     #         train_sig_mask = data["train_y"] == sig_idx
#     #         train_bkg_mask = data["train_y"] == bkg_idx
#     #         val_sig_mask = data["val_y"] == sig_idx
#     #         val_bkg_mask = data["val_y"] == bkg_idx


#     #         kolS, kolB = postTrainingToolkit.KS_test(
#     #             train_score=np.concatenate((train_score[train_sig_mask, sig_idx], train_score[train_bkg_mask, bkg_idx]), axis=0),
#     #             val_score=np.concatenate((val_score[val_sig_mask, sig_idx], val_score[val_bkg_mask, bkg_idx]), axis=0),
#     #             train_w=np.concatenate((data["train_weight"][train_sig_mask], data["train_weight"][train_bkg_mask]), axis=0),
#     #             val_w=np.concatenate((data["val_weight"][val_sig_mask], data["val_weight"][val_bkg_mask]), axis=0),
#     #             train_y=np.concatenate((np.zeros(train_score[train_sig_mask].shape[0]), np.ones(train_score[train_bkg_mask].shape[0])), axis=0),
#     #             val_y=np.concatenate((np.zeros(val_score[val_sig_mask].shape[0]), np.ones(val_score[val_bkg_mask].shape[0])), axis=0),
#     #             plotPath=model_save_path,
#     #             postfix=f"score_{sig_idx}_sig_{sig_idx}_bkg_{bkg_idx}"
#     #         )
#     #         print(f"  - KS statistic (Signal): {kolS}, (Background): {kolB}")
#     with task("KS test and plotting"):

#         def sigmoid_stable(x: np.ndarray) -> np.ndarray:
#             # overflow/underflow에 강한 시그모이드
#             out = np.empty_like(x, dtype=np.float64)
#             pos = x >= 0
#             neg = ~pos
#             out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
#             e = np.exp(x[neg])
#             out[neg] = e / (1.0 + e)
#             return out

#         num_class = train_score.shape[1]
#         print(f"Number of classes: {num_class}")

#         for sig_idx, bkg_idx in combinations(range(num_class), 2):
#             # 두 클래스 서브셋만 선택
#             tr_mask = (data["train_y"] == sig_idx) | (data["train_y"] == bkg_idx)
#             vl_mask = (data["val_y"]   == sig_idx) | (data["val_y"]   == bkg_idx)

#             # logits (또는 log-prob) 취득
#             z_tr_sig = train_score[tr_mask, sig_idx]
#             z_tr_bkg = train_score[tr_mask, bkg_idx]
#             z_vl_sig =  val_score[vl_mask,  sig_idx]
#             z_vl_bkg =  val_score[vl_mask,  bkg_idx]

#             # pair-wise margin = z_sig - z_bkg
#             tr_margin = z_tr_sig - z_tr_bkg
#             vl_margin = z_vl_sig - z_vl_bkg
#             # (선택) 극단값 클립: tr_margin = np.clip(tr_margin, -80, 80); vl_margin = np.clip(vl_margin, -80, 80)

#             # pair-wise score = sigmoid(margin)
#             tr_s = sigmoid_stable(tr_margin)
#             vl_s = sigmoid_stable(vl_margin)

#             # 라벨: sig=0, bkg=1  (이전 코드와 동일)
#             tr_y = np.where(data["train_y"][tr_mask] == sig_idx, 0, 1).astype(np.int8)
#             vl_y = np.where(data["val_y"][vl_mask]   == sig_idx, 0, 1).astype(np.int8)

#             tr_w = data["train_weight"][tr_mask]
#             vl_w = data["val_weight"][vl_mask]

#             # KS 테스트 & 플로팅 (기존 툴킷 사용)
#             kolS, kolB = postTrainingToolkit.KS_test(
#                 train_score=tr_s,
#                 val_score=vl_s,
#                 train_w=tr_w,
#                 val_w=vl_w,
#                 train_y=tr_y,
#                 val_y=vl_y,
#                 plotPath=model_save_path,
#                 postfix=f"pair_sig{sig_idx}_bkg{bkg_idx}",
#                 use_weight=False
#             )
#             print(f"  - p-value (Signal): {kolS:.3g}, (Background): {kolB:.3g}")

#     with task("Explainability (masks) and saving"):
#         res_explain, res_masks = explain_fast(model, data["test_features"], normalize=False, batch_size = BATCH_LOAD)
#         np.save(os.path.join(model_save_path, "explain.npy"), res_explain)
#         np.save(os.path.join(model_save_path, "mask.npy"),    res_masks)
#         np.save(os.path.join(model_save_path, "y.npy"),       data["train_y"])
#         M_explain = res_explain
#         sum_explain = M_explain.sum(axis=0)
#         feature_importances_ = sum_explain / np.sum(sum_explain)
#         print(feature_importances_)


#     with task("Variable list loading and cleaning"):
#         info_arr = np.load(os.path.join(model_save_path, "info.npy"), allow_pickle=True)
#         info_arr = info_arr[()]
#         varlist = list(info_arr['data_info']['varlist'])
#         if "weight" in varlist:
#             varlist.remove("weight")

#     with task("Feature importance plotting (mplhep)"):
#         if len(varlist) != len(feature_importances_):
#             varlist.append("year_index")
#         postTrainingToolkit.draw_feature_importance_mplhep(
#             varlist=varlist,
#             feature_importance=feature_importances_,
#             plot_dir=model_save_path,
#             fname_base="feature_importance",
#         )


def plot(model_save_path, checkpoint_path=None):
    import os, re, numpy as np
    from itertools import combinations

    import postTrainingToolkit
    from pytorch_tabnet.tab_model import TabNetClassifier  # 기존과 동일 가정
    from scipy.special import softmax, logsumexp

    BATCH_LOAD = 8192 * 64

    with task(f"[{os.path.basename(model_save_path)}] Loading dataset (data.npz)"):
        data = load_dataset_npz_json(model_save_path)

    def sigmoid_stable(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        e = np.exp(x[neg])
        out[neg] = e / (1.0 + e)
        return out

    # ---------- single-run core (reused per fold) ----------
    def _run_one(
        run_dir,
        out_parent_dir,
        fold,
        checkpoint_override=None,
        TC_local=None,
        class_labels=None,
    ):
        # out_parent_dir: 폴드별 plot 저장할 부모 디렉토리
        if TC_local is None:
            raise ValueError("TC_local (TrainingConfig module) must be provided.")

        with task(f"[{os.path.basename(run_dir)}] Loading dataset (data.npz)"):
            data_k = view_fold(data, fold)

        with task(f"[{os.path.basename(run_dir)}] Model loading"):
            model = TabNetClassifier()
            if checkpoint_override is None:
                # dir 내에서 적절한 zip 선택
                ckpt = _pick_zip(run_dir)
                if ckpt is None:
                    raise FileNotFoundError(f"No model .zip file found in {run_dir}")
                model.load_model(ckpt)
                out_dir = os.path.join(out_parent_dir)
                os.makedirs(out_dir, exist_ok=True)
                print(f"  - Loaded from: {ckpt}")
            else:
                model.load_model(checkpoint_override)
                m = re.search(
                    r"model_epoch(\d+)\.zip$", os.path.basename(checkpoint_override)
                )
                epoch_str = m.group(1) if m else "UNK"
                out_dir = os.path.join(out_parent_dir, f"ckpt{epoch_str}")
                os.makedirs(out_dir, exist_ok=True)
                print(f"  - Loaded from: {checkpoint_override}")

        # ---------- ROC on test ----------
        with task(f"[{os.path.basename(run_dir)}] ROC AUC evaluation and plotting"):

            # binary 가정: class 0에 대한 점수 사용 (기존 코드 유지)
            y_inv = np.logical_not(data_k["val_y"]).astype(int)
            logit = predict_logit_fast(
                model, data_k["val_features"], batch_size=BATCH_LOAD
            )
            proba0 = softmax(logit, axis=1)[:, 0]
            num_class = logit.shape[1]
            label_str = f"{class_labels[0]} (0) vs Others"
            postTrainingToolkit.ROC_AUC(
                score=proba0,
                y=y_inv,
                plot_path=out_dir,
                fname="ROC.png",
                extra_text=label_str,
                # weight=data.get("test_weight", None),
            )
            postTrainingToolkit.ROC_AUC(
                score=proba0,
                y=y_inv,
                plot_path=out_dir,
                fname="ROC_log.png",
                scale="log",
                extra_text=label_str,
                # weight=data.get("test_weight", None),
            )

            scores_by_bkg = []
            labels_by_bkg = []
            yinv_by_bkg = []

            for bkg_idx in range(1, num_class):
                sig_idx = 0
                local_signal_mask = data_k["val_y"] == sig_idx
                local_bkg_mask = data_k["val_y"] == bkg_idx
                local_mask = local_signal_mask | local_bkg_mask
                local_y = np.where(data_k["val_y"][local_mask] == sig_idx, 1, 0).astype(
                    np.int8
                )
                local_margin = logit[local_mask, sig_idx] - logit[local_mask, bkg_idx]
                local_score = sigmoid_stable(local_margin)
                label_str = f"{class_labels[sig_idx]} ({sig_idx}) vs {class_labels[bkg_idx]} ({bkg_idx})"

                scores_by_bkg.append(local_score)
                yinv_by_bkg.append(local_y)

                local_auc = postTrainingToolkit.ROC_AUC(
                    score=local_score,
                    y=local_y,
                    plot_path=out_dir,
                    fname=f"ROC_{sig_idx}_VS_{bkg_idx}.png",
                    extra_text=label_str,
                    # weight=data.get("test_weight", None),
                )

                labels_by_bkg.append(label_str + f" (AUC: {local_auc:.4f})")

                postTrainingToolkit.ROC_AUC(
                    score=local_score,
                    y=local_y,
                    plot_path=out_dir,
                    fname=f"ROC_{sig_idx}_VS_{bkg_idx}_log.png",
                    scale="log",
                    extra_text=label_str,
                    # weight=data.get("test_weight", None),
                )

            postTrainingToolkit.ROC_AUC(
                score=scores_by_bkg,
                y=yinv_by_bkg,
                plot_path=out_dir,
                fname=f"ROC_class_compare.png",
                labels=labels_by_bkg,
            )
            postTrainingToolkit.ROC_AUC(
                score=scores_by_bkg,
                y=yinv_by_bkg,
                plot_path=out_dir,
                fname=f"ROC_class_compare_log.png",
                labels=labels_by_bkg,
                scale="log",
            )

        # ---------- KS (pairwise margin, multi-class 지원) ----------
        with task(f"[{os.path.basename(run_dir)}] Train/Validation predictions"):
            train_score = predict_logit_fast(
                model, data_k["train_features"], batch_size=BATCH_LOAD
            )
            val_score = predict_logit_fast(
                model, data_k["val_features"], batch_size=BATCH_LOAD
            )

        with task(f"[{os.path.basename(run_dir)}] KS test and plotting"):
            num_class = train_score.shape[1]
            print(f"Number of classes: {num_class}")

            tr_s = softmax(train_score, axis=1)[:, 0]
            vl_s = softmax(val_score, axis=1)[:, 0]

            tr_y = np.where(data_k["train_y"] == sig_idx, 0, 1).astype(np.int8)
            vl_y = np.where(data_k["val_y"] == sig_idx, 0, 1).astype(np.int8)

            tr_w = data_k.get("train_weight", np.ones_like(tr_y, dtype=np.float64))
            vl_w = data_k.get("val_weight", np.ones_like(vl_y, dtype=np.float64))

            kolS, kolB = postTrainingToolkit.KS_test(
                train_score=tr_s,
                val_score=vl_s,
                train_w=tr_w,
                val_w=vl_w,
                train_y=tr_y,
                val_y=vl_y,
                plotPath=out_dir,
                postfix=f"pair_sig{sig_idx}_bkg{bkg_idx}",
                use_weight=False,
            )
            print(f"  - p-value (Signal): {kolS:.3g}, (Background): {kolB:.3g}")

            for sig_idx, bkg_idx in combinations(range(num_class), 2):
                tr_mask = (data_k["train_y"] == sig_idx) | (
                    data_k["train_y"] == bkg_idx
                )
                vl_mask = (data_k["val_y"] == sig_idx) | (data_k["val_y"] == bkg_idx)

                z_tr_sig = train_score[tr_mask, sig_idx]
                z_tr_bkg = train_score[tr_mask, bkg_idx]
                z_vl_sig = val_score[vl_mask, sig_idx]
                z_vl_bkg = val_score[vl_mask, bkg_idx]

                tr_margin = z_tr_sig - z_tr_bkg
                vl_margin = z_vl_sig - z_vl_bkg

                tr_s = sigmoid_stable(tr_margin)
                vl_s = sigmoid_stable(vl_margin)

                tr_y = np.where(data_k["train_y"][tr_mask] == sig_idx, 0, 1).astype(
                    np.int8
                )
                vl_y = np.where(data_k["val_y"][vl_mask] == sig_idx, 0, 1).astype(
                    np.int8
                )

                tr_w = data_k.get("train_weight", np.ones_like(tr_y, dtype=np.float64))[
                    tr_mask
                ]
                vl_w = data_k.get("val_weight", np.ones_like(vl_y, dtype=np.float64))[
                    vl_mask
                ]

                kolS, kolB = postTrainingToolkit.KS_test(
                    train_score=tr_s,
                    val_score=vl_s,
                    train_w=tr_w,
                    val_w=vl_w,
                    train_y=tr_y,
                    val_y=vl_y,
                    plotPath=out_dir,
                    postfix=f"pair_sig{sig_idx}_bkg{bkg_idx}",
                    use_weight=False,
                )
                print(f"  - p-value (Signal): {kolS:.3g}, (Background): {kolB:.3g}")

        # ---------- Explainability & Feature importance ----------
        with task(f"[{os.path.basename(run_dir)}] Explainability (masks) and saving"):
            res_explain, res_masks = explain_fast(
                model, data_k["val_features"], normalize=False, batch_size=BATCH_LOAD
            )
            np.save(os.path.join(out_dir, "explain.npy"), res_explain)
            np.save(os.path.join(out_dir, "mask.npy"), res_masks)
            # NOTE: 기존 코드 유지 (train_y 저장). 필요하면 test_y로 바꿔도 됨.
            np.save(os.path.join(out_dir, "y.npy"), data_k["val_y"])

            M_explain = res_explain
            sum_explain = M_explain.sum(axis=0)
            feature_importances_ = sum_explain / np.sum(sum_explain)
            print(feature_importances_)

        with task(
            f"[{os.path.basename(run_dir)}] Variable list loading and FI plotting (mplhep)"
        ):
            info_arr = np.load(os.path.join(run_dir, "info.npy"), allow_pickle=True)[()]
            varlist = list(info_arr["data_info"]["varlist"])
            if "weight" in varlist:
                varlist.remove("weight")
            if len(varlist) != len(feature_importances_):
                varlist.append("year_index")
            postTrainingToolkit.draw_feature_importance_mplhep(
                varlist=varlist,
                feature_importance=feature_importances_,
                plot_dir=out_dir,
                fname_base="feature_importance",
            )

        # 반환: 테스트 ROC용 (결합 계산)
        return {
            "out_dir": out_dir,
            "val_score": proba0,
            "val_y_inv": y_inv,
            # "test_w": data.get("test_weight", None),  # 필요시 사용
        }

    # =========================
    # main flow: fold-aware
    # =========================

    folds = _discover_folds(model_save_path)
    if not folds:
        raise RuntimeError(f"No folds found under {model_save_path}.")
    print(f"[fold-aware] Found {len(folds)} folds: {[name for _, name, _ in folds]}")

    scores_per_fold = []
    yinv_per_fold = []
    labels = []

    for idx, name, run_dir in folds:
        fold_plot_dir = os.path.join(run_dir, "plot")
        os.makedirs(fold_plot_dir, exist_ok=True)

        cfg_path = os.path.join(run_dir, "TabNetTrainConfig.py")
        TC_local_cfg = _load_training_config_module(cfg_path)
        obj = TC_local_cfg.TabNetTrainConfig()
        _, class_labels = obj.build_input_tuple("","","")

        res = _run_one(
            run_dir=run_dir,
            out_parent_dir=fold_plot_dir,
            checkpoint_override=checkpoint_path,
            fold=idx,
            TC_local=TC_local_cfg,
            class_labels=class_labels,
        )
        scores_per_fold.append(res["val_score"])
        yinv_per_fold.append(res["val_y_inv"])
        labels.append(f"Fold {idx}")

    with task("[per-fold] ROC AUC overlay over all folds"):
        summary_dir = os.path.join(model_save_path, "plot", "_folds")
        os.makedirs(summary_dir, exist_ok=True)

        aucs_lin = postTrainingToolkit.ROC_AUC(
            score=scores_per_fold,
            y=yinv_per_fold,
            weight=None,
            plot_path=summary_dir,
            fname="ROC_folds_linear.png",
            scale="linear",
            labels=labels,
        )
        aucs_log = postTrainingToolkit.ROC_AUC(
            score=scores_per_fold,
            y=yinv_per_fold,
            weight=None,
            plot_path=summary_dir,
            fname="ROC_folds_logx.png",
            scale="log",
            labels=labels,
        )

        from sklearn.metrics import roc_auc_score

        for lab, s, yy in zip(labels, scores_per_fold, yinv_per_fold):
            print(f"  - {lab}: AUC = {roc_auc_score(yy, s):.4f}")

        print(f"[per-fold] Saved overlay ROC plots to: {summary_dir}")


def infer_and_write(root_file, input_model_path, new_branch_name, model_folder):
    folds = _discover_folds(input_model_path)
    if not folds:
        raise RuntimeError(f"No folds found under {input_model_path}.")
    print(f"[fold-aware] Found {len(folds)} folds: {[name for _, name, _ in folds]}")
    model_folds = {}

    for idx, name, run_dir in folds:
        this_model = TabNetClassifier()
        model_zip = _pick_zip(run_dir)
        if model_zip is None:
            raise FileNotFoundError(f"No model .zip file found in {run_dir}")
        this_model.load_model(model_zip)
        model_folds[idx] = this_model
        print(f"  - Loaded fold {idx} from: {model_zip}")

    first_run_dir = folds[0][2]
    data_info = np.load(os.path.join(first_run_dir, "info.npy"), allow_pickle=True)
    data_info = data_info[()]
    data_info = data_info["data_info"]
    num_class = len(data_info["tree_path_filter_str"])
    data_info["tree_path_filter_str"] = [[(root_file, "Result_Tree", "")]]
    data_info["infer_mode"] = True
    data = load_data_kfold(**data_info)
    if len(model_folds) != len(data["folds"]):
        raise RuntimeError(
            f"Fold count mismatch: model has {len(model_folds)} folds but data has {len(np.unique(data['folds']))} folds"
        )

    result_arr = np.zeros((data["X"].shape[0], num_class), dtype=np.float64)
    all_infer_check = np.zeros(data["X"].shape[0], dtype=bool)
    for idx, name, run_dir in folds:
        data_k = view_fold(data, idx)
        arr = data_k["val_features"]
        model = model_folds[idx]
        pred = predict_log_proba(model, arr) if arr.shape[0] > 0 else np.empty((0, num_class))
        pred = np.asarray(pred)

        _, fold_idx = data["folds"][idx]
        all_infer_check[fold_idx] = True
        result_arr[fold_idx] = pred
    if any(~all_infer_check):
        raise RuntimeError(
            f"Some events were not inferred. Check fold assignment and data loading."
        )

    # --- 쓰기 시작 ---
    tf = None
    try:
        tf = ROOT.TFile.Open(root_file, "UPDATE")
        if not tf or tf.IsZombie():
            raise RuntimeError(f"Failed to open for UPDATE: {root_file}")

        tree = tf.Get("Result_Tree")
        if not tree:
            raise RuntimeError(f"'Result_Tree' not found in {root_file}")

        n_entries = tree.GetEntries()
        if n_entries != result_arr.shape[0]:
            raise RuntimeError(
                f"Entry mismatch: tree entries={n_entries} vs pred rows={result_arr.shape[0]} in {root_file}"
            )

        # 브랜치 버퍼/생성
        bufs = [array.array("f", [0.0]) for _ in range(num_class)]
        branches = []
        for cls in range(num_class):
            bname = f"{new_branch_name}_log_prob_{cls}"
            branches.append(tree.Branch(bname, bufs[cls], f"{bname}/F"))

        # 채우기
        for i in range(n_entries):
            for cls in range(num_class):
                bufs[cls][0] = float(result_arr[i, cls])
                branches[cls].Fill()

        # 쓰기
        tree.Write("", ROOT.TObject.kOverwrite)

        # 성공 시 메시지(선택)
        return f"OK: {root_file}"

    finally:
        if tf:
            tf.Close()


def safe_infer(args):
    f, input_model_path, new_branch_name, model_folder = args
    logs = []

    def log(msg):
        logs.append(msg)

    try:
        log(f"[{os.getpid()}] start {f}")
        msg = infer_and_write(f, input_model_path, new_branch_name, model_folder)
        log(f"[{os.getpid()}] done  {f}: {msg}")
        return ("ok", f, logs)
    except Exception:
        import traceback

        logs.append(traceback.format_exc())
        return ("err", f, logs)


def infer(input_root_file, input_model_path, branch_name="template_score"):
    import array, shutil, time, tempfile
    from pathlib import Path
    import multiprocessing as mp  # torch.multiprocessing가 꼭 필요 없으면 표준 mp가 덜 까다롭습니다.

    start_time = time.time()
    input_root_file = str(Path(input_root_file).resolve())
    print(input_model_path)

    model_folder = str(Path(input_model_path).parent)
    outname = "_".join(Path(input_root_file).with_suffix("").parts[-5:])

    # spawn 설정은 여기서 중복 호출시 에러 -> 무시
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    tmp_dir = Path(input_root_file).parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)  # 없으면 생성

    tmp_root = Path(tempfile.mkdtemp(prefix=f"tmp_infer_{outname}_", dir=tmp_dir))
    print(f"[tmp] working dir: {tmp_root}")

    try:
        new_branch_name = branch_name

        print(f"Start to process {input_root_file}")
        input_file = ROOT.TFile.Open(input_root_file, "READ")
        if not input_file or input_file.IsZombie():
            raise RuntimeError(f"Failed to open: {input_root_file}")
        print(f"Opened {input_root_file}")

        output_files = []

        # 1) 상위 디렉터리 순회
        for ch_key in input_file.GetListOfKeys():
            ch_obj = ch_key.ReadObj()
            if not ch_obj.InheritsFrom("TDirectory"):
                continue

            chdirname = tmp_root / ch_obj.GetName()
            print(f"Will create a new directory: {chdirname}")
            chdirname.mkdir(parents=True, exist_ok=True)

            # 2) 하위 디렉터리 순회
            for key in ch_obj.GetListOfKeys():
                obj = key.ReadObj()
                if not obj.InheritsFrom("TDirectory"):
                    continue

                out_path = chdirname / f"{obj.GetName()}.root"
                output_file = ROOT.TFile.Open(str(out_path), "RECREATE")
                output_files.append(str(out_path))

                # 입력 디렉토리로 cd
                input_file.cd(f"{ch_obj.GetName()}/{obj.GetName()}")

                for inner_key in ROOT.gDirectory.GetListOfKeys():
                    inner_obj = inner_key.ReadObj()

                    output_file.cd()
                    prefix = str(new_branch_name)

                    if inner_obj.InheritsFrom("TTree"):
                        tree = inner_obj
                        # 어떤 브랜치가 지워질지 미리 로깅
                        branches = tree.GetListOfBranches()
                        names = [
                            branches.At(i).GetName() for i in range(branches.GetSize())
                        ]
                        matches = [n for n in names if n.startswith(prefix)]

                        # 예외가 나도 상태 원복되도록
                        tree.SetBranchStatus("*", 1)
                        try:
                            if matches:
                                print(
                                    f"[{ch_obj.GetName()}/{obj.GetName()}]"
                                    f" Tree '{tree.GetName()}': delete {len(matches)} branches with prefix '{prefix}': {matches}"
                                )
                                tree.SetBranchStatus(prefix + "*", 0)
                            cloned = tree.CloneTree(-1, "fast")
                            cloned.SetName(tree.GetName())
                            cloned.Write("", ROOT.TObject.kOverwrite)
                        finally:
                            tree.SetBranchStatus("*", 1)
                    else:
                        inner_obj.Write(inner_obj.GetName(), ROOT.TObject.kOverwrite)

                output_file.Close()

        # 입력 파일은 더 이상 사용 안 하므로 즉시 닫기
        input_file.Close()

        # 3) 멀티프로세싱 추론
        errors, success = [], []

        args = [
            (f, input_model_path, new_branch_name, model_folder) for f in output_files
        ]

        with mp.Pool(processes=7) as pool:
            for status, f, logs in pool.imap_unordered(safe_infer, args, chunksize=1):
                for line in logs:
                    print(line, flush=True)
                if status == "ok":
                    success.append((f, "\n".join(logs)))
                else:
                    errors.append((f, "\n".join(logs)))

        if errors:
            print("Errors encountered during processing:")
            for f, tb in errors:
                print(f"[{f}]")
                print(tb)
            print("[ABORT] Keeping original ROOT file intact; not merging/replacing.")
            return

        if success:
            print("Successfully processed:")
            for f, msg in success:
                print(f"- {f} {('→ ' + msg) if msg else ''}")  # msg가 list
        # 4) 병합
        merged_path = str(
            Path(input_root_file).with_name(Path(input_root_file).stem + "_merged.root")
        )
        merged_file = ROOT.TFile.Open(merged_path, "RECREATE")
        if not merged_file or merged_file.IsZombie():
            raise RuntimeError(f"Failed to create merged file: {merged_path}")

        for f in output_files:
            src = ROOT.TFile.Open(f, "READ")
            if not src or src.IsZombie():
                raise RuntimeError(f"Failed to open split file: {f}")
            # chdir/dir명은 파일 경로로부터
            chdir = Path(f).parent.name
            filedir = Path(f).stem

            src.cd()
            for inner_key in ROOT.gDirectory.GetListOfKeys():
                inner_obj = inner_key.ReadObj()
                merged_file.cd()
                # 중첩 디렉토리 확보
                full_dir = f"{chdir}/{filedir}"
                d = merged_file.GetDirectory(full_dir)
                if not d:
                    d = merged_file.mkdir(full_dir)
                d.cd()
                if inner_obj.InheritsFrom("TTree"):
                    inner_obj.SetBranchStatus("*", 1)
                    cloned_tree = inner_obj.CloneTree(-1, "fast")
                    cloned_tree.Write()
                else:
                    inner_obj.Write()
            src.Close()

        merged_file.Close()

        # 5) 원본 교체 + 정리
        shutil.move(merged_path, input_root_file)

        end_time = time.time()
        print(f"Elapsed Time: {end_time - start_time:.2f} seconds")

    except Exception as e:
        raise


# patch_list는 네가 준 그대로라고 가정
patch_list = {
    "2016preVFP": [
        "QCD_bEnriched_HT2000toInf",
    ],
    "2016postVFP": [
        "DYJets_MG",
        "SingleTop_sch_Lep",
        "SingleTop_tch_antitop_Incl",
        "SingleTop_tch_top_Incl",
        "WZ_pythia",
        "ZZ_pythia",
        "ttZToQQ_ll",
    ],
    "2017": [
        "DYJets_MG",
        "SingleTop_sch_Lep",
        "SingleTop_tch_antitop_Incl",
        "SingleTop_tch_top_Incl",
        "WZ_pythia",
        "ZZ_pythia",
        "ttZToQQ_ll",
    ],
    "2018": [
        "DYJets_MG",
        "SingleTop_sch_Lep",
        "SingleTop_tch_antitop_Incl",
        "SingleTop_tch_top_Incl",
        "WZ_pythia",
        "ZZ_pythia",
        "ttZToQQ_ll",
    ],
}

# --- 루프 위(한 번만)에서 역인덱스 준비 ---
allowed_by_sample = {}
for era, samples in patch_list.items():
    for s in samples:
        allowed_by_sample.setdefault(s, set()).add(era)

# 긴 이름(예: TTLL_powheg_CP5Down) 먼저 매칭되도록 정렬
samples_by_len = sorted(allowed_by_sample.keys(), key=len, reverse=True)


def infer_with_iter(input_folder, input_model_path, branch_name, result_folder_name):
    import htcondor, shutil, ROOT, pathlib

    log_path = os.path.join(
        os.environ["DIR_PATH"],
        "TabNet_template",
        pathlib.Path(input_model_path).parent.absolute(),
        "infer_log",
    )

    # if os.path.isdir(log_path):
    #    shutil.rmtree(log_path)
    # os.makedirs(log_path)
    eras = [era] if era != "All" else ["2016preVFP", "2016postVFP", "2017", "2018"]
    chs = ["Mu", "El"]
    for e in eras:
        # for ch in chs:
        print(os.path.join(input_folder, e, result_folder_name))
        if not os.path.isdir(os.path.join(input_folder, e, result_folder_name)):
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
                if not file.endswith(".root"):
                    continue
                # ##########
                # ######clear residuals
                # ##########

                if "temp" in file or "update" in file:
                    os.remove(file)
                    continue
                # matched = next((s for s in samples_by_len if s in file), None)
                # if matched is None:
                #     # patch_list 어떤 샘플명도 포함 안 함 -> 스킵
                #     continue
                # if e not in allowed_by_sample[matched]:
                #     continue

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
                            if ("TTLJ_powheg" in outname or "TTLL_powheg" in outname)
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
                        "on_exit_hold": "(ExitBySignal == True) || (ExitCode != 0)",
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
        help="add year index",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        type=str,
        default=None,
        help="path to checkpoint model",
    )
    parser.add_argument(
        "--fold",
        dest="fold",
        type=int,
        default=0,
        help="k-fold cross validation fold index",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        default=None,
        help="Path to TrainingConfig.py. If omitted, imports the installed TrainingConfig module.",
    )

    # Parse the arguments from the command line
    args = parser.parse_args()
    era = args.era

    # Handle the selected working mode
    if args.add_year_index and (
        args.working_mode != "train" and args.working_mode != "train_submit"
    ):
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
            add_year_index=args.add_year_index,
            fold=args.fold,
            TC=_load_training_config_module(args.config),
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
