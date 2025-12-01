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
import tempfile
import time

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
import TrainingConfig

# Local
from helpers import (
    compute_class_counts,
    pick_best_device,
    task,
    predict_logit_fast,
    explain_fast,
    predict_proba_fast,
    predict_log_proba_fast,
    _discover_folds,
    _pick_zip,
    _load_training_config_module,
    view_fold,
    _iter_with_rich_progress,
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
                fname="ROC.pdf",
                extra_text=label_str,
                # weight=data.get("test_weight", None),
            )
            postTrainingToolkit.ROC_AUC(
                score=proba0,
                y=y_inv,
                plot_path=out_dir,
                fname="ROC_log.pdf",
                scale="log",
                extra_text=label_str,
                legend_loc="upper left",
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
                    fname=f"ROC_{sig_idx}_VS_{bkg_idx}.pdf",
                    extra_text=label_str,
                    # weight=data.get("test_weight", None),
                )

                labels_by_bkg.append(label_str + f" (AUC: {local_auc:.4f})")

                postTrainingToolkit.ROC_AUC(
                    score=local_score,
                    y=local_y,
                    plot_path=out_dir,
                    fname=f"ROC_{sig_idx}_VS_{bkg_idx}_log.pdf",
                    scale="log",
                    extra_text=label_str,
                    # weight=data.get("test_weight", None),
                )

            postTrainingToolkit.ROC_AUC(
                score=scores_by_bkg,
                y=yinv_by_bkg,
                plot_path=out_dir,
                fname=f"ROC_class_compare.pdf",
                labels=labels_by_bkg,
            )
            postTrainingToolkit.ROC_AUC(
                score=scores_by_bkg,
                y=yinv_by_bkg,
                plot_path=out_dir,
                fname=f"ROC_class_compare_log.pdf",
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
            fname="ROC_folds_linear.pdf",
            scale="linear",
            labels=labels,
        )
        aucs_log = postTrainingToolkit.ROC_AUC(
            score=scores_per_fold,
            y=yinv_per_fold,
            weight=None,
            plot_path=summary_dir,
            fname="ROC_folds_logx.pdf",
            scale="log",
            labels=labels,
        )

        from sklearn.metrics import roc_auc_score

        for lab, s, yy in zip(labels, scores_per_fold, yinv_per_fold):
            print(f"  - {lab}: AUC = {roc_auc_score(yy, s):.4f}")

        print(f"[per-fold] Saved overlay ROC plots to: {summary_dir}")


def infer_and_write(root_file, input_model_path, new_branch_name, model_folder, backend="pytorch"):

    folds, model_folds, meta = _load_models_cached(input_model_path, backend=backend)
    if not folds:
        raise RuntimeError(f"No folds found under {input_model_path}.")
    print(
        f"[fold-aware] ({backend}) Found {len(folds)} folds: {[name for _, name, _ in folds]}"
    )

    data_info = meta["data_info"].copy()
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
        runner = model_folds[idx]
        pred = runner.predict_log_proba(arr) if arr.shape[0] > 0 else np.empty((0, num_class))
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


def _shutdown_pool(pool, label: str, timeout: float = 15.0):
    """
    Terminate a multiprocessing pool quickly so the main process can exit.
    We avoid the blocking Pool.join() to prevent hangs seen after inference.
    """
    try:
        pool.close()
    except Exception:
        pass

    try:
        pool.terminate()
    except Exception:
        pass

    deadline = time.time() + timeout
    for proc in getattr(pool, "_pool", []):
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            proc.join(timeout=remaining)
        except Exception:
            pass

    alive = [p for p in getattr(pool, "_pool", []) if p.is_alive()]
    if alive:
        print(f"[warn] forcing {len(alive)} stuck worker(s) to exit for {label}")
        for proc in alive:
            try:
                proc.join(timeout=1.0)
            except Exception:
                pass


def safe_infer(args):
    f, input_model_path, new_branch_name, model_folder, backend = args
    logs = []

    def log(msg):
        logs.append(msg)

    try:
        log(f"[{os.getpid()}] start {f}")
        msg = infer_and_write(
            f, input_model_path, new_branch_name, model_folder, backend=backend
        )
        log(f"[{os.getpid()}] done  {f}: {msg}")
        return ("ok", f, logs)
    except Exception:
        import traceback

        logs.append(traceback.format_exc())
        return ("err", f, logs)


def infer(input_root_file, input_model_path, branch_name="template_score", backend="pytorch"):
    import array, shutil, time, tempfile
    from pathlib import Path
    import multiprocessing as mp  # torch.multiprocessing가 꼭 필요 없으면 표준 mp가 덜 까다롭습니다.

    start_time = time.time()
    input_root_file = str(Path(input_root_file).resolve())
    print(input_model_path)

    model_folder = str(Path(input_model_path).parent)
    outname = "_".join(Path(input_root_file).with_suffix("").parts[-5:])
    backend = _normalize_backend(backend)
    print(f"[infer] backend={backend}")

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
            (f, input_model_path, new_branch_name, model_folder, backend)
            for f in output_files
        ]

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=1)
        try:
            for status, f, logs in pool.imap_unordered(safe_infer, args, chunksize=1):
                for line in logs:
                    print(line, flush=True)
                if status == "ok":
                    success.append((f, "\n".join(logs)))
                else:
                    errors.append((f, "\n".join(logs)))
        finally:
            _shutdown_pool(pool, label="infer")

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

# parallel_score_friend.py
import os, shutil, tempfile, array
from pathlib import Path
import numpy as np
import multiprocessing as mp
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.ROOT.EnableImplicitMT(0)

# ==== 기존 유틸이 있다고 가정 ====
# TabNetClassifier, _discover_folds, _pick_zip, load_data_kfold, view_fold, predict_log_proba_fast

_MODEL_CACHE = {}

_BACKEND_ALIASES = {
    "torch": "pytorch",
    "pt": "pytorch",
    "pytorch": "pytorch",
    "onnx": "onnx",
    "onnx-int8": "onnx-int8",
    "onnx_int8": "onnx-int8",
    "trt": "tensorrt",
    "trt-int8": "tensorrt-int8",
    "trt_int8": "tensorrt-int8",
    "tensorrt": "tensorrt",
    "tensorrt-int8": "tensorrt-int8",
    "tensorrt_int8": "tensorrt-int8",
}


def _normalize_backend(name: Optional[str]) -> str:
    backend = "pytorch" if not name else _BACKEND_ALIASES.get(name.lower(), name.lower())
    valid = {"pytorch", "onnx", "onnx-int8", "tensorrt", "tensorrt-int8"}
    if backend not in valid:
        raise ValueError(f"Unsupported backend '{name}'. Choose from {sorted(valid)}")
    return backend


def _prefer_int8(backend: str) -> bool:
    return "int8" in backend


def _dl_workers_for_infer() -> int:
    env_v = os.getenv("TABNET_INFER_DL_WORKERS")
    if env_v is not None:
        try:
            return max(0, int(env_v))
        except Exception:
            pass
    try:
        import multiprocessing as omp

        if omp.current_process().daemon:
            return 0  # daemonic process cannot spawn DataLoader workers
    except Exception:
        pass
    return 0  # safe default


def _log_softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.size == 0:
        # keep shape-compatible empty
        if logits.ndim == 1:
            return logits.reshape(0, 0)
        return logits.reshape(logits.shape[0], logits.shape[1] if logits.ndim > 1 else 0)
    maxv = np.max(logits, axis=1, keepdims=True)
    stable = logits - maxv
    sumexp = np.exp(stable).sum(axis=1, keepdims=True)
    return stable - np.log(sumexp)


def _batch_size_from_env(keys: tuple[str, ...], default: int) -> int:
    for k in keys:
        v = os.getenv(k)
        if v is None:
            continue
        try:
            return max(1, int(v))
        except Exception:
            pass
    return default


def _iter_batches(arr: np.ndarray, batch_size: int):
    if batch_size is None or batch_size <= 0:
        yield arr
        return
    n = arr.shape[0]
    for start in range(0, n, batch_size):
        yield arr[start : min(n, start + batch_size)]


def _resolve_onnx_path(model_root: str, fold_idx: int, prefer_int8: bool = False) -> str:
    root = Path(model_root)
    onnx_dir = root / "onnx"
    base = f"tabnet_fold{fold_idx}"
    candidates = []
    if prefer_int8:
        candidates.extend(
            [
                onnx_dir / f"{base}.int8.onnx",
                onnx_dir / f"{base}_int8.onnx",
                root / f"{base}.int8.onnx",
            ]
        )
    candidates.extend(
        [
            onnx_dir / f"{base}.onnx",
            root / f"{base}.onnx",
        ]
    )
    for cand in candidates:
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        f"ONNX model for fold {fold_idx} not found. Looked for: {', '.join(map(str, candidates))}"
    )


def _resolve_trt_plan_path(model_root: str, fold_idx: int, prefer_int8: bool = False) -> Optional[str]:
    root = Path(model_root)
    trt_dir = root / "onnx"
    base = f"tabnet_fold{fold_idx}"
    candidates = []
    if prefer_int8:
        candidates.extend(
            [
                trt_dir / f"{base}.int8.plan",
                trt_dir / f"{base}_int8.plan",
            ]
        )
    candidates.extend([trt_dir / f"{base}.plan", root / f"{base}.plan"])
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None


class _TorchFoldRunner:
    def __init__(self, model: TabNetClassifier):
        self.model = model

    def predict_log_proba(self, arr: np.ndarray) -> np.ndarray:
        return predict_log_proba_fast(
            self.model, arr, num_workers=_dl_workers_for_infer()
        )


class _OnnxFoldRunner:
    def __init__(self, onnx_path: str, prefer_trt_provider: bool = False, batch_size: Optional[int] = None):
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise ImportError(
                f"onnxruntime is required for ONNX/TensorRT backends: {exc}"
            )

        available = ort.get_available_providers()
        provider_order = []
        if prefer_trt_provider:
            provider_order.append("TensorrtExecutionProvider")
        provider_order.extend(["CUDAExecutionProvider", "CPUExecutionProvider"])
        providers = [p for p in provider_order if p in available] or available

        so = ort.SessionOptions()
        if "TensorrtExecutionProvider" in providers:
            cache_dir = Path(onnx_path).parent / "trt_cache"
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                so.add_session_config_entry("trt_engine_cache_enable", "1")
                so.add_session_config_entry("trt_engine_cache_path", str(cache_dir))
            except Exception:
                pass

        self.session = ort.InferenceSession(
            onnx_path, providers=providers, sess_options=so
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.batch_size = batch_size or _batch_size_from_env(
            ("ONNX_BATCH_INFER", "BATCH_INFER"), 4096
        )
        print(f"[onnx] Loaded {onnx_path} with providers={providers}")

    def predict_log_proba(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32, order="C")
        n = arr.shape[0]
        if n == 0:
            return np.empty((0, arr.shape[1] if arr.ndim > 1 else 0), dtype=np.float64)

        outputs = []
        for start in _iter_with_rich_progress(
            range(0, n, self.batch_size), desc="Predict log proba (onnx)"
        ):
            end = min(n, start + self.batch_size)
            chunk = arr[start:end]
            logits = self.session.run(
                [self.output_name], {self.input_name: chunk}
            )[0]
            outputs.append(_log_softmax_np(logits))

        return np.concatenate(outputs, axis=0) if outputs else np.empty(
            (0, arr.shape[1] if arr.ndim > 1 else 0), dtype=np.float64
        )


class _TrtPlanFoldRunner:
    def __init__(self, plan_path: str, batch_size: Optional[int] = None):
        try:
            import tensorrt as trt  # type: ignore
            import pycuda.driver as cuda  # type: ignore
            import pycuda.autoinit  # type: ignore  # noqa: F401
        except Exception as exc:
            raise ImportError(f"TensorRT plan backend requires tensorrt + pycuda: {exc}")

        self.cuda = cuda
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(plan_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine at {plan_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.batch_size = batch_size or _batch_size_from_env(
            ("TRT_BATCH_INFER", "ONNX_BATCH_INFER", "BATCH_INFER"), 1024
        )

        self.trt = trt
        self.cuda = cuda
        self._use_binding_api = hasattr(self.engine, "num_bindings")
        self.input_binding = None
        self.output_binding = None
        self.input_name = None
        self.output_name = None
        self.input_dtype = None
        self.output_dtype = None
        self._output_bindings: list[tuple[int, str]] = []
        self._output_names_all: list[str] = []
        self._output_dtypes: dict[str, np.dtype] = {}

        def _pick_output_name(names: list[str]) -> str:
            lower = [n.lower() for n in names]
            for target in ("logit", "prob"):
                for name, lname in zip(names, lower):
                    if target in lname:
                        return name
            return names[0]

        if self._use_binding_api:
            for i in range(self.engine.num_bindings):
                bname = self.engine.get_binding_name(i)
                if self.engine.binding_is_input(i):
                    self.input_binding = i
                    self.input_name = bname
                else:
                    self._output_bindings.append((i, bname))
            if self.input_binding is None:
                raise RuntimeError("TensorRT plan must have at least one input.")
            if not self._output_bindings:
                raise RuntimeError("TensorRT plan must have at least one output.")
            if len(self._output_bindings) == 1:
                self.output_binding, self.output_name = self._output_bindings[0]
            else:
                chosen = _pick_output_name([n for _, n in self._output_bindings])
                self.output_binding, self.output_name = next((i, n) for i, n in self._output_bindings if n == chosen)
            self.input_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.input_binding)))
            self.output_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.output_binding)))
        else:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            if len(inputs) != 1:
                raise RuntimeError(f"TensorRT plan must have exactly one input tensor. inputs={inputs}")
            if not outputs:
                raise RuntimeError("TensorRT plan must have at least one output tensor.")
            self.input_name = inputs[0]
            self.output_name = _pick_output_name(outputs)
            self._output_names_all = outputs
            self._output_dtypes = {
                name: np.dtype(trt.nptype(self.engine.get_tensor_dtype(name))) for name in outputs
            }
            self.input_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
            self.output_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.output_name)))
        self._num_classes = None
        print(f"[trt] Loaded TensorRT plan from {plan_path}")

    def predict_log_proba(self, arr: np.ndarray) -> np.ndarray:
        cuda = self.cuda
        trt = self.trt
        arr = np.asarray(arr, dtype=self.input_dtype, order="C")
        n = arr.shape[0]
        if n == 0:
            n_class = self._num_classes or (arr.shape[1] if arr.ndim > 1 else 0)
            return np.empty((0, n_class), dtype=np.float64)

        outputs = []
        for start in _iter_with_rich_progress(
            range(0, n, self.batch_size), desc="Predict log proba (trt)"
        ):
            end = min(n, start + self.batch_size)
            chunk = arr[start:end]

            if self._use_binding_api:
                self.context.set_binding_shape(self.input_binding, tuple(chunk.shape))
                out_host = None
                bindings = [None] * self.engine.num_bindings
                # input binding
                d_in = cuda.mem_alloc(chunk.nbytes)
                bindings[self.input_binding] = int(d_in)

                # outputs: allocate per-output device buffer
                d_out_primary = None
                out_host = None
                for ob_idx, ob_name in self._output_bindings:
                    shape = tuple(
                        (chunk.shape[0] if dim == -1 else dim)
                        for dim in self.context.get_binding_shape(ob_idx)
                    )
                    dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(ob_idx)))
                    d_buf = cuda.mem_alloc(int(np.prod(shape)) * dtype.itemsize)
                    bindings[ob_idx] = int(d_buf)
                    if ob_idx == self.output_binding:
                        d_out_primary = d_buf
                        out_host = np.empty(shape, dtype=self.output_dtype)
                    else:
                        # store to free later
                        pass

                cuda.memcpy_htod_async(d_in, chunk, self.stream)
                self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
                if d_out_primary is not None:
                    cuda.memcpy_dtoh_async(out_host, d_out_primary, self.stream)
                self.stream.synchronize()

                # free allocations
                d_in.free()
                for ptr in bindings:
                    if ptr is None or ptr == int(d_in):
                        continue
                    try:
                        cuda.mem_free(ptr)
                    except Exception:
                        pass
            else:
                self.context.set_input_shape(self.input_name, tuple(chunk.shape))
                out_shapes = {}
                d_outputs = {}
                out_host = None

                for name in self._output_names_all:
                    shape = tuple(
                        (chunk.shape[0] if dim == -1 else dim)
                        for dim in self.context.get_tensor_shape(name)
                    )
                    out_shapes[name] = shape
                    dtype = self._output_dtypes.get(name, self.output_dtype)
                    d_buf = cuda.mem_alloc(int(np.prod(shape)) * dtype.itemsize)
                    d_outputs[name] = (d_buf, dtype)
                    self.context.set_tensor_address(name, int(d_buf))

                d_in = cuda.mem_alloc(chunk.nbytes)
                self.context.set_tensor_address(self.input_name, int(d_in))

                cuda.memcpy_htod_async(d_in, chunk, self.stream)
                self.context.execute_async_v3(stream_handle=self.stream.handle)

                if self.output_name in d_outputs:
                    out_buf, _dtype = d_outputs[self.output_name]
                    out_host = np.empty(out_shapes[self.output_name], dtype=self.output_dtype)
                    cuda.memcpy_dtoh_async(out_host, out_buf, self.stream)
                self.stream.synchronize()

                d_in.free()
                for buf, _dtype in d_outputs.values():
                    try:
                        buf.free()
                    except Exception:
                        pass

            if self._num_classes is None and out_host.ndim > 1:
                self._num_classes = out_host.shape[1]
            outputs.append(_log_softmax_np(out_host))

        if outputs:
            return np.concatenate(outputs, axis=0)
        # graceful empty result
        n_class = self._num_classes or (arr.shape[1] if arr.ndim > 1 else 0)
        return np.empty((0, n_class), dtype=np.float64)


def _maybe_create_trt_plan_runner(plan_path: Optional[str]) -> Optional[_TrtPlanFoldRunner]:
    if plan_path is None:
        return None
    try:
        return _TrtPlanFoldRunner(plan_path)
    except Exception as exc:
        print(f"[trt] Failed to use plan {plan_path}: {exc}")
        return None


def _load_models_cached(input_model_path: str, backend: str = "pytorch"):
    backend = _normalize_backend(backend)
    cache_key = (backend, input_model_path)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    folds = _discover_folds(input_model_path)
    if not folds:
        raise RuntimeError(f"No folds under {input_model_path}")

    model_folds = {}
    prefer_int8 = _prefer_int8(backend)
    for idx, name, run_dir in folds:
        if backend == "pytorch":
            model = TabNetClassifier()
            z = _pick_zip(run_dir)
            if z is None:
                raise FileNotFoundError(f"No model .zip in {run_dir}")
            model.load_model(z)
            model_folds[idx] = _TorchFoldRunner(model)
        elif backend.startswith("onnx"):
            onnx_path = _resolve_onnx_path(input_model_path, idx, prefer_int8=prefer_int8)
            model_folds[idx] = _OnnxFoldRunner(onnx_path, prefer_trt_provider=False)
        elif backend.startswith("tensorrt"):
            runner = _maybe_create_trt_plan_runner(
                _resolve_trt_plan_path(input_model_path, idx, prefer_int8=prefer_int8)
            )
            if runner is None:
                onnx_path = _resolve_onnx_path(input_model_path, idx, prefer_int8=prefer_int8)
                runner = _OnnxFoldRunner(onnx_path, prefer_trt_provider=True)
            model_folds[idx] = runner
        else:
            raise ValueError(f"Unknown backend: {backend}")

    meta = np.load(os.path.join(folds[0][2], "info.npy"), allow_pickle=True)[()]
    _MODEL_CACHE[cache_key] = (folds, model_folds, meta)
    return _MODEL_CACHE[cache_key]

def _iter_subdirs_with_result_tree(in_root: str, tree_name="Result_Tree"):
    f = ROOT.TFile.Open(in_root, "READ");  assert f and not f.IsZombie()
    pairs = []
    tree_name = tree_name.lstrip("/")
    for k in f.GetListOfKeys():
        obj = k.ReadObj()
        if not obj.InheritsFrom("TDirectory"): continue
        ch = obj.GetName()
        for k2 in obj.GetListOfKeys():
            obj2 = k2.ReadObj()
            if not obj2.InheritsFrom("TDirectory"): continue
            sub = obj2.GetName()
            f.cd(f"{ch}/{sub}")
            t = ROOT.gDirectory.Get(tree_name)
            if t and t.InheritsFrom("TTree"):
                pairs.append((ch, sub))
    if not pairs:
        tree = f.Get(tree_name)
        if tree and tree.InheritsFrom("TTree"):
            pairs.append(("", ""))
    f.Close()
    return pairs

def _infer_logp(
    shard_file: str,
    input_model_path: str,
    tree_name: str = "Result_Tree",
    tree_path: Optional[str] = None,
    backend: str = "pytorch",
) -> np.ndarray:
    folds, model_folds, meta = _load_models_cached(input_model_path, backend=backend)
    data_info = meta["data_info"].copy()
    K = len(data_info["tree_path_filter_str"])
    data_info["infer_mode"] = True
    # tree_path가 주어지면 그대로 사용하고, 없으면 tree_name만 사용
    clean_tree_name = tree_name.lstrip("/")
    use_tree_path = tree_path.lstrip("/") if tree_path else clean_tree_name
    print(f"Loading data from: {shard_file} with tree path: {use_tree_path}")
    data_info["tree_path_filter_str"] = [[(shard_file, use_tree_path, "")]]
    data = load_data_kfold(**data_info)
    N = data["X"].shape[0]
    out = np.zeros((N, K), dtype=np.float64)
    mask = np.zeros(N, dtype=bool)
    for idx, name, _ in folds:
        d = view_fold(data, idx)
        arr = d["val_features"]
        pred = (
            model_folds[idx].predict_log_proba(arr) if arr.shape[0] > 0 else np.empty((0, K))
        )
        pred = np.asarray(pred)
        _, fold_idx = data["folds"][idx]
        mask[fold_idx] = True
        out[fold_idx] = pred
    if not mask.all():
        raise RuntimeError("Incomplete inference coverage.")
    return out  # (N,K)

def _write_logp_tree(dest_file: str, chdir: str, subdir: str, logp: np.ndarray, tree_name="Result_Tree"):
    # dest_file는 미리 RECREATE로 만들어져 있다고 가정
    f = ROOT.TFile.Open(str(dest_file), "UPDATE");  assert f and not f.IsZombie()
    current_dir = f  # root or subdir
    if chdir:
        current_dir = f.GetDirectory(chdir) or f.mkdir(chdir)
    current_dir.cd()
    if subdir:
        current_dir = current_dir.GetDirectory(subdir) or current_dir.mkdir(subdir)
    current_dir.cd()
    t = ROOT.TTree(tree_name, "log_prob_* only")
    N, K = logp.shape
    bufs = [array.array("f", [0.0]) for _ in range(K)]
    for k in range(K):
        b = f"log_prob_{k}"
        t.Branch(b, bufs[k], f"{b}/F")
    for i in range(N):
        row = logp[i]
        for k in range(K):
            bufs[k][0] = float(row[k])
        t.Fill()
    t.Write("", ROOT.TObject.kOverwrite)
    f.Close()


def _merge_parts_to_dest(dest_file: str, parts_dir: str, pairs, tree_name="Result_Tree"):
    tree_name = tree_name.lstrip("/")
    ROOT.TFile.Open(dest_file, "RECREATE").Close()
    fout = ROOT.TFile.Open(dest_file, "UPDATE");  assert fout and not fout.IsZombie()
    try:
        for chdir, subdir in pairs:
            p = Path(parts_dir) / f"{chdir}__{subdir}.root"
            if not p.exists():
                raise RuntimeError(f"Missing part: {p}")
            src = ROOT.TFile.Open(str(p), "READ");  assert src and not src.IsZombie()
            if chdir or subdir:
                path = os.path.join(*[q for q in (chdir, subdir) if q])
                src.cd(path)
                t = ROOT.gDirectory.Get(tree_name)
            else:
                t = src.Get(tree_name)
            assert t and t.InheritsFrom("TTree")
            if chdir:
                d1 = fout.GetDirectory(chdir) or fout.mkdir(chdir)
            else:
                d1 = fout
            d1.cd()
            if subdir:
                d2 = d1.GetDirectory(subdir) or d1.mkdir(subdir)
            else:
                d2 = d1
            d2.cd()
            t.CloneTree(-1, "fast").Write(tree_name, ROOT.TObject.kOverwrite)
            src.Close()
            fout.cd()
    finally:
        fout.Close()


def _worker_infer_part(args):
    (
        input_root_file,
        input_model_path,
        part_path,
        chdir,
        subdir,
        tree_name,
        backend,
    ) = args
    tree_path = "/".join([p for p in (chdir, subdir, tree_name) if p])
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        logp = _infer_logp(
            input_root_file,
            input_model_path,
            tree_name=tree_name,
            tree_path=tree_path,
            backend=backend,
        )
        ROOT.TFile.Open(part_path, "RECREATE").Close()
        _write_logp_tree(part_path, chdir, subdir, logp, tree_name=tree_name)
        return ("ok", tree_path, part_path)
    except Exception:
        import traceback

        return ("err", tree_path, traceback.format_exc())


def make_score_friend_file_parallel(input_root_file: str,
                                    input_model_path: str,
                                    out_group_name: str,
                                    tree_name: str = "Result_Tree",
                                    backend: str = "pytorch",
                                    num_workers: int = 2,
                                    cleanup_parts: bool = True) -> str:
    """
    원본 옆 out_group_name/ 폴더에 같은 파일명으로 score-only ROOT 생성.
    내부 구조는 원본과 동일(chdir/subdir/Result_Tree) + 브랜치는 log_prob_*.
    트리별로 병렬 추론 → part 파일 → 병합 순서로 동작.
    """
    backend = _normalize_backend(backend)
    input_root_file = str(Path(input_root_file).resolve())
    tree_name = tree_name.lstrip("/")
    base_dir = Path(input_root_file).parent
    src_name = Path(input_root_file).name

    out_dir = base_dir / out_group_name
    out_dir.mkdir(exist_ok=True)
    parts_dir = out_dir / f".parts__{Path(src_name).stem}"
    parts_dir.mkdir(parents=True, exist_ok=True)

    pairs = _iter_subdirs_with_result_tree(input_root_file, tree_name=tree_name)
    if not pairs:
        alt_tree = (
            "Result_Tree"
            if tree_name != "Result_Tree"
            else "Template_Training_Tree"
        )
        if alt_tree != tree_name:
                alt_pairs = _iter_subdirs_with_result_tree(
                    input_root_file, tree_name=alt_tree
                )
                if alt_pairs:
                    tree_name = alt_tree
                    pairs = alt_pairs
    if not pairs:
        raise RuntimeError(f"No subdirs with {tree_name} found.")

    # 결과 파일 생성 준비
    dest_file = str(out_dir / src_name)

    tasks = [
        (
            input_root_file,
            input_model_path,
            str(parts_dir / f"{ch}__{sub}.root"),
            ch,
            sub,
            tree_name,
            backend,
        )
        for ch, sub in pairs
    ]

    errs, oks = [], []
    worker_count = max(1, num_workers)
    if worker_count == 1:
        for t in tasks:
            status, tag, payload = _worker_infer_part(t)
            if status == "ok":
                print(f"[OK ] {tag}  → {payload}")
                oks.append(payload)
            else:
                print(f"[ERR] {tag}\n{payload}")
                errs.append(tag)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=worker_count)
        try:
            for status, tag, payload in pool.imap_unordered(_worker_infer_part, tasks, chunksize=1):
                if status == "ok":
                    print(f"[OK ] {tag}  → {payload}")
                    oks.append(payload)
                else:
                    print(f"[ERR] {tag}\n{payload}")
                    errs.append(tag)
        finally:
            _shutdown_pool(pool, label="score-friend")

    if errs:
        raise RuntimeError(f"Part failures: {errs}")

    _merge_parts_to_dest(dest_file, str(parts_dir), pairs, tree_name=tree_name)

    if cleanup_parts:
        shutil.rmtree(parts_dir, ignore_errors=True)

    return dest_file

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


def infer_with_iter(input_folder, input_model_path, branch_name, result_folder_name, backend="pytorch"):
    import htcondor, shutil, ROOT, pathlib

    backend = _normalize_backend(backend)
    log_path = os.path.join(
        os.environ["DIR_PATH"],
        "TabNet_template",
        pathlib.Path(input_model_path).parent.absolute(),
        "infer_log",
    )

    eras = [era] if era != "All" else ["2016preVFP", "2016postVFP", "2017", "2018"]

    def _collect_legacy_entries(e):
        base = os.path.join(input_folder, e, result_folder_name)
        if not os.path.isdir(base):
            return []
        systs = [
            f
            for f in os.listdir(base)
            if os.path.isdir(os.path.join(base, f)) and "." not in f
        ]
        entries = []
        for syst in systs:
            syst_dir = os.path.join(base, syst)
            files = [
                os.path.join(syst_dir, f)
                for f in os.listdir(syst_dir)
                if f.endswith(".root")
            ]
            if files:
                entries.extend([(file, syst) for file in files])
        return entries

    def _collect_2024_entries():
        channel_names = ["Mu_TemplateTraining", "El_TemplateTraining"]
        candidate_dirs = []
        if result_folder_name:
            candidate = os.path.join(input_folder, result_folder_name)
            if os.path.isdir(candidate):
                candidate_dirs.append(candidate)
        if os.path.isdir(input_folder):
            candidate_dirs.append(input_folder)
        seen = set()
        for candidate in candidate_dirs:
            norm_candidate = os.path.normpath(candidate)
            if norm_candidate in seen:
                continue
            seen.add(norm_candidate)
            base_name = os.path.basename(norm_candidate)
            if base_name in channel_names:
                era_dir = os.path.join(norm_candidate, "2024")
                if os.path.isdir(era_dir):
                    yield from _iter_root_files(era_dir, base_name)
            else:
                for channel_name in channel_names:
                    era_dir = os.path.join(norm_candidate, channel_name, "2024")
                    if os.path.isdir(era_dir):
                        yield from _iter_root_files(era_dir, channel_name)

    def _iter_root_files(directory, label):
        for entry in sorted(os.listdir(directory)):
            if not entry.endswith(".root"):
                continue
            yield (os.path.join(directory, entry), label)
    Arg_list_file = "infer_arg_list.txt"
    #make empty log file
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, Arg_list_file), "w") as f:
        f.write("")
    for e in eras:
        print(e)
        entries = []
        if e == "2024":
            entries.extend(list(_collect_2024_entries()))
        entries.extend(_collect_legacy_entries(e))
        if not entries:
            continue


        for file, syst in entries:
            if "temp" in file or "update" in file:
                os.remove(file)
                continue

            print(file)
            outname = file.split("/")
            outname[-1] = outname[-1].replace(".root", "")
            outname = "_".join(outname[-5:])
            with open(os.path.join(log_path, Arg_list_file), "a") as f:
                f.write(f"{input_model_path} {file} {branch_name} {e} {backend}\n")
            job = htcondor.Submit(
                {
                    "universe": "vanilla",
                    "getenv": True,
                    "jobbatchname": f"Vcb_infer_{e}_{syst}_{outname}",
                    "executable": "/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/infer_write.sh",
                    "arguments": f"{input_model_path} {file} {branch_name} {e} {backend}",
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
        "--backend",
        dest="backend",
        type=str,
        default="tensorrt",
        help="inference backend: pytorch, onnx, onnx-int8, tensorrt, tensorrt-int8",
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
            backend=args.backend,
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
        make_score_friend_file_parallel(
            input_root_file=args.input_root_file,
            input_model_path=args.input_model,
            out_group_name=args.branch_name,
            tree_name="Template_Training_Tree" if args.era == "2024" else "Result_Tree",
            backend=args.backend,
        )
        #infer(args.input_root_file, args.input_model, args.branch_name)
    else:
        print("Wrong working mode")
