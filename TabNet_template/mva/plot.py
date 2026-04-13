import sys
from pathlib import Path

# Allow imports when running from outside the project root
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from helpers import (
    _discover_folds,
    _load_training_config_module,
    _pick_zip,
    explain_fast,
    predict_logit_fast,
    task,
    view_fold,
)
from input_preprocessing import apply_feature_preprocess
from tabnet_compat import load_tabnet_model
from data.root_data_loader_awk import load_dataset_npz_json


def _load_fold_preprocess_info(run_dir: str):
    import os
    import numpy as np

    info_path = os.path.join(run_dir, "info.npy")
    if not os.path.exists(info_path):
        return None
    info = np.load(info_path, allow_pickle=True)[()]
    return info.get("preprocess_info")


def _pretty_class_label(label: str) -> str:
    import re

    text = str(label).strip()
    if text.startswith("$") and text.endswith("$"):
        text = text[1:-1]

    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    text = text.replace(r"\to", "→")
    text = text.replace(r"\bar b", "b")
    text = text.replace(r"\,", " ")
    text = text.replace(r"\ ", " ")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("(reco correct)", "(Reco. correct)")
    text = text.replace("(reco wrong)", "(Reco. wrong)")
    text = re.sub(r"\s*\+\s*", " + ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _pair_label(sig_label: str, bkg_label: str) -> str:
    return f"{_pretty_class_label(sig_label)} vs {_pretty_class_label(bkg_label)}"


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
            preprocess_info = _load_fold_preprocess_info(run_dir)
            data_k["train_features"] = apply_feature_preprocess(
                data_k["train_features"], preprocess_info
            )
            data_k["val_features"] = apply_feature_preprocess(
                data_k["val_features"], preprocess_info
            )

        with task(f"[{os.path.basename(run_dir)}] Model loading"):
            model = TabNetClassifier()
            if checkpoint_override is None:
                # dir 내에서 적절한 zip 선택
                ckpt = _pick_zip(run_dir)
                if ckpt is None:
                    raise FileNotFoundError(f"No model .zip file found in {run_dir}")
                load_tabnet_model(model, ckpt)
                out_dir = os.path.join(out_parent_dir)
                os.makedirs(out_dir, exist_ok=True)
                print(f"  - Loaded from: {ckpt}")
            else:
                load_tabnet_model(model, checkpoint_override)
                m = re.search(
                    r"model_epoch(\d+)\.zip$", os.path.basename(checkpoint_override)
                )
                epoch_str = m.group(1) if m else "UNK"
                out_dir = os.path.join(out_parent_dir, f"ckpt{epoch_str}")
                os.makedirs(out_dir, exist_ok=True)
                print(f"  - Loaded from: {checkpoint_override}")

        # ---------- ROC on test ----------
        with task(f"[{os.path.basename(run_dir)}] ROC AUC evaluation and plotting"):
            pretty_labels = [_pretty_class_label(label) for label in class_labels]

            # binary 가정: class 0에 대한 점수 사용 (기존 코드 유지)
            y_inv = np.logical_not(data_k["val_y"]).astype(int)
            logit = predict_logit_fast(
                model, data_k["val_features"], batch_size=BATCH_LOAD
            )
            proba = softmax(logit, axis=1)
            proba0 = proba[:, 0]
            num_class = logit.shape[1]
            label_str = f"{pretty_labels[0]} vs Others"
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

            if num_class > 1:
                y_inv_1 = (data_k["val_y"] == 1).astype(int)
                proba1 = proba[:, 1]
                label_str_1 = f"{pretty_labels[1]} vs Others"
                postTrainingToolkit.ROC_AUC(
                    score=proba1,
                    y=y_inv_1,
                    plot_path=out_dir,
                    fname="ROC_class1.pdf",
                    extra_text=label_str_1,
                    # weight=data.get("test_weight", None),
                )
                postTrainingToolkit.ROC_AUC(
                    score=proba1,
                    y=y_inv_1,
                    plot_path=out_dir,
                    fname="ROC_class1_log.pdf",
                    scale="log",
                    extra_text=label_str_1,
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
                label_str = _pair_label(class_labels[sig_idx], class_labels[bkg_idx])

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

                labels_by_bkg.append(f"{pretty_labels[bkg_idx]} (AUC: {local_auc:.4f})")

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
                extra_text=f"Signal: {pretty_labels[0]}",
                legend_loc="upper center",
                legend_bbox_to_anchor=(0.5, -0.12),
                legend_ncols=2,
                legend_fontsize=13,
            )
            postTrainingToolkit.ROC_AUC(
                score=scores_by_bkg,
                y=yinv_by_bkg,
                plot_path=out_dir,
                fname=f"ROC_class_compare_log.pdf",
                labels=labels_by_bkg,
                scale="log",
                log_xmin=1e-3,
                extra_text=f"Signal: {pretty_labels[0]}",
                legend_loc="upper center",
                legend_bbox_to_anchor=(0.5, -0.12),
                legend_ncols=2,
                legend_fontsize=13,
            )

            if num_class > 1:
                scores_by_bkg_1 = []
                labels_by_bkg_1 = []
                yinv_by_bkg_1 = []

                sig_idx_local = 1
                for bkg_idx in range(num_class):
                    if bkg_idx == sig_idx_local:
                        continue
                    local_signal_mask = data_k["val_y"] == sig_idx_local
                    local_bkg_mask = data_k["val_y"] == bkg_idx
                    local_mask = local_signal_mask | local_bkg_mask
                    local_y = np.where(
                        data_k["val_y"][local_mask] == sig_idx_local, 1, 0
                    ).astype(np.int8)
                    local_margin = (
                        logit[local_mask, sig_idx_local] - logit[local_mask, bkg_idx]
                    )
                    local_score = sigmoid_stable(local_margin)
                    label_str = _pair_label(
                        class_labels[sig_idx_local], class_labels[bkg_idx]
                    )

                    scores_by_bkg_1.append(local_score)
                    yinv_by_bkg_1.append(local_y)

                    local_auc = postTrainingToolkit.ROC_AUC(
                        score=local_score,
                        y=local_y,
                        plot_path=out_dir,
                        fname=f"ROC_{sig_idx_local}_VS_{bkg_idx}.pdf",
                        extra_text=label_str,
                        # weight=data.get("test_weight", None),
                    )

                    labels_by_bkg_1.append(
                        f"{pretty_labels[bkg_idx]} (AUC: {local_auc:.4f})"
                    )

                    postTrainingToolkit.ROC_AUC(
                        score=local_score,
                        y=local_y,
                        plot_path=out_dir,
                        fname=f"ROC_{sig_idx_local}_VS_{bkg_idx}_log.pdf",
                        scale="log",
                        extra_text=label_str,
                        # weight=data.get("test_weight", None),
                    )

                postTrainingToolkit.ROC_AUC(
                    score=scores_by_bkg_1,
                    y=yinv_by_bkg_1,
                    plot_path=out_dir,
                    fname="ROC_class1_compare.pdf",
                    labels=labels_by_bkg_1,
                    extra_text=f"Signal: {pretty_labels[sig_idx_local]}",
                    legend_loc="upper center",
                    legend_bbox_to_anchor=(0.5, -0.12),
                    legend_ncols=2,
                    legend_fontsize=13,
                )
                postTrainingToolkit.ROC_AUC(
                    score=scores_by_bkg_1,
                    y=yinv_by_bkg_1,
                    plot_path=out_dir,
                    fname="ROC_class1_compare_log.pdf",
                    labels=labels_by_bkg_1,
                    scale="log",
                    extra_text=f"Signal: {pretty_labels[sig_idx_local]}",
                    legend_loc="upper center",
                    legend_bbox_to_anchor=(0.5, -0.12),
                    legend_ncols=2,
                    legend_fontsize=13,
                )

        # ---------- Confusion matrix ----------
        with task(f"[{os.path.basename(run_dir)}] Confusion matrix plotting"):
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix

            y_true = data_k["val_y"]
            y_pred = np.argmax(logit, axis=1)
            weights = data_k.get("val_weight", None)
            labels_idx = np.arange(num_class)
            cm = confusion_matrix(
                y_true, y_pred, labels=labels_idx, sample_weight=weights
            )

            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(
                cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0
            )

            if class_labels is None or len(class_labels) != num_class:
                cm_labels = [str(i) for i in labels_idx]
            else:
                cm_labels = class_labels

            def _plot_cm(matrix, fname, title, fmt):
                fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=150)
                im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
                ax.figure.colorbar(im, ax=ax)
                ax.set(
                    xticks=labels_idx,
                    yticks=labels_idx,
                    xticklabels=cm_labels,
                    yticklabels=cm_labels,
                    xlabel="Predicted",
                    ylabel="True",
                    title=title,
                )
                plt.setp(
                    ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
                )
                thresh = np.nanmax(matrix) / 2.0 if matrix.size else 0.0
                for i in range(num_class):
                    for j in range(num_class):
                        val = matrix[i, j]
                        text = "nan" if np.isnan(val) else fmt.format(val)
                        ax.text(
                            j,
                            i,
                            text,
                            ha="center",
                            va="center",
                            color="white" if val > thresh else "black",
                            fontsize=8,
                        )
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, fname))
                plt.close(fig)

            count_fmt = "{:.0f}" if np.issubdtype(cm.dtype, np.integer) else "{:.2f}"
            _plot_cm(cm, "confusion_matrix.pdf", "Confusion Matrix", count_fmt)
            _plot_cm(cm_norm, "confusion_matrix_norm.pdf", "Confusion Matrix (Normalized)", "{:.2f}")

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
