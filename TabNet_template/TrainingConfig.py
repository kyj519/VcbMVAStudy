# --- 필요한 의존성 ---
import os
import numpy as np
import torch
import inspect
import sys
import shutil
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

from eval_functions import ClassBalancedFocalLoss, make_maxsig_metric_cls, make_cb_focal_metric_cls

from helpers import pick_best_device, compute_class_counts, SaveEachEpochCallback
from pytorch_tabnet.tab_model import TabNetClassifier

@dataclass
class TabNetTrainConfig:
    # model & optim
    n_d: int = 16
    n_a: int = 16
    n_steps: int = 5
    lambda_sparse: float = 1e-4
    tabnet_gamma: float = 1.5
    mask_type: str = "entmax"

    lr: float = 2e-3
    weight_decay: float = 5e-3
    betas: Tuple[float, float] = (0.9, 0.999)

    # sched
    T0: int = 10
    eta_min: float = 1e-5
    warm_restart_mult: int = 2
    fine_tune: bool = False
    fine_tune_Tmax: int = 30
    fine_tune_eta_min: float = 1e-6

    # training
    batch_size: int = 8192 * 8
    num_virtual_minibatches: int = 32
    num_workers: int = 32
    patience: int = 30
    compute_importance: bool = False

    # loss/metrics
    floss_gamma: float = 0.0
    use_cb_focal: bool = True
    use_asimov_metric: bool = True
    pretrained_model: Optional[str] = None  # if set → fine_tune=True

    # varlist
    varlist: List[str] = field(default_factory=lambda: [
        "pt_w_u", "pt_w_d", "pt_had_t_b", "pt_lep_t_b",
        "m_had_t", "m_had_w",
        "bvsc_had_t_b", "bvsc_lep_t_b", "bvsc_w_u", "bvsc_w_d",
        "n_bjets", "n_jets",
        # "n_cjets", "cvsb_had_t_b", "cvsb_lep_t_b", "cvsl_had_t_b", "cvsl_lep_t_b", "cvsl_w_u", "cvsl_w_d", "cvsb_w_u", "cvsb_w_d",
        "best_mva_score", "least_dr_bb", "least_m_bb", "pt_tt", "ht",
    ])

    log_columns: List[str] = field(default_factory=lambda: [
        "pt_w_u", "pt_w_d", "pt_had_t_b", "pt_lep_t_b",
        "least_m_bb", "pt_tt", "ht",
    ])

    winsorize_columns: List[Tuple[str, Tuple[float, float]]] = field(default_factory=lambda: [
        ("m_had_t", (0, 400)),
        ("m_had_w", (0, 300))
    ])
    


    @property
    def config_name(self) -> str:
        """저장 시 파일명 등에 쓰기 좋은 설정 이름 (서브클래스가 자동 반영)."""
        return self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """dataclass → dict (JSON 직렬화용)."""
        from dataclasses import asdict
        d = asdict(self)
        # 필요하면 민감/거대한 필드 제거 가능
        return d

    def save_config_source(self, dst_dir: str) -> str:
        """
        이 설정 클래스가 정의된 .py 파일을 그대로 복사.
        - 모듈 파일이 없거나 인터프리터 상 정의되었으면 클래스 소스를 덤프.
        반환: 저장된 파일 경로
        """
        os.makedirs(dst_dir, exist_ok=True)
        out_path = os.path.join(dst_dir, f"{self.config_name}.py")
        try:
            mod = sys.modules[self.__class__.__module__]
            src_path = inspect.getsourcefile(mod) or inspect.getfile(mod)
            if src_path and os.path.exists(src_path):
                shutil.copy2(src_path, out_path)
                return out_path
        except Exception:
            pass  # fallback으로 넘어감

        # 모듈 파일을 못 찾은 경우: 클래스 소스를 직접 덤프
        try:
            src = inspect.getsource(self.__class__)
        except Exception:
            src = f"# Source unavailable for {self.config_name}\n"
        with open(out_path, "w") as f:
            f.write(src)
        return out_path

    # ---------- (1) 입력 튜플 빌더 ----------
    @staticmethod
    def _eras_list(era: str) -> List[str]:
        return [era] if era != "All" else ["2016preVFP", "2016postVFP", "2017", "2018"]

    @staticmethod
    def _cuts(rel_iso_muon: float = 0.15, met_pt: float = 25) -> Dict[str, str]:
        D_COND_MU = f"(met_pt>{met_pt})&&(lepton_rel_iso<{rel_iso_muon})"
        D_COND_EL = f"(met_pt>{met_pt})&&((electron_id_bit & (1 << 5)) != 0)"
        HF_BB_MASK = "(((genTtbarId%100) >= 51) && ((genTtbarId%100) <= 55))"
        HF_CC_MASK = "(((genTtbarId%100) >= 41) && ((genTtbarId%100) <= 45))"
        CX_DECAY_MASK = "((decay_mode//10)%10 == 4)"
        UX_DECAY_MASK = "((decay_mode//10)%10 == 2)"
        HF_MASK = f"{HF_BB_MASK}||{HF_CC_MASK}"
        return dict(D_COND_MU=D_COND_MU, D_COND_EL=D_COND_EL,
                    HF_BB_MASK=HF_BB_MASK, HF_CC_MASK=HF_CC_MASK, HF_MASK=HF_MASK, CX_DECAY_MASK=CX_DECAY_MASK, UX_DECAY_MASK=UX_DECAY_MASK)

    def build_input_tuple(self,
                          sample_folder_loc: str,
                          result_folder_name: str,
                          era: str,
                          include_extra_bkgs: bool = False) -> List[List[Tuple[str, str, str]]]:
        cuts = self._cuts()
        eras = self._eras_list(era)
        cls0, cls1, cls2, cls3, cls4, cls5 = [], [], [], [], [], []
        # class name
        class_labels = [
            r"$W\to c\bar b$",
            r"$W\to c(s/d)+\,extra b$",
            r"$W\to u(s/d)+\,extra b$",
            r"$W\to c(s/d)+\,extra c$",
            r"$W\to c(s/d)+\,\mathrm{light}$",
            r"$W\to u(s/d)+\,\mathrm{no}\ extra b$",
        ]
        for e in eras:
            # class 0: signal (TTLJ_WtoCB)
            cls0 += [
                (f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root",
                 "Mu/Central/Result_Tree", cuts["D_COND_MU"]),
                (f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root",
                 "El/Central/Result_Tree", cuts["D_COND_EL"]),
            ]
            # class 1: TTLJ(W->CS, CD) + BB
            cls1 += [
                (f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                 "Mu/Central/Result_Tree", f'{cuts["D_COND_MU"]}&&({cuts["HF_BB_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
                (f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                 "El/Central/Result_Tree", f'{cuts["D_COND_EL"]}&&({cuts["HF_BB_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
            ]
            # class 2: TTLJ(W->US, UD) + BB
            cls2 += [
                (f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                 "Mu/Central/Result_Tree", f'{cuts["D_COND_MU"]}&&({cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})'),
                (f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                 "El/Central/Result_Tree", f'{cuts["D_COND_EL"]}&&({cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})'),
            ]
            # class 3: TTLJ(W->CS, CD) + CC
            cls3 += [
                (f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                 "Mu/Central/Result_Tree", f'{cuts["D_COND_MU"]}&&({cuts["HF_CC_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
                (f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                 "El/Central/Result_Tree", f'{cuts["D_COND_EL"]}&&({cuts["HF_CC_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
            ]
            # class 4: TTLJ(W->CS, CD) + light
            cls4 += [
                (f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                 "Mu/Central/Result_Tree", f'{cuts["D_COND_MU"]}&&(!{cuts["HF_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
                (f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                 "El/Central/Result_Tree", f'{cuts["D_COND_EL"]}&&(!{cuts["HF_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
            ]
            # class 5: TTLJ(W->US, UD) + not BB
            cls5 += [
                (f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                 "Mu/Central/Result_Tree", f'{cuts["D_COND_MU"]}&&(!{cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})'),
                (f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                 "El/Central/Result_Tree", f'{cuts["D_COND_EL"]}&&(!{cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})'),
            ]
            
            # # (옵션) non-TT 배경을 class3에 추가하려면 include_extra_bkgs=True로 넘겨
            # if include_extra_bkgs and not self.fine_tune:
            #     extra = (
            #         "Vcb_DYJets_MG.root",
            #         "Vcb_QCD_bEnriched_HT1000to1500.root",
            #         "Vcb_QCD_bEnriched_HT1500to2000.root",
            #         "Vcb_QCD_bEnriched_HT2000toInf.root",
            #         "Vcb_QCD_bEnriched_HT200to300.root",
            #         "Vcb_QCD_bEnriched_HT300to500.root",
            #         "Vcb_QCD_bEnriched_HT500to700.root",
            #         "Vcb_QCD_bEnriched_HT700to1000.root",
            #         "Vcb_SingleTop_sch_Lep.root",
            #         "Vcb_SingleTop_tW_antitop_NoFullyHad.root",
            #         "Vcb_SingleTop_tW_top_NoFullyHad.root",
            #         "Vcb_SingleTop_tch_antitop_Incl.root",
            #         "Vcb_SingleTop_tch_top_Incl.root",
            #         "Vcb_TTLL_powheg.root",
            #         "Vcb_WJets_HT100to200.root",
            #         "Vcb_WJets_HT1200to2500.root",
            #         "Vcb_WJets_HT200to400.root",
            #         "Vcb_WJets_HT2500toInf.root",
            #         "Vcb_WJets_HT400to600.root",
            #         "Vcb_WJets_HT600to800.root",
            #         "Vcb_WJets_HT800to1200.root",
            #         "Vcb_WW_pythia.root",
            #         "Vcb_WZ_pythia.root",
            #         "Vcb_ZZ_pythia.root",
            #         "Vcb_ttHToNonbb.root",
            #         "Vcb_ttHTobb.root",
            #         "Vcb_ttWToLNu.root",
            #         "Vcb_ttWToQQ.root",
            #         "Vcb_ttZToLLNuNu.root",
            #         "Vcb_ttZToQQ.root",
            #         "Vcb_ttZToQQ_ll.root",
            #     )
            #     for fname in extra:
            #         if "TTLJ" in fname:
            #             continue
            #         base = f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/{fname}"
            #         cls3 += [
            #             (base, "Mu/Central/Result_Tree", cuts["D_COND_MU"]),
            #             (base, "El/Central/Result_Tree", cuts["D_COND_EL"]),
            #         ]

        return_arr = [cls0, cls1, cls2, cls3, cls4, cls5]
        if len(return_arr) != len(class_labels):
            raise ValueError("Class labels length mismatch.")
        return return_arr, class_labels

    def make_data_info(self,
                       sample_folder_loc: str,
                       result_folder_name: str,
                       era: str,
                       add_year_index: int = 0,
                       sample_bkg: int = 1,
                       include_extra_bkgs: bool = False) -> Dict[str, Any]:
        input_tuple, class_labels = self.build_input_tuple(sample_folder_loc, result_folder_name, era,
                                             include_extra_bkgs=include_extra_bkgs)
        varlist = self.varlist
        if add_year_index != 0:
            varlist = varlist + ["year_index"]
        log_columns = self.log_columns if self.log_columns else None
        winsorize_cols = self.winsorize_columns if self.winsorize_columns else None

        return dict(
            tree_path_filter_str=tuple(input_tuple),
            varlist=varlist,
            log_columns=log_columns,
            winsorize_columns=winsorize_cols,
            #test_ratio=test_ratio,
            #val_ratio=val_ratio,
            sample_bkg=sample_bkg,
        )

    # ---------- (2) 모델/로스/메트릭/트레이닝 인포 ----------
    def build_model(self, data: Dict[str, Any],
                    device: Optional[torch.device] = None,
                    checkpoint: Optional[str] = None) -> Tuple[TabNetClassifier, Dict[str, Any]]:
        device = device or pick_best_device(6.0)

        model_info = dict(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            lambda_sparse=self.lambda_sparse, gamma=self.tabnet_gamma,
            mask_type=self.mask_type, verbose=1,
            cat_idxs=list(data["cat_idxs"]),
            cat_dims=list(data["cat_dims"]),
            cat_emb_dim=1,
            device_name=str(device),
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=self.lr, weight_decay=self.weight_decay, betas=self.betas),
        )

        if not self.fine_tune:
            model_info.update(
                scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                scheduler_params=dict(T_0=self.T0, T_mult=self.warm_restart_mult,
                                      eta_min=self.eta_min, last_epoch=-1, verbose=False)
            )
        else:
            def build_warmup_cosine(optimizer, *,
                        warmup_epochs: int = 5,
                        T_max: int = 30,
                        eta_min: float = 1e-6,
                        last_epoch: int = -1,
                        verbose: bool = False):
                # 1) 선형 워밍업: epoch 0..warmup_epochs-1 에서 0→1로 증가
                warmup = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda e: min((e + 1) / float(warmup_epochs), 1.0),
                    verbose=verbose
                )
                # 2) 코사인 어닐링
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=T_max, eta_min=eta_min, verbose=verbose
                )
                # 3) 시퀀셜로 연결
                sched = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                    last_epoch=last_epoch,
                    verbose=verbose
                )
                return sched
            model_info.update(
                scheduler_fn=build_warmup_cosine,
                scheduler_params=dict(warmup_epochs=5, T_max=self.fine_tune_Tmax,
                                      eta_min=self.fine_tune_eta_min,
                                      last_epoch=-1, verbose=False)
            )

        clf = TabNetClassifier(**model_info) if self.pretrained_model is None else TabNetClassifier()
        if self.pretrained_model is not None:
            clf.load_model(self.pretrained_model)
            clf.optimizer_fn = model_info["optimizer_fn"]
            clf.optimizer_params = model_info["optimizer_params"]
            clf.scheduler_fn = model_info["scheduler_fn"]
            clf.scheduler_params = model_info["scheduler_params"]

        if checkpoint is not None and self.pretrained_model is None:
            print(f"Loading checkpoint: {checkpoint}")
            clf.load_model(checkpoint)

        # 디바이스 확실히 밀착
        clf.device_name = str(device)
        clf.device = torch.device(clf.device_name)

        return clf, model_info

    @staticmethod
    def _log_train_stats(y, w):
        labels, inv = np.unique(y, return_inverse=True)
        sums = np.bincount(inv, weights=w, minlength=labels.size)
        betas = (sums - 1.0) / np.maximum(sums, 1.0)
        ws = (1.0 - betas) / (1.0 - np.power(betas, np.maximum(sums, 1.0)))
        print(f"[train] sumW: {sums}  ws(norm*K): {ws/ws.sum()*len(ws)}")

    def build_loss_and_metrics(self, data: Dict[str, Any], device: Optional[torch.device] = None):
        device = device or pick_best_device(6.0)
        num_classes = int(np.max(data["train_y"])) + 1
        counts_train = compute_class_counts(data["train_y"], data["train_weight"], num_classes)
        self._log_train_stats(data["train_y"], data["train_weight"])

        loss_fn = None
        if self.use_cb_focal:
            loss_fn = ClassBalancedFocalLoss(
                counts=counts_train, num_classes=num_classes,
                gamma=self.floss_gamma, device=device, force_cpu=False
            )

        eval_metrics = []
        if self.use_asimov_metric:
            eval_metrics.append(make_maxsig_metric_cls(bins=100, mode='asimov', clamp_nonneg=True, clip01=True))
        if self.use_cb_focal:
            eval_metrics.append(make_cb_focal_metric_cls(num_classes=num_classes, gamma=self.floss_gamma, device=torch.device("cpu")))
        return loss_fn, eval_metrics, counts_train, num_classes

    def build_train_info(self, data: Dict[str, Any],
                         loss_fn,
                         eval_metrics,
                         model_save_path: str) -> Dict[str, Any]:
        callbacks = [SaveEachEpochCallback(save_dir=os.path.join(model_save_path, "checkpoints"))]
        return dict(
            X_train=data["train_features"],
            y_train=data["train_y"],
            w_train=data["train_weight"],
            eval_set=[(data["val_features"], data["val_y"], data["val_weight"])],
            eval_metric=eval_metrics,
            max_epochs=1000,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            virtual_batch_size=self.batch_size // self.num_virtual_minibatches,
            patience=self.patience,
            loss_fn=loss_fn,
            callbacks=callbacks,
            compute_importance=self.compute_importance,
        )

    @staticmethod
    def write_info_files(model_save_path: str,
                         data_info: Dict[str, Any],
                         model_info: Dict[str, Any],
                         train_info: Dict[str, Any],
                         floss_gamma: float,
                         counts_train: np.ndarray,
                         *,
                         cfg: "TabNetTrainConfig" = None,   # ← 추가: 현재 설정 인스턴스
                         save_config_py: bool = True,       # ← 추가: .py 복사 여부
                         save_config_json: bool = True):    # ← 추가: JSON 저장 여부
        os.makedirs(model_save_path, exist_ok=True)

        # info.txt
        with open(os.path.join(model_save_path, "info.txt"), "w") as f:
            f.write("Training info\n")
            for k, v in train_info.items():
                if k in ["X_train", "y_train", "w_train", "eval_set", "loss_fn", "callbacks", "eval_metric"]:
                    continue
                f.write(f"{k}: {v}\n")
            f.write(f"gamma: {floss_gamma}\n")
            f.write(f"train_counts: {counts_train.tolist()}\n")
            f.write("###########################################################\n")
            f.write("model info\n")
            for k, v in model_info.items():
                f.write(f"{k}: {v}\n")
            f.write("###########################################################\n")
            f.write("data info\n")
            for k, v in data_info.items():
                f.write(f"{k}: {v}\n")

        # info.npy (그대로 유지)
        np.save(
            os.path.join(model_save_path, "info.npy"),
            np.array(
                {
                    "train_info": {k: v for k, v in train_info.items()
                                   if k not in ["X_train", "y_train", "w_train", "eval_set", "loss_fn", "callbacks", "eval_metric"]},
                    "model_info": model_info,
                    "data_info": data_info,
                },
                dtype=object,
            ),
        )

        # (추가) 설정 JSON 저장
        if save_config_json and cfg is not None:
            with open(os.path.join(model_save_path, f"{cfg.config_name}.json"), "w") as jf:
                json.dump(cfg.to_dict(), jf, indent=2, ensure_ascii=False)

        # (추가) 현재 설정 .py 파일 그대로 복사
        if save_config_py and cfg is not None:
            cfg.save_config_source(model_save_path) 

            
# ----------------------------
# Convenience: fetch one fold's (train/val) arrays
# ----------------------------

@staticmethod
def get_fold(dataset: Dict[str, np.ndarray], k: int = 0):
    """Return (Xtr, ytr, wtr, Xval, yval, wval) for fold k."""
    X, y, w = dataset["X"], dataset["y"], dataset["weight"]
    folds: List[Tuple[np.ndarray, np.ndarray]] = dataset["folds"]
    if not (0 <= k < len(folds)):
        raise IndexError(f"k={k} out of range for {len(folds)} folds")
    train_idx, val_idx = folds[k]
    return X[train_idx], y[train_idx], w[train_idx], X[val_idx], y[val_idx], w[val_idx]

@staticmethod
def view_fold(dataset: Dict[str, Any], k: int) -> Dict[str, Any]:
    Xtr, ytr, wtr, Xval, yval, wval = get_fold(dataset, k)
    return dict(
        train_features=Xtr,
        train_y=ytr,
        train_weight=wtr,
        val_features=Xval,
        val_y=yval,
        val_weight=wval,
        # 모델 빌드에 그대로 전달
        cat_idxs=dataset["cat_idxs"],
        cat_dims=dataset["cat_dims"],
    )