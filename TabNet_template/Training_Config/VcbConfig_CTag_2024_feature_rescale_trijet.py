from dataclasses import dataclass, field
import sys, os
from typing_extensions import override

sys.path.append(os.environ.get("DIR_PATH") + "/TabNet_template")
from TrainingConfig import TabNetTrainConfig as Base
from typing import List, Optional, Tuple, Dict


@dataclass
class TabNetTrainConfig(Base):

    # --- [1] TabNet 구조적 하이퍼파라미터 튜닝 ---
    n_d: int = 64
    n_a: int = 64
    n_steps: int = 4  # 기존 8 -> 4: 과적합 방지 및 연산 속도 대폭 향상
    lambda_sparse: float = (
        1e-3  # Base(1e-4) 오버라이드: 물리 변수 중 노이즈성 피처 가지치기 강화
    )

    # --- [2] 학습 파라미터 및 Ghost Batch Norm 세팅 ---
    T0: int = 20
    warmup_epochs: int = 10
    patience: int = 100
    batch_size: int = 8192 * 4  # 32,768
    num_virtual_minibatches: int = (
        128  # 32768 / 128 = 가상 배치 사이즈 256 (TabNet 권장치)
    )
    lr: float = 2e-3

    # --- [3] 범주형 변수 세팅 ---
    categorical_columns: Optional[List[str]] = field(
        default_factory=lambda: ["Cat_w_u", "Cat_w_d"]
    )
    categorical_dims: Optional[Dict[str, int]] = field(
        default_factory=lambda: {"Cat_w_u": 12, "Cat_w_d": 12}
    )
    cat_emb_dim: int = 4  # 기존 2 -> 4: 12개 범주 정보를 더 넉넉하게 임베딩

    # --- [4] 가중치 및 변수 리스트 ---
    weight_fields: List[str] = field(
        default_factory=lambda: [
            "weight_Central",
            "sign(weight_mc)",
        ]
    )

    varlist: List[str] = field(
        default_factory=lambda: [
            "m_had_w",
            "pt_w_u",
            "pt_w_d",
            "eta_w_u",
            "eta_w_d",
            "Cat_w_u",
            "Cat_w_d",
            # "logp_class_0", "logp_class_1", "logp_class_2",
            # "logp_class_3", "logp_class_4", "logp_class_5",
            "pre_softmax_tt_Wcb",
            "pre_softmax_ttLF",
            "pre_softmax_ttcj",
            "pre_softmax_tt2c",
            "pre_softmax_ttcc",
            "pre_softmax_ttbj",
            "pre_softmax_tt2b",
            "pre_softmax_ttbb",
            "detection_score_logp",
            "assignment_logp",
            "n_bjets",
            "n_jets",
            "n_cjets",
            "ht",
            "Met_Pt",
        ]
    )

    preprocess_mode: Optional[str] = None

    log_columns: List[str] = field(default_factory=list)

    norm_columns: List[str] = field(
        default_factory=lambda: [
            "eta_w_u",
            "eta_w_d",
            # "logp_class_0",
            # "logp_class_1",
            # "logp_class_2",
            # "logp_class_3",
            # "logp_class_4",
            # "logp_class_5",
            "pre_softmax_tt_Wcb",
            "pre_softmax_ttLF",
            "pre_softmax_ttcj",
            "pre_softmax_tt2c",
            "pre_softmax_ttcc",
            "pre_softmax_ttbj",
            "pre_softmax_tt2b",
            "pre_softmax_ttbb",
            "detection_score_logp",
            "assignment_logp",
        ]
    )

    log_norm_columns: List[str] = field(
        default_factory=lambda: ["pt_w_u", "pt_w_d", "ht", "Met_Pt"]
    )

    winsorize_columns: List[Tuple[str, Tuple[float, float]]] = field(
        default_factory=list
    )

    winsorize_norm_columns: List[Tuple[str, Tuple[float, float]]] = field(
        default_factory=lambda: [("m_had_w", (0, 300))]
    )

    assume_btag_mode: bool = False
    n_folds: int = 4

    @override
    @staticmethod
    def _cuts(rel_iso_muon: float = 0.15, met_pt: float = 25) -> Dict[str, str]:
        HF_BB_MASK = "(((genTtbarId%100) >= 51) && ((genTtbarId%100) <= 55))"
        HF_CC_MASK = "(((genTtbarId%100) >= 41) && ((genTtbarId%100) <= 45))"
        CX_DECAY_MASK = "((decay_mode//10)%10 == 4)"
        UX_DECAY_MASK = "((decay_mode//10)%10 == 2)"
        HF_MASK = f"{HF_BB_MASK}||{HF_CC_MASK}"
        RECO_CORRECT_MASK = "(chk_reco_correct==1)"
        return dict(
            HF_BB_MASK=HF_BB_MASK,
            HF_CC_MASK=HF_CC_MASK,
            HF_MASK=HF_MASK,
            CX_DECAY_MASK=CX_DECAY_MASK,
            UX_DECAY_MASK=UX_DECAY_MASK,
            RECO_CORRECT_MASK=RECO_CORRECT_MASK,
        )

    @override
    def build_input_tuple(
        self,
        sample_folder_loc: str,
        result_folder_name: str,
        era: str,
        include_extra_bkgs: bool = False,
    ) -> List[List[Tuple[str, str, str]]]:
        cuts = self._cuts()
        eras = self._eras_list(era)

        cls0, cls1, cls2, cls3, cls4, cls5, cls6 = [], [], [], [], [], [], []

        class_labels = [
            r"$W\to c\bar b (reco correct)$",
            r"$W\to c\bar b (reco wrong)$",
            r"$W\to c(s/d)+\,extra b$",
            r"$W\to u(s/d)+\,extra b$",
            r"$W\to c(s/d)+\,extra c$",
            r"$W\to c(s/d)+\,\mathrm{light}$",
            r"$W\to u(s/d)+\,\mathrm{no}\ extra b$",
        ]

        for e in eras:
            # 경로 중복을 최소화하여 가독성 개선
            base_mu = f"{sample_folder_loc}/{result_folder_name}/Mu_Unmapped_TriJet_TemplateTraining_PseudoCont/{e}"
            base_el = f"{sample_folder_loc}/{result_folder_name}/El_Unmapped_TriJet_TemplateTraining_PseudoCont/{e}"

            vcb_mu = (f"{base_mu}/TTLJ_Vcb_powheg.root", "Template_Training_Tree")
            vcb_el = (f"{base_el}/TTLJ_Vcb_powheg.root", "Template_Training_Tree")
            ttlj_mu = (f"{base_mu}/TTLJ_powheg.root", "Template_Training_Tree")
            ttlj_el = (f"{base_el}/TTLJ_powheg.root", "Template_Training_Tree")

            cls0.extend(
                [
                    (*vcb_mu, cuts["RECO_CORRECT_MASK"]),
                    (*vcb_el, cuts["RECO_CORRECT_MASK"]),
                ]
            )
            cls1.extend(
                [
                    (*vcb_mu, f"!{cuts['RECO_CORRECT_MASK']}"),
                    (*vcb_el, f"!{cuts['RECO_CORRECT_MASK']}"),
                ]
            )

            cls2.extend(
                [
                    (*ttlj_mu, f"({cuts['HF_BB_MASK']})&&({cuts['CX_DECAY_MASK']})"),
                    (*ttlj_el, f"({cuts['HF_BB_MASK']})&&({cuts['CX_DECAY_MASK']})"),
                ]
            )

            cls3.extend(
                [
                    (*ttlj_mu, f"({cuts['HF_BB_MASK']})&&({cuts['UX_DECAY_MASK']})"),
                    (*ttlj_el, f"({cuts['HF_BB_MASK']})&&({cuts['UX_DECAY_MASK']})"),
                ]
            )

            cls4.extend(
                [
                    (*ttlj_mu, f"({cuts['HF_CC_MASK']})&&({cuts['CX_DECAY_MASK']})"),
                    (*ttlj_el, f"({cuts['HF_CC_MASK']})&&({cuts['CX_DECAY_MASK']})"),
                ]
            )

            cls5.extend(
                [
                    (*ttlj_mu, f"(!{cuts['HF_MASK']})&&({cuts['CX_DECAY_MASK']})"),
                    (*ttlj_el, f"(!{cuts['HF_MASK']})&&({cuts['CX_DECAY_MASK']})"),
                ]
            )

            cls6.extend(
                [
                    (*ttlj_mu, f"(!{cuts['HF_BB_MASK']})&&({cuts['UX_DECAY_MASK']})"),
                    (*ttlj_el, f"(!{cuts['HF_BB_MASK']})&&({cuts['UX_DECAY_MASK']})"),
                ]
            )

        return [cls0, cls1, cls2, cls3, cls4, cls5, cls6], class_labels
