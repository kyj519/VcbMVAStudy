from dataclasses import dataclass, field
import sys,os
from typing_extensions import override
sys.path.append(os.environ.get('DIR_PATH')+'/TabNet_template')
from TrainingConfig import TabNetTrainConfig as Base 
from typing import List, Optional, Tuple, Dict

@dataclass
class TabNetTrainConfig(Base):
    
    weight_fields: List[str] = field(
        default_factory=lambda: [
            "weight_Central",
            "sign(weight_mc)",
        ]
    )

    n_d: int = 64
    n_a: int = 64
    n_steps: int = 10
    T0: int = 10
    patience: int = 100
    batch_size: int = 8192*4
    lr: float = 2e-3 
    cat_emb_dim: int = 2 
     
    categorical_columns: Optional[List[str]] = field(
        default_factory=lambda: ["Cat_w_u", "Cat_w_d"]
    )
    categorical_dims: Optional[Dict[str, int]] = field(
        default_factory=lambda: {"Cat_w_u": 12, "Cat_w_d": 12}
    )
    
    varlist: List[str] = field(
        default_factory=lambda: [
            "m_had_w",
            "pt_w_u",
            "pt_w_d",
            "eta_w_u",
            "eta_w_d",
            "ilr_dim1_w_u",
            "ilr_dim1_w_d",
            "ilr_dim2_w_u",
            "ilr_dim2_w_d",
            # "Cat_w_u",
            # "Cat_w_d",
            # "N0_w_u",
            # "L0_w_u",
            # "C0_w_u",
            # "C1_w_u",
            # "C2_w_u",
            # "C3_w_u",
            # "C4_w_u",
            # "B0_w_u",
            # "B1_w_u",
            # "B2_w_u",
            # "B3_w_u",
            # "B4_w_u",
            # "N0_w_d",
            # "L0_w_d",
            # "C0_w_d",
            # "C1_w_d",
            # "C2_w_d",
            # "C3_w_d",
            # "C4_w_d",
            # "B0_w_d",
            # "B1_w_d",
            # "B2_w_d",
            # "B3_w_d",
            # "B4_w_d",
            "logp_class_0",
            "logp_class_1",
            "logp_class_2",
            "logp_class_3",
            "logp_class_4",
            "logp_class_5",
            "detection_score_logp",
            "assignment_logp"
        ]
    )

    log_columns: List[str] = field(default_factory=lambda: [
        "pt_w_u", "pt_w_d"
    ])

    winsorize_columns: List[Tuple[str, Tuple[float, float]]] = field(default_factory=lambda: [
        ("m_had_w", (0, 300))
    ])
    
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
        return dict(HF_BB_MASK=HF_BB_MASK, HF_CC_MASK=HF_CC_MASK,
                    HF_MASK=HF_MASK, CX_DECAY_MASK=CX_DECAY_MASK, UX_DECAY_MASK=UX_DECAY_MASK,
                    RECO_CORRECT_MASK=RECO_CORRECT_MASK)


    @override
    def build_input_tuple(self,
                          sample_folder_loc: str,
                          result_folder_name: str,
                          era: str,
                          include_extra_bkgs: bool = False) -> List[List[Tuple[str, str, str]]]:
        cuts = self._cuts()
        eras = self._eras_list(era)
        cls0, cls1, cls2, cls3, cls4, cls5, cls6 = [], [], [], [], [], [], []
        # class name
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
            cls0 += [
                (f"{sample_folder_loc}/{result_folder_name}/Mu_OTv3_QuadJet_TemplateTraining/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
                 "Template_Training_Tree", cuts['RECO_CORRECT_MASK']),
                (f"{sample_folder_loc}/{result_folder_name}/El_OTv3_QuadJet_TemplateTraining/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
                 "Template_Training_Tree", cuts['RECO_CORRECT_MASK']),
            ]
            cls1 += [
                (f"{sample_folder_loc}/{result_folder_name}/Mu_OTv3_QuadJet_TemplateTraining/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
                 "Template_Training_Tree", f'!{cuts["RECO_CORRECT_MASK"]}'),
                (f"{sample_folder_loc}/{result_folder_name}/El_OTv3_QuadJet_TemplateTraining/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
                 "Template_Training_Tree", f'!{cuts["RECO_CORRECT_MASK"]}'),
            ]

            # class 1: TTLJ(W->CS, CD) + BB
            cls2 += [
                (f"{sample_folder_loc}/{result_folder_name}/Mu_OTv3_QuadJet_TemplateTraining/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_powheg.root",
                 "Template_Training_Tree", f'({cuts["HF_BB_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
                (f"{sample_folder_loc}/{result_folder_name}/El_OTv3_QuadJet_TemplateTraining/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_powheg.root",
                 "Template_Training_Tree", f'({cuts["HF_BB_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
            ]
            # class 2: TTLJ(W->US, UD) + BB
            cls3 += [
                (f"{sample_folder_loc}/{result_folder_name}/Mu_OTv3_QuadJet_TemplateTraining/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_powheg.root",
                 "Template_Training_Tree", f'({cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})'),
                (f"{sample_folder_loc}/{result_folder_name}/El_OTv3_QuadJet_TemplateTraining/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_powheg.root",
                 "Template_Training_Tree", f'({cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})'),
            ]
            # class 3: TTLJ(W->CS, CD) + CC
            cls4 += [
                (f"{sample_folder_loc}/{result_folder_name}/Mu_OTv3_QuadJet_TemplateTraining/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_powheg.root",
                 "Template_Training_Tree", f'({cuts["HF_CC_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
                (f"{sample_folder_loc}/{result_folder_name}/El_OTv3_QuadJet_TemplateTraining/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_powheg.root",
                 "Template_Training_Tree", f'({cuts["HF_CC_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
            ]
            # class 4: TTLJ(W->CS, CD) + light
            cls5 += [
                (f"{sample_folder_loc}/{result_folder_name}/Mu_OTv3_QuadJet_TemplateTraining/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_powheg.root",
                 "Template_Training_Tree", f'(!{cuts["HF_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
                (f"{sample_folder_loc}/{result_folder_name}/El_OTv3_QuadJet_TemplateTraining/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_powheg.root",
                 "Template_Training_Tree", f'(!{cuts["HF_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
            ]
            # class 5: TTLJ(W->US, UD) + not BB
            cls6 += [
                (f"{sample_folder_loc}/{result_folder_name}/Mu_OTv3_QuadJet_TemplateTraining/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_powheg.root",
                 "Template_Training_Tree", f'(!{cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})'),
                (f"{sample_folder_loc}/{result_folder_name}/El_OTv3_QuadJet_TemplateTraining/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_powheg.root",
                 "Template_Training_Tree", f'(!{cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})'),
            ]

            # cls0 += [
            #     (f"{sample_folder_loc}/{result_folder_name}/Mu_OTv3_OutputTrees/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
            #      "Mu_Inclusive_Central_TTLJ_Vcb", cuts['RECO_CORRECT_MASK']),
            #     (f"{sample_folder_loc}/{result_folder_name}/El_OTv3_OutputTrees/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
            #      "El_Inclusive_Central_TTLJ_Vcb", cuts['RECO_CORRECT_MASK']),
            # ]
            # cls1 += [
            #     (f"{sample_folder_loc}/{result_folder_name}/Mu_OTv3_OutputTrees/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
            #      "Mu_Inclusive_Central_TTLJ_Vcb", f'!{cuts["RECO_CORRECT_MASK"]}'),
            #     (f"{sample_folder_loc}/{result_folder_name}/El_OTv3_OutputTrees/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
            #      "El_Inclusive_Central_TTLJ_Vcb", f'!{cuts["RECO_CORRECT_MASK"]}'),
            # ]

            # # class 1: TTLJ(W->CS, CD) + BB
            # cls2 += [
            #     (f"{sample_folder_loc}/{result_folder_name}/Mu_OTv3_OutputTrees/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
            #      "Mu_Inclusive_Central_TTLJ", f'({cuts["HF_BB_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
            #     (f"{sample_folder_loc}/{result_folder_name}/El_OTv3_OutputTrees/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
            #      "El_Inclusive_Central_TTLJ", f'({cuts["HF_BB_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
            # ]
            # # class 2: TTLJ(W->US, UD) + BB
            # cls3 += [
            #     (f"{sample_folder_loc}/{result_folder_name}/Mu_OTv3_OutputTrees/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
            #      "Mu_Inclusive_Central_TTLJ", f'({cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})'),
            #     (f"{sample_folder_loc}/{result_folder_name}/El_OTv3_OutputTrees/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
            #      "El_Inclusive_Central_TTLJ", f'({cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})'),
            # ]
            # # class 3: TTLJ(W->CS, CD) + CC
            # cls4 += [
            #     (f"{sample_folder_loc}/{result_folder_name}/Mu_OTv3_OutputTrees/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
            #      "Mu_Inclusive_Central_TTLJ", f'({cuts["HF_CC_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
            #     (f"{sample_folder_loc}/{result_folder_name}/El_OTv3_OutputTrees/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
            #      "El_Inclusive_Central_TTLJ", f'({cuts["HF_CC_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
            # ]
            # # class 4: TTLJ(W->CS, CD) + light
            # cls5 += [
            #     (f"{sample_folder_loc}/{result_folder_name}/Mu_OTv3_OutputTrees/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
            #      "Mu_Inclusive_Central_TTLJ", f'(!{cuts["HF_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
            #     (f"{sample_folder_loc}/{result_folder_name}/El_OTv3_OutputTrees/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
            #      "El_Inclusive_Central_TTLJ", f'(!{cuts["HF_MASK"]})&&({cuts["CX_DECAY_MASK"]})'),
            # ]
            # # class 5: TTLJ(W->US, UD) + not BB
            # cls6 += [
            #     (f"{sample_folder_loc}/{result_folder_name}/Mu_OTv3_OutputTrees/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
            #      "Mu_Inclusive_Central_TTLJ", f'(!{cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})'),
            #     (f"{sample_folder_loc}/{result_folder_name}/El_OTv3_OutputTrees/{e}/Skim_Vcb_SL_Skim_v2_TTLJ_Vcb_powheg.root",
            #      "El_Inclusive_Central_TTLJ", f'(!{cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})'),
            # ]
            
        return_arr = [cls0, cls1, cls2, cls3, cls4, cls5, cls6]
        if len(return_arr) != len(class_labels):
            raise ValueError("Class labels length mismatch.")
        return return_arr, class_labels
