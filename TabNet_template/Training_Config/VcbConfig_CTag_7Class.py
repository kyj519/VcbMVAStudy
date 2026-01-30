import sys, os
from dataclasses import dataclass, field
from typing_extensions import override

sys.path.append(os.environ.get("DIR_PATH") + "/TabNet_template")
from TrainingConfig import TabNetTrainConfig as Base
from typing import List, Tuple, Dict

@dataclass
class TabNetTrainConfig(Base):
    n_d: int = 32
    n_a: int = 32
    n_steps: int = 5
    assume_btag_mode: bool = False
    cat_emb_dim: int = 3
    varlist: List[str] = field(
        default_factory=lambda: [
            "pt_w_u",
            "pt_w_d",
            "pt_had_t_b",
            "pt_lep_t_b",
            "m_had_t",
            "m_had_w",
            "bvsc_had_t_b",
            "bvsc_lep_t_b",
            "bvsc_w_u",
            "bvsc_w_d",
            "n_bjets",
            "n_jets",
            "n_cjets", "cvsb_had_t_b", "cvsb_lep_t_b", "cvsl_had_t_b", "cvsl_lep_t_b", "cvsl_w_u", "cvsl_w_d", "cvsb_w_u", "cvsb_w_d",
            "best_mva_score",
            "least_dr_bb",
            "least_m_bb",
            "pt_tt",
            "ht"
        ]
    )

    @override
    @staticmethod
    def _cuts(rel_iso_muon: float = 0.15, met_pt: float = 25) -> Dict[str, str]:
        D_COND_MU = f"(met_pt>{met_pt})&&(lepton_rel_iso<{rel_iso_muon})"
        D_COND_EL = f"(met_pt>{met_pt})&&((electron_id_bit & (1 << 5)) != 0)"
        HF_BB_MASK = "(((genTtbarId%100) >= 51) && ((genTtbarId%100) <= 55))"
        HF_CC_MASK = "(((genTtbarId%100) >= 41) && ((genTtbarId%100) <= 45))"
        CX_DECAY_MASK = "((decay_mode//10)%10 == 4)"
        UX_DECAY_MASK = "((decay_mode//10)%10 == 2)"
        HF_MASK = f"{HF_BB_MASK}||{HF_CC_MASK}"
        RECO_CORRECT_MASK = "(chk_reco_correct==1)"
        return dict(
            D_COND_MU=D_COND_MU,
            D_COND_EL=D_COND_EL,
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
            # class 0: signal (TTLJ_WtoCB)
            cls0 += [
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root",
                    "Mu/Central/Result_Tree",
                    f"{cuts['D_COND_MU']}&&({cuts['RECO_CORRECT_MASK']})",
                ),
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root",
                    "El/Central/Result_Tree",
                    f"{cuts['D_COND_EL']}&&({cuts['RECO_CORRECT_MASK']})",
                ),
            ]
            cls1 += [
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root",
                    "Mu/Central/Result_Tree",
                    f'{cuts["D_COND_MU"]}&&(!{cuts["RECO_CORRECT_MASK"]})',
                ),
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_WtoCB_powheg.root",
                    "El/Central/Result_Tree",
                    f'{cuts["D_COND_EL"]}&&(!{cuts["RECO_CORRECT_MASK"]})',
                ),
            ]

            # class 1: TTLJ(W->CS, CD) + BB
            cls2 += [
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "Mu/Central/Result_Tree",
                    f'{cuts["D_COND_MU"]}&&({cuts["HF_BB_MASK"]})&&({cuts["CX_DECAY_MASK"]})',
                ),
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "El/Central/Result_Tree",
                    f'{cuts["D_COND_EL"]}&&({cuts["HF_BB_MASK"]})&&({cuts["CX_DECAY_MASK"]})',
                ),
            ]
            # class 2: TTLJ(W->US, UD) + BB
            cls3 += [
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "Mu/Central/Result_Tree",
                    f'{cuts["D_COND_MU"]}&&({cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})',
                ),
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "El/Central/Result_Tree",
                    f'{cuts["D_COND_EL"]}&&({cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})',
                ),
            ]
            # class 3: TTLJ(W->CS, CD) + CC
            cls4 += [
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "Mu/Central/Result_Tree",
                    f'{cuts["D_COND_MU"]}&&({cuts["HF_CC_MASK"]})&&({cuts["CX_DECAY_MASK"]})',
                ),
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "El/Central/Result_Tree",
                    f'{cuts["D_COND_EL"]}&&({cuts["HF_CC_MASK"]})&&({cuts["CX_DECAY_MASK"]})',
                ),
            ]
            # class 4: TTLJ(W->CS, CD) + light
            cls5 += [
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "Mu/Central/Result_Tree",
                    f'{cuts["D_COND_MU"]}&&(!{cuts["HF_MASK"]})&&({cuts["CX_DECAY_MASK"]})',
                ),
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "El/Central/Result_Tree",
                    f'{cuts["D_COND_EL"]}&&(!{cuts["HF_MASK"]})&&({cuts["CX_DECAY_MASK"]})',
                ),
            ]
            # class 5: TTLJ(W->US, UD) + not BB
            cls6 += [
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "Mu/Central/Result_Tree",
                    f'{cuts["D_COND_MU"]}&&(!{cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})',
                ),
                (
                    f"{sample_folder_loc}/{e}/{result_folder_name}/Central_Syst/Vcb_TTLJ_powheg.root",
                    "El/Central/Result_Tree",
                    f'{cuts["D_COND_EL"]}&&(!{cuts["HF_BB_MASK"]})&&({cuts["UX_DECAY_MASK"]})',
                ),
            ]

        return_arr = [cls0, cls1, cls2, cls3, cls4, cls5, cls6]
        if len(return_arr) != len(class_labels):
            raise ValueError("Class labels length mismatch.")
        return return_arr, class_labels
