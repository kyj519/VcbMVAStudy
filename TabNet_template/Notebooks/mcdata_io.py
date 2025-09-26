# mcdata_io.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import uproot
import awkward as ak

# ----------------------------
# Public API
# ----------------------------

__all__ = [
    "build_classif_keys",
    "need_branches",
    "open_tree",
    "load_arrays",
    "build_weights_dict",
    "load_signal_and_background",
]

# ----------------------------
# Config / constants
# ----------------------------

BASE_BRANCHES_COMMON = [
    "weight_el_id", "weight_mc", "weight_lumi", "weight_mu_id", "weight_mu_iso",
    "weight_el_reco", "weight_pileup", "weight_prefire", "weight_sl_trig", "weight_top_pt",
    "genTtbarId",
]

B_MODE_BRANCHES = [
    "weight_b_tag", "weight_b_tag_up_cferr1", "weight_b_tag_up_cferr2", "weight_b_tag_up_hf",
]

C_MODE_BRANCHES = [
    "weight_c_tag",
    "weight_c_tag_up_stat",
    "weight_c_tag_up_xsec_brunc_dyjets_b",
    "weight_c_tag_up_xsec_brunc_dyjets_c",
    "weight_c_tag_up_xsec_brunc_wjets_c",
]

@dataclass
class DatasetSpec:
    """Path spec in the form 'file.root:Tree/Path'."""
    spec: str

# ----------------------------
# Branch helpers
# ----------------------------

def build_classif_keys(branch_name: str, num_classes: int) -> List[str]:
    """
    Given a base branch prefix (e.g., 'template_score_MultiClass'),
    return the list of K log-prob branch names:
      f"{branch_name}_log_prob_0", ..., f"..._{K-1}"
    """
    return [f"{branch_name}_log_prob_{i}" for i in range(num_classes)]

def need_branches(mode: str, classif_keys: Iterable[str]) -> List[str]:
    """
    mode ∈ {'b','c'}
    Returns the full branch list needed to build weights + classifier logits.
    """
    out = list(BASE_BRANCHES_COMMON) + list(classif_keys)
    if mode == 'b':
        out += B_MODE_BRANCHES
    elif mode == 'c':
        out += C_MODE_BRANCHES
    else:
        raise ValueError("mode must be 'b' or 'c'")
    return out

# ----------------------------
# I/O
# ----------------------------

def open_tree(spec: str):
    """
    Open a tree given 'file.root:Tree/Path'
    """
    if ":" not in spec:
        raise ValueError("Path must look like '/path/file.root:Tree/Path'")
    return uproot.open(spec)

def load_arrays(spec: str, branches: Iterable[str]) -> Dict[str, np.ndarray]:
    """
    Read selected branches and convert awkward arrays to numpy.
    """
    tree = open_tree(spec)
    arrs = tree.arrays(list(branches), library="ak")
    out = {k: arrs[k].to_numpy() for k in arrs.fields}
    return out

# ----------------------------
# Weight building
# ----------------------------

def _add_tt_masks(d: Dict[str, np.ndarray]) -> None:
    if "genTtbarId" in d:
        ttbar = d["genTtbarId"]
        d["TTBB_Mask"] = (ttbar % 100 >= 51) & (ttbar % 100 <= 55)
        d["TTCC_Mask"] = (ttbar % 100 >= 41) & (ttbar % 100 <= 45)
    else:
        # fall back to a bool array of zeros
        any_key = next(iter(d))
        d["TTBB_Mask"] = d["TTCC_Mask"] = np.zeros_like(d[any_key], dtype=bool)
        
def _scale_tthf(d: Dict[str, np.ndarray]) -> None:
    if "weight_base" in d and "TTBB_Mask" in d:
        d["weight_base"] *= np.where(d["TTBB_Mask"], 1.36, 1.)
    else :
        raise RuntimeError("Cannot scale tthf: missing weight_base or TTBB_Mask")
    if "weight_base" in d and "TTCC_Mask" in d:
        d["weight_base"] *= np.where(d["TTCC_Mask"], 1.11, 1.)
    else :
        raise RuntimeError("Cannot scale tthf: missing weight_base or TTCC_Mask")

def build_weights_dict(raw: Dict[str, np.ndarray], mode: str,
                       classif_keys: Iterable[str]) -> Dict[str, np.ndarray]:
    """
    From raw branches -> add:
      - weight_base (+ systematics variations depending on mode)
      - ttbb/ttcc mask variations
      - classif_{i} = log-prob branches copied to fixed names
    Returns a new dict (does not mutate the input).
    """
    d = dict(raw)  # shallow copy

    _add_tt_masks(d)

    # base common
    base_common = (
        d["weight_el_id"] * d["weight_mc"] * d["weight_lumi"] * d["weight_mu_id"] * d["weight_mu_iso"] *
        d["weight_el_reco"] * d["weight_pileup"] * d["weight_prefire"] * d["weight_sl_trig"] * d["weight_top_pt"]
    )

    if mode == 'b':
        base = d["weight_b_tag"] * base_common
        d["weight_base"] = base
        safe = np.maximum(d["weight_b_tag"], 1e-12)
        d["weight_cferr1_up"] = base * d["weight_b_tag_up_cferr1"] / safe
        d["weight_cferr2_up"] = base * d["weight_b_tag_up_cferr2"] / safe
        d["weight_hf_up"]     = base * d["weight_b_tag_up_hf"]    / safe
        d["weight_top_pt_up"] = base / np.maximum(d["weight_top_pt"], 1e-12)
    elif mode == 'c':
        base = d["weight_c_tag"] * base_common
        d["weight_base"] = base
        safe = np.maximum(d["weight_c_tag"], 1e-12)
        d["weight_c_tag_stat_up"]             = base / safe * d["weight_c_tag_up_stat"]
        d["weight_c_tag_xsec_dyjets_b_up"]    = base / safe * d["weight_c_tag_up_xsec_brunc_dyjets_b"]
        d["weight_c_tag_xsec_dyjets_c_up"]    = base / safe * d["weight_c_tag_up_xsec_brunc_dyjets_c"]
        d["weight_c_tag_xsec_wjets_c_up"]     = base / safe * d["weight_c_tag_up_xsec_brunc_wjets_c"]
    else:
        raise ValueError("mode must be 'b' or 'c'")

    # apply tthf scaling
    
    # flavor fraction tweaks
    d["weight_ttbb_up"] = base * np.where(d["TTBB_Mask"], 1.1, 1.0)
    d["weight_ttcc_up"] = base * np.where(d["TTCC_Mask"], 1.15, 1.0)

    # classifier logits copied to canonical keys
    for i, key in enumerate(classif_keys):
        d[f"classif_{i}"] = d[key]

    return d

# ----------------------------
# One-shot convenience
# ----------------------------

def load_signal_and_background(
    ttlj_spec: str,
    vcb_spec: str,
    mode: str,
    branch_name: str = "template_score_MultiClass",
    num_classes: int = 4,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    High-level loader: returns (TTLJ_dict, Vcb_dict) with weights and classifier fields ready.

    - Validates mode
    - Builds branch list (incl. classifier keys)
    - Reads arrays (uproot+awkward → numpy)
    - Builds weight/systematic fields and canonical classif_{i} fields
    """
    classif_keys = build_classif_keys(branch_name, num_classes)
    branches = need_branches(mode, classif_keys)

    ttlj_raw = load_arrays(ttlj_spec, branches)
    vcb_raw  = load_arrays(vcb_spec,  branches)

    TTLJ = build_weights_dict(ttlj_raw, mode, classif_keys)
    Vcb  = build_weights_dict(vcb_raw,  mode, classif_keys)
    return TTLJ, Vcb