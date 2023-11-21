#!/bin/bash

dir=$DIR_PATH
method='TabNet_template'
#root -l -b -q "${dir}/$method/Permutation_MVA.cxx($n_jet, $pre_kin)"
python3 ${dir}/$method/MVA.py --working_mode train --out_path ${dir}/$method/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/model_2017_mu+el_sample_calssweight_RFCompleted_RECOMVAincluded_swap_nstep2_smallBN_only45
