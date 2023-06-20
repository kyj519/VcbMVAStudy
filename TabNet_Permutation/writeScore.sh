#!/bin/bash
dir=$DIR_PATH
method='TabNet_template'
#root -l -b -q "${dir}/$method/Permutation_MVA.cxx($n_jet, $pre_kin)"
python3 ${dir}/$method/MVA.py --working_mode infer --input_root_file $GV0/Vcb/Sample --input_model '/data6/Users/yeonjoon/VcbMVAStudy/TabNet_template/model_only_correct_reco'