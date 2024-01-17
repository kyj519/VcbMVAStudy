#!/bin/bash

dir=$DIR_PATH
method='TabNet_Permutation'
#root -l -b -q "${dir}/$method/Permutation_MVA.cxx($n_jet, $pre_kin)"
python3 ${dir}/$method/MVA.py --working_mode train --out_path ${dir}/$method/model --n_jets $1
