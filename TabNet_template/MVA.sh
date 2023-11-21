#!/bin/bash

dir=$DIR_PATH
method='TabNet_template'
#root -l -b -q "${dir}/$method/Permutation_MVA.cxx($n_jet, $pre_kin)"
python3 ${dir}/$method/MVA.py --working_mode train --out_path ${dir}/$method/$1 --floss_gamma $2
#python3 ${dir}/$method/MVA.py --working_mode train --out_path ${dir}/$method/test