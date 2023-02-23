#!/bin/bash


dir=$DIR_PATH
cd $dir
method='TabNet_Permutation'
#root -l -b -q "${dir}/$method/Permutation_MVA.cxx($n_jet, $pre_kin)"
python3 "${dir}/$method/MVA.py"

