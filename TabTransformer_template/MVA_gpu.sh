#!/bin/bash


dir=$DIR_PATH
method='TabTransformer_template'
#root -l -b -q "${dir}/$method/Permutation_MVA.cxx($n_jet, $pre_kin)"
python3 "${dir}/$method/MVA.py"


