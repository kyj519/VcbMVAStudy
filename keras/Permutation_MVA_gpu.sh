#!/bin/bash


dir=$DIR_PATH
cd $dir
method='keras'
#root -l -b -q "${dir}/$method/Permutation_MVA.cxx($n_jet, $pre_kin)"
python3 "${dir}/$method/Permutation_MVA_subprocess_runner.py"

