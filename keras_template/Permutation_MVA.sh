#!/bin/bash

if [ $1 -gt 5 ]; then
    echo "Out of range"
    exit
fi

n_jet=`expr $1 % 3 + 4`
pre_kin=`expr $1 / 3`

dir=$DIR_PATH
cd $dir
method='keras_template'
#root -l -b -q "${dir}/$method/Permutation_MVA.cxx($n_jet, $pre_kin)"
python ${dir}/$method/Permutation_MVA.py 
