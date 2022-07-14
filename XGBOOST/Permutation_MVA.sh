#!/bin/bash

if [ $1 -gt 5 ]; then
    echo "Out of range"
    exit
fi

n_jet=`expr $1 % 3 + 4`
pre_kin=`expr $1 / 3`

dir=$DIR_PATH
method='XGBOOST'
#root -l -b -q "${dir}/$method/Permutation_MVA.cxx($n_jet, $pre_kin)"
python ${dir}/$method/Permutation_MVA.py --n_jet $n_jet --pre_kin $pre_kin

# ls dataset/weights/TMVAClassification*

# if [ $pre_kin -eq 0 ]; then 
#     mkdir -p ${dir}/result/$method/${n_jet}Jets/weights/
#     mv dataset/weights/TMVAClassification_* ${dir}/result/${method}/${n_jet}Jets/weights/
# elif [ $pre_kin -eq 1 ]; then
#     mkdir -p ${dir}/result/$method/Pre_Kin_${n_jet}Jets/weights/
#     mv dataset/weights/TMVAClassification_* ${dir}/result/${method}/Pre_Kin_${n_jet}Jets/weights/
# fi

#cd ${dir}
