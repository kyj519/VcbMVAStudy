#!/bin/bash

if [ $1 -gt 5 ]; then
    echo "Out of range"
    exit
fi

n_jet=`expr $1 % 3 + 4`
pre_kin=`expr $1 / 3`

dir=$DIR_PATH

root -l -b -q "${dir}/BDT/Permutation_MVA.cxx($n_jet, $pre_kin)"


ls dataset/weights/TMVAClassification*

if [ $pre_kin -eq 0 ]; then 
    mkdir -p ${dir}/${n_jet}Jets/weights/
    mv dataset/weights/TMVAClassification_* ${dir}/${n_jet}Jets/weights/
elif [ $pre_kin -eq 1 ]; then
    mkdir -p ${dir}/Pre_Kin_${n_jet}Jets/weights/
    mv dataset/weights/TMVAClassification_* ${dir}/Pre_Kin_${n_jet}Jets/weights/
fi

#cd ${dir}
