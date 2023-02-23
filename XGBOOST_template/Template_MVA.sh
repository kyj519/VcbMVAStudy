#!/bin/bash

dir=$DIR_PATH
method='XGBOOST_template'
#root -l -b -q "${dir}/$method/Permutation_MVA.cxx($n_jet, $pre_kin)"
python ${dir}/$method/Template_MVA.py 

# ls dataset/weights/TMVAClassification*

# if [ $pre_kin -eq 0 ]; then 
#     mkdir -p ${dir}/result/$method/${n_jet}Jets/weights/
#     mv dataset/weights/TMVAClassification_* ${dir}/result/${method}/${n_jet}Jets/weights/
# elif [ $pre_kin -eq 1 ]; then
#     mkdir -p ${dir}/result/$method/Pre_Kin_${n_jet}Jets/weights/
#     mv dataset/weights/TMVAClassification_* ${dir}/result/${method}/Pre_Kin_${n_jet}Jets/weights/
# fi

#cd ${dir}
