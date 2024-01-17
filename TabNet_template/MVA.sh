#!/bin/bash

dir=$DIR_PATH
method='TabNet_template'
#root -l -b -q "${dir}/$method/Permutation_MVA.cxx($n_jet, $pre_kin)"
if [ -e "$6" ]; then
	echo "Training with pretrained model"
	python3 ${dir}/$method/MVA.py --working_mode train --model_out_path ${dir}/$method/$1 --floss_gamma $2 --rf_file_name $3 --result_folder_name $4 --sample_folder_loc $5 --pretrained_model $6 
else
	echo "Training without pretrained model"
	python3 ${dir}/$method/MVA.py --working_mode train --model_out_path ${dir}/$method/$1 --floss_gamma $2 --rf_file_name $3 --result_folder_name $4 --sample_folder_loc $5 
fi
#python3 ${dir}/$method/MVA.py --working_mode train --out_path ${dir}/$method/testa