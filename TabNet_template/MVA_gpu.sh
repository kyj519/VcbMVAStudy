#!/bin/bash

dir=$DIR_PATH
method='TabNet_template'
#root -l -b -q "${dir}/$method/Permutation_MVA.cxx($n_jet, $pre_kin)"
python3 ${dir}/$method/MVA.py --working_mode train --out_path ${dir}/$method/model_only_correct_reco_mu+el_weighted_loss
