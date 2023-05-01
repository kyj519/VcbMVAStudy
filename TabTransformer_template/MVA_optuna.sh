#!/bin/bash
export X509_USER_PROXY=/cms/ldap_home/yeonjoon/proxy.cert
export CUDA_LAUNCH_BLOCKING=1
dir=$DIR_PATH
method='TabNet_template'
#root -l -b -q "${dir}/$method/Permutation_MVA.cxx($n_jet, $pre_kin)"
python3 -u "${dir}/$method/MVA_optuna.py"

