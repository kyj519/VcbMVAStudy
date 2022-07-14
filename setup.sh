#!/bin/bash

export DIR_PATH=`pwd`
export WtoCB_PATH='/gv0/Users/yeonjoon/'
conda activate cms-py3
python3 unset_cvmfs.py
source unsetter.sh
#### use cvmfs for root ####
# export CMS_PATH=/cvmfs/cms.cern.ch
# source $CMS_PATH/cmsset_default.sh
# export SCRAM_ARCH=slc7_amd64_gcc900
# export cmsswrel='cmssw/CMSSW_12_1_0_pre4_ROOT624'
# cd /cvmfs/cms.cern.ch/$SCRAM_ARCH/cms/$cmsswrel/src
# echo "@@@@ SCRAM_ARCH = "$SCRAM_ARCH
# echo "@@@@ cmsswrel = "$cmsswrel
# echo "@@@@ scram..."
# eval `scramv1 runtime -sh`
# cd -
#source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.26.04/x86_64-centos8-gcc85-opt/bin/thisroot.sh
