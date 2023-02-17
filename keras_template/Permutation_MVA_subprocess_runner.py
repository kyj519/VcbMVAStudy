import subprocess
from itertools import product
import sys

n_jet = [4]
pre_kin = [1]
procs = []

for tup in product(n_jet,pre_kin):
    proc = subprocess.Popen(f'python3 /data6/Users/yeonjoon/VcbMVAStudy/keras/Permutation_MVA.py --n_jet {tup[0]} --pre_kin {tup[1]}'.split()
                            , stdout=sys.stdout, stderr=sys.stderr)
    procs.append(proc)

    
procs[0].communicate()
assert procs[0]