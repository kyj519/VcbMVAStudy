import os
f = open('unsetter.sh','w')
os.system('chmod 755 unsetter.sh')
f.write('#!/bin/bash\n')
for key, item in os.environ.items():
    if '_' == key[0]: continue
    envs = item.split(':')
    for env in envs: 
        if 'cvmfs' in env: envs.remove(env)
    if len(envs)==0:
        print(f"unset {key}")
        f.write(f'unset {key}\n')
    else:
        newenv = ":".join(envs)
        f.write(f"export {key}='{newenv}'\n")
f.close()
            