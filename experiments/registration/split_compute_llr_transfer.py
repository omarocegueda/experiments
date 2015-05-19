import os
import errno
import subprocess
header = '''#!/bin/bash
####################################################
#Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l mem=3GB
#PBS -l pmem=3GB
#PBS -l vmem=3GB
#PBS -l nodes=1:ppn=1
#PBS -l walltime=06:00:00
#PBS -N Optimal LLR-T
'''
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

for radius in range(2,11):
    dirname = str(radius)
    mkdir_p(dirname)
    f = open(dirname+'/'+'job_llr_transfer.sh','w')
    f.write(header)
    f.write('python compute_llr_transfer.py %d %s'%(radius, 'random'))
    f.close()
    subprocess.call('chmod +x '+dirname+'/'+'job_llr_transfer.sh', shell=True)
    subprocess.call('ln ../compute_llr_transfer.py '+dirname, shell=True)

