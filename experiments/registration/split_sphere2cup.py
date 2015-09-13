import sys
import os
from experiments.registration.split_tools import (mkdir_p)
import subprocess
step_length_list = [25, 35, 45, 55, 65, 75]
inv_iter_list = [20, 40, 100, 200]
inv_tol_list = [2, 3, 4, 5, 6]

python_script_name = '../sphere2cup.py'
cluster_job_name = '../job_sphere2cup.sh'
configname = 'config.txt'
idx = 0
for step_length_100 in step_length_list:
    for inv_iter in inv_iter_list:
        for inv_tol_exponent in inv_tol_list:
            step_length = step_length_100 /100.0
            inv_tol = float('1e-%d'%(inv_tol_exponent))
            run_name='%d_%03d_%d'%(step_length_100, inv_iter, inv_tol_exponent)
            
            dirname = '%03d'%(idx)
            mkdir_p(dirname)
            
            subprocess.call('ln '+python_script_name+' '+dirname, shell=True)
            subprocess.call('ln '+cluster_job_name+' '+dirname, shell=True)
            
            with open(os.path.join(dirname, configname), 'w') as f:
                f.write('experiment_name\t%s\n'%(run_name,))
                f.write('inv_iter\t%d\n'%(inv_iter,))
                f.write('inv_tol\t%e\n'%(inv_tol,))
                f.write('step_length\t%e\n'%(step_length,))
            idx += 1
