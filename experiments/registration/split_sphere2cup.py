import sys
import os
from experiments.registration.split_tools import (mkdir_p, clean_working_dirs, query_yes_no)
import subprocess
import fnmatch

cmd = sys.argv[1]
if cmd == 's':
    #step_length_list = [15, 25, 35, 45]
    #step_length_list= [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    step_length_list = [30]
    #inv_iter_list = [10, 20, 30, 40]
    #inv_iter_list = [30]
    inv_iter_list = [50]
    #inv_tol_list = [2, 3, 4, 5, 6]
    #inv_tol_list = [3]
    inv_tol_list = range(1,21)

    python_script_name = '../sphere2cup.py'
    cluster_job_name = '../job_sphere2cup.sh'
    configname = 'config.txt'
    idx = 0
    for step_length_100 in step_length_list:
        for inv_iter in inv_iter_list:
            for inv_tol_exponent in inv_tol_list:
                step_length = step_length_100 /100.0
                #inv_tol = float('1e-%d'%(inv_tol_exponent))
                inv_tol = 1.0/(2**inv_tol_exponent)
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
elif cmd == 'c':
    if not os.path.isdir('results'):
        cleanAnyway=query_yes_no("It seems like you have not collected the results yet. Clean anyway? (y/n)")
        if not cleanAnyway:
            sys.exit(0)
    clean_working_dirs()
elif cmd == 'u':
    dirNames=[name for name in os.listdir(".") if os.path.isdir(name) and fnmatch.fnmatch(name, '[0-9]*')]
    for name in dirNames:
        os.chdir('./'+name)
        subprocess.call('qsub job*.sh -d . -q batch', shell=True)
        os.chdir('./..')
elif cmd == 'o':
    mkdir_p('results')
    dirNames=[name for name in os.listdir(".") if os.path.isdir(name) and fnmatch.fnmatch(name, '[0-9]*')]
    for name in dirNames:
        subprocess.call('mv '+os.path.join(name,'*.p')+' results', shell=True)
