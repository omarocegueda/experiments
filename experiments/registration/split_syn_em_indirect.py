import sys
import os
import experiments.registration.split_tools as st
if __name__=="__main__":
    req_dir = os.path.dirname(os.path.realpath(__file__))
    required_files = [req_dir+'/dipyreg.py',
                      req_dir+'/job_syn_em_indirect.sh']
    st.split_script(sys.argv, required_files, 'indirect')
