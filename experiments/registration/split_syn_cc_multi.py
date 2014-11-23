import sys
import os
import experiments.registration.split_tools as st
if __name__=="__main__":
    req_dir = os.path.dirname(os.path.realpath(__file__))
    required_files = [req_dir+'/dipyreg.py',
                      req_dir+'/job_syn_cc.sh']
    st.split_all_pairs_multi_modal(sys.argv, required_files)
