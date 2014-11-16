import sys
import os
import experiments.registration.split_tools as st
if __name__=="__main__":
    req_dir = os.path.dirname(os.path.realpath(__file__))
    required_files = [req_dir+'/semi_synthetic.py',
                      req_dir+'/job_semi_synthetic.sh']
    st.split_script(sys.argv, required_files)
