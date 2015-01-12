import sys
import os
import experiments.registration.split_tools as st
if __name__=="__main__":
    req_dir = os.path.dirname(os.path.realpath(__file__))
    required_files = [req_dir+'/nipyreg_affine.py',
                      req_dir+'/job_nipy_affine_mi.sh']
    st.split_script(sys.argv, required_files)
