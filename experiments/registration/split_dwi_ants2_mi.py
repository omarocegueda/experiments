import sys
import os
import experiments.registration.split_tools as st
if __name__=="__main__":
    req_dir = os.path.dirname(os.path.realpath(__file__))
    required_files = [req_dir+'/job_ants2_rigid_mi.sh']
    st.split_dwi(sys.argv, required_files)
