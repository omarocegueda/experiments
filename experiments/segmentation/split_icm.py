import sys
import os
import experiments.segmentation.split_tools as st
if __name__=="__main__":
    req_dir = os.path.dirname(os.path.realpath(__file__))
    required_files = [os.path.join(req_dir, 'job_icm.sh'),
                      os.path.join(req_dir, 'icm.py'),
                      os.path.join(req_dir, 'jaccard.py')]
    st.split_script(sys.argv, required_files)
