import sys
import experiments.registration.IBSR18.split_tools as st
if __name__=="__main__":
    required_files = ['../dipyreg.py',
                      'job_syn_cc.sh']
    st.split_script(sys.argv, required_files)
