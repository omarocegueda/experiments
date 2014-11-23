import sys
import os
import experiments.registration.split_tools as st
from dipy.fixes import argparse as arg

parser = arg.ArgumentParser(
    description=""
)

parser.add_argument(
    'mod1', action = 'store', metavar = 'mod1',
    help = '''Modality identifier. It will be added as suffix to the filename
    of each input file, e.g. "", "_t2", "_pd" ''')

parser.add_argument(
    'mod2', action = 'store', metavar = 'mod2',
    help = '''Modality identifier. It will be added as suffix to the filename
    of each input file, e.g. "", "_t2", "_pd" ''')

if __name__=="__main__":
    req_dir = os.path.dirname(os.path.realpath(__file__))
    required_files = [req_dir+'/dipyreg.py',
                      req_dir+'/job_syn_cc.sh']

    params = parser.parse_args()
    params.mod1
    st.split_all_pairs_multi_modal(sys.argv, required_files, params.mod1, params.mod2)
