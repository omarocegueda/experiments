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

parser.add_argument(
    'action', action = 'store', metavar = 'action',
    help = '''Any of ["s", "s2", "c", "o", "u"] to split all pairs, split
    corresponding pairs, clean, collect or submit, respectively''')

parser.add_argument(
    'names_from', action = 'store', metavar = 'names_from',
    help = '''Text file containing the input file names''')

parser.add_argument(
    '-nt', '--names_to', action = 'store', metavar = 'names_to',
    help = '''Text file containing the input file names''')

if __name__=="__main__":
    req_dir = os.path.dirname(os.path.realpath(__file__))
    required_files = [req_dir+'/dipyreg.py',
                      req_dir+'/job_syn_ecc.sh']

    params = parser.parse_args()
    params.mod1
    st.split_script(["", params.action, params.names_from, params.names_to], required_files, 'multi', params.mod1, params.mod2)
