import sys
import os
import numpy as np
import experiments.segmentation.split_tools as st
if __name__=="__main__":
    if sys.argv[1]=='c':
        st.split_script(sys.argv, None)
    if sys.argv[1]=='s' and len(sys.argv)>2:
        sys.argv[2] = '../%s'%(sys.argv[2],)

    # Create one full experiment per parameter value
    start_path = os.getcwd()
    for i, beta in enumerate(np.linspace(0.05, 1, 20)):
        directory = '%02d'%(1 + i,)
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.chdir(directory)
        print "Processing: %d. Beta=%0.6f"%(i, beta)

        with open("beta.txt","w") as f:
            f.write(str(beta)+'\n')
        req_dir = os.path.dirname(os.path.realpath(__file__))
        required_files = [os.path.join(req_dir, '../job_icm.sh'),
                          os.path.join(req_dir, '../icm.py'),
                          os.path.join(req_dir, '../jaccard.py'),
                          os.path.join(os.getcwd(), 'beta.txt')]
        st.split_script(sys.argv, required_files)
        os.chdir(start_path)

