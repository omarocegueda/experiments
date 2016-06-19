import sys
import numpy as np
import nibabel as nib
import experiments.segmentation.evaluation as eval
import time
from dipy.segment.tissue import TissueClassifierHMRF
# Get the file names from command line
img_fname = sys.argv[1]
out_fname = sys.argv[2]
with open('beta.txt','r') as f:
    beta = float(f.readline())

# Load the data
img_nib = nib.load(img_fname)
t1 = img_nib.get_data().squeeze().astype(np.float64)

# Execute the segmentation
nclass = 3
#beta = 0.1
tolerance = 0.00001

t0 = time.time()
hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(t1, nclass, beta, tolerance)
final_segmentation = np.array(final_segmentation)
t1 = time.time()
total_time = t1-t0
print('Total time:' + str(total_time))

# Convert numpy array to Nifti image and save
out_nib = nib.Nifti1Image(final_segmentation, img_nib.get_affine())
out_nib.to_filename(out_fname)
