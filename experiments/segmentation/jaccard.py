import sys
import numpy as np
import nibabel as nib
import experiments.segmentation.evaluation as eval

# Get the file names from command line
seg_fname = sys.argv[1]
gt_fname = sys.argv[2]
out_fname = sys.argv[3]

# Load the data
seg_nib = nib.load(seg_fname)
seg = seg_nib.get_data().squeeze()
gt_fname = nib.load(gt_fname)
gt = gt_nib.get_data().squeeze()

# Compute the scores
jaccard = eval.compute_jaccard(seg, gt)
jaccard = np.array(jaccard)

# Write results
with open(out_fname, 'w') as f:
    for i in range(jaccard.shape[0]):
        f.write("%0.4f\n" % (jaccard[i],))
