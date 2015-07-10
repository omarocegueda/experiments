import sys
import numpy as np
import nibabel as nib
import experiments.segmentation.evaluation as eval
# Get the file names from command line
img_fname = sys.argv[1]
out_fname = sys.argv[2]

# Load the data
img_nib = nib.load(img_fname)
img = img_nib.get_data().squeeze().astype(np.float64)

# Execute the segmentation
out, means = eval.baseline_segmentation(img, 4) # 4 classes, including background
out = np.array(out)

# Convert numpy array to Nifti image and save
out_nib = nib.Nifti1Image(out, img_nib.get_affine())
out_nib.to_filename(out_fname)
