#!/bin/bash
####################################################
#Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l mem=4GB
#PBS -l pmem=4GB
#PBS -l vmem=4GB
#PBS -l nodes=1:ppn=1
#PBS -l walltime=01:00:00
#PBS -N seg-baseline
date
img=$(ls {img/*.nii.gz,img/*.img} 2>/dev/null | xargs -n1 basename)
seg=$(ls {seg/*.nii.gz,seg/*.img} 2>/dev/null | xargs -n1 basename)
imgbase="${img%.*}"
imgbase="${imgbase%.*}"
segbase="${reference%.*}"
segbase="${segbase%.*}"
# Execute icm segmentation, request to save the file as 'seg_X.nii.gz' where X is the name of the input image (after removing the .nii.gz extension)
python icm.py img/$img seg_$imgbase.nii.gz
# Evaluation: we assume that the segmented file was named 'seg_X.nii.gz', as described above
# Save the results as 'jaccard_X.txt'
python jaccard.py seg_$imgbase.nii.gz seg/$seg jaccard_$imgbase.txt
date
