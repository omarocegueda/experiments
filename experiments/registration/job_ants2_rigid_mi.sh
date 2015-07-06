#!/bin/bash
####################################################
#Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l mem=8GB
#PBS -l pmem=8GB
#PBS -l vmem=8GB
#PBS -l nodes=1:ppn=1
#PBS -l walltime=04:00:00
#PBS -N ANTS2-RIGID-MI

# Configure your environment
export DIPY_DIR="$HOME/opt/dipy"
export EXPERIMENTS_DIR="$HOME/opt/experiments"
###################################
echo "=====Dipy commit ====="
(cd $DIPY_DIR && git branch|grep "*")
(cd $DIPY_DIR && git show --stat|grep "commit\|Date")
echo "====Experiments commit===="
(cd $EXPERIMENTS_DIR && git branch|grep "*")
(cd $EXPERIMENTS_DIR && git show --stat|grep "commit\|Date")
echo "======================"
date
reference=$(ls reference)
target=$(ls target)
extension="${target##*.}"
targetbase="${target%.*}"
targetbase="${targetbase%.*}"
referencebase="${reference%.*}"
referencebase="${referencebase%.*}"
#Affine registration using Mutual information with ANTS
affine="${targetbase}_${referencebase}Affine.txt"
affine0="${targetbase}_${referencebase}0GenericAffine.mat"

sampling="regular, 0.3"
#sampling="None"
op="${targetbase}_${referencebase}"
if ! [ -r $affine ]; then
    exe="antsRegistration -d 3 -r [ reference/$reference, target/$target, 1 ] \
                      -m mattes[ reference/$reference, target/$target, 1 , 32, $sampling] \
                      -t translation[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 6x4x2 -l 1 \
                      -m mattes[ reference/$reference, target/$target, 1 , 32, $sampling] \
                      -t rigid[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 3x2x1 -l 1 \
                      -o [${op}]"
    echo " $exe "
    $exe
    ConvertTransformFile 3 ${affine0} ${affine}
else
    echo "Affine mapping found ($affine). Skipping affine registration."
fi

oname=warpedDiff_${targetbase}_${referencebase}.nii.gz
antsApplyTransforms -d 3 -i target/$target -o $oname -r reference/$reference -n Linear --float -t $affine

for towarp in $( ls warp ); do
    towarpbase="${towarp%.*}"
    towarpbase="${towarpbase%.*}"
    oname=warpedAff_${towarpbase}_${referencebase}.nii.gz
    antsApplyTransforms -d 3 -i warp/$towarp -o $oname -r reference/$reference -n NearestNeighbor --float -t $affine
done
python -c 'from experiments.registration.dipyreg_affine import *; compute_scores()'


