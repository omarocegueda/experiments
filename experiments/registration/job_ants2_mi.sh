#!/bin/bash
####################################################
#Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l mem=3GB
#PBS -l pmem=3GB
#PBS -l vmem=3GB
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00
#PBS -N ANTS2-MI

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
affinePrecomputed="../affine/${affine}"
if [ -r $affinePrecomputed ]; then
    cp $affinePrecomputed .
fi

op="${targetbase}_${referencebase}"
if ! [ -r $affine ]; then
    exe="antsRegistration -d 3 -r [ reference/$reference, target/$target, 1 ] \
                      -m mattes[ reference/$reference, target/$target, 1 , 32, regular, 0.3 ] \
                      -t translation[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 6x4x2 -l 1 \
                      -m mattes[ reference/$reference, target/$target, 1 , 32, regular, 0.3 ] \
                      -t rigid[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 3x2x1 -l 1 \
                      -m mattes[ reference/$reference, target/$target, 1 , 32, regular, 0.3 ] \
                      -t affine[ 0.1 ] \
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

#Diffeomorphic registration
deformationField=${targetbase}_${referencebase}Warp.nii.gz
inverseField=${targetbase}_${referencebase}InverseWarp.nii.gz
if [ -r $deformationField ]; then
    echo "Deformation found. Registration skipped."
else
    exe="antsRegistration -d 3 -r $affine \
                      -m mattes[reference/$reference, target/$target, 1 , 32 ] \
                      -t syn[ .25, 3, 0 ] \
                      -c [ 1x1x0,1e-5,12 ] \
                      -s 1x0.5x0vox \
                      -f 4x2x1\
                      -o [${op}, warpedDiff_${op}.nii.gz, warpedDiff_${op}.nii.gz]"
    echo " $exe "
    $exe
fi

date

for towarp in $( ls warp ); do
    towarpbase="${towarp%.*}"
    towarpbase="${towarpbase%.*}"
    oname=warpedDiff_${towarpbase}_${referencebase}.nii.gz
    deformationField=${targetbase}_${referencebase}1Warp.nii.gz
    antsApplyTransforms -d 3 -i warp/$towarp -o $oname -r reference/$reference -n NearestNeighbor --float -t $deformationField -t $affine
done
rm $deformationField
rm $inverseField
python -c 'from experiments.registration.dipyreg import *; compute_scores()'


