#!/bin/bash
####################################################
#Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l mem=8GB
#PBS -l pmem=8GB
#PBS -l vmem=8GB
#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00
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
ref_fixed=$(ls {ref_fixed/*.nii.gz,ref_fixed/*.img} 2>/dev/null | xargs -n1 basename)
ref_moving=$(ls {ref_moving/*.nii.gz,ref_moving/*.img} 2>/dev/null | xargs -n1 basename)
target=$(ls {target/*.nii.gz,target/*.img} 2>/dev/null | xargs -n1 basename)

ref_fixed_base="${ref_fixed%.*}"
ref_fixed_base="${ref_fixed_base%.*}"
ref_moving_base="${ref_moving%.*}"
ref_moving_base="${ref_moving_base%.*}"
targetbase="${target%.*}"
targetbase="${targetbase%.*}"

#Affine registration (target towards ref_fixed) using Mutual information with ANTS
affine_ref_fixed="${targetbase}_${ref_fixed_base}Affine.txt"
affine_ref_fixed0="${targetbase}_${ref_fixed_base}0GenericAffine.mat"
affine_ref_fixed_precomputed="../affine/${affine_ref_fixed}"
if [ -r $affine_ref_fixed_precomputed ]; then
    cp $affine_ref_fixed_precomputed .
fi

op="${targetbase}_${ref_fixed_base}"
if ! [ -r $affine_ref_fixed ]; then
    exe="antsRegistration -d 3 -r [ ref_fixed/$ref_fixed, target/$target, 1 ] \
                      -m mattes[ ref_fixed/$ref_fixed, target/$target, 1 , 32, regular, 0.3 ] \
                      -t translation[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 6x4x2 -l 1 \
                      -m mattes[ ref_fixed/$ref_fixed, target/$target, 1 , 32, regular, 0.3 ] \
                      -t rigid[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 3x2x1 -l 1 \
                      -m mattes[ ref_fixed/$ref_fixed, target/$target, 1 , 32, regular, 0.3 ] \
                      -t affine[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 3x2x1 -l 1 \
                      -o [${op}]"
    echo " $exe "
    $exe
    ConvertTransformFile 3 ${affine_ref_fixed0} ${affine_ref_fixed}
else
    echo "Affine mapping found ($affine_ref_fixed). Skipping affine registration."
fi

#Affine registration (target towards ref_moving) using Mutual information with ANTS
affine_ref_moving="${targetbase}_${ref_moving_base}Affine.txt"
affine_ref_moving0="${targetbase}_${ref_moving_base}0GenericAffine.mat"
affine_ref_moving_precomputed="../affine/${affine_ref_moving}"
if [ -r $affine_ref_moving_precomputed ]; then
    cp $affine_ref_moving_precomputed .
fi

op="${targetbase}_${ref_moving_base}"
if ! [ -r $affine_ref_moving ]; then
    exe="antsRegistration -d 3 -r [ ref_moving/$ref_moving, target/$target, 1 ] \
                      -m mattes[ ref_moving/$ref_moving, target/$target, 1 , 32, regular, 0.3 ] \
                      -t translation[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 6x4x2 -l 1 \
                      -m mattes[ ref_moving/$ref_moving, target/$target, 1 , 32, regular, 0.3 ] \
                      -t rigid[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 3x2x1 -l 1 \
                      -m mattes[ ref_moving/$ref_moving, target/$target, 1 , 32, regular, 0.3 ] \
                      -t affine[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 3x2x1 -l 1 \
                      -o [${op}]"
    echo " $exe "
    $exe
    ConvertTransformFile 3 ${affine_ref_moving0} ${affine_ref_moving}
else
    echo "Affine mapping found ($affine_ref_moving). Skipping affine registration."
fi




#Diffeomorphic registration (target towards ref_fixed) using Mutual information with ANTS
deformationField=${targetbase}_${ref_fixed_base}1Warp.nii.gz
inverseField=${targetbase}_${ref_fixed_base}1InverseWarp.nii.gz
op="${targetbase}_${ref_fixed_base}"
if [ -r $deformationField ]; then
    echo "Deformation found. Registration skipped."
else
    exe="antsRegistration -d 3 -r $affine_ref_fixed \
                      -m mattes[ref_fixed/$ref_fixed, target/$target, 1, 32 ] \
                      -t syn[ .25, 3, 0 ] \
                      -c [ 100x100x25,1e-5,12 ] \
                      -s 1x0.5x0vox \
                      -f 4x2x1\
                      -o [${op}, warpedDiff_${op}.nii.gz, warpedDiff_${op}.nii.gz]"
    echo " $exe "
    $exe
fi



#Diffeomorphic registration (target towards ref_moving) using Mutual information with ANTS
deformationField=${targetbase}_${ref_moving_base}1Warp.nii.gz
inverseField=${targetbase}_${ref_moving_base}1InverseWarp.nii.gz
op="${targetbase}_${ref_moving_base}"
if [ -r $deformationField ]; then
    echo "Deformation found. Registration skipped."
else
    exe="antsRegistration -d 3 -r $affine_ref_moving \
                      -m mattes[ref_moving/$ref_moving, target/$target, 1, 32 ] \
                      -t syn[ .25, 3, 0 ] \
                      -c [ 100x100x25,1e-5,12 ] \
                      -s 1x0.5x0vox \
                      -f 4x2x1\
                      -o [${op}, warpedDiff_${op}.nii.gz, warpedDiff_${op}.nii.gz]"
    echo " $exe "
    $exe
fi
date

oname=warpedDiff_${ref_moving_base}_${ref_fixed_base}.nii.gz
phi2=${targetbase}_${ref_moving_base}1InverseWarp.nii.gz
aff2=${affine_ref_moving}
phi1=${targetbase}_${ref_fixed_base}1Warp.nii.gz
aff1=${affine_ref_fixed}
antsApplyTransforms -d 3 -i ref_moving/$ref_moving -o $oname -r ref_fixed/$ref_fixed -n Linear --float -t ${phi1} -t ${aff1} -t ["${aff2}", 1] -t ${phi2} 

for towarp in $( ls warp ); do
    towarpbase="${towarp%.*}"
    towarpbase="${towarpbase%.*}"
    oname=warpedDiff_${towarpbase}_${ref_fixed}.nii.gz
    antsApplyTransforms -d 3 -i warp/$towarp -o $oname -r ref_fixed/$ref_fixed -n NearestNeighbor --float -t ${phi1} -t ${aff1} -t ["${aff2}", 1] -t ${phi2}
done
rm $phi1
rm $phi2
rm ${targetbase}_${ref_fixed_base}1InverseWarp.nii.gz
rm ${targetbase}_${ref_moving_base}1Warp.nii.gz
python -c 'from experiments.registration.dipyreg import *; compute_scores()'


