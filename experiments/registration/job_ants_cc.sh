#!/bin/bash
####################################################
#Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l mem=3GB
#PBS -l pmem=3GB
#PBS -l vmem=3GB
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00
#PBS -N ANTs-CC

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
reference=$(ls {reference/*.nii.gz,reference/*.img} 2>/dev/null | xargs -n1 basename)
target=$(ls {target/*.nii.gz,target/*.img} 2>/dev/null | xargs -n1 basename)
extension="${target##*.}"
targetbase="${target%.*}"
targetbase="${targetbase%.*}"
referencebase="${reference%.*}"
referencebase="${referencebase%.*}"
#Affine registration using Mutual information with ANTS
affine="${targetbase}_${referencebase}Affine.txt"
affinePrecomputed="../affine/${affine}"
if [ -r $affinePrecomputed ]; then
    cp $affinePrecomputed .
fi
if ! [ -r $affine ]; then
    ANTS 3 -m MI[reference/$reference, target/$target, 1, 32] -i 0 -o ${targetbase}_${referencebase}
else
    echo "Affine mapping found ($affine). Skipping affine registration."
fi
#Diffeomorphic registration
deformationField=${targetbase}_${referencebase}Warp.nii.gz
inverseField=${targetbase}_${referencebase}InverseWarp.nii.gz
if [ -r $deformationField ]; then
    echo "Deformation found. Registration skipped."
else
    exe="ANTS 3 -m  CC[reference/$reference,target/$target,1,4] -t SyN[0.25] -a ${affine} -r Gauss[3,0] -o ${targetbase}_${referencebase} -i 100x100x25 --continue-affine false"
    echo " $exe "
    $exe
fi
oname=warpedDiff_${targetbase}_${referencebase}.nii.gz
WarpImageMultiTransform 3 target/$target $oname $deformationField $affine -R reference/$reference
for towarp in $( ls warp ); do
    towarpbase="${towarp%.*}"
    towarpbase="${towarpbase%.*}"
    oname=warpedDiff_${towarpbase}_${referencebase}.nii.gz
    deformationField=${targetbase}_${referencebase}Warp.nii.gz
    WarpImageMultiTransform 3 warp/$towarp $oname $deformationField $affine -R reference/$reference --use-NN
done
rm $deformationField
rm $inverseField
python -c 'from experiments.registration.dipyreg import *; compute_scores()'
date

