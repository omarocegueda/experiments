#!/bin/bash
####################################################
#Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l mem=2GB
#PBS -l pmem=2GB
#PBS -l vmem=2GB
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00
#PBS -N FLIRT-RIGID-MI

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

op="${targetbase}_${referencebase}"
matname="${targetbase}_${referencebase}Affine.txt"
metric="mutualinfo"
nbins="32"
dof="6"
oname=warpedAff_${targetbase}_${referencebase}.nii.gz
STARTTIME=$(date +%s)
if ! [ -r $affine ]; then
    exe="flirt -in target/$target -ref reference/$reference -out $oname -omat $matname -bins $nbins -cost $metric -dof $dof"
    echo " $exe "
    $exe
else
    echo "Affine mapping found ($affine). Skipping affine registration."
fi
ENDTIME=$(date +%s)
for towarp in $( ls warp ); do
    towarpbase="${towarp%.*}"
    towarpbase="${towarpbase%.*}"
    oname=warpedAff_${towarpbase}_${referencebase}.nii.gz
    flirt -in warp/$towarp -ref reference/$reference -out $oname -applyxfm -init $matname -interp nearestneighbour
done
python -c 'from experiments.registration.dipyreg_affine import *; compute_scores()'
echo "Time elapsed (sec.): $(($ENDTIME - $STARTTIME))"
