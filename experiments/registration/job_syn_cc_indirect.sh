#!/bin/bash
####################################################
#Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l mem=3GB
#PBS -l pmem=3GB
#PBS -l vmem=3GB
#PBS -l nodes=1:ppn=1
#PBS -l walltime=08:00:00
#PBS -N SyNECC

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
lscpu
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
affine_ref_fixed_precomputed="../affine/${affine_ref_fixed}"
if [ -r $affine_ref_fixed_precomputed ]; then
    cp $affine_ref_fixed_precomputed .
fi
if ! [ -r $affine_ref_fixed ]; then
    ANTS 3 -m MI[ref_fixed/$ref_fixed, target/$target, 1, 32] -i 0 -o ${targetbase}_${ref_fixed_base}
else
    echo "Affine mapping found ($affine_ref_fixed). Skipping target->ref_fixed affine registration."
fi

affine_ref_moving="${targetbase}_${ref_moving_base}Affine.txt"
affine_ref_moving_precomputed="../affine/${affine_ref_moving}"
if [ -r $affine_ref_moving_precomputed ]; then
    cp $affine_ref_moving_precomputed .
fi
if ! [ -r $affine_ref_moving ]; then
    ANTS 3 -m MI[ref_moving/$ref_moving, target/$target, 1, 32] -i 0 -o ${targetbase}_${ref_moving_base}
else
    echo "Affine mapping found ($affine_ref_moving). Skipping target->ref_moving affine registration."
fi

#Diffeomorphic registration
affine='""'
python dipyreg.py ref_moving/$ref_moving ref_fixed/$ref_fixed $affine warp --intermediate=target/$target --inter_to_reference_aff=$affine_ref_fixed --inter_to_target_aff=$affine_ref_moving --metric=CC[1.7,4] --iter=100,100,25 --step_length=0.25
date
