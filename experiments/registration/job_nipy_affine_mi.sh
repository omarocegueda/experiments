#!/bin/bash
####################################################
#Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l mem=2GB
#PBS -l pmem=2GB
#PBS -l vmem=2GB
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00
#PBS -N Nipy-MI

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
reference=$(ls {reference/*.nii.gz,reference/*.img} 2>/dev/null | xargs -n1 basename)
target=$(ls {target/*.nii.gz,target/*.img} 2>/dev/null | xargs -n1 basename)
extension="${target##*.}"
targetbase="${target%.*}"
targetbase="${targetbase%.*}"
referencebase="${reference%.*}"
referencebase="${referencebase%.*}"
#Affine registration with dipy
#metric="mi[32]"
metric="cr[32]"
python nipyreg_affine.py target/$target reference/$reference warp --metric=$metric
date

