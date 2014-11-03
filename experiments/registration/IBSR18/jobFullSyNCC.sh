#!/bin/bash
####################################################
#Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l mem=2GB
#PBS -l pmem=2GB
#PBS -l vmem=2GB
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00
#PBS -N SyNCC

# Configure your environment
export ANACONDA_BIN_DIR="/home/omar/anaconda/bin"
export DIPY_DIR = "~/opt/dipy"
export EXPERIMENTS_DIR = "~/experiments/"
# ===

export PATH=$ANACONDA_BIN_DIR:$PATH
export PYTHONPATH=$DIPY_DIR:$EXPERIMENTS_DIR:$PYTHONPATH
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
python dipyreg.py target/$target reference/$reference $affine warp --metric=CC[1.7,4] --iter=10,10,5 --step_length=0.25
date
