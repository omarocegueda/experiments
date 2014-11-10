import numpy as np
import os
import nibabel as nib
from dipyreg import compute_jaccard, compute_target_overlap

def getBaseFileName(fname):
    base=os.path.basename(fname)
    noExt=os.path.splitext(base)[0]
    while(noExt!=base):
        base=noExt
        noExt=os.path.splitext(base)[0]
    return noExt


def decompose_path(fname):
    dirname=os.path.dirname(fname)+'/'
    base=os.path.basename(fname)
    no_ext = os.path.splitext(base)[0]
    while(no_ext !=base):
        base=no_ext
        no_ext =os.path.splitext(base)[0]
    ext = os.path.basename(fname)[len(no_ext):]
    return dirname, base, ext


def readAntsAffine(fname):
    try:
        with open(fname) as f:
            lines=[line.strip() for line in f.readlines()]
    except IOError:
        print 'Can not open file: ', fname
        return
    if not (lines[0]=="#Insight Transform File V1.0"):
        print 'Unknown file format'
        return
    if lines[1]!="#Transform 0":
        print 'Unknown transformation type'
        return
    A=np.zeros((3,3))
    b=np.zeros((3,))
    c=np.zeros((3,))
    for line in lines[2:]:
        data=line.split()
        if data[0]=='Transform:':
            if data[1]!='MatrixOffsetTransformBase_double_3_3' and data[1]!='AffineTransform_double_3_3':
                print 'Unknown transformation type'
                return
        elif data[0]=='Parameters:':
            parameters=np.array([float(s) for s in data[1:]], dtype=np.float64)
            A=parameters[:9].reshape((3,3))
            b=parameters[9:]
        elif data[0]=='FixedParameters:':
            c=np.array([float(s) for s in data[1:]], dtype=np.float64)
    T=np.ndarray(shape=(4,4), dtype=np.float64)
    T[:3,:3]=A[...]
    T[3,:]=0
    T[3,3]=1
    T[:3,3]=b+c-A.dot(c)
    ############This conversion is necessary for compatibility between itk and nibabel#########
    conversion=np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    T=conversion.dot(T.dot(conversion))
    ###########################################################################################
    return T


def compute_scores(pairs_fname = 'jaccard_pairs.lst'):
    with open(pairs_fname) as input:
        names = [s.split() for s in input.readlines()]
        for r in names:
            moving_dir, moving_base, moving_ext = decompose_path(r[0])
            fixed_dir, fixed_base, fixed_ext = decompose_path(r[1])
            warped_name = "warpedDiff_"+moving_base+"_"+fixed_base+".nii.gz"
            computeJacard(r[2], warped_name)

if __name__=='__main__':
    compute_scores('jaccard_pairs.lst')
