import numpy as np
import os
import nibabel as nib

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


def readAntsAffine(fname, coordinate_system='LPS'):
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
    conversion=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    if coordinate_system[0]=='L':
        conversion[0, 0] = -1
    if coordinate_system[0]=='P':
        conversion[1, 1] = -1
    if coordinate_system[2]=='I':
        conversion[2, 2] = -1
    T=conversion.dot(T.dot(conversion))
    ###########################################################################################
    return T
