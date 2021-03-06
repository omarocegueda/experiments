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
    dirname=os.path.dirname(fname)
    if len(dirname)>0:
        dirname += '/'

    base=os.path.basename(fname)
    no_ext = os.path.splitext(base)[0]
    while(no_ext !=base):
        base=no_ext
        no_ext =os.path.splitext(base)[0]
    ext = os.path.basename(fname)[len(no_ext):]
    return dirname, base, ext


def readAntsAffine(fname, ref_coordinate_system='LPS', tgt_coordinate_system='LPS'):
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
    ref_conversion=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    tgt_conversion=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    if ref_coordinate_system[0]=='L':
        ref_conversion[0, 0] = -1
    if ref_coordinate_system[1]=='P':
        ref_conversion[1, 1] = -1
    if ref_coordinate_system[2]=='I':
        ref_conversion[2, 2] = -1

    if tgt_coordinate_system[0]=='L':
        tgt_conversion[0, 0] = -1
    if tgt_coordinate_system[1]=='P':
        tgt_conversion[1, 1] = -1
    if tgt_coordinate_system[2]=='I':
        tgt_conversion[2, 2] = -1
    T=tgt_conversion.dot(T.dot(ref_conversion))
    ###########################################################################################
    return T


def add_random_rigid_transforms(dwi, grid2world):
    from dipy.align.transforms import RigidTransform3D
    from dipy.align.vector_fields import warp_3d_affine
    if grid2world is None:
        grid2world = np.eye(4)
        world2grid = np.eye(4)
    else:
        world2grid = np.linalg.inv(grid2world)

    # sigmas in degrees
    sigmas = np.array([3, 3, 3, 2, 2, 2], dtype=np.float64)
    # to radians
    sigmas[:3] *= np.pi/180.0
    n = dwi.shape[3]
    rigid = RigidTransform3D()
    transforms = []
    jitter = np.empty_like(dwi)
    for i in range(n):
        theta = np.random.randn(6)
        theta *= sigmas
        gt = rigid.param_to_matrix(theta)
        gt_inv = np.linalg.inv(gt)
        M = world2grid.dot(gt_inv.dot(grid2world))
        in_vol = dwi[...,i].astype(np.float32)
        out_shape = np.array(dwi[...,i].shape, dtype=np.int32)
        w = warp_3d_affine(in_vol, out_shape, M)
        jitter[...,i] = w[...]
        transforms.append(gt)
    return jitter, transforms


