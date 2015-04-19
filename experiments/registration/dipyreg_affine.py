"""
This script is the main launcher of the multi-modal non-linear image
registration
"""
from __future__ import print_function
import sys
import os
import numpy as np
import nibabel as nib
from dipy.align.imaffine import MattesMIMetric, AffineRegistration, aff_warp
from dipy.align.transforms import regtransforms
import dipy.align.imwarp as imwarp
import dipy.align.metrics as metrics
from dipy.fixes import argparse as arg
from dipy.align import VerbosityLevels
from experiments.registration.rcommon import getBaseFileName, decompose_path, readAntsAffine
import experiments.registration.evaluation as evaluation

parser = arg.ArgumentParser(
    description="Affine image registration."
    )

parser.add_argument(
    'moving', action = 'store', metavar = 'moving',
    help = '''Nifti1 image or other formats supported by Nibabel''')

parser.add_argument(
    'static', action = 'store', metavar = 'static',
    help = '''Nifti1 image or other formats supported by Nibabel''')

parser.add_argument(
    'warp_dir', action = 'store', metavar = 'warp_dir',
    help = '''Directory (relative to ./ ) containing the images to be warped
           with the obtained deformation field''')

parser.add_argument(
    '-mm', '--moving_mask', action = 'store', metavar = 'moving_mask',
    help = '''Nifti1 image or other formats supported by Nibabel''',
    default=None)

parser.add_argument(
    '-sm', '--static_mask', action = 'store', metavar = 'static_mask',
    help = '''Nifti1 image or other formats supported by Nibabel''',
    default=None)

parser.add_argument(
    '-m', '--metric', action = 'store', metavar = 'metric',
    help = '''Any of {MI[L]} specifying the metric to be used
    MI= mutual information and the comma-separated (WITH NO SPACES) parameter
    list L:
    MI[nbins]
        nbins: number of histogram bins
    ''',
    default = 'MI[32]')

parser.add_argument(
    '-t', '--transforms', action = 'store', metavar = 'transforms',
    help = '''A comma-separated (WITH NO SPACES) list of transform names,
    each being any of {TRANSLATION, ROTATION, RIGID, SCALING, AFFINE} specifying
    the desired sequence of transformation types
    ''',
    default = 'RIGID,AFFINE')

parser.add_argument(
    '-i', '--iter', action = 'store', metavar = 'i_0,i_1,...,i_n',
    help = '''A comma-separated (WITH NO SPACES) list of integers indicating the
           maximum number of iterations at each level of the Gaussian Pyramid
           (similar to ANTS), e.g. 10,100,100 (NO SPACES)''',
    default = '25,100,100')

parser.add_argument(
    '-method', '--method', action = 'store',
    metavar = 'method',
    help = '''Optimization method''',
    default = 'CGGS')

parser.add_argument(
    '-f', '--factors', action = 'store',
    metavar = 'factors',
    help = '''Shrink factors for the scale space''',
    default = '4,2,1')

parser.add_argument(
    '-s', '--sigmas', action = 'store',
    metavar = 'sigmas',
    help = '''Smoothin kernel's standard deviations for scale space''',
    default = '3,1,0')

parser.add_argument(
    '-sssf', '--ss_sigma_factor', action = 'store',
    metavar = 'ss_sigma_factor',
    help = '''parameter of the scale-space smoothing kernel. For example, the
           std. dev. of the kernel will be factor*(2^i) in the isotropic case
           where i=0,1,..,n_scales is the scale''',
    default = '1.0')

parser.add_argument(
    '-mask0', '--mask0', action = 'store_true',
    help = '''Set to zero all voxels of the scale space that are zero in the
           original image''')


def print_arguments(params):
    r'''
    Verify all arguments were correctly parsed and interpreted
    '''
    print('========================Parameters========================')
    print('moving: ', params.moving)
    print('static: ', params.static)
    print('moving_mask: ', params.moving_mask)
    print('static_mask: ', params.static_mask)
    print('warp_dir: ', params.warp_dir)
    print('metric: ', params.metric)
    print('transforms: ', params.transforms)
    print('iter:', params.iter)
    print('ss_sigma_factor', params.ss_sigma_factor)
    print('factors:', params.factors)
    print('sigmas:', params.sigmas)
    print('method:', params.method)
    print('mask0',params.mask0)


def compute_jaccard(aname, bname, keep_existing = True):
    baseA=getBaseFileName(aname)
    baseB=getBaseFileName(bname)
    oname="jaccard_"+baseA+"_"+baseB+".txt"
    if keep_existing and os.path.exists(oname):
        print('Jaccard overlap found. Skipped computation.')
        jaccard=np.loadtxt(oname)
        return jaccard
    nib_A=nib.load(aname)
    affineA=nib_A.get_affine()
    A=nib_A.get_data().squeeze().astype(np.int32)
    A=np.copy(A, order='C')
    print("A range:",A.min(), A.max())
    nib_B=nib.load(bname)
    #newB=nib.Nifti1Image(nib_B.get_data(),affineA)
    #newB.to_filename(bname)
    B=nib_B.get_data().squeeze().astype(np.int32)
    B=np.copy(B, order='C')
    print("B range:",B.min(), B.max())
    jaccard=np.array(evaluation.compute_jaccard(A,B))
    print("Jaccard range:",jaccard.min(), jaccard.max())
    np.savetxt(oname,jaccard)
    return jaccard


def compute_target_overlap(aname, bname, keep_existing = True):
    baseA=getBaseFileName(aname)
    baseB=getBaseFileName(bname)
    oname="t_overlap_"+baseA+"_"+baseB+".txt"
    if keep_existing and os.path.exists(oname):
        print('Target overlap overlap found. Skipped computation.')
        socres=np.loadtxt(oname)
        return socres
    nib_A=nib.load(aname)
    affineA=nib_A.get_affine()
    A=nib_A.get_data().squeeze().astype(np.int32)
    A=np.copy(A, order='C')
    print("A range:",A.min(), A.max())
    nib_B=nib.load(bname)
    #newB=nib.Nifti1Image(nib_B.get_data(),affineA)
    #newB.to_filename(bname)
    B=nib_B.get_data().squeeze().astype(np.int32)
    B=np.copy(B, order='C')
    print("B range:",B.min(), B.max())
    socres=np.array(evaluation.compute_target_overlap(A,B))
    print("Target overlap range:",socres.min(), socres.max())
    np.savetxt(oname,socres)
    return socres


def compute_scores(pairs_fname = 'jaccard_pairs.lst'):
    with open(pairs_fname) as input:
        names = [s.split() for s in input.readlines()]
        for r in names:
            moving_dir, moving_base, moving_ext = decompose_path(r[0])
            static_dir, static_base, static_ext = decompose_path(r[1])
            warped_name = "warpedAff_"+moving_base+"_"+static_base+".nii.gz"
            compute_jaccard(r[2], warped_name, False)
            compute_target_overlap(r[2], warped_name, False)


def save_registration_results(affine, params):
    r'''
    Warp the moving image using the obtained affine transform
    '''
    warp_dir = params.warp_dir

    base_static = getBaseFileName(params.static)
    static_nib = nib.load(params.static)
    static = static_nib.get_data().squeeze().astype(np.float64)
    static_affine = static_nib.get_affine()
    static_shape = np.array(static.shape, dtype=np.int32)

    base_moving = getBaseFileName(params.moving)
    moving_nib = nib.load(params.moving)
    moving = moving_nib.get_data().squeeze().astype(np.float64)
    moving_affine = moving_nib.get_affine()
    moving_shape = np.array(moving.shape, dtype=np.int32)

    dim = len(static.shape)
    static_affine = static_affine[:(dim + 1), :(dim + 1)]
    moving_affine = moving_affine[:(dim + 1), :(dim + 1)]

    warped = aff_warp(static, static_affine, moving, moving_affine, affine, False)

    img_affine = np.eye(4,4)
    img_affine[:(dim + 1), :(dim + 1)] = static_affine[...]
    img_warped = nib.Nifti1Image(warped, img_affine)
    img_warped.to_filename('warpedAff_'+base_moving+'_'+base_static+'.nii.gz')
    #---warp all volumes in the warp directory using NN interpolation
    names = [os.path.join(warp_dir, name) for name in os.listdir(warp_dir)]
    for name in names:
        to_warp_nib = nib.load(name)
        to_warp_affine = to_warp_nib.get_affine()
        img_affine = to_warp_affine[:(dim + 1), :(dim + 1)]

        to_warp = to_warp_nib.get_data().squeeze().astype(np.int32)
        base_warp = getBaseFileName(name)
        print(static.dtype, static_affine.dtype, to_warp.dtype, img_affine.dtype, affine.dtype)
        warped = aff_warp(static, static_affine, to_warp, img_affine, affine, True)
        img_affine = np.eye(4,4)
        img_affine[:(dim + 1), :(dim + 1)] = static_affine[...]
        img_warped = nib.Nifti1Image(warped, img_affine)
        img_warped.to_filename('warpedAff_'+base_warp+'_'+base_static+'.nii.gz')
    #---now the jaccard indices
    if os.path.exists('jaccard_pairs.lst'):
        with open('jaccard_pairs.lst','r') as f:
            for line in f.readlines():
                aname, bname, cname= line.strip().split()
                abase = getBaseFileName(aname)
                bbase = getBaseFileName(bname)
                aname = 'warpedAff_'+abase+'_'+bbase+'.nii.gz'
                if os.path.exists(aname) and os.path.exists(cname):
                    compute_jaccard(cname, aname, False)
                    compute_target_overlap(cname, aname, False)
                else:
                    print('Pair not found ['+cname+'], ['+aname+']')
    #---finally, the optional output
    oname = base_moving+'_'+base_static+'Affine.txt'
    np.savetxt(oname, affine)


def register_3d(params):
    r'''
    Runs affine registration with the parsed parameters
    '''
    print('Registering %s to %s'%(params.moving, params.static))
    sys.stdout.flush()
    metric_name=params.metric[0:params.metric.find('[')]
    metric_params_list=params.metric[params.metric.find('[')+1:params.metric.find(']')].split(',')
    moving_mask = None
    static_mask = None
    #Initialize the appropriate metric
    if metric_name == 'MI':
        nbins=int(metric_params_list[0])
        metric = MattesMIMetric(nbins)
    else:
        raise ValueError('Unknown metric: %s'%(metric_name,))

    #Initialize the optimizer
    opt_tol = 1e-5
    opt_iter = [int(i) for i in params.iter.split(',')]
    transforms = [t for t in params.transforms.split(',')]
    ss_sigma_factor = float(params.ss_sigma_factor)
    factors = [int(i) for i in params.factors.split(',')]
    sigmas = [float(i) for i in params.sigmas.split(',')]
    #method = 'CGGS'
    method = params.method
    affreg = AffineRegistration(metric, method, opt_iter, opt_tol, ss_sigma_factor,
                                factors, sigmas, None)

    #Load the data
    moving_nib = nib.load(params.moving)
    moving_affine = moving_nib.get_affine()
    moving = moving_nib.get_data().squeeze().astype(np.float64)

    static_nib = nib.load(params.static)
    static_affine = static_nib.get_affine()
    static = static_nib.get_data().squeeze().astype(np.float64)

    dim = len(static.shape)
    static_affine = static_affine[:(dim + 1), :(dim + 1)]
    moving_affine = moving_affine[:(dim + 1), :(dim + 1)]

    #Preprocess the data
    moving = (moving-moving.min())/(moving.max()-moving.min())
    static = (static-static.min())/(static.max()-static.min())

    #Run the registration
    sol = np.eye(dim + 1)
    prealign = 'mass'

    for transform_name in transforms:
        transform = regtransforms[(transform_name, dim)]
        print('Optimizing: %s'%(transform_name,))
        x0 = None
        sol = affreg.optimize(static, moving, transform, x0,
                              static_affine, moving_affine, prealign = prealign)
        prealign = sol.copy()

    save_registration_results(sol, params)
    print('Solution: ', sol)


if __name__ == '__main__':
      import time
      params = parser.parse_args()
      print_arguments(params)
      tic = time.clock()
      register_3d(params)
      toc = time.clock()
      print('Time elapsed (sec.): ',toc - tic)
