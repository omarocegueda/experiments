"""
This script is the main launcher of the multi-modal non-linear image
registration
"""
from __future__ import print_function
import sys
import os
import numpy as np
import nibabel as nib
from dipy.fixes import argparse as arg
from experiments.registration.rcommon import getBaseFileName, decompose_path
import experiments.registration.evaluation as evaluation
from nipy.algorithms.registration import HistogramRegistration, resample
from nipy.io.files import nipy2nifti, nifti2nipy

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
    default = 'crl1')

parser.add_argument(
    '-t', '--transforms', action = 'store', metavar = 'transforms',
    help = '''A comma-separated (WITH NO SPACES) list of transform names,
    each being any of {TRANSLATION, ROTATION, RIGID, SCALING, AFFINE} specifying
    the desired sequence of transformation types
    ''',
    default = 'affine')

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
    default = 'powell')

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


def save_registration_results(sol, params):
    r'''
    Warp the moving image using the obtained solution from Nipy registration
    '''
    warp_dir = params.warp_dir

    base_static = getBaseFileName(params.static)
    static_nib = nib.load(params.static)
    static_nib = nib.Nifti1Image(static_nib.get_data().squeeze(), static_nib.get_affine())
    static = nifti2nipy(static_nib)
    static_affine = static_nib.get_affine()
    static_shape = np.array(static.shape, dtype=np.int32)

    base_moving = getBaseFileName(params.moving)
    moving_nib = nib.load(params.moving)
    moving_nib = nib.Nifti1Image(moving_nib.get_data().squeeze(), moving_nib.get_affine())
    moving = nifti2nipy(moving_nib)
    moving_affine = moving_nib.get_affine()
    moving_shape = np.array(moving.shape, dtype=np.int32)

    dim = len(static.shape)
    static_affine = static_affine[:(dim + 1), :(dim + 1)]
    moving_affine = moving_affine[:(dim + 1), :(dim + 1)]

    warped = resample(moving, sol.inv(), reference=static, interp_order=1)
    fmoved = 'warpedAff_'+base_moving+'_'+base_static+'.nii.gz'
    nib.save(nipy2nifti(warped, strict=True), fmoved)
    #---warp all volumes in the warp directory using NN interpolation
    names = [os.path.join(warp_dir, name) for name in os.listdir(warp_dir)]
    for name in names:
        to_warp_nib = nib.load(name)
        to_warp_nib = nib.Nifti1Image(to_warp_nib.get_data().squeeze(), to_warp_nib.get_affine())
        to_warp_affine = to_warp_nib.get_affine()
        img_affine = to_warp_affine[:(dim + 1), :(dim + 1)]

        to_warp = nifti2nipy(to_warp_nib)
        base_warp = getBaseFileName(name)
        warped = resample(to_warp, sol.inv(), reference=static, interp_order=0)
        fmoved = 'warpedAff_'+base_warp+'_'+base_static+'.nii.gz'
        nib.save(nipy2nifti(warped, strict=True), fmoved)
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
    #oname = base_moving+'_'+base_static+'Affine.txt'
    #np.savetxt(oname, affine)


def register_3d(params):
    r'''
    Runs affine registration with the parsed parameters
    '''
    # Default parameters
    renormalize = False
    interp = 'tri'

    metric_name=params.metric[0:params.metric.find('[')]
    metric_params_list=params.metric[params.metric.find('[')+1:params.metric.find(']')].split(',')
    nbins=int(metric_params_list[0])

    optimizer = params.method
    
    print('Registering %s to %s'%(params.moving, params.static))
    sys.stdout.flush()
    moving_mask = None
    static_mask = None

    #Load the data
    moving_nib = nib.load(params.moving)
    moving_nib = nib.Nifti1Image(moving_nib.get_data().squeeze(), moving_nib.get_affine())
    static_nib = nib.load(params.static)
    static_nib = nib.Nifti1Image(static_nib.get_data().squeeze(), static_nib.get_affine())

    moving= nifti2nipy(moving_nib)
    static= nifti2nipy(static_nib)

    # Register
    tic = time.time()
    R = HistogramRegistration(moving, static, from_bins=nbins, to_bins=nbins, 
                              similarity=metric_name, interp=interp, 
                              renormalize=renormalize)

    T = R.optimize('affine', optimizer=optimizer)
    toc = time.time()    

    save_registration_results(T, params)


if __name__ == '__main__':
      import time
      params = parser.parse_args()
      print_arguments(params)
      tic = time.clock()
      register_3d(params)
      toc = time.clock()
      print('Time elapsed (sec.): ',toc - tic)
