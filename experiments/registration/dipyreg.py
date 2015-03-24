"""
This script is the main launcher of the multi-modal non-linear image
registration
"""
from __future__ import print_function
import sys
import os
import numpy as np
import nibabel as nib
import dipy.align.imwarp as imwarp
from dipy.align import floating
import dipy.align.metrics as metrics
from dipy.fixes import argparse as arg
from dipy.align import VerbosityLevels
import dipy.align.vector_fields as vfu
from experiments.registration.rcommon import getBaseFileName, decompose_path, readAntsAffine
import experiments.registration.evaluation as evaluation

parser = arg.ArgumentParser(
    description=
        "Multi-modal, non-linear image registration. By default, it does NOT "
        "save the resulting deformation field but only the deformed target "
        "image using tri-linear interpolation under two transformations: "
        "(1)the affine transformation used to pre-register the images and (2) "
        "the resulting displacement field. If a warp directory is specified, "
        "then each image inside the directory will be warped using "
        "nearest-neighbor interpolation under the resulting deformation field."
        " The name of the warped target image under the AFFINE transformation"
        "will be 'warpedAffine_'+baseTarget+'_'+baseReference+'.nii.gz', "
        "where baseTarget is the file name of the target image (excluding all "
        "characters at and after the first '.'), baseReference is the file "
        "name of the reference image (excluding all characters at and after "
        "the first '.'). For example, if target is 'a.nii.gz' and reference is"
        " 'b.nii.gz', then the resulting deformation field will be saved as "
        "'dispDiff_a_b.npy'.\n"
        "The name of the warped target image under the NON-LINEAR "
        "transformation will be "
        "'warpedAffine_'+baseTarget+'_'+baseReference+'.nii.gz', with the "
        "convention explained above.\n"

        "Similarly, the name of each warped image inside the warp directory "
        "will be 'warpedDiff_'+baseWarp+'_'+baseReference+'.nii.gz' where "
        "baseWarp is the name of the corresponding image following the same "
        "convetion as above.\n"
        "Example:\n"
        "python dipyreg.py target.nii.gz reference.nii.gz "
        "ANTS_affine_target_reference.txt --smooth=25.0 --iter 10,50,50"
    )

parser.add_argument(
    'target', action = 'store', metavar = 'target',
    help = '''Nifti1 image or other formats supported by Nibabel''')

parser.add_argument(
    'reference', action = 'store', metavar = 'reference',
    help = '''Nifti1 image or other formats supported by Nibabel''')

parser.add_argument(
    'affine', action = 'store', metavar = 'affine',
    help = '''ANTS affine registration matrix (.txt) that registers target to
           reference''')

parser.add_argument(
    'warp_dir', action = 'store', metavar = 'warp_dir',
    help = '''Directory (relative to ./ ) containing the images to be warped
           with the obtained deformation field''')

parser.add_argument(
    '-inter', '--intermediate', action = 'store', metavar = 'intermediate',
    help = '''Nifti1 image or other formats supported by Nibabel''',
    default=None)

parser.add_argument(
    '-i2r', '--inter_to_reference_aff', action = 'store', metavar = 'inter_to_reference_aff',
    help = '''ANTS affine registration matrix (.txt) that registers target to
           reference''',
    default=None)

parser.add_argument(
    '-i2t', '--inter_to_target_aff', action = 'store', metavar = 'inter_to_target_aff',
    help = '''ANTS affine registration matrix (.txt) that registers target to
           reference''',
    default=None)

parser.add_argument(
    '-tm', '--target_mask', action = 'store', metavar = 'target_mask',
    help = '''Nifti1 image or other formats supported by Nibabel''',
    default=None)

parser.add_argument(
    '-rm', '--reference_mask', action = 'store', metavar = 'reference_mask',
    help = '''Nifti1 image or other formats supported by Nibabel''',
    default=None)

parser.add_argument(
    '-m', '--metric', action = 'store', metavar = 'metric',
    help = '''Any of {EM[L], CC[L]} specifying the metric to be used
    SSD=sum of squared diferences (monomodal), EM=Expectation Maximization
    to fit the transfer functions (multimodal), CC=Cross Correlation (monomodal
    and some multimodal) and the comma-separated (WITH NO SPACES) parameter
    list L:
    SSD[lambda,max_inner_iter,step_type]
        lambda: the smoothing parameter (the greater the smoother)
        max_inner_iter: maximum number of iterations of each level of the
        multi-resolution Gauss-Seidel algorithm step_type : energy minimization
        step, either 'gauss_newton' (optimized using multi-resolution
        Gauss Seidel) or 'demons' e.g.: SSD[25.0,20,'gauss_newton'] (NO SPACES),
        SSD[2.0,20,'demons'] (NO SPACES)
    EM[lambda,qLevels,max_inner_iter,step_type]
        lambda: the smoothing parameter (the greater the smoother)
        qLevels: number of quantization levels (hidden variables) in the EM
        formulation max_inner_iter: maximum number of iterations of each level
        of the multi-resolution Gauss-Seidel algorithm step_type : energy
        minimization step, either 'gauss_newton' (optimized using
        multi-resolution Gauss Seidel) or 'demons' e.g.:
        EM[25.0,256,20,'gauss_newton'] (NO SPACES), EM[2.0,256,20,'demons']
        (NO SPACES)
    CC[sigma_smooth,neigh_radius]
        sigma_smooth: std. dev. of the smoothing kernel to be used to smooth the
        gradient at each step neigh_radius: radius of the squared neighborhood
        to be used to compute the Cross Correlation at each voxel e.g.:CC[3.0,4]
        (NO SPACES)
    ECC[sigma_smooth,neigh_radius,qLevels]
        sigma_smooth: std. dev. of the smoothing kernel to be used to smooth the
        gradient at each step neigh_radius: radius of the squared neighborhood
        to be used to compute the Cross Correlation at each voxel qLevels:
        number of quantization levels (hidden variables) in the ECC formulation
        e.g.:CC[3.0,4,256] (NO SPACES)
    ''',
    default = 'CC[3.0,4]')

parser.add_argument(
    '-i', '--iter', action = 'store', metavar = 'i_0,i_1,...,i_n',
    help = '''A comma-separated (WITH NO SPACES) list of integers indicating the
           maximum number of iterations at each level of the Gaussian Pyramid
           (similar to ANTS), e.g. 10,100,100 (NO SPACES)''',
    default = '25,100,100')

parser.add_argument(
    '-stepl', '--step_length', action = 'store',
    metavar = 'step_length',
    help = '''The length of the maximum displacement vector of the update
           displacement field at each iteration''',
    default = '0.25')

parser.add_argument(
    '-sssf', '--ss_sigma_factor', action = 'store',
    metavar = 'ss_sigma_factor',
    help = '''parameter of the scale-space smoothing kernel. For example, the
           std. dev. of the kernel will be factor*(2^i) in the isotropic case
           where i=0,1,..,n_scales is the scale''',
    default = '0.2')

parser.add_argument(
    '-inv_iter', '--inversion_iter', action = 'store', metavar = 'max_iter',
    help = '''The maximum number of iterations for the displacement field
           inversion algorithm''',
    default='20')

parser.add_argument(
    '-inv_tol', '--inversion_tolerance', action = 'store',
    metavar = 'tolerance',
    help = '''The tolerance for the displacement field inversion algorithm''',
    default = '1e-3')

parser.add_argument(
    '-ii', '--inner_iter', action = 'store', metavar = 'max_iter',
    help = '''The number of Gauss-Seidel iterations to be performed to minimize
           each linearized energy''',
    default='20')

parser.add_argument(
    '-ql', '--quantization_levels', action = 'store', metavar = 'qLevels',
    help = '''The number levels to be used for the EM quantization''',
    default = '256')

parser.add_argument(
    '-single', '--single_gradient', action = 'store_true',
    help = '''The number levels to be used for the EM quantization''')

parser.add_argument(
    '-mask0', '--mask0', action = 'store_true',
    help = '''Set to zero all voxels of the scale space that are zero in the
           original image''')

parser.add_argument(
    '-rs', '--report_status', action = 'store_true',
    help = '''Instructs the algorithm to show the overlaid registered images
           after each pyramid level''')

parser.add_argument(
    '-aff', '--affine_only', dest = 'output_list',
    action = 'append_const', const='affine_only',
    help = r'''Indicates that only affine registration (provided as parameter)
           will be performed to warp the target images''')

parser.add_argument(
    '-sd', '--save_displacement', dest = 'output_list',
    action = 'append_const', const='displacement',
    help = r'''Specifies that the displacement field must be saved. The
           displacement field will be saved in .npy format. The filename will
           be the concatenation:
           'dispDiff_'+baseTarget+'_'+baseReference+'.npy'
           where baseTarget is the file name of the target image (excluding all
           characters at and after the first '.'), baseReference is the file
           name of the reference image (excluding all characters at and after
           the first '.'). For example, if target is 'a.nii.gz' and reference
           is 'b.nii.gz', then the resulting deformation field will be saved as
           'dispDiff_a_b.npy'.''')

parser.add_argument(
    '-si', '--save_inverse', dest = 'output_list', action = 'append_const',
    const = 'inverse',
    help = r'''Specifies that the inverse displacement field must be saved.
           The displacement field will be saved in .npy format. The filename
           will be the concatenation:
           'invDispDiff_'+baseTarget+'_'+baseReference+'.npy'
           where baseTarget is the file name of the target image (excluding all
           characters at and after the first '.'), baseReference is the file
           name of the reference image (excluding all characters at and after
           the first '.'). For example, if target is 'a.nii.gz' and reference
           is 'b.nii.gz', then the resulting deformation field will be saved as
           'invDispDiff_a_b.npy'.''')


def print_arguments(params):
    r'''
    Verify all arguments were correctly parsed and interpreted
    '''
    print('========================Parameters========================')
    print('target: ', params.target)
    print('reference: ', params.reference)
    print('intermediate: ', params.intermediate)
    print('inter_to_reference_aff: ', params.inter_to_reference_aff)
    print('inter_to_target_aff: ', params.inter_to_target_aff)
    print('target_mask: ', params.target_mask)
    print('reference_mask: ', params.reference_mask)
    print('affine: ', params.affine)
    print('warp_dir: ', params.warp_dir)
    print('metric: ', params.metric)
    print('iter:', params.iter)
    print('step_length:', params.step_length)
    print('ss_sigma_factor', params.ss_sigma_factor)
    print('inversion_iter', params.inversion_iter)
    print('inversion_tolerance', params.inversion_tolerance)
    print('single_gradient', params.single_gradient)
    print('mask0',params.mask0)
    print('report_status', params.report_status)
    print('---------Output requested--------------')
    print(params.output_list)
    print('==========================================================')



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
        scores=np.loadtxt(oname)
        return scores
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
    scores=np.array(evaluation.compute_target_overlap(A,B))
    print("Target overlap range:",scores.min(), scores.max())
    np.savetxt(oname,scores)
    return scores


def compute_scores(pairs_fname = 'jaccard_pairs.lst'):
    with open(pairs_fname) as input:
        names = [s.split() for s in input.readlines()]
        for r in names:
            moving_dir, moving_base, moving_ext = decompose_path(r[0])
            fixed_dir, fixed_base, fixed_ext = decompose_path(r[1])
            warped_name = "warpedDiff_"+moving_base+"_"+fixed_base+".nii.gz"
            compute_jaccard(r[2], warped_name, False)
            compute_target_overlap(r[2], warped_name, False)


def save_registration_results(mapping, params, mapping2=None):
    r'''
    Warp the target image using the obtained deformation field
    '''
    fixed = nib.load(params.reference)
    fixed_affine = fixed.get_affine()
    reference_shape = np.array(fixed.shape, dtype=np.int32)
    warp_dir = params.warp_dir
    base_moving = getBaseFileName(params.target)
    base_fixed = getBaseFileName(params.reference)
    moving = nib.load(params.target)
    moving_affine = moving.get_affine()
    moving = moving.get_data().squeeze().astype(np.float64)
    moving = moving.copy(order='C')
    if mapping2 is None:
        warped = np.array(mapping.transform(moving, 'linear'))
    else:
        # Warp by composition: mapping(mapping2_inv(x))
        # mapping is static to intermediate
        # mapping2 is moving to intermediate
        # The final composition is phi1(phi2(x))

        phi1 = mapping2.get_backward_field() # from intermediate to moving
        phi2 = mapping.get_forward_field() # from static to intermediate
        A = None # We have a post-align here (phi2 is forward-field of an inverse)
        B = fixed_affine # grid-to-space of phi2's domain (static domain)
        C = np.linalg.inv(moving_affine).dot(mapping2.prealign.dot(mapping.prealign_inv))
        D = mapping2.prealign.dot(mapping.prealign_inv)
        E = np.linalg.inv(moving_affine)
        warped = vfu.warp_with_composition_trilinear(phi1, phi2, A, B, C, D, E,
                                                     moving.astype(floating),
                                                     reference_shape)
    img_warped = nib.Nifti1Image(warped, fixed_affine)
    img_warped.to_filename('warpedDiff_'+base_moving+'_'+base_fixed+'.nii.gz')
    #---warp all volumes in the warp directory using NN interpolation
    names = [os.path.join(warp_dir, name) for name in os.listdir(warp_dir)]
    for name in names:
        to_warp = nib.load(name).get_data().squeeze().astype(np.int32)
        to_warp = to_warp.copy(order='C')
        base_warp = getBaseFileName(name)
        if mapping2 is None:
            warped = np.array(mapping.transform(to_warp, 'nearest')).astype(np.int16)
        else:
            # Warp by composition: mapping(mapping2_inv(x))
            warped = np.array(vfu.warp_with_composition_nn(phi1, phi2, A, B, C, D, E,
                                                         to_warp.astype(np.int32),
                                                         reference_shape)).astype(np.int16)
        img_warped = nib.Nifti1Image(warped, fixed_affine)
        img_warped.to_filename('warpedDiff_'+base_warp+'_'+base_fixed+'.nii.gz')
    #---now the jaccard indices
    if os.path.exists('jaccard_pairs.lst'):
        with open('jaccard_pairs.lst','r') as f:
            for line in f.readlines():
                aname, bname, cname= line.strip().split()
                abase = getBaseFileName(aname)
                bbase = getBaseFileName(bname)
                aname = 'warpedDiff_'+abase+'_'+bbase+'.nii.gz'
                if os.path.exists(aname) and os.path.exists(cname):
                    compute_jaccard(cname, aname, False)
                    compute_target_overlap(cname, aname, False)
                else:
                    print('Pair not found ['+cname+'], ['+aname+']')
    #---finally, the optional output
    if mapping2 is None:
        if params.output_list == None:
            return
        if 'lattice' in params.output_list:
            save_deformed_lattice_3d(
                mapping.forward,
                'latticeDispDiff_'+base_moving+'_'+base_fixed+'.nii.gz')
        if 'inv_lattice' in params.output_list:
            save_deformed_lattice_3d(
                mapping.backward, 'invLatticeDispDiff_'+base_moving+'_'+base_fixed+'.nii.gz')
        if 'displacement' in params.output_list:
            np.save('dispDiff_'+base_moving+'_'+base_fixed+'.npy', mapping.forward)
        if 'inverse' in params.output_list:
            np.save('invDispDiff_'+base_moving+'_'+base_fixed+'.npy', mapping.backward)


def load_nifti(fname):
    image = nib.load(fname)
    affine = image.get_affine()
    image = image.get_data().squeeze().astype(np.float64)
    if fname[-3:] == 'img': # Analyze: w.r.t the center of the image
        offset = affine[:3,:3].dot(np.array(image.shape)//2)
        affine[:3,3] += offset
    return image, affine


def register_3d(params):
    r'''
    Runs the non-linear registration with the parsed parameters
    '''
    print('Registering %s to %s'%(params.target, params.reference))
    sys.stdout.flush()
    metric_name=params.metric[0:params.metric.find('[')]
    metric_params_list=params.metric[params.metric.find('[')+1:params.metric.find(']')].split(',')
    moving_mask = None
    static_mask = None
    #Initialize the appropriate metric
    if metric_name == 'SSD':
        smooth=float(metric_params_list[0])
        inner_iter=int(metric_params_list[1])
        iter_type = metric_params_list[2]
        similarity_metric = metrics.SSDMetric(
            3, smooth, inner_iter, iter_type)
    elif metric_name=='EM':
        smooth=float(metric_params_list[0])
        q_levels=int(metric_params_list[1])
        inner_iter=int(metric_params_list[2])
        iter_type = metric_params_list[3]
        double_gradient=False if params.single_gradient else True
        similarity_metric = metrics.EMMetric(
            3, smooth, inner_iter, q_levels, double_gradient, iter_type)
        similarity_metric.mask0 = params.mask0
    elif metric_name=='CC':
        sigma_diff = float(metric_params_list[0])
        radius = int(metric_params_list[1])
        similarity_metric = metrics.CCMetric(3, sigma_diff, radius)
    elif metric_name=='ECC':
        from dipy.align.ECCMetric import ECCMetric
        sigma_diff = float(metric_params_list[0])
        radius = int(metric_params_list[1])
        q_levels = int(metric_params_list[2])
        similarity_metric = ECCMetric(3, sigma_diff, radius, q_levels)

        if params.target_mask is not None and os.path.isfile(params.target_mask):
            moving_mask = nib.load(params.target_mask)
            moving_mask = moving_mask.get_data().squeeze()
            moving_mask = (moving_mask>0).astype(np.int32)

        if params.reference_mask is not None and os.path.isfile(params.reference_mask):
            static_mask = nib.load(params.reference_mask)
            static_mask = static_mask.get_data().squeeze()
            static_mask = (static_mask>0).astype(np.int32)

    #Initialize the optimizer
    opt_iter = [int(i) for i in params.iter.split(',')]
    step_length = float(params.step_length)
    opt_tol = 1e-5
    inv_iter = int(params.inversion_iter)
    inv_tol = float(params.inversion_tolerance)
    ss_sigma_factor = float(params.ss_sigma_factor)
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, opt_iter, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol, None)

    #Load the data
    moving, moving_affine = load_nifti(params.target)
    fixed, fixed_affine = load_nifti(params.reference)

    # If internediate is not None, we need to run two registrations
    if params.intermediate is not None:
        intermediate, intermediate_affine = load_nifti(params.intermediate)
        inter_to_reference_aff = readAntsAffine(params.inter_to_reference_aff,
                                                'LPS', 'LPS')
        inter_to_target_aff = readAntsAffine(params.inter_to_target_aff,
                                                'LPS', 'LPS')
        registration_optimizer.verbosity = VerbosityLevels.DEBUG
        inter_to_ref = registration_optimizer.optimize(fixed, intermediate,
                                                       fixed_affine,
                                                       intermediate_affine,
                                                       inter_to_reference_aff)
        inter_to_moving = registration_optimizer.optimize(moving, intermediate,
                                                          moving_affine,
                                                          intermediate_affine,
                                                          inter_to_target_aff)
        save_registration_results(inter_to_ref, params, inter_to_moving)


    else:
        print('Affine:', params.affine)
        if not params.affine:
            transform = np.eye(4)
        else:
            #http://fieldtrip.fcdonders.nl/faq/how_are_the_different_head_and_mri_coordinate_systems_defined
            if params.reference[:-3] == 'img': # Analyze
                ref_coordinate_system = 'LAS'
            else: # DICOM
                ref_coordinate_system = 'LPS'

            if params.target[:-3] == 'img': # Analyze
                tgt_coordinate_system = 'LAS'
            else: # DICOM
                tgt_coordinate_system = 'LPS'
            transform = readAntsAffine(params.affine, ref_coordinate_system, tgt_coordinate_system)
        init_affine = np.linalg.inv(moving_affine).dot(transform.dot(fixed_affine))
        #Preprocess the data
        moving = (moving-moving.min())/(moving.max()-moving.min())
        fixed = (fixed-fixed.min())/(fixed.max()-fixed.min())
        #Run the registration
        if params.output_list is not None and 'affine_only' in params.output_list:
            print('Applying affine only')
            sh_direct=fixed.shape + (3,)
            sh_inv=moving.shape + (3,)
            direct = np.zeros(shape = sh_direct, dtype=np.float32)
            inv = np.zeros(shape = sh_inv, dtype=np.float32)
            mapping=imwarp.DiffeomorphicMap(3, direct, inv, None, init_affine)
        else:
            registration_optimizer.verbosity = VerbosityLevels.DEBUG
            mapping = registration_optimizer.optimize(fixed, moving, fixed_affine, moving_affine, transform)
        del registration_optimizer
        del similarity_metric
        save_registration_results(mapping, params)


if __name__ == '__main__':
      import time
      params = parser.parse_args()
      print_arguments(params)
      tic = time.clock()
      register_3d(params)
      toc = time.clock()
      print('Time elapsed (sec.): ',toc - tic)
