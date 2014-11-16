import os
import experiments.registration.dataset_info as info
import nibabel as nib
import dipy.align.metrics as metrics
import dipy.align.imwarp as imwarp
from dipy.align import VerbosityLevels
from experiments.registration.rcommon import getBaseFileName, decompose_path, readAntsAffine

parser = arg.ArgumentParser(
    description=""
    )

parser.add_argument(
    'real', action = 'store', metavar = 'real',
    help = '''Real image, reference modality''')

parser.add_argument(
    'template', action = 'store', metavar = 'template',
    help = '''Template, reference modality''')

parser.add_argument(
    'prealign', action = 'store', metavar = 'prealign',
    help = '''ANTS affine registration matrix (.txt) that registers template towards real''')

parser.add_argument(
    'warp_dir', action = 'store', metavar = 'warp_dir',
    help = '''Directory (relative to ./ ) containing the template images in other modalities of interest''')

def print_arguments(params):
    r'''
    Verify all arguments were correctly parsed and interpreted
    '''
    print('========================Parameters========================')
    print('target: ', params.target)
    print('reference: ', params.reference)
    print('affine: ', params.affine)
    print('warp_dir: ', params.warp_dir)


def get_mean_transfer(A, B):
    means = np.array([B[A == label].mean() for label in range(256) ])
    means[np.isnan(means)] = 0
    vars = np.array([B[A == label].var() for label in range(256) ])
    vars[np.isnan(vars)] = 0
    return means, vars


def create_semi_synthetic(params):
    r''' Create semi-synthetic image using real_mod1 as anatomy and tmp_mod2 template as intensity model
    Template tmp_mod1 is registered towards real_mod1 (which are assumed of the same modality) using SyN-CC.
    The transformation is applied to template tmp_mod2 (which is assumed to be perfectly aligned with tmp_mod1).
    The transfer function is computed from real_mod1 to warped tmp_mod2 and applied to real_mod1.
    '''
    real_mod1 = params.real
    tmp_mod1 = params.template
    prealign_name = params.prealign
    tmp_mod2_list = [os.path.join(params.warp_dir, name) for name in os.listdir(params.warp_dir)]
    
    base_moving = getBaseFileName(tmp_mod1)
    base_fixed = getBaseFileName(real_mod1)
    oname = 'warpedDiff_'+base_moving+'_'+base_fixed
    if real_mod1[:-3] == 'img': # Analyze
        oname += '.nii.gz'
    else:
        oname += '.img'

    #Load input images
    real_nib = nib.load(real_mod1)
    real_aff = real_nib.get_affine()
    real = real_nib.get_data().squeeze()

    t_mod1_nib = nib.load(tmp_mod1)
    t_mod1_aff = t_mod1_nib.get_affine()
    t_mod1 = t_mod1_nib.get_data().squeeze()

    t_mod2_nib = nib.load(tmp_mod2)
    t_mod2_aff = t_mod2_nib.get_affine()
    t_mod2 = t_mod2_nib.get_data().squeeze()

    #Load pre-align matrix
    print('Pre-align:', prealign_name)
    if not prealign_name:
        prealign = np.eye(4)
    else:
        if real_mod1[:-3] == 'img': # Analyze
            prealign = readAntsAffine(prealign_name, 'LAS')
        else: # DICOM
            prealign = readAntsAffine(prealign_name, 'LPS')

    #Configure CC metric
    sigma_diff = 1.7
    radius = 4
    similarity_metric = metrics.CCMetric(3, sigma_diff, radius)

    #Configure optimizer
    opt_iter = [1, 1, 0]
    step_length = 0.25
    opt_tol = 1e-5
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.2
    syn = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric, 
                                                    opt_iter, 
                                                    step_length, 
                                                    ss_sigma_factor, 
                                                    opt_tol, 
                                                    inv_iter, 
                                                    inv_tol, 
                                                    callback = None)
    #Run registration
    syn.verbosity = VerbosityLevels.DEBUG
    mapping = syn.optimize(real, t_mod1, real_aff, t_mod1_aff, prealign)

    #Transform template (opposite modality)
    warped = mapping.transform(t_mod2)
    
    #Compute transfer function
    means, vars = get_mean_transfer(real, warped)
    
    #Apply transfer to real
    real = means[real]
    
    #Save semi_synthetic
    real_nib.to_filename(oname)


if __name__ == '__main__':
      import time
      params = parser.parse_args()
      print_arguments(params)
      tic = time.clock()
      create_semi_synthetic(params)
      toc = time.clock()
      print('Time elapsed (sec.): ',toc - tic)
