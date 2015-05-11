import os
import os.path
import numpy as np
import experiments.registration.dataset_info as info
import nibabel as nib
import dipy.align.metrics as metrics
import dipy.align.imwarp as imwarp
from dipy.align import VerbosityLevels
from experiments.registration.rcommon import getBaseFileName, decompose_path, readAntsAffine
from dipy.fixes import argparse as arg
from experiments.registration.evaluation import (compute_densities,
                                                 sample_from_density,
                                                 create_ss_de)

parser = arg.ArgumentParser(
    description=""
    )

parser.add_argument(
    'template', action = 'store', metavar = 'template',
    help = '''Template, reference modality''')


parser.add_argument(
    'real', action = 'store', metavar = 'real',
    help = '''Real image, reference modality''')


parser.add_argument(
    'prealign', action = 'store', metavar = 'prealign',
    help = '''ANTS affine registration matrix (.txt) that registers template towards real''')

parser.add_argument(
    'warp_dir', action = 'store', metavar = 'warp_dir',
    help = '''Directory (relative to ./ ) containing the template images in other modalities of interest''')





def draw_boxplots(means, vars, nrows=1, ncols=1, selected=1, fig=None):
    n = len(means)
    bpd = np.zeros(shape = (3, n), dtype = np.float64)

    for i, mu in enumerate(means):
        delta = np.sqrt(vars[i]/2)
        bpd[0][i] = mu - delta
        bpd[1][i] = mu
        bpd[2][i] = mu + delta

    if fig is None:
        fig = plt.figure(1)
    ax = fig.add_subplot(nrows, ncols, selected)
    bp = ax.boxplot(bpd)

    m = np.min(means)
    M = np.max(means)
    print("Range: [%0.2f, %0.2f]" % (m, M))
    xpos = range(0, n, 10)
    ypos = range(int(m), int(M), 10)
    xlabels = [str(i) for i in range(0, n, 10)]
    ylabels = [str(i) for i in range(int(m), int(M), 10)]
    plt.xticks(xpos, xlabels, fontsize=20)
    plt.yticks(ypos, ylabels, fontsize=20)
    plt.plot(np.array(range(n))+1, means)
    plt.grid()


def draw_boxplot_series(means, vars):
    colors = ["#96b4e6", "#faaa82", "#bebebe", "#fad278"]
    nseries = means.shape[0]
    n = means.shape[1]
    bpd = np.zeros(shape = (3, n*nseries), dtype = np.float64)
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    m = np.min(means)
    M = np.max(means)
    print("Range: [%0.2f, %0.2f]" % (m, M))
    ax.set_xticks(np.arange(m, M, 5))
    ax.set_yticks(np.arange(m, M, 5))

    for s in range(nseries):
        for i, mu in enumerate(means[s]):
            delta = np.sqrt(vars[s,i]/2)
            bpd[0][s*n + i] = mu - delta
            bpd[1][s*n + i] = mu
            bpd[2][s*n + i] = mu + delta

    bp = ax.boxplot(bpd, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors*nseries):
        patch.set_facecolor(color)



def print_arguments(params):
    r'''
    Verify all arguments were correctly parsed and interpreted
    '''
    print('========================Parameters========================')
    print('real: ', params.real)
    print('template: ', params.template)
    print('prealign: ', params.prealign)
    print('warp_dir: ', params.warp_dir)


def get_mean_transfer(A, B):
    means = np.zeros(1 + A.max(), dtype = np.float64)
    vars = np.zeros(1 + A.max(), dtype = np.float64)
    for label in range(A.min(), 1 + A.max()):
        if (A == label).sum() > 0:
            means[label] = np.array(B[A == label].mean())
            vars[label] = np.array(B[A == label].var())
    means[np.isnan(means)] = 0
    vars[np.isnan(vars)] = 0
    return means, vars

def create_semi_synthetic(params):
    r''' Create semi-synthetic image using real_mod1 as anatomy and tmp_mod2 template as intensity model
    Template tmp_mod1 is registered towards real_mod1 (which are assumed of the same modality) using SyN-CC.
    The transformation is applied to template tmp_mod2 (which is assumed to be perfectly aligned with tmp_mod1).
    The transfer function is computed from real_mod1 to warped tmp_mod2 and applied to real_mod1.
    '''
    real_mod1 = params.real
    base_fixed = getBaseFileName(real_mod1)
    tmp_mod1 = params.template
    prealign_name = params.prealign
    tmp_mod2_list = [os.path.join(params.warp_dir, name) for name in os.listdir(params.warp_dir)]
    tmp_mod2_list = [tmp_mod1] + tmp_mod2_list
    # Check if all warpings are already done
    warp_done = os.path.isfile('mask_'+base_fixed+'.nii.gz')
    if warp_done:
        for tmp_mod2 in tmp_mod2_list:
            base_moving = getBaseFileName(tmp_mod2)
            wname = 'warpedDiff_'+base_moving+'_'+base_fixed
            if real_mod1[-3:] == 'img': # Analyze
                wname += '.img'
            else:
                wname += '.nii.gz'
            if not os.path.isfile(wname):
                warp_done = False
                break

    #Load input images
    real_nib = nib.load(real_mod1)
    real_aff = real_nib.get_affine()
    real = real_nib.get_data().squeeze()
    if real_mod1[-3:] == 'img': # Analyze: move reference from center to corner
        offset = real_aff[:3,:3].dot(np.array(real.shape)//2)
        real_aff[:3,3] += offset

    t_mod1_nib = nib.load(tmp_mod1)
    t_mod1_aff = t_mod1_nib.get_affine()
    t_mod1 = t_mod1_nib.get_data().squeeze()
    if tmp_mod1[-3:] == 'img': # Analyze: move reference from center to corner
        offset = t_mod1_aff[:3,:3].dot(np.array(t_mod1.shape)//2)
        t_mod1_aff[:3,3] += offset

    #Load pre-align matrix
    print('Pre-align:', prealign_name)
    if not prealign_name:
        prealign = np.eye(4)
    else:
        if real_mod1[-3:] == 'img': # Analyze
            ref_coordinate_system = 'LAS'
        else: # DICOM
            ref_coordinate_system = 'LPS'

        if tmp_mod1[-3:] == 'img': # Analyze
            tgt_coordinate_system = 'LAS'
        else: # DICOM
            tgt_coordinate_system = 'LPS'
        prealign = readAntsAffine(prealign_name, ref_coordinate_system, tgt_coordinate_system)
    #Configure CC metric
    sigma_diff = 1.7
    radius = 4
    similarity_metric = metrics.CCMetric(3, sigma_diff, radius)

    #Configure optimizer
    opt_iter = [100, 100, 50]
    step_length = 0.25
    opt_tol = 1e-5
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.2
    if not warp_done:
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
        #Save the warped template (so we can visually check the registration result)
        warped = mapping.transform(t_mod1)
        base_moving = getBaseFileName(tmp_mod1)
        oname = 'warpedDiff_'+base_moving+'_'+base_fixed
        if real_mod1[-3:] == 'img': # Analyze
            oname += '.img'
        else:
            oname += '.nii.gz'
        real[...] = warped[...]
        real_nib.to_filename(oname)
        mask = (t_mod1 > 0).astype(np.int32)
        wmask = mapping.transform(mask, 'nearest')
        wmask_nib = nib.Nifti1Image(wmask, t_mod1_aff)
        wmask_nib.to_filename('mask_'+base_fixed+'.nii.gz')
    else:
        wmask_nib = nib.load('mask_'+base_fixed+'.nii.gz')
        wmask = wmask_nib.get_data().squeeze()

    #Compute and save the semi-synthetic images in different modalities
    for tmp_mod2 in tmp_mod2_list:
        print('Warping: '+tmp_mod2)
        t_mod2_nib = nib.load(tmp_mod2)
        t_mod2_aff = t_mod2_nib.get_affine()
        t_mod2 = t_mod2_nib.get_data().squeeze()

        base_moving = getBaseFileName(tmp_mod2)
        oname = base_moving+'_'+base_fixed
        if real_mod1[-3:] == 'img': # Analyze
            oname += '.img'
        else:
            oname += '.nii.gz'

        if not warp_done:
            # Save warped image
            warped = mapping.transform(t_mod2)
            wnib = nib.Nifti1Image(warped, t_mod2_aff)
            wnib.to_filename('warpedDiff_'+oname)
        else:
            wnib = nib.load('warpedDiff_'+oname)
            warped = wnib.get_data().squeeze()

        real_nib = nib.load(real_mod1)
        real = real_nib.get_data().squeeze()

        use_density_estimation = True
        nbins = 100
        if use_density_estimation:
            print('Using density sampling.')
            oname = 'ssds_' + oname
            # Compute marginal distributions
            densities = np.array(compute_densities(real.astype(np.int32), warped.astype(np.float64), nbins, wmask))
            # Sample the marginal distributions
            real[...] = create_ss_de(real.astype(np.int32), densities)
        else:
            print('Using mean transfer.')
            oname = 'ssmt_' + oname
            #Compute transfer function
            means, vars = get_mean_transfer(real, warped)
            #Apply transfer to real
            real[...] = means[real]

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

