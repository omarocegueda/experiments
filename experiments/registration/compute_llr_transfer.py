import os
import sys
import numpy as np
import nibabel as nib
from dipy.core.optimize import Optimizer
import experiments.registration.dataset_info as info
import experiments.registration.semi_synthetic as ss
import dipy.align.cc_residuals as ccr

if __name__ == '__main__':
    radius = 4
    init_from = 'random'
    if len(sys.argv) > 1:
        radius = int(sys.argv[1])
    if len(sys.argv) > 2:
        init_from = sys.argv[2]
    # Load data
    print('Loading data...')
    t1_name = t1_name = info.get_brainweb("t1", "strip")
    t1_nib = nib.load(t1_name)
    t1 = t1_nib.get_data().squeeze()
    t1_lab = t1.astype(np.int32)
    t1 = t1.astype(np.float64)
    t1 = (t1.astype(np.float64) - t1.min())/(t1.max()-t1.min())

    t2_name = t2_name = info.get_brainweb("t2", "strip")
    t2_nib = nib.load(t2_name)
    t2 = t2_nib.get_data().squeeze()
    t2_lab = t2.astype(np.int32)
    t2 = t2.astype(np.float64)
    t2 = (t2.astype(np.float64) - t2.min())/(t2.max()-t2.min())

    # Compute transfer for optimal local-linearity
    print('Computing mean transfers...')
    init_from_mean = True
    nlabels_t1 = 1 + np.max(t1_lab)
    nlabels_t2 = 1 + np.max(t2_lab)

    means_t1t2, vars_t1t2 = ss.get_mean_transfer(t1_lab, t2)
    fmean_t1lab = np.array(means_t1t2)

    means_t2t1, vars_t2t1 = ss.get_mean_transfer(t2_lab, t1)
    fmean_t2lab = np.array(means_t2t1)

    def value_and_gradient_t1lab(x, *args):
        val, grad = ccr.compute_transfer_value_and_gradient(x, t1_lab, nlabels_t1, t2, radius)
        print("Energy:%f"%(val,))
        return val, np.array(grad)

    def value_and_gradient_t2lab(x, *args):
        val, grad = ccr.compute_transfer_value_and_gradient(x, t2_lab, nlabels_t2, t1, radius)
        print("Energy:%f"%(val,))
        return val, np.array(grad)

    
    options = {'maxiter': 20}
    if init_from == 'mean':
        f0 = fmean_t1lab
    elif init_from == 'random':
        f0 = np.random.uniform(0,1,len(fmean_t1lab))
    else:
        raise ValueError('Unknown initial point type:%s'%(init_from,))

    ofname = 'fopt_t1lab_'+str(radius)+'_'+init_from+'.npy'
    fopt_t1lab = None
    if os.path.isfile(ofname):
        print('Found precomputed t1lab transfer. Skipping optimization.')
        fopt_t1lab = np.load(ofname)
    else:
        print('Computing optimal llr transfer, t1lab. Radius=%d, init:%s'%(radius, init_from))
        opt_t1lab = Optimizer(value_and_gradient_t1lab, f0, "BFGS", jac = True, options=options)
        fopt_t1lab = opt_t1lab.xopt
        np.save(ofname, fopt_t1lab)

    options = {'maxiter': 20}
    if init_from == 'mean':
        f0 = fmean_t2lab
    elif init_from == 'random':
        f0 = np.random.uniform(0,1,len(fmean_t2lab))
    else:
        raise ValueError('Unknown initial point type:%s'%(init_from,))

    ofname = 'fopt_t2lab_'+str(radius)+'_'+init_from+'.npy'
    fopt_t2lab = None
    if os.path.isfile(ofname):
        print('Found precomputed t2lab transfer. Skipping optimization.')
        fopt_t2lab = np.load(ofname)
    else:
        print('Computing optimal llr transfer, t2lab. Radius=%d, init:%s'%(radius, init_from))
        opt_t2lab = Optimizer(value_and_gradient_t2lab, f0, "BFGS", jac = True, options=options)
        fopt_t2lab = opt_t2lab.xopt
        np.save(ofname, fopt_t2lab)


