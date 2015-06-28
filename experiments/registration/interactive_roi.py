from experiments.registration.dataset_info import *
from experiments.registration.semi_synthetic import *
import nibabel as nib
import cc_residuals as ccr
from experiments.registration.rcommon import readAntsAffine
import dipy.align.vector_fields as vfu
from scipy.linalg import eig, eigvals, solve, lstsq
from dipy.core.optimize import Optimizer
import os.path
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import dipy.viz.regtools as rt
import experiments.registration.regviz as rv

from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
from dipy.align.imaffine import (align_centers_of_mass,
                                 transform_image,
                                 MattesMIMetric,
                                 LocalCCMetric,
                                 AffineRegistration)
from dipy.align.transforms import regtransforms

def check_affine_fit(x, y, x_name="", y_name="", force_reference=-1):
    alpha, beta, x_fit, y_fit, ref, mse = ccr.affine_fit(x, y, force_reference)
    x_fit = np.array(x_fit)
    y_fit = np.array(y_fit)
    #verify optimality
    main_title = ""
    if ref == 0: # fit y as a function of x
        main_title = y_name+" as a function of "+x_name
        assert_almost_equal(x.dot(x)*alpha + x.sum()*beta - x.dot(y), 0)
        assert_almost_equal(x.sum()*alpha + len(y) * beta - y.sum(), 0)
        figure()
        scatter(x,y)
        plot(x_fit, y_fit)
        title(main_title)
        xlabel(x_name)
        ylabel(y_name)
    else: # Fit x as a function of y
        main_title = x_name+" as a function of "+y_name
        assert_almost_equal(y.dot(y)*alpha + y.sum()*beta - y.dot(x), 0)
        assert_almost_equal(y.sum()*alpha + len(x) * beta - x.sum(), 0)
        figure()
        scatter(y,x)
        plot(y_fit, x_fit)
        title(main_title)
        xlabel(y_name)
        ylabel(x_name)
    print(main_title + ". MSE: "+str(mse)+". alpha: "+str(alpha)+". beta: "+str(beta) )


def check_linear_fit(x, y, x_name="", y_name="", force_reference=-1):
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    alpha, x_fit, y_fit, ref, mse = ccr.linear_fit(x, y, force_reference)
    x_fit = np.array(x_fit)
    y_fit = np.array(y_fit)
    #verify optimality
    main_title = ""
    if ref == 0: # fit y as a function of x
        main_title = y_name+" as a function of "+x_name
        assert_almost_equal(x.dot(y) - alpha * x.dot(x), 0)
        figure()
        scatter(x,y)
        plot(x_fit, y_fit)
        title(main_title)
        xlabel(x_name)
        ylabel(y_name)
    else: # Fit x as a function of y
        main_title = x_name+" as a function of "+y_name
        assert_almost_equal(x.dot(y) - alpha * y.dot(y), 0)
        figure()
        scatter(y,x)
        plot(y_fit, x_fit)
        title(main_title)
        xlabel(y_name)
        ylabel(x_name)
    print(main_title + ". MSE: "+str(mse)+". alpha: "+str(alpha))


def analyze_roi(p, delta, mod1, ss_mod1, mod1_name, mod2, ss_mod2, mod2_name, residuals, check_function):
    px = p[0]
    py = p[1]
    pz = p[2]
    roi = np.zeros_like(residuals)
    roi[(px-delta):(px+delta+1), (py-delta):(py+delta+1), (pz-delta):(pz+delta+1)] = 1
    rt.plot_slices(residuals_nb, cmap=None)
    rt.plot_slices((roi+0.1)*residuals_nb, cmap=None)

    samples = roi*residuals_nb
    samples = samples[roi!=0].reshape(-1)

    x = mod1[roi!=0].reshape(-1)
    y = mod2[roi!=0].reshape(-1)

    check_function(x, y, mod1_name, mod2_name, 0)
    check_function(x, y, mod1_name, mod2_name, 1)

    x_ss_t1 = mod1[roi!=0].reshape(-1)
    y_ss_t1 = ss_mod1[roi!=0].reshape(-1)
    check_function(x_ss_t1, y_ss_t1, mod1_name, "F["+mod2_name+"]", 0)
    check_function(x_ss_t1, y_ss_t1, mod1_name, "F["+mod2_name+"]", 1)

    x_ss_t2 = mod2[roi!=0].reshape(-1)
    y_ss_t2 = ss_mod2[roi!=0].reshape(-1)
    check_function(x_ss_t2, y_ss_t2, mod2_name, "F["+mod1_name+"]", 0)
    check_function(x_ss_t2, y_ss_t2, mod2_name, "F["+mod1_name+"]", 1)

def draw_affine_fit(ax, x, x_name, y, y_name):
    # Affine fit
    alpha, beta, x_fit, y_fit, ref, mse = ccr.affine_fit(x, y, 0)
    x_fit, y_fit = np.array(x_fit), np.array(y_fit)

    font = {'family' : 'serif',
            'color'  : 'black',
            'weight' : 'normal',
            'size'   : 16,}

    main_title = y_name+" as a function of "+x_name
    ax.cla()
    ax.scatter(x, y)
    ax.plot(x_fit, y_fit)
    ax.grid(True)
    ax.set_title(main_title, fontdict=font)
    ax.set_xlabel(x_name, fontdict=font)
    ax.set_ylabel(y_name, fontdict=font)


def draw_dwi_gp(ax, sel_signal, dwi_name):
    font = {'family' : 'serif',
            'color'  : 'black',
            'weight' : 'normal',
            'size'   : 16,}

    main_title = dwi_name
    ax.cla()
    # Plot required info
    x_name = ''
    y_name = ''
    ax.plot(sel_signal)
    #--
    ax.grid(True)
    ax.set_title(main_title, fontdict=font)
    ax.set_xlabel(x_name, fontdict=font)
    ax.set_ylabel(y_name, fontdict=font)


def draw_rect_image_pair(event):
    global global_figure
    global sel_x
    global sel_y
    if event.inaxes != global_figure.axes[0]:
        return
    global global_map
    side = 9
    px, py = int(event.xdata), int(event.ydata)

    img0 = global_map['img0']
    img0_name = global_map['img0_name']
    img1 = global_map['img1']
    img1_name = global_map['img1_name']
    shape = img0.shape
    residuals = global_map['residuals']
    slice_type = global_map['slice_type']
    slice_index = global_map['slice_index']
    vmin = global_map['vmin']
    vmax = global_map['vmax']
    if slice_index is None:
        slice_index = img0.shape[slice_type]//2

    subsample0=None
    subsample1=None
    x, y, z = None, None, None
    ax = global_figure.add_subplot(1,3,1)
    if slice_type==0:
        ax.imshow(residuals[slice_index,:,:].T, origin='lower', vmin=vmin, vmax=vmax)
        x, y, z = slice_index, px, py
    elif slice_type==1:
        ax.imshow(residuals[:,slice_index,:].T, origin='lower', vmin=vmin, vmax=vmax)
        x, y, z = px, slice_index, py
    else:
        ax.imshow(residuals[:,:,slice_index].T, origin='lower', vmin=vmin, vmax=vmax)
        x, y, z = px, py, slice_index
    print("V[%d,%d,%d]=%f\n"%(x,y,z,residuals[x,y,z]))
    minx, maxx = max(0, x-side//2), min(shape[0]-1, x+side//2)
    miny, maxy = max(0, y-side//2), min(shape[1]-1, y+side//2)
    minz, maxz = max(0, z-side//2), min(shape[2]-1, z+side//2)

    sel_x=img0[minx:(maxx+1), miny:(maxy+1), minz:(maxz+1)]
    sel_x = sel_x.reshape(-1)
    sel_y=img1[minx:(maxx+1), miny:(maxy+1), minz:(maxz+1)]
    sel_y = sel_y.reshape(-1)
    ax = global_figure.add_subplot(1,3,2)
    draw_affine_fit(ax, sel_x, img0_name, sel_y, img1_name)
    ax = global_figure.add_subplot(1,3,3)
    draw_affine_fit(ax, sel_y, img1_name, sel_x, img0_name)

    ax = global_figure.get_axes()[0]
    R = Rectangle((px-side//2,py-side//2), side, side, facecolor='none', linewidth=3, edgecolor='#DD0000')
    if len(ax.artists)>0:
        ax.artists[-1].remove()
    ax.add_artist(R)
    draw()

def draw_rect_dwi(event):
    global global_figure
    global sel_signal
    if event.inaxes != global_figure.axes[0]:
        return
    global global_map
    side = 9
    px, py = int(event.xdata), int(event.ydata)

    dwi = global_map['dwi']
    dwi_name = global_map['dwi_name']
    shape = dwi.shape
    residuals = global_map['residuals']
    slice_type = global_map['slice_type']
    slice_index = global_map['slice_index']
    vmin = global_map['vmin']
    vmax = global_map['vmax']
    if slice_index is None:
        slice_index = dwi.shape[slice_type]//2

    subsample0=None
    subsample1=None
    x, y, z = None, None, None
    ax = global_figure.add_subplot(1,2,1)
    if slice_type==0:
        ax.imshow(residuals[slice_index,:,:].T, origin='lower', vmin=vmin, vmax=vmax)
        x, y, z = slice_index, px, py
    elif slice_type==1:
        ax.imshow(residuals[:,slice_index,:].T, origin='lower', vmin=vmin, vmax=vmax)
        x, y, z = px, slice_index, py
    else:
        ax.imshow(residuals[:,:,slice_index].T, origin='lower', vmin=vmin, vmax=vmax)
        x, y, z = px, py, slice_index
    print("V[%d,%d,%d]=%f\n"%(x,y,z,residuals[x,y,z]))
    minx, maxx = max(0, x-side//2), min(shape[0]-1, x+side//2)
    miny, maxy = max(0, y-side//2), min(shape[1]-1, y+side//2)
    minz, maxz = max(0, z-side//2), min(shape[2]-1, z+side//2)

    sel_signal=dwi[x,y,z,:].copy()
    ax = global_figure.add_subplot(1,2,2)
    #draw_affine_fit(ax, sel_x, img0_name, sel_y, img1_name)
    draw_dwi_gp(ax, sel_signal, dwi_name)

    ax = global_figure.get_axes()[0]
    R = Rectangle((px-side//2,py-side//2), side, side, facecolor='none', linewidth=3, edgecolor='#DD0000')
    if len(ax.artists)>0:
        ax.artists[-1].remove()
    ax.add_artist(R)
    draw()

def run_interactive_pair(img0, img0_name, img1, img1_name, residuals,
                         slice_type=1, slice_index=None, vmin=None, vmax=None):
    """ Starts the interactive ROI analysis tool

    Requires the following global variables:
    1. global_figure : this is needed because the `draw_rect_image_pair` event only receives the click event
        as parameter, so it would otherwise be unable to know where to write the rectangle.
    2. global_map : this is needed for the same reason as `global_figure`. Actually,
    `global_figure` might be part of this map (refactor required)
    """
    global global_figure
    global global_map

    if vmin is None:
        vmin = residuals.min()
    if vmax is None:
        vmax = residuals.max()

    global_figure = figure()
    ax = global_figure.add_subplot(1,3,1)

    if slice_type==0:
        if slice_index is None:
            slice_index =  residuals.shape[0]//2
        mappable = ax.imshow(residuals[slice_index,:,:].transpose(), origin='lower',
                             vmin=vmin, vmax=vmax)
    elif slice_type==1:
        if slice_index is None:
            slice_index =  residuals.shape[1]//2
        mappable = ax.imshow(residuals[:,residuals.shape[1]//2,:].transpose(), origin='lower',
                             vmin=vmin, vmax=vmax)
    else:
        if slice_index is None:
            slice_index =  residuals.shape[2]//2
        mappable = ax.imshow(residuals[:,:,slice_index].transpose(), origin='lower',
                             vmin=vmin, vmax=vmax)

    global_figure.colorbar(mappable)
    global_map = {'img0':img0,
                  'img0_name':img0_name,
                  'img1':img1,
                  'img1_name':img1_name,
                  'residuals':residuals,
                  'slice_type':slice_type,
                  'slice_index':slice_index,
                  'vmin':vmin,
                  'vmax':vmax}
    global_figure.canvas.mpl_connect('button_press_event', draw_rect_image_pair)


def run_interactive_dwi(dwi, dwi_name, residuals,
                        slice_type=1, slice_index=None, vmin=None, vmax=None):
    """ Starts the interactive ROI analysis tool for DWI data

    Requires the following global variables:
    1. global_figure : this is needed because the `draw_rect_dwi` event only receives the click event
        as parameter, so it would otherwise be unable to know where to write the rectangle.
    2. global_map : this is needed for the same reason as `global_figure`. Actually,
    `global_figure` might be part of this map (refactor required)
    """
    global global_figure
    global global_map

    # Display residuals
    if vmin is None:
        vmin = residuals.min()
    if vmax is None:
        vmax = residuals.max()

    global_figure = figure()
    ax = global_figure.add_subplot(1,3,1)

    if slice_type==0:
        if slice_index is None:
            slice_index =  residuals.shape[0]//2
        mappable = ax.imshow(residuals[slice_index,:,:].transpose(), origin='lower',
                             vmin=vmin, vmax=vmax)
    elif slice_type==1:
        if slice_index is None:
            slice_index =  residuals.shape[1]//2
        mappable = ax.imshow(residuals[:,residuals.shape[1]//2,:].transpose(), origin='lower',
                             vmin=vmin, vmax=vmax)
    else:
        if slice_index is None:
            slice_index =  residuals.shape[2]//2
        mappable = ax.imshow(residuals[:,:,slice_index].transpose(), origin='lower',
                             vmin=vmin, vmax=vmax)
    global_figure.colorbar(mappable)

    # Set the global map's properties
    global_map = {'dwi':dwi,
                  'dwi_name':dwi_name,
                  'residuals':residuals,
                  'slice_type':slice_type,
                  'slice_index':slice_index,
                  'vmin':vmin,
                  'vmax':vmax}
    global_figure.canvas.mpl_connect('button_press_event', draw_rect_image_pair)


def demo_t1_t2():
    # Load data
    t1_name = t1_name = get_brainweb("t1", "strip")
    t1_nib = nib.load(t1_name)
    t1 = t1_nib.get_data().squeeze()
    t1_lab = t1.astype(np.int32)
    t1 = t1.astype(np.float64)
    #t1 = (t1.astype(np.float64) - t1.min())/(t1.max()-t1.min())


    t2_name = t2_name = get_brainweb("t2", "strip")
    t2_nib = nib.load(t2_name)
    t2 = t2_nib.get_data().squeeze()
    t2_lab = t2.astype(np.int32)
    t2 = t2.astype(np.float64)
    #t2 = (t2.astype(np.float64) - t2.min())/(t2.max()-t2.min())



    # Prepare interactive graphs
    global_figure = None
    global_map = None
    sel_x = None
    sel_y = None


    # Compute residuals of locally affine fit on T1 and T2
    radius = 4
    residuals_nb = ccr.compute_cc_residuals(t1, t2, radius, 1)
    residuals_nb = np.array(residuals_nb)

    # Apply the transfer
    means_t2t1, vars_t2t1 = get_mean_transfer(t2_lab, t1)
    ss_t1 = means_t2t1[t2_lab]

    means_t1t2, vars_t1t2 = get_mean_transfer(t1_lab, t2)
    ss_t2 = means_t1t2[t1_lab]

    # Compute residuals of locally affine fit on T1 and F[T2]
    radius = 4
    residuals_t1 = ccr.compute_cc_residuals(t1, ss_t1, radius, 1)
    residuals_t1 = np.array(residuals_t1)

    # Compute residuals of locally affine fit on F(T1) and T2
    radius = 4
    residuals_t2 = ccr.compute_cc_residuals(ss_t2, t2, radius, 1)
    residuals_t2 = np.array(residuals_t2)

    slice_type = 1
    slice_index = residuals_nb.shape[slice_type]//2
    if slice_type == 0:
        max_val = np.max([residuals_nb[slice_index,:,:].max(), residuals_t1[slice_index,:,:].max(), residuals_t2[slice_index,:,:].max()])
    elif slice_type == 1:
        max_val = np.max([residuals_nb[:,slice_index,:].max(), residuals_t1[:,slice_index,:].max(), residuals_t2[:,slice_index,:].max()])
    else:
        max_val = np.max([residuals_nb[:,:,slice_index].max(), residuals_t1[:,:,slice_index].max(), residuals_t2[:,:,slice_index].max()])

    run_interactive_pair(t1, "T1", t2, "T2", residuals_nb, slice_type, slice_index, vmin=0, vmax=max_val)
    run_interactive_pair(t1, "T1", ss_t1, "F[T2]", residuals_t1, slice_type, slice_index, vmin=0, vmax=max_val)
    run_interactive_pair(t2, "T2", ss_t2, "F[T1]", residuals_t2, slice_type, slice_index, vmin=0, vmax=max_val)



def demo_near_dwi(idx=-1):
    # Plot local linear reconstruction error from the pair of images whose corresponding
    # diffusion encoding vectors' dot product rank `idx` from lowest to highest, e.g.
    # demo_near_dwi(0) shows the error for the "least coherent" pair, while
    # demo_near_dwi(-1) shows the error for the "most coherent" pair
    # Load data
    dwi_fname = 'usmcuw_dipy.nii.gz'
    dwi_nib = nib.load(dwi_fname)
    dwi = dwi_nib.get_data().squeeze()
    B_name = 'B_dipy.txt'
    B = np.loadtxt(B_name)
    n = B.shape[0]

    pp = []
    for i in range(1,n):
        for j in range(i+1, n):
            p = np.abs(B[i,:3].dot(B[j,:3]))
            pp.append((p, (i,j)))
    pp.sort()
    sel = pp[idx][1]

    t1 = dwi[...,sel[0]]
    t1_lab = t1.astype(np.int32)
    t1 = t1.astype(np.float64)
    t1 = (t1.astype(np.float64) - t1.min())/(t1.max()-t1.min())


    t2 = dwi[...,sel[1]]
    t2_lab = t2.astype(np.int32)
    t2 = t2.astype(np.float64)
    t2 = (t2.astype(np.float64) - t2.min())/(t2.max()-t2.min())



    # Prepare interactive graphs
    global_figure = None
    global_map = None
    sel_x = None
    sel_y = None


    # Compute residuals of locally affine fit on T1 and T2
    radius = 4
    residuals_nb = ccr.compute_cc_residuals(t1, t2, radius, 1)
    residuals_nb = np.array(residuals_nb)

    # Apply the transfer
    means_t2t1, vars_t2t1 = get_mean_transfer(t2_lab, t1)
    ss_t1 = means_t2t1[t2_lab]

    means_t1t2, vars_t1t2 = get_mean_transfer(t1_lab, t2)
    ss_t2 = means_t1t2[t1_lab]

    # Compute residuals of locally affine fit on T1 and F[T2]
    radius = 4
    residuals_t1 = ccr.compute_cc_residuals(t1, ss_t1, radius, 1)
    residuals_t1 = np.array(residuals_t1)

    # Compute residuals of locally affine fit on F(T1) and T2
    radius = 4
    residuals_t2 = ccr.compute_cc_residuals(ss_t2, t2, radius, 1)
    residuals_t2 = np.array(residuals_t2)

    slice_type = 2
    slice_index = residuals_nb.shape[slice_type]//2
    if slice_type == 0:
        max_val = np.max([residuals_nb[slice_index,:,:].max(), residuals_t1[slice_index,:,:].max(), residuals_t2[slice_index,:,:].max()])
    elif slice_type == 1:
        max_val = np.max([residuals_nb[:,slice_index,:].max(), residuals_t1[:,slice_index,:].max(), residuals_t2[:,slice_index,:].max()])
    else:
        max_val = np.max([residuals_nb[:,:,slice_index].max(), residuals_t1[:,:,slice_index].max(), residuals_t2[:,:,slice_index].max()])

    run_interactive_pair(t1, "T1", t2, "T2", residuals_nb, slice_type, slice_index, vmin=0, vmax=max_val)
    #run_interactive_pair(t1, "T1", ss_t1, "F[T2]", residuals_t1, slice_type, slice_index, vmin=0, vmax=max_val)
    #run_interactive_pair(t2, "T2", ss_t2, "F[T1]", residuals_t2, slice_type, slice_index, vmin=0, vmax=max_val)


def register(metric_name, static, moving, static_grid2space, moving_grid2space):
    if metric_name == 'LCC':
        radius = 4
        metric = LocalCCMetric(radius)
    elif metric_name == 'MI':
        nbins = 32
        sampling_prop = None
        metric = MattesMIMetric(nbins, sampling_prop)
    else:
        raise ValueError("Unknown metric "+metric_name)

    align_centers = True
    #schedule = ['TRANSLATION', 'RIGID', 'AFFINE']
    schedule = ['TRANSLATION','RIGID']
    if True:
        level_iters = [100, 100, 100]
        sigmas = [3.0, 1.0, 0.0]
        factors = [4, 2, 1]
    else:
        level_iters = [100]
        sigmas = [0.0]
        factors = [1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    out = np.eye(4)
    if align_centers:
        print('Aligning centers of mass')
        c_static = ndimage.measurements.center_of_mass(np.array(static))
        c_static = static_grid2space.dot(c_static+(1,))
        original_static = static_grid2space.copy()
        static_grid2space = static_grid2space.copy()
        static_grid2space[:3,3] -= c_static[:3]
        out = align_centers_of_mass(static, static_grid2space, moving, moving_grid2space)

    for step in schedule:
        print('Optimizing: %s'%(step,))
        transform = regtransforms[(step, 3)]
        params0 = None
        out = affreg.optimize(static, moving, transform, params0,
                              static_grid2space, moving_grid2space,
                              starting_affine=out)
    if align_centers:
        print('Updating center-of-mass reference')
        T = np.eye(4)
        T[:3,3] = -1*c_static[:3]
        out = out.dot(T)
    return out


def create_correction_schedule(bvecs):
    # Create registration schedule
    n = bvecs.shape[0]
    A = np.abs(bvecs.dot(bvecs.T))
    in_set = np.zeros(n)
    set_sim = A[0,:].copy()
    in_set[0] = 1
    set_sim[0] = -1
    set_size = 1
    regs = []
    while(set_size < n):
        sel = np.argmax(set_sim)
        closest = -1
        for i in range(n):
            if in_set[i] == 0:
                set_sim[i] = max([set_sim[i], A[i, sel]])
            else:
                if closest == -1 or A[sel, i] > A[sel, closest]:
                    closest = i
        in_set[sel] = 1
        set_size += 1
        set_sim[sel] = -1
        regs.append((closest, sel))

    # Find the centroid
    A[...] = -1
    for i in range(n):
        A[i, i] = 0
    for reg in regs:
        A[reg[0], reg[1]] = 1
        A[reg[1], reg[0]] = 1

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if A[i, k] > 0 and A[k, j] > 0:
                    opt = A[i, k] + A[k, j]
                    if A[i, j] == -1 or opt < A[i, j]:
                        A[i, j] = opt
    steps = A.sum(1)
    centroid = np.argmin(steps)

    # Build correction schedule
    paths = {}
    paths[centroid] = [centroid]
    while(len(paths)<n):
        for reg in regs:
            if reg[0] in paths and reg[1] not in paths:
                paths[reg[1]] = paths[reg[0]] + [reg[1]]
            elif reg[1] in paths and reg[0] not in paths:
                paths[reg[0]] = paths[reg[1]] + [reg[0]]

    return regs, centroid, paths

def execute_path(path):
    n = len(path)
    affine = np.eye(4)
    for i in range(1,n):
        static = path[i-1]
        moving = path[i]
        fname = 'align_'+str(static)+'_'+str(moving)+'.npy'
        if os.path.isfile(fname):
            new = np.load(fname)
        else:
            fname = 'align_'+str(moving)+'_'+str(static)+'.npy'
            if os.path.isfile(fname):
                new = np.load(fname)
                new = np.linalg.inv(new)
            else:
                raise ValueError("Path broken")
        affine = affine.dot(new)
    return affine


def check_direct(a, b):
    static = dwi[...,a]
    moving = dwi[...,b]
    static_grid2space = dwi_nib.get_affine()
    moving_grid2space = dwi_nib.get_affine()
    rt.overlay_slices(static, moving, None, 0, "Static", "Warped LCC")
    rt.overlay_slices(static, moving, None, 1, "Static", "Warped LCC")
    rt.overlay_slices(static, moving, None, 2, "Static", "Warped LCC")
    affine_lcc = register('LCC', static, moving, static_grid2space, moving_grid2space)
    warped_lcc = transform_image(static, static_grid2space, moving, moving_grid2space, affine_lcc)
    rt.overlay_slices(static, warped_lcc, None, 0, "Static", "Warped LCC")
    rt.overlay_slices(static, warped_lcc, None, 1, "Static", "Warped LCC")
    rt.overlay_slices(static, warped_lcc, None, 2, "Static", "Warped LCC")
    return affine_lcc

w_8_32 = transform_image(dwi[...,8], dwi_nib.get_affine(), dwi[...,32], dwi_nib.get_affine(), lcc_8_32)
w_8_1 = transform_image(dwi[...,8], dwi_nib.get_affine(), dwi[...,1], dwi_nib.get_affine(), lcc_8_1)
w_1_32 = transform_image(dwi[...,1], dwi_nib.get_affine(), dwi[...,32], dwi_nib.get_affine(), lcc_1_32)
wp_8_32 = transform_image(dwi[...,8], dwi_nib.get_affine(), dwi[...,32], dwi_nib.get_affine(), lcc_8_1.dot(lcc_1_32))
wrong = transform_image(dwi[...,8], dwi_nib.get_affine(), dwi[...,32], dwi_nib.get_affine(), lcc_1_32.dot(lcc_8_1))

rt.overlay_slices(dwi[...,8], dwi[...,32], None, 2, "Static", "Moving")
rt.overlay_slices(dwi[...,8], w_8_32, None, 2, "Static", "W-direct")
rt.overlay_slices(dwi[...,8], wp_8_32, None, 2, "Static", "W-path")
rt.overlay_slices(dwi[...,8], wrong, None, 2, "Static", "Wrong")

rt.overlay_slices(wp_8_32, wrong, None, 2, "Static", "Wrong")

#dwi_fname = 'usmcuw_dipy.nii.gz'
#B_name = 'B_dipy.txt'
dwi_fname = 'Diffusion.nii.gz'
B_name = 'B.txt'
dwi_nib = nib.load(dwi_fname)
dwi = dwi_nib.get_data().squeeze()

B = np.loadtxt(B_name)
n = B.shape[0]

pp = []
for i in range(1,n):
    for j in range(i+1, n):
        p = np.abs(B[i,:3].dot(B[j,:3]))
        pp.append((p, (i,j)))
pp.sort()
far = pp[0][1]
near = pp[-1][1]
rt.overlay_slices(dwi[...,near[0]], dwi[...,near[1]], slice_type=0)
rt.overlay_slices(dwi[...,near[0]], dwi[...,near[1]], slice_type=1)
rt.overlay_slices(dwi[...,near[0]], dwi[...,near[1]], slice_type=2)


rt.overlay_slices(dwi[...,far[0]], dwi[...,far[1]], slice_type=0)
rt.overlay_slices(dwi[...,far[0]], dwi[...,far[1]], slice_type=1)
rt.overlay_slices(dwi[...,far[0]], dwi[...,far[1]], slice_type=2)


static = dwi[...,far[0]]
moving = dwi[...,far[1]]
static_grid2space = dwi_nib.get_affine()
moving_grid2space = dwi_nib.get_affine()

affine_lcc = register('LCC', static, moving, static_grid2space, moving_grid2space)
warped_lcc = transform_image(static, static_grid2space, moving, moving_grid2space, affine_lcc)
rt.overlay_slices(static, warped_lcc, None, 0, "Static", "Warped LCC")
rt.overlay_slices(static, warped_lcc, None, 1, "Static", "Warped LCC")
rt.overlay_slices(static, warped_lcc, None, 2, "Static", "Warped LCC")

affine_mi = register('MI', static, moving, static_grid2space, moving_grid2space)
warped_mi = transform_image(static, static_grid2space, moving, moving_grid2space, affine_mi)
rt.overlay_slices(static, warped_mi, None, 0, "Static", "Warped MI")
rt.overlay_slices(static, warped_mi, None, 1, "Static", "Warped MI")
rt.overlay_slices(static, warped_mi, None, 2, "Static", "Warped MI")

rt.overlay_slices(warped_mi, warped_lcc, None, 0, "Warped MI", "Warped LCC")
rt.overlay_slices(warped_mi, warped_lcc, None, 1, "Warped MI", "Warped LCC")
rt.overlay_slices(warped_mi, warped_lcc, None, 2, "Warped MI", "Warped LCC")

# Create schedule and execute
regs, centroid, paths = create_correction_schedule(B[:,:3])
for a, b in regs:
    fname = 'align_'+str(a)+'_'+str(b)+'.npy'
    print('Computing:',fname)
    affine = register('MI', dwi[...,a], dwi[...,b], dwi_nib.get_affine(), dwi_nib.get_affine())
    np.save(fname, affine)


demo_near_dwi(4)


A_path = execute_path(paths[13])
w_8_13_path = transform_image(dwi[...,8], dwi_nib.get_affine(), dwi[...,13], dwi_nib.get_affine(), A_path)
rt.overlay_slices(dwi[...,8], dwi[...,13], None, 2, "dwi[8]", "dwi[13]")
rt.overlay_slices(dwi[...,8], w_8_13_path, None, 0, "dwi[8]", "dwi[13]-->...-->dwi[8]")
rt.overlay_slices(dwi[...,8], w_8_13_path, None, 1, "dwi[8]", "dwi[13]-->...-->dwi[8]")
rt.overlay_slices(dwi[...,8], w_8_13_path, None, 2, "dwi[8]", "dwi[13]-->...-->dwi[8]")

A_direct = register('LCC', dwi[...,8], dwi[...,13], dwi_nib.get_affine(), dwi_nib.get_affine())
w_8_13_direct = transform_image(dwi[...,8], dwi_nib.get_affine(), dwi[...,13], dwi_nib.get_affine(), A_direct)
rt.overlay_slices(dwi[...,8], w_8_13_direct, None, 0, "dwi[8]", "dwi[13]-->dwi[8]")
rt.overlay_slices(dwi[...,8], w_8_13_direct, None, 1, "dwi[8]", "dwi[13]-->dwi[8]")
rt.overlay_slices(dwi[...,8], w_8_13_direct, None, 2, "dwi[8]", "dwi[13]-->dwi[8]")


#w_path = np.zeros(dwi.shape, dtype=np.float32)
w_path = np.empty_like(dwi)
w_path[...,0] = 0
for i in range(1, 33):
    A = execute_path(paths[i])
    w = transform_image(dwi[...,8], dwi_nib.get_affine(), dwi[...,i], dwi_nib.get_affine(), A, True)
    w_path[...,i] = w[...]
    w_path[...,0] += w
w_path /= 32
w_path_nib = nib.Nifti1Image(w_path, dwi_nib.get_affine())
w_path_nib.to_filename('w_path.nii.gz')


P = np.random.permutation(range(1,33))
for i in range(32):
    s = w_path.shape[2]//2
    figure()
    subplot(1,2,1)
    imshow(w_path[:,:,s,P[i]].T)
    subplot(1,2,2)
    imshow(dwi[:,:,s,P[i]].T)


for i in range(2, 4):
    rt.plot_slices(dwi[...,i])




def optimal_vs_average_experiment():
    # Compute transfer for optimal local-linearity
    radius = 5
    init_from_mean = True
    nlabels_t1 = 1 + np.max(t1_lab)
    nlabels_t2 = 1 + np.max(t2_lab)

    means_t1t2, vars_t1t2 = get_mean_transfer(t1_lab, t2)
    fmean_t1lab = np.array(means_t1t2)

    means_t2t1, vars_t2t1 = get_mean_transfer(t2_lab, t1)
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
    if init_from_mean:
        f0 = fmean_t1lab
    else:
        f0 = np.random.uniform(0,1,len(fmean_t1lab))

    ofname = 'fopt_t1lab_'+str(radius)+'.npy'
    fopt_t1lab = None
    if os.path.isfile(ofname):
        fopt_t1lab = np.load(ofname)
    else:
        opt_t1lab = Optimizer(value_and_gradient_t1lab, f0, "BFGS", jac = True, options=options)
        fopt_t1lab = opt_t1lab.xopt
        np.save(ofname, fopt_t1lab)

    options = {'maxiter': 20}
    if init_from_mean:
        f0 = fmean_t2lab
    else:
        f0 = np.random.uniform(0,1,len(fmean_t2lab))

    ofname = 'fopt_t2lab_'+str(radius)+'.npy'
    fopt_t2lab = None
    if os.path.isfile(ofname):
        fopt_t2lab = np.load(ofname)
    else:
        opt_t2lab = Optimizer(value_and_gradient_t2lab, f0, "BFGS", jac = True, options=options)
        fopt_t2lab = opt_t2lab.xopt
        np.save(ofname, fopt_t2lab)


    # Compare mean vs. optimal transfers : t2lab
    markers = ['o','D','s','^']
    linestyles = ['-', '--', '--', '--']
    def compare_transfers(ax, fmean, fopt, fmean_legend, fopt_legend, legend_location, xlabel='', ylabel=''):
        line, = ax.plot(range(0, len(fmean)), fmean, linestyle=linestyles[1])
        line.set_label(fmean_legend)
        line, = ax.plot(range(0, len(fmean)), fopt, linestyle=linestyles[0])
        line.set_label(fopt_legend)
        ax.legend(loc=legend_location, fontsize=16)
        plt.grid()
        plt.xlim(0,len(fmean) - 1)
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)

    def normalize(fopt, fmean=None, flip=False):
        if flip:
            fopt_norm = fopt * -1
        else:
            fopt_norm = fopt.copy()
        min_opt = fopt_norm.min()
        delta_opt = fopt_norm.max() - min_opt
        fopt_norm = (fopt_norm - min_opt)/delta_opt
        fopt_nz = fopt_norm.copy()
        if fmean is not None:
            fopt_nz[fmean == 0] = 0
        return fopt_nz

    min_mean = fmean_t1lab.min()
    delta_mean = fmean_t1lab.max() - min_mean
    fmean_t1lab_norm = (fmean_t1lab - min_mean)/(fmean_t1lab.max() - min_mean)



    # Normalize fopt_t2lab and fmean_t2lab
    fopt_t2lab_norm = fopt_t2lab * -1 # flip opt
    min_opt = fopt_t2lab_norm.min()
    delta_opt = fopt_t2lab_norm.max() - min_opt
    fopt_t2lab_norm = (fopt_t2lab_norm - min_opt)/(fopt_t2lab_norm.max()-min_opt)
    fopt_t2lab_norm[fmean_t2lab == 0] = 0# Set the value of empty iso-sets to zero
    fopt_t2lab_nz = fopt_t2lab
    fopt_t2lab_nz[fmean_t2lab == 0] = 0
    min_mean = fmean_t2lab.min()
    delta_mean = fmean_t2lab.max() - min_mean
    fmean_t2lab_norm = (fmean_t2lab - min_mean)/(fmean_t2lab.max() - min_mean)

    t2lab_fig = plt.figure()
    ax = t2lab_fig.add_subplot(1,2,1)
    compare_transfers(ax, fmean_t2lab, fopt_t2lab_nz, "Iso-set mean", "Optimized", 2, 'T2', 'F[T2]')
    ax = t2lab_fig.add_subplot(1,2,2)
    compare_transfers(ax, fmean_t2lab_norm, fopt_t2lab_norm, "Iso-set mean (normalized to [0,1])", "Optimized (flipped & normalized to [0,1])", 1, 'T2', 'F[T2]')

    # Normalize fopt_t1lab and fmean_t1lab
    fopt_t1lab_norm = fopt_t1lab * -1 # flip opt
    min_opt = fopt_t1lab_norm.min()
    delta_opt = fopt_t1lab_norm.max() - min_opt
    fopt_t1lab_norm = (fopt_t1lab_norm - min_opt)/(fopt_t1lab_norm.max()-min_opt)
    fopt_t1lab_norm[fmean_t1lab == 0] = 0# Set the value of empty iso-sets to zero
    fopt_t1lab_nz = fopt_t1lab
    fopt_t1lab_nz[fmean_t1lab == 0] = 0
    min_mean = fmean_t1lab.min()
    delta_mean = fmean_t1lab.max() - min_mean
    fmean_t1lab_norm = (fmean_t1lab - min_mean)/(fmean_t1lab.max() - min_mean)

    t1lab_fig = plt.figure()
    ax = t1lab_fig.add_subplot(1,2,1)
    compare_transfers(ax, fmean_t1lab, fopt_t1lab_nz, "Iso-set mean", "Optimized", 2, 'T1', 'F[T1]')
    ax = t1lab_fig.add_subplot(1,2,2)
    compare_transfers(ax, fmean_t1lab_norm, fopt_t1lab_norm, "Iso-set mean (normalized to [0,1])", "Optimized (flipped & normalized to [0,1])", 1, 'T1', 'F[T1]')





def optimal_transfer_different_window_sizes():
    # Check transfer functions with different window sizes
    opt_list_t1lab = {}
    diff_t1lab = {}
    opt_list_t2lab = {}
    diff_t2lab = {}
    max_size = 9
    for s in range(2,max_size+1):
        #t1lab
        fname = 'fopt_t1lab_'+str(s)+'.npy'
        fopt = np.load(fname)
        opt_list_t1lab[s] = fopt
        diff_t1lab[s] = np.sqrt((np.abs(fmean_t1lab - fopt)**2).sum())

        fname = 'fopt_t2lab_'+str(s)+'.npy'
        fopt = np.load(fname)
        opt_list_t2lab[s] = fopt
        diff_t2lab[s] = np.sqrt((np.abs(fmean_t2lab - fopt)**2).sum())

    # Plot RMSE graphs
    rmse_t1lab = [diff_t1lab[i] for i in range(2, max_size+1)]
    rmse_t2lab = [diff_t2lab[i] for i in range(2, max_size+1)]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xticks(range(2,max_size+1))
    ax.set_xlabel('Window size', fontsize=24)
    ax.set_ylabel('RMSE', fontsize=24)
    line, = plot(range(2, max_size+1), rmse_t1lab)
    line.set_label('T2 as a function of T1')
    line, = plot(range(2, max_size+1), rmse_t2lab)
    line.set_label('T1 as a function of T2')
    ax.legend(loc=1, fontsize=24)
    plt.grid()


def plot_average_and_optimized_transfers():
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    compare_transfers(ax, fmean_t1lab, opt_list_t1lab[4], "Iso-set mean", "Optimized", 1, 'T1', 'F[T1]')
    ax = fig.add_subplot(1,2,2)
    compare_transfers(ax, fmean_t2lab, opt_list_t2lab[4], "Iso-set mean", "Optimized", 1, 'T2', 'F[T2]')


    # Apply the transfer
    t2_lab = t2.astype(np.int32)
    means_t2t1, vars_t2t1 = get_mean_transfer(t2_lab, t1)
    ss_t1 = means_t2t1[t2_lab]

    t1_lab = t1.astype(np.int32)
    means_t1t2, vars_t1t2 = get_mean_transfer(t1_lab, t2)
    ss_t2 = means_t1t2[t1_lab]



    # Compute residuals of locally affine fit on T1 and T2
    radius = 4
    residuals_nb = ccr.compute_cc_residuals(t1, t2, radius, 1)
    residuals_nb = np.array(residuals_nb)

    # Compute residuals of locally affine fit on F(T1) and T2
    radius = 4
    residuals_t2 = ccr.compute_cc_residuals(ss_t2, t2, radius, 1)
    residuals_t2 = np.array(residuals_t2)

    # Compute residuals of locally affine fit on T1 and F[T2]
    radius = 4
    residuals_t1 = ccr.compute_cc_residuals(t1, ss_t1, radius, 1)
    residuals_t1 = np.array(residuals_t1)
