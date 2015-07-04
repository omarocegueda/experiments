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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import experiments.registration.gproc as gp

from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
from dipy.align.transforms import regtransforms

def draw_dwi_gp(ax, points, dwi_name):
    font = {'family' : 'serif',
            'color'  : 'black',
            'weight' : 'normal',
            'size'   : 16,}

    main_title = dwi_name
    ax.cla()
    # Plot required info
    x_name = ''
    y_name = ''

    ax.scatter(points[:,0].copy(), points[:,1].copy(), points[:,2].copy(), c='r')
    ax.scatter(points[:,0].copy(), points[:,1].copy(), points[:,2].copy(), c='r')
    #--
    ax.grid(True)
    ax.set_title(main_title, fontdict=font)
    ax.set_xlabel(x_name, fontdict=font)
    ax.set_ylabel(y_name, fontdict=font)

def draw_rect_dwi(event):
    global global_figure
    global sel_signal
    if event.inaxes != global_figure.axes[0]:
        return
    global global_map
    side = 2
    px, py = int(event.xdata), int(event.ydata)

    dwi = global_map['dwi']
    bvecs = global_map['bvecs']
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
    plt.clf()
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
    print(sel_signal)
    # Duplicate the points in the opposite direction
    points = np.empty((bvecs.shape[0] * 2, 3), dtype=np.float64)
    points[:bvecs.shape[0],:] = diag(sel_signal).dot(bvecs)
    points[bvecs.shape[0]:,:] = diag(sel_signal).dot(bvecs)*-1

    # Fit Gaussian process
    mean_signal = sel_signal.mean()
    f_in = sel_signal - mean_signal
    x_in = bvecs.copy()
    sigmasq_signal = f_in.var()
    sigmasq_noise = 100.0 # We need to estimate this from the data
    # Create sampling points
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    x_out = np.array([x.reshape(-1),y.reshape(-1),z.reshape(-1)]).T
    # Get the conditional mean and covariance matrix
    mean_out, S_out = gp.spherical_poly_conditional(x_in, f_in, x_out, sigmasq_signal, sigmasq_noise)
    mean_out = np.array(mean_out)
    # Get the predicted signal at the new sampling points
    predicted = mean_out.reshape(x.shape)
    predicted += mean_signal
    x *= predicted
    y *= predicted
    z *= predicted
    ax = global_figure.add_subplot(122, projection='3d')
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2, shade=True)
    points = diag(sel_signal).dot(x_in)
    ax.scatter(points[:,0].copy(), points[:,1].copy(), points[:,2].copy(), c='r', s=40)
    ax.scatter(-1*points[:,0].copy(), -1*points[:,1].copy(), -1*points[:,2].copy(), c='r', s=40)

    ax = global_figure.get_axes()[0]
    R = Rectangle((px-side//2,py-side//2), side, side, facecolor='none', linewidth=3, edgecolor='#DD0000')
    if len(ax.artists)>0:
        ax.artists[-1].remove()
    ax.add_artist(R)
    draw()


def run_interactive_dwi(dwi, bvecs, dwi_name, residuals,
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
    ax = global_figure.add_subplot(1,2,1)

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
                  'bvecs':bvecs,
                  'dwi_name':dwi_name,
                  'residuals':residuals,
                  'slice_type':slice_type,
                  'slice_index':slice_index,
                  'vmin':vmin,
                  'vmax':vmax}
    global_figure.canvas.mpl_connect('button_press_event', draw_rect_dwi)


def demo_dwi_roi(idx=-1):
    dwi_fname = 'Diffusion.nii.gz'
    #dwi_fname = 'NoArtifacts.nii.gz'
    B_name = 'B.txt'
    dataset_name = 'Challenge 2015'
    #dataset_name = 'Ramon DWI'
    #dwi_fname = 'Ramon_dwi.nii.gz'
    #B_name = 'Ramon_dwi.bvecs'

    dwi_nib = nib.load(dwi_fname)
    dwi = dwi_nib.get_data().squeeze()
    B = np.loadtxt(B_name)

    remove_first = True
    if remove_first:
        B = B[1:,...]
        dwi = dwi[...,1:]
    bvecs = B[...,:3]
    n = B.shape[0]

    residuals = dwi[...,0].copy()
    run_interactive_dwi(dwi, bvecs, dataset_name, residuals,
                        slice_type=1, slice_index=None, vmin=None, vmax=None)


