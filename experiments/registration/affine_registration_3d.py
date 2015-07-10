"""
==========================================
Affine Registration in 3D
==========================================
This example explains how to compute an affine transformation to register two 3D
volumes by maximization of their Mutual Information [Mattes03]_. The optimization
strategy is similar to that implemented in ANTS [Avants11]_.
"""

import numpy as np
import nibabel as nib
import os.path
import scipy
import scipy.ndimage as ndimage
from dipy.viz import regtools as rt
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.segment.mask import median_otsu
from dipy.align.transforms import regtransforms
from dipy.align.imaffine import (align_centers_of_mass,
                                 MattesMIMetric,
                                 AffineRegistration)



jit_nib = nib.load('jittered.nii.gz')
jit = jit_nib.get_data().squeeze()
static = jit[...,0]
moving = jit[...,2]
aff = jit_nib.get_affine()
rt.overlay_slices(static, moving, slice_type=2)
# Bring the center of mass to the origin
c_static = ndimage.measurements.center_of_mass(np.array(static))
c_static = aff.dot(c_static+(1,))
correction = np.eye(4, dtype=np.float64)
correction[:3,3] = -1 * c_static[:3]
new_aff = correction.dot(aff)

com = align_centers_of_mass(static, new_aff, moving, new_aff)
warped = com.transform(moving)
rt.overlay_slices(static, warped, slice_type=2)

# Create the metric
nbins = 32
sampling_prop = None
metric = MattesMIMetric(nbins, sampling_prop)

# Create the optimizer
level_iters = [10000, 1000, 100]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]
affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)

# Translation
transform = regtransforms[('TRANSLATION', 3)]
params0 = None
starting_affine = com.affine
trans = affreg.optimize(static, moving, transform, params0,
                        new_aff, new_aff,
                        starting_affine=starting_affine)
warped = trans.transform(moving)
rt.overlay_slices(static, warped, None, 0, "Static", "Warped", "warped_trans_0.png")
rt.overlay_slices(static, warped, None, 1, "Static", "Warped", "warped_trans_1.png")
rt.overlay_slices(static, warped, None, 2, "Static", "Warped", "warped_trans_2.png")

# Rigid
transform = regtransforms[('RIGID', 3)]
params0 = None
starting_affine = trans.affine
rigid = affreg.optimize(static, moving, transform, params0,
                        new_aff, new_aff,
                        starting_affine=starting_affine)
# fix solution
backup = rigid.affine
fixed = rigid.affine.dot(correction)
rigid.set_affine(fixed)
rigid.domain_grid2world = aff

warped = rigid.transform(moving)
rt.overlay_slices(static, warped, None, 0, "Static", "Warped", "warped_trans_0.png")
rt.overlay_slices(static, warped, None, 1, "Static", "Warped", "warped_trans_1.png")
rt.overlay_slices(static, warped, None, 2, "Static", "Warped", "warped_trans_2.png")

"""
Let's fetch two b0 volumes, the static image will be the b0 from the Stanford
HARDI dataset
"""

fetch_stanford_hardi()
nib_stanford, gtab_stanford = read_stanford_hardi()
static = np.squeeze(nib_stanford.get_data())[..., 0]
static_grid2space = nib_stanford.get_affine()

correction = None
c_static_space = np.zeros(4)
if correction == 'mass':
    import scipy.ndimage as ndimage
    c_static = ndimage.measurements.center_of_mass(np.array(static)) + (1,)
    c_static_space = static_grid2space.dot(c_static)
elif correction == 'imcenter':
    c_static = np.array(static.shape + (2.0,)) * 0.5
    c_static_space = static_grid2space.dot(c_static)
elif correction == 'corner':
    c_static = np.array(static.shape + (0.5,)) * 2.0
    c_static_space = static_grid2space.dot(c_static)

original_grid2space = static_grid2space.copy()
static_grid2space[:3,3] -= c_static_space[:3]


"""
Now the moving image
"""

fetch_syn_data()
nib_syn_t1, nib_syn_b0 = read_syn_data()
moving = np.array(nib_syn_b0.get_data())
moving_grid2space = nib_syn_b0.get_affine()

"""
We can obtain a very rough (and fast) registration by just aligning the centers of mass
of the two images
"""

c_of_mass = align_centers_of_mass(static, static_grid2space, moving, moving_grid2space)

"""
We can now warp the moving image and draw it on top of the static image, registration
is not likely to be good, but at least they will occupy roughly the same space
"""

warped = transform_image(static, static_grid2space, moving, moving_grid2space, c_of_mass)
regtools.overlay_slices(static, warped, None, 0, "Static", "Warped", "warped_com_0.png")
regtools.overlay_slices(static, warped, None, 1, "Static", "Warped", "warped_com_1.png")
regtools.overlay_slices(static, warped, None, 2, "Static", "Warped", "warped_com_2.png")

"""
.. figure:: warped_com_0.png
   :align: center
.. figure:: warped_com_1.png
   :align: center
.. figure:: warped_com_2.png
   :align: center

   **Registration result by simply aligning the centers of mass of the images**.
"""

"""
This was just a translation of the moving image towards the static image, now we will
refine it by looking for an affine transform. We first create the similarity metric
(Mutual Information) to be used. We need to specify the number of bins to be used to
discretize the joint and marginal probability distribution functions (PDF), a typical
value is 32. We also need to specify the percentage (an integer in (0, 100])of voxels
to be used for computing the PDFs, the most accurate registration will be obtained by
using all voxels, but it is also the most time-consuming choice. We specify full
sampling by passing None instead of an integer
"""

nbins = 32
sampling_prop = None
#metric = MattesMIMetric(nbins, sampling_prop)
metric = LocalCCMetric(4)
#Optimal transform found with MI
#c_of_mass = np.array([[1.02783543e+00, -4.83019053e-02, -6.07735639e-02, -2.57654118e+00],
#                      [4.34051706e-03, 9.41918267e-01, -2.66525861e-01, 3.23579799e+01],
#                      [5.34288908e-02, 2.90262026e-01, 9.80820307e-01, -1.46216651e+01],
#                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#Optimal translation found with MI
#c_of_mass = array([[  1.        ,   0.        ,   0.        ,  -2.55666434],
#                   [  0.        ,   1.        ,   0.        ,  32.76111558],
#                   [  0.        ,   0.        ,   1.        , -20.32769812],
#                   [  0.        ,   0.        ,   0.        ,   1.        ]])

"""
To avoid getting stuck at local optima, and to accelerate convergence, we use a
multi-resolution strategy (similar to ANTS [Avants11]_) by building a Gaussian Pyramid.
To have as much flexibility as possible, the user can specify how this Gaussian Pyramid
is built. First of all, we need to specify how many resolutions we want to use. This is
indirectly specified by just providing a list of the number of iterations we want to
perform at each resolution. Here we will just specify 3 resolutions and a large number
of iterations, 10000 at the coarsest resolution, 1000 at the medium resolution and 100
at the finest. These are the default settings
"""

level_iters = [10000, 1000, 100]

"""
To compute the Gaussian pyramid, the original image is first smoothed at each level
of the pyramid using a Gaussian kernel with the requested sigma. A good initial choice
is [3.0, 1.0, 0.0], this is the default
"""

sigmas = [3.0, 1.0, 0.0]

"""
Now we specify the sub-sampling factors. A good configuration is [4, 2, 1], which means
that, if the original image shape was (nx, ny, nz) voxels, then the shape of the coarsest
image will be about (nx//4, ny//4, nz//4), the shape in the middle resolution will be
about (nx//2, ny//2, nz//2) and the image at the finest scale has the same size as the
original image. This set of factors is the default
"""

factors = [4, 2, 1]

"""
Now we go ahead and instantiate the registration class with the configuration we just
prepared
"""

affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)

"""
Using AffineRegistration we can register our images in as many stages as we want, providing
previous results as initialization for the next (the same logic as in ANTS). The reason why
it is useful is that registration is a non-convex optimization problem (it may have more
than one local optima), which means that it is very important to initialize as close to the
solution as possible. For example, lets start with our (previously computed) rough
transformation aligning the centers of mass of our images, and then refine it in three
stages. First look for an optimal translation. The dictionary regtransforms contains all
available transforms, we obtain one of them by providing its name and the dimension
(either 2 or 3) of the image we are working with (since we are aligning volumes, the
dimension is 3)
"""

transform = regtransforms[('TRANSLATION', 3)]
params0 = None
starting_affine = c_of_mass
trans = affreg.optimize(static, moving, transform, params0,
                        static_grid2space, moving_grid2space,
                        starting_affine=starting_affine)

"""
If we look at the result, we can see that this translation is much better than simply
aligning the centers of mass
"""

warped = transform_image(static, static_grid2space, moving, moving_grid2space, trans)
regtools.overlay_slices(static, warped, None, 0, "Static", "Warped", "warped_trans_0.png")
regtools.overlay_slices(static, warped, None, 1, "Static", "Warped", "warped_trans_1.png")
regtools.overlay_slices(static, warped, None, 2, "Static", "Warped", "warped_trans_2.png")

"""
.. figure:: warped_trans_0.png
   :align: center
.. figure:: warped_trans_1.png
   :align: center
.. figure:: warped_trans_2.png
   :align: center

   **Registration result by just translating the moving image, using Mutual Information**.
"""

"""
Now lets refine with a rigid transform (this may even modify our previously found
optimal translation)
"""

transform = regtransforms[('RIGID', 3)]
params0 = None
starting_affine = trans
rigid = affreg.optimize(static, moving, transform, params0,
                        static_grid2space, moving_grid2space,
                        starting_affine=starting_affine)

"""
This produces a slight rotation, and the images are now better aligned
"""

warped = transform_image(static, static_grid2space, moving, moving_grid2space, rigid)
regtools.overlay_slices(static, warped, None, 0, "Static", "Warped", "warped_rigid_0.png")
regtools.overlay_slices(static, warped, None, 1, "Static", "Warped", "warped_rigid_1.png")
regtools.overlay_slices(static, warped, None, 2, "Static", "Warped", "warped_rigid_2.png")

"""
.. figure:: warped_rigid_0.png
   :align: center
.. figure:: warped_rigid_1.png
   :align: center
.. figure:: warped_rigid_2.png
   :align: center

   **Registration result with a rigid transform, using Mutual Information**.
"""

"""
Finally, lets refine with a full affine transform (translation, rotation, scale and
shear), it is safer to fit more degrees of freedom now, since we must be very close
to the optimal transform
"""

transform = regtransforms[('AFFINE', 3)]
params0 = None
starting_affine = rigid
affine = affreg.optimize(static, moving, transform, params0,
                         static_grid2space, moving_grid2space,
                         starting_affine=starting_affine)

"""
This results in a slight shear and scale
"""
static_grid2space = original_grid2space.copy()  # restore affine
# Correct transform
P = np.eye(4)
P[:3,3] = -1 * c_static_space[:3]
affine = affine.dot(P)
warped = transform_image(static, static_grid2space, moving, moving_grid2space, affine)
regtools.overlay_slices(static, warped, None, 0, "Static", "Warped", "warped_affine_0.png")
regtools.overlay_slices(static, warped, None, 1, "Static", "Warped", "warped_affine_1.png")
regtools.overlay_slices(static, warped, None, 2, "Static", "Warped", "warped_affine_2.png")

#metric = MattesBase(nbins, sampling_prop)
#metric.setup(static, moving)
metric.update_pdfs_dense(static.astype(double), warped.astype(double))
metric.update_mi_metric()
metric.metric_val
# 1.2794398720727664 # as-is
# 1.2794398720700166 # center of mass
# 1.2793007928708782 # imcenter
# 1.279439872073092  # corner (0,0,0)
# 1.2793695855254934 # corner shape
# 1.2794398720699167

# direct image gradient
pre_align_cc = np.array([[  1.02797013e+00,  -4.46004768e-02,  -4.45651097e-02,  -2.34941661e+00],
                         [ -2.84847443e-03,   9.30403951e-01,  -2.58580228e-01,   3.15228099e+01],
                         [  3.30472637e-02,   2.85205574e-01,   9.63671270e-01,  -1.29812789e+01],
                         [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

# gradient of warped image
#1.2793362473847294
pre_align_cc = np.array([[  1.02308414e+00,  -2.80493738e-02,  -3.84216071e-02,  -3.29484759e+00],
                         [ -3.30231901e-02,   9.33749783e-01,  -2.47734290e-01,   3.20531970e+01],
                         [  4.51376961e-02,   2.69667242e-01,   9.58674621e-01,  -1.27004183e+01],
                         [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])


warped_cc = transform_image(static, static_grid2space, moving, moving_grid2space, pre_align_cc)
regtools.overlay_slices(static, warped_cc, None, 0, "Static", "Warped", "warped_affine_0.png")
regtools.overlay_slices(static, warped_cc, None, 1, "Static", "Warped", "warped_affine_1.png")
regtools.overlay_slices(static, warped_cc, None, 2, "Static", "Warped", "warped_affine_2.png")
metric.update_pdfs_dense(static.astype(double), warped_cc.astype(double))
metric.update_mi_metric()
metric.metric_val
# 1.2792485621027512




pre_align = np.array([[1.02783543e+00, -4.83019053e-02, -6.07735639e-02, -2.57654118e+00],
                      [4.34051706e-03, 9.41918267e-01, -2.66525861e-01, 3.23579799e+01],
                      [5.34288908e-02, 2.90262026e-01, 9.80820307e-01, -1.46216651e+01],
                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
warped_ants = transform_image(static, static_grid2space, moving, moving_grid2space, pre_align)
regtools.overlay_slices(static, warped_ants, None, 0, "Static", "Warped", "warped_affine_0.png")
regtools.overlay_slices(static, warped_ants, None, 1, "Static", "Warped", "warped_affine_1.png")
regtools.overlay_slices(static, warped_ants, None, 2, "Static", "Warped", "warped_affine_2.png")
metric.update_pdfs_dense(static.astype(double), warped_ants.astype(double))
metric.update_mi_metric()
metric.metric_val
# 1.2793543179896139  # ANTS


regtools.overlay_slices(warped, warped_ants, None, 0, "Static", "Warped", "warped_affine_0.png")
regtools.overlay_slices(warped, warped_ants, None, 1, "Static", "Warped", "warped_affine_1.png")
regtools.overlay_slices(warped, warped_ants, None, 2, "Static", "Warped", "warped_affine_2.png")
"""
.. figure:: warped_affine_0.png
   :align: center
.. figure:: warped_affine_1.png
   :align: center
.. figure:: warped_affine_2.png
   :align: center

   **Registration result with an affine transform, using Mutual Information**.

.. [Mattes03] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank, W. (2003). PET-CT image registration in the chest using free-form deformations. IEEE Transactions on Medical Imaging, 22(1), 120-8.
.. [Avants11] Avants, B. B., Tustison, N., & Song, G. (2011). Advanced Normalization Tools ( ANTS ), 1-35.

.. include:: ../links_names.inc

"""
