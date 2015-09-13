import os
import pickle
from dipy.align import VerbosityLevels
from dipy.align.metrics import SSDMetric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from inverse.common import invert_vector_field_fixed_point_3d
from dipy.align.vector_fields import compose_vector_fields_3d
from experiments.registration.images2gif import writeGif
from inverse.dfinverse_3d import warp_points_3d, revolution_solid
from dipy.data import get_data
import time
def get_deformed_grid(field, zlist=None, npoints=None, x0=None, x1=None, y0=None, y1=None):
    if x0 is None:
        x0 = 0
    if y0 is None:
        y0 = 0
    if x1 is None:
        x1 = field.shape[0]-1
    if y1 is None:
        y1 = field.shape[1]-1
    if npoints is None:
        npoints = 1+np.max([x1, y1])
    if zlist is None:
        zlist = [field.shape[1]//2]
    
    x = np.linspace(x0, x1, npoints)
    y = np.linspace(y0, y1, npoints)

    for z0 in zlist:
        hlines = []
        for i in range(npoints):
            hline = [(x[i], y[j], z0) for j in range(npoints)]
            hlines.append(np.array(hline))
        vlines = []
        for i in range(npoints):
            vline = [(x[j], y[i], z0) for j in range(npoints)]
            vlines.append(np.array(vline))
        whlines = []
        for line  in hlines:
            warped = np.array(warp_points_3d(line, field))
            whlines.append(warped)
        wvlines = []
        for line  in vlines:
            warped = np.array(warp_points_3d(line, field))
            wvlines.append(warped)
    return whlines, wvlines


cup_fname = 'cup_256.npy'
sphere_fname = 'sphere_256.npy'
fname_circle = get_data('reg_o')
fname_c = get_data('reg_c')
if os.path.isfile(cup_fname) and os.path.isfile(sphere_fname):
    sphere = np.load(sphere_fname)
    cup = np.load(cup_fname)
else:
    circle = np.load(fname_circle)
    sphere = np.array(revolution_solid(circle.astype(np.float64)))
    c = np.load(fname_c)
    cup = np.array(revolution_solid(c.astype(np.float64)))

# Read options
config_fname = 'config.txt'
if os.path.isfile(config_fname):
    with open(config_fname,'r') as f:
        options = [tuple(line.strip().split()) for line in f.readlines()]
    options = {opt[0]:opt[1] for opt in options}
else:
    options = {}

step_length = float(options.get('step_length', '0.25'))
inv_tol = float(options.get('inv_tol', '1e-3'))
inv_iter = int(options.get('inv_iter', '20'))
experiment_name = options.get('experiment_name')

#level_iters = [200, 100, 50, 25]
level_iters = [1, 1, 0, 0]
dim = 3
metric = SSDMetric(dim, smooth=3)
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter=inv_iter, inv_tol=inv_tol, opt_tol=1e-6, step_length=step_length, ss_sigma_factor=0.2)
sdr.verbosity = VerbosityLevels.DIAGNOSE

start = time.time()
mapping = sdr.optimize(cup, sphere)
end = time.time()
elapsed = end - start
print('Elapsed: %e'%(elapsed,))

fwd = np.array(mapping.forward)
bwd = np.array(mapping.backward)

# Get deformed mid slices
z0 = v.shape[2]//2
whlines, wvlines = get_deformed_grid(fwd, [z0])
fwd_fname = 'fwd_lines_%s.npy'%(experiment_name,)
pickle.dump((whlines, wvlines), open(fwd_fname,'wb'))


whlines, wvlines = get_deformed_grid(bwd, [z0])
bwd_fname = 'bwd_lines_%s.npy'%(experiment_name,)
pickle.dump((whlines, wvlines), open(bwd_fname,'wb'))
