#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

cimport numpy as cnp
import numpy as np

def compute_jaccard(int[:,:,:] A, int[:,:,:] B):
    r""" Computes the Jaccard index of each region of A and B

    parameters
    ----------
    A: array, shape (S, R, C)
        annotated volume. Each integer value is regarded as a region label
    B: array, shape (S, R, C) (same dimensions as A)
        annotated volume. Each integer value is regarded as a region label

    returns
    -------
    jaccard: array, shape (nlabels,)
        the jaccard index of each region of A and B. Integer values in A and B
        are regarded as annotated regions. The number of labels is 1 + M where
        M is the maximum value in A and B.
    """
    cdef:
        cnp.npy_intp nslices = A.shape[0]
        cnp.npy_intp nrows = A.shape[1]
        cnp.npy_intp ncols = A.shape[2]
        cnp.npy_intp i, j, k
        int a, b
        int nlabels = 1 + np.max([np.max(A), np.max(B)])
        int[:] union = np.zeros(shape=(nlabels,), dtype=np.int32)
        int[:] intersection = np.zeros(shape=(nlabels,), dtype=np.int32)
        double[:] jaccard = np.zeros(shape=(nlabels,), dtype=np.float64)

    with nogil:
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if A[k, i, j]<B[k, i, j]:
                        a = A[k, i, j]
                        b = B[k, i, j]
                    else:
                        a = B[k, i, j]
                        b = A[k, i, j]
                    union[a] += 1
                    if a == b:
                        intersection[a] += 1
                    else:
                        union[b] += 1

        for i in range(nlabels):
            if union[i] > 0:
                jaccard[i]=<double>intersection[i]/<double>union[i]

    return jaccard


def compute_target_overlap(int[:,:,:] T, int[:,:,:] S):
    r""" Computes the target overlap index of each region of T and S

    The target overlap between two sets T, S, where T is not empty, is:

    |Intersection(T, S)|/|T|

    parameters
    ----------
    T: array, shape (K, I, J)
        annotated volume. Each integer value is regarded as a region label
    S: array, shape (K, I, J) (same dimensions as T)
        annotated volume. Each integer value is regarded as a region label

    returns
    -------
    scores: array, shape (nlabels,)
        the target overlap of each region of T and S. Integer values in T and S
        are regarded as annotated regions. The number of labels is 1 + M where
        M is the maximum value in T and S.
    """
    cdef:
        cnp.npy_intp nslices = T.shape[0]
        cnp.npy_intp nrows = T.shape[1]
        cnp.npy_intp ncols = T.shape[2]
        cnp.npy_intp i, j, k
        int a, b
        int nlabels = 1 + np.max([np.max(T), np.max(S)])
        int[:] target_size = np.zeros(shape=(nlabels,), dtype=np.int32)
        int[:] intersection = np.zeros(shape=(nlabels,), dtype=np.int32)
        double[:] scores = np.zeros(shape=(nlabels,), dtype=np.float64)

    with nogil:
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    a = T[k, i, j]
                    b = S[k, i, j]
                    target_size[a] += 1
                    if a == b:
                        intersection[a] += 1


        for i in range(nlabels):
            if target_size[i] > 0:
                scores[i]=<double>intersection[i]/<double>target_size[i]

    return scores

cdef inline double _cubic_spline(double x) nogil:
    r''' Cubic B-Spline evaluated at x
    See eq. (3) of [Matttes03].

    References
    ----------
    [Mattes03] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K.,
               & Eubank, W. PET-CT image registration in the chest using
               free-form deformations. IEEE Transactions on Medical Imaging,
               22(1), 120-8, 2003.
    '''
    cdef:
        double absx = -x if x < 0.0 else x
        double sqrx = x * x

    if absx < 1.0:
        return (4.0 - 6.0 * sqrx + 3.0 * sqrx * absx) / 6.0
    elif absx < 2.0:
        return (8.0 - 12 * absx + 6.0 * sqrx - sqrx * absx) / 6.0
    return 0.0


def compute_separate_densities(int[:,:,:] labels, double[:,:,:] vol, int nbins):
    cdef:
        int ns = vol.shape[0]
        int nr = vol.shape[1]
        int nc = vol.shape[2]
        int padding = 2
        int k, i, j, offset, lab, bin
        double x, spline_arg, val
        double[:] min_val
        double[:] max_val
        double[:] delta
        double[:] sums
        double[:,:] densities
        int max_label
    with nogil:
        max_label = 0
        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    if labels[k,i,j] > max_label:
                        max_label = labels[k,i,j]

    min_val = np.empty(1 + max_label, dtype=np.float64)
    max_val = np.empty(1 + max_label, dtype=np.float64)
    delta = np.empty(1 + max_label, dtype=np.float64)
    sums = np.zeros(1 + max_label, dtype=np.float64)
    densities = np.zeros(shape=(1 + max_label, nbins), dtype=np.float64)

    with nogil:
        for i in range(1 + max_label):
            min_val[i] = 1e16
            max_val[i] = -1e16

        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    val = vol[k,i,j]
                    lab = labels[k,i,j]
                    if val < min_val[lab]:
                        min_val[lab] = val
                    if val > max_val[lab]:
                        max_val[lab] = val

        for i in range(1 + max_label):
            delta[i] = (max_val[i] - min_val[i])/(nbins - 2 * padding)
            min_val[i] = min_val[i]/delta[i] - padding

        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    lab = labels[k,i,j]
                    # find the bin corresponding to intensity samples[i]
                    x = vol[k,i,j] / delta[lab] - min_val[lab]  # Normalized intensity
                    bin = <cnp.npy_intp>(x)  # Histogram bin
                    if bin < padding:
                        bin = padding
                    if bin > nbins - 1 - padding:
                        bin = nbins - 1 - padding

                    # Add contribution of vol[k,i,j] to the histogram
                    spline_arg = (bin - 2) - x

                    for offset in range(-2, 3):
                        val = _cubic_spline(spline_arg)
                        densities[lab, bin + offset] += val
                        sums[lab] += val
                        spline_arg += 1.0

        for i in range(1 + max_label):
            if sums[i] > 0:
                for j in range(nbins):
                    densities[i, j] /= sums[i]
    return densities


def compute_densities(int[:,:,:] labels, double[:,:,:] vol, int nbins, int[:,:,:] mask=None):
    cdef:
        int ns = vol.shape[0]
        int nr = vol.shape[1]
        int nc = vol.shape[2]
        int padding = 2
        int k, i, j, offset, lab, bin
        double x, spline_arg, val
        double min_val, max_val, delta
        double[:] sums
        double[:,:] densities
        int max_label
    with nogil:
        max_label = 0
        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    if labels[k,i,j] > max_label:
                        max_label = labels[k,i,j]

    sums = np.zeros(1 + max_label, dtype=np.float64)
    densities = np.zeros(shape=(1 + max_label, nbins), dtype=np.float64)

    with nogil:
        min_val = 1e16
        max_val = -1e16

        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    if mask is None or mask[k,i,j] == 0:
                        continue
                    val = vol[k,i,j]
                    if val < min_val:
                        min_val = val
                    if val > max_val:
                        max_val = val

        delta = (max_val - min_val)/(nbins - 2 * padding)
        min_val = min_val/delta - padding

        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    if mask is None or mask[k,i,j] == 0:
                        continue
                    lab = labels[k,i,j]
                    # find the bin corresponding to intensity samples[i]
                    x = vol[k,i,j] / delta - min_val  # Normalized intensity
                    bin = <cnp.npy_intp>(x)  # Histogram bin
                    if bin < padding:
                        bin = padding
                    if bin > nbins - 1 - padding:
                        bin = nbins - 1 - padding

                    # Add contribution of vol[k,i,j] to the histogram
                    spline_arg = (bin - 2) - x

                    for offset in range(-2, 3):
                        val = _cubic_spline(spline_arg)
                        densities[lab, bin + offset] += val
                        sums[lab] += val
                        spline_arg += 1.0

        for i in range(1 + max_label):
            if sums[i] > 0:
                for j in range(nbins):
                    densities[i, j] /= sums[i]
    return densities


cdef double _sample_from_density(double[:] s, double u)nogil:
    cdef:
        double a, b, mid, frac, eval
        double epsilon = 1e-6
        int bin
    a = 0
    b = s.shape[0] - 1
    while (b - a > epsilon):
        mid = 0.5 * (a + b)
        # Interpolate at mid
        bin = <cnp.npy_intp>(mid)
        frac = mid-bin
        eval = (1.0 - frac) * s[bin] + frac * s[bin + 1]
        if eval <= u:
            a = mid
        if eval >= u:
            b = mid
    return 0.5 * (a + b)


def sample_from_density(double[:] density, int nsamples):
    cdef:
        int nbins = density.shape[0]
        int i, bin
        double a, b, mid, epsilon, frac, eval, target
        double[:] s = np.zeros(1 + nbins, dtype=np.float64)
        double[:] samples = np.empty(nsamples, dtype=np.float64)
        double[:] unif

    unif = np.random.uniform(0, 1, (nsamples,))
    with nogil:
        for i in range(1, 1 + nbins):
            s[i] = s[i-1] + density[i-1]

        for i in range(nsamples):
            samples[i] = _sample_from_density(s, unif[i])
    return samples, unif


def create_ss_de(int[:,:,:] labels, double[:,:] densities):
    cdef:
        int k, i, j, lab
        int ns = labels.shape[0]
        int nr = labels.shape[1]
        int nc = labels.shape[2]
        int nlabels = densities.shape[0]
        int nbins = densities.shape[1]
        double[:,:] s = np.zeros((nlabels, 1 + nbins), dtype=np.float64)
        double[:,:,:] out

    out = np.random.uniform(0, 1, (ns,nr,nc))
    with nogil:
        for lab in range(nlabels):
            for i in range(1, 1 + nbins):
                s[lab, i] = s[lab, i-1] + densities[lab, i-1]

        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    lab = labels[k,i,j]
                    if lab == 0:
                        out[k,i,j] = 0
                    else:
                        out[k,i,j] = _sample_from_density(s[lab], out[k,i,j])
    return out


def create_ss_mode(int[:,:,:] labels, double[:,:] densities):
    cdef:
        int k, i, j, lab
        int ns = labels.shape[0]
        int nr = labels.shape[1]
        int nc = labels.shape[2]
        int nlabels = densities.shape[0]
        int nbins = densities.shape[1]
        int[:] modes = np.empty(nlabels, dtype=np.int32)
        double[:,:,:] out

    out = np.random.uniform(0, 1, (ns,nr,nc))
    with nogil:
        modes[0] = 0
        for lab in range(1, nlabels):
            modes[lab] = 0
            for i in range(1, nbins):
                if densities[lab, modes[lab]] < densities[lab, i]:
                    modes[lab] = i

        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    lab = labels[k,i,j]
                    out[k,i,j] = modes[lab]
    return out


def create_ss_median(int[:,:,:] labels, double[:,:] densities):
    cdef:
        int k, i, j, lab
        int ns = labels.shape[0]
        int nr = labels.shape[1]
        int nc = labels.shape[2]
        int nlabels = densities.shape[0]
        int nbins = densities.shape[1]
        double[:,:] s = np.zeros((nlabels, 1 + nbins), dtype=np.float64)
        double[:,:,:] out = np.empty((ns, nr, nc), dtype=np.float64)
        double[:] medians = np.empty(nlabels, dtype=np.float64)

    with nogil:
        for lab in range(nlabels):
            for i in range(1, 1 + nbins):
                s[lab, i] = s[lab, i-1] + densities[lab, i-1]
        medians[0] = 0
        for lab in range(1, nlabels):
            medians[lab] = _sample_from_density(s[lab], 0.5)

        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    lab = labels[k,i,j]
                    out[k,i,j] = medians[lab]
    return out


def create_ss_mean(int[:,:,:] labels, double[:,:] densities):
    cdef:
        int k, i, j, lab
        int ns = labels.shape[0]
        int nr = labels.shape[1]
        int nc = labels.shape[2]
        int nlabels = densities.shape[0]
        int nbins = densities.shape[1]
        double[:] means = np.zeros(nlabels, dtype=np.float64)
        double[:,:,:] out = np.empty((ns, nr, nc), dtype=np.float64)

    with nogil:
        for lab in range(1, nlabels):
            for i in range(1, nbins):
                means[lab] += densities[lab, i] * i

        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    lab = labels[k,i,j]
                    out[k,i,j] = means[lab]
    return out

