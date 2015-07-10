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

cdef inline double ABS(double x)nogil:
    if x < 0:
        return -x
    return x

def baseline_segmentation(double[:,:,:] img, int nclasses, int max_iter=20):
    """ Simple k-mean clustering, starting with uniformly spaced intensities
    """
    cdef:
        int nx = img.shape[0]
        int ny = img.shape[1]
        int nz = img.shape[2]
        int i, j, k, ii, sel, iter
        double max_val, min_val, delta, best_diff
        double d
        double[:] means = np.empty(nclasses, dtype=np.float64)
        int[:] counts = np.empty(nclasses, dtype=np.int32)
        int[:,:,:] seg = np.empty((nx, ny, nz), dtype=np.int32)
    with nogil:
        min_val = img[0, 0, 0]
        max_val = img[0, 0, 0]
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if img[i, j, k] < min_val:
                        min_val = img[i, j, k]
                    if img[i, j, k] > max_val:
                        max_val = img[i, j, k]

        delta = (max_val - min_val) / (nclasses + 1)
        for i in range(1, 1 + nclasses):
            means[i - 1] = i * delta
        
        for iter in range(max_iter):
            # maximum likelihood assignment
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        sel = 0
                        best_diff = ABS(img[i,j,k] - means[0])
                        for ii in range(nclasses):
                            d = ABS(img[i,j,k] - means[ii])
                            if d < best_diff:
                                best_diff = d
                                sel = ii
                        seg[i,j,k] = sel

            # update the means
            counts[:] = 0
            means[:] = 0
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        counts[seg[i,j,k]] += 1
                        means[seg[i,j,k]] += img[i,j,k]
            for ii in range(nclasses):
                if counts[ii] > 0:
                    means[ii] /= counts[ii]
    return seg, means            
