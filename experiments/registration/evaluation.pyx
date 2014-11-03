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
