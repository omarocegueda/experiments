#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
import numpy as np
cimport cython
import scipy as sp
import scipy
import scipy.sparse
from dipy.align.fused_types cimport floating
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

# This is a cython bug, we define a global variable as a work around
bug = 'ImportError: dynamic module does not define init function (initsplines)'  

cdef extern from "math.h":
    double sqrt(double x) nogil
    double floor(double x) nogil
    double log(double x) nogil
    double exp(double x) nogil

cdef _estimate_b0(double[:,:,:,:] dwi, double[:,:] bvecs, double[:] bvals,
                  double[:,:,:] out):
    """
    Parameters
    ----------
    dwi: array, shape(nx, ny, nz, n)
        dw-mri volume (only weighted images)
    bvecs: array, shape(n, 3)
        n b-vectors
    bvals: array, shape(n,)
        n b-values
    out: array, shape(nx, ny, nz)
        the buffer where the estimated b0 will be written
    """
    cdef:
        int n=bvecs.shape[0]
        int nx=dwi.shape[0]
        int ny=dwi.shape[1]
        int nz=dwi.shape[2]
        int i, j, k, ii
        double[:,:] X = np.ndarray((n, 7), dtype=np.float64)
        double[:,:] XtX
        double[:,:] H
        double[:] h
        double log_b0, y
        double epsilon = 1e-9
    # Build the extended design matrix
    with nogil:
        for i in range(n):
            X[i, 0] = -bvals[i] * bvecs[i,0] * bvecs[i,0]
            X[i, 1] = -bvals[i] * bvecs[i,1] * bvecs[i,1]
            X[i, 2] = -bvals[i] * bvecs[i,2] * bvecs[i,2]
            X[i, 3] = -2 * bvals[i] * bvecs[i,0] * bvecs[i,1]
            X[i, 4] = -2 * bvals[i] * bvecs[i,1] * bvecs[i,2]
            X[i, 5] = -2 * bvals[i] * bvecs[i,0] * bvecs[i,2]
            X[i, 6] = 1
    XtX = np.dot(X.T, X)
    print(XtX)
    H = np.dot(np.linalg.inv(XtX), X.T)
    # We only need the last coefficient
    h = H[6,:]
    print(h)

    # Solve linear systems
    with nogil:
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    log_b0 = 0
                    for ii in range(n):
                        if dwi[i,j,k,ii] > epsilon:
                            y = log(dwi[i,j,k,ii])
                        else:
                            y = log(epsilon)
                        log_b0 += h[ii] * y
                    out[i,j,k] = exp(log_b0)

def estimate_b0(double[:,:,:,:] dwi, double[:,:] bvecs, double[:] bvals):
    cdef:
        int nx = dwi.shape[0]
        int ny = dwi.shape[1]
        int nz = dwi.shape[2]
        double[:,:,:] out = np.ndarray((nx, ny, nz), dtype=np.float64)
    _estimate_b0(dwi, bvecs, bvals, out)
    return out
