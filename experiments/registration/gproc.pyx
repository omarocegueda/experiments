#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
import numpy.random as random
import numpy.linalg as npl


from dipy.align.transforms cimport (Transform)

cdef extern from "math.h" nogil:
    double cos(double)
    double sin(double)
    double log(double)
    double exp(double)

cdef void _squared_exponential(double[:] input, double ell, double[:,:] out)nogil:
    cdef:
        int n = input.shape[0]
        int i, j
        double p
    ell *= - 2.0
    for i in range(n):
        for j in range(n):
            p = input[i] - input[j] 
            out[i,j] = exp((p * p) / ell)
            
def squared_exponential_conditional(double[:] x_in, double[:] f_in, double[:] x_out, double ell, double sigma_noise):
    cdef:
        int n = x_out.shape[0]
        int m = x_in.shape[0]
        double[:,:] S_in = np.empty((m,m), dtype=np.float64)
        double[:,:] S_out_in = np.empty((n,m), dtype=np.float64)
        double[:,:] tmp = np.empty((n,n), dtype=np.float64)
        double[:,:] S_out = np.empty((n,n), dtype=np.float64)
        double[:] mean_out = np.empty((n,), dtype=np.float64)
        int i, j, k
        double p
    ell *= - 2.0
    # S_in
    with nogil:
        for i in range(m):
            for j in range(m):
                p = x_in[i] - x_in[j] 
                S_in[i,j] = exp((p * p) / ell)
            S_in[i,i] += sigma_noise * sigma_noise
    S_in = np.linalg.inv(S_in)
    
    #S_out_in
    with nogil:
        for i in range(n):
            for j in range(m):
                p = x_out[i] - x_in[j] 
                S_out_in[i,j] = exp((p * p) / ell)
    
    # S_out
    with nogil:
        for i in range(n):
            for j in range(n):
                p = x_out[i] - x_out[j] 
                S_out[i,j] = exp((p * p) / ell)
    
    tmp = np.dot(S_out_in, S_in).dot(np.transpose(S_out_in))
    mean_out = np.dot(S_out_in, S_in).dot(f_in)
    with nogil:
        for i in range(n):
            for j in range(n):
                S_out[i,j] -= tmp[i,j]
    return mean_out, S_out


def spherical_poly_conditional(double[:,:] x_in, double[:] f_in, double[:,:] x_out, double sigmasq_signal, double sigmasq_noise):
    cdef:
        int n = x_out.shape[0]
        int m = x_in.shape[0]
        double[:,:] S_in = np.empty((m,m), dtype=np.float64)
        double[:,:] S_out_in = np.empty((n,m), dtype=np.float64)
        double[:,:] tmp = np.empty((n,n), dtype=np.float64)
        double[:,:] S_out = np.empty((n,n), dtype=np.float64)
        double[:] mean_out = np.empty((n,), dtype=np.float64)
        int i, j, k
        double p
    # S_in
    with nogil:
        for i in range(m):
            for j in range(m):
                p = (x_in[i, 0] * x_in[j, 0]) + (x_in[i, 1] * x_in[j, 1]) + (x_in[i, 2] * x_in[j, 2])
                S_in[i,j] = sigmasq_signal * ( 0.54 + 1.54 * p * p)
            S_in[i,i] += sigmasq_noise
    S_in = np.linalg.inv(S_in)
    
    #S_out_in
    with nogil:
        for i in range(n):
            for j in range(m):
                p = (x_out[i, 0] * x_in[j, 0]) + (x_out[i, 1] * x_in[j, 1]) + (x_out[i, 2] * x_in[j, 2])
                S_out_in[i,j] = sigmasq_signal * ( 0.54 + 1.54 * p * p)
    
    # S_out
    with nogil:
        for i in range(n):
            for j in range(n):
                p = (x_out[i, 0] * x_out[j, 0]) + (x_out[i, 1] * x_out[j, 1]) + (x_out[i, 2] * x_out[j, 2])
                S_out[i,j] = sigmasq_signal * ( 0.54 + 1.54 * p * p)
    
    tmp = np.dot(S_out_in, S_in).dot(np.transpose(S_out_in))
    mean_out = np.dot(S_out_in, S_in).dot(f_in)
    with nogil:
        for i in range(n):
            for j in range(n):
                S_out[i,j] -= tmp[i,j]
    return mean_out, S_out


def squared_exp(input, ell):
    cdef:
        int n = input.shape[0]
        double[:,:] out = np.empty((n,n), dtype=np.float64)
    _squared_exponential(input, ell, out)
    return out

def sample_gp(sigma):
    pass
