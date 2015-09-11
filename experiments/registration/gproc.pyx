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

def gram_matrix_poly(double[:,:] x_in, double sigmasq_signal):
    cdef:
        int m = x_in.shape[0]
        double[:,:] out = np.empty((m,m), dtype=np.float64)
        int i, j
        double p
    with nogil:
        for i in range(m):
            for j in range(m):
                p = (x_in[i, 0] * x_in[j, 0]) + (x_in[i, 1] * x_in[j, 1]) + (x_in[i, 2] * x_in[j, 2])
                out[i,j] = sigmasq_signal * ( 0.54 + 1.54 * p * p)
    return out


def predict_dwi(double[:,:,:,:] dwi_in, double[:,:] K, double sigmasq_noise):
    cdef:
        int nx = dwi_in.shape[0]
        int ny = dwi_in.shape[1]
        int nz = dwi_in.shape[2]
        int n = dwi_in.shape[3]
        int x, y, z, i, j
        double s
        double[:,:] iK
        double[:,:] P
        double[:,:,:,:] out = np.empty_like(dwi_in)
    iK = np.linalg.inv(K + np.diag(np.ones(n)*sigmasq_noise))
    P = np.dot(K, iK)
    with nogil:
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    for i in range(n):
                        s = 0;
                        for j in range(n):
                            s += P[i, j] * dwi_in[x, y, z, j]
                        out[x,y,z,i] = s
    return out



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

def get_wm_variance(double[:,:,:,:] dwi, int[:,:,:] mask):
    cdef:
        double[:,:,:] var_vol
    var_vol = np.var(dwi, 3) * mask


def get_ec_fields(int[:]domain_shape, int[:]codomain_shape, double[:] params, double[:] pe_dir):
    cdef:
        int nx = domain_shape[0]
        int ny = domain_shape[1]
        int nz = domain_shape[2]
        int cnx = codomain_shape[0]
        int cny = codomain_shape[1]
        int cnz = codomain_shape[2]
        int x, y, z
        double pp, den
        double[:,:,:] field = np.empty((domain_shape[0], domain_shape[1], domain_shape[2]), dtype=np.float64)
        double[:,:,:] ifield = np.empty((codomain_shape[0], codomain_shape[1], codomain_shape[2]), dtype=np.float64)

    with nogil:
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    pp = x * params[0] + y*params[1] + z*params[2]
                    field[x,y,z] = pp + params[3]

        den = 1.0 + params[0]*pe_dir[0] + params[1]*pe_dir[1] + params[2]*pe_dir[2]
        for x in range(cnx):
            for y in range(cny):
                for z in range(cnz):
                    pp = x * params[0] + y*params[1] + z*params[2]
                    ifield[x,y,z] = -1.0 * (pp + params[3])/den
    return field, ifield

def warp_model_to_scan(double[:,:,:] img, double[:,:] img_grid2world, double[:] pe_dir,
                       double[:,:] mov_affine, double[:] ec_params, double hz2vox,
                       int[:] out_shape, double[:,:] out_grid2world):
    r""" Transform a model prediction to scan space

    Parameters
    img:
        the input image, in model space

    pe_dir:
        phase-encode direction in grid coordinates
    """
    cdef:
        int nx = out_shape[0]
        int ny = out_shape[1]
        int nz = out_shape[2]
        int x, y, z

    with nogil:
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    # The Eddy-current parameters provide the field in Hz.

                    # The ec_params describe the field in model space
                    # we need the inverse field in scan space
                    # the ec_field is F(x) = hz2vox*[(a^T)x + c], in voxels
                    #
                    pass
    pass

