#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from dipy.align.fused_types cimport floating, number
cdef extern from "dpy_math.h" nogil:
    double floor(double)
    double sqrt(double)
    double floor(double x) nogil
    double ceil(double x) nogil
    double INFINITY


cdef:
    int *DX = [0, 1, 1, 0, 0, 1, 1, 0]
    int *DY = [0, 0, 1, 1, 0, 0, 1, 1]
    int *DZ = [0, 0, 0, 0, 1, 1, 1, 1]

cdef inline int _argmax(double a, double b, double c)nogil:
    if ABS(a) > ABS(b):  # It cannot be b
        if ABS(a) > ABS(c):
            return 0
        return 2
    elif ABS(b) > ABS(c):  # it cannot be a nor c
        return 1
    return 2


cdef inline double ABS(double x)nogil:
    if x<0:
        return -x
    return x


cdef inline int _solve_pivoting(double **M, double *b, double *out)nogil:
    cdef:
        int i, j, k, p,q
        int pivot0, pivot1, pivot2
        double den
    # Choose first pivot
    pivot0 = _argmax(M[0][0], M[1][0], M[2][0])
    den = M[pivot0][0]
    if den == 0:
        return -1
    p = pivot0+1
    q = pivot0+2
    if p > 2:
        p -= 3
    if q > 2:
        q -= 3
    # Divide pivotal equation
    M[pivot0][1] /= den
    M[pivot0][2] /= den
    b[pivot0] /= den

    # Add pivotal equation to the others
    M[p][1] -= M[p][0] * M[pivot0][1]
    M[p][2] -= M[p][0] * M[pivot0][2]
    b[p] -= M[p][0] * b[pivot0]

    M[q][1] -= M[q][0] * M[pivot0][1]
    M[q][2] -= M[q][0] * M[pivot0][2]
    b[q] -= M[q][0] * b[pivot0]

    # Choose second pivot
    if ABS(M[p][1]) > ABS(M[q][1]):
        pivot1 = p
        pivot2 = q
    else:
        pivot1 = q
        pivot2 = p

    # Divide pivotal equation
    den = M[pivot1][1]
    if den == 0:
        return -1
    M[pivot1][2] /= den
    b[pivot1] /= den

    # Add pivotal equation to the last one
    M[pivot2][2] -= M[pivot2][1]*M[pivot1][2]
    b[pivot2] -= M[pivot2][1]*b[pivot1]

    # Back substitution
    if M[pivot2][2] == 0:
        out[2] = 0.0
    else:
        out[2] = b[pivot2] / M[pivot2][2]
    out[1] = b[pivot1] - M[pivot1][2]*out[2]
    out[0] = b[pivot0] - out[1]*M[pivot0][1] - out[2]*M[pivot0][2]
    return 0


cdef int _solve_trilinear_cube(double **P, double x, double y, double z, int maxiter, double* out)nogil:
    cdef:
        double *_M = [0,0,0,0,0,0,0,0,0]
        double *residual = [0,0,0]
        double *sol = [0,0,0]
        double **M = [_M, _M+3, _M+6]
        double alpha=0.5, beta=0.5, gamma=0.5
        int ii, jj, iter, retval
        double epsilon = 1e-7
        double px, py, pz, sd

    alpha, beta, gamma = 0.5, 0.5, 0.5
    for iter in range(maxiter):
        # Residual
        for ii in range(3):
            residual[ii] = (gamma    *(beta*(alpha*P[0][ii] + (1-alpha)*P[1][ii]) + (1-beta)*(alpha*P[3][ii] + (1-alpha)*P[2][ii])) +
                            (1-gamma)*(beta*(alpha*P[4][ii] + (1-alpha)*P[5][ii]) + (1-beta)*(alpha*P[7][ii] + (1-alpha)*P[6][ii])))
        residual[0] -= x
        residual[1] -= y
        residual[2] -= z
        # Jacobian matrix
        for ii in range(3):
            M[ii][0] = gamma*(beta*(P[0][ii]-P[1][ii])+(1-beta)*(P[3][ii]-P[2][ii])) + (1-gamma)*(beta*(P[4][ii]-P[5][ii])+(1-beta)*(P[7][ii]-P[6][ii]))
            M[ii][1] = gamma*((alpha*P[0][ii]+(1-alpha)*P[1][ii]) - (alpha*P[3][ii]+(1-alpha)*P[2][ii])) + (1-gamma)*((alpha*P[4][ii]+(1-alpha)*P[5][ii]) - (alpha*P[7][ii]+(1-alpha)*P[6][ii]))
            M[ii][2] = (beta*(alpha*P[0][ii] + (1-alpha)*P[1][ii]) + (1-beta)*(alpha*P[3][ii] + (1-alpha)*P[2][ii])) - (beta*(alpha*P[4][ii] + (1-alpha)*P[5][ii]) + (1-beta)*(alpha*P[7][ii] + (1-alpha)*P[6][ii]))
        #retval = _inverse(M, Minv)
        retval = _solve_pivoting(M, residual, sol)
        if retval < 0:
            return -1
        alpha -= sol[0]
        beta -= sol[1]
        gamma -= sol[2]
        sd = ABS(sol[0])+ABS(sol[1])+ABS(sol[2])
        if sd < epsilon:
            break
    out[0] = alpha
    out[1] = beta
    out[2] = gamma
    if -epsilon <= alpha and alpha <= 1+epsilon and -epsilon <= beta and beta <= 1+epsilon and -epsilon <= gamma and gamma <= 1+epsilon:
        for ii in range(3):
            residual[ii] = (gamma    *(beta*(alpha*P[0][ii] + (1-alpha)*P[1][ii]) + (1-beta)*(alpha*P[3][ii] + (1-alpha)*P[2][ii])) +
                            (1-gamma)*(beta*(alpha*P[4][ii] + (1-alpha)*P[5][ii]) + (1-beta)*(alpha*P[7][ii] + (1-alpha)*P[6][ii])))
        sd = ABS(residual[0] - x) + ABS(residual[1] - y) + ABS(residual[2] - z)
        if sd < epsilon:
            return 1
    return 0


cdef void _mark_full_box(int maxiter, floating[:,:,:,:] v, int xq, int yq, int zq, char[:,:,:] solved, floating[:,:,:,:] out)nogil:
    cdef:
        double *_P = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        double **P = [_P, _P+3, _P+6, _P+9, _P+12, _P+15, _P+18, _P+21]
        int ii, jj, kk, idx, d
        double x, y, z
        int *opts = [0,0]
        int sol_idx
        int flags, retval
        int start_h, end_h, start_v, end_v
        double *N = [0,0,0]
        double *sol0 = [0,0]
        double *sol1 = [0,0]
        double **sols = [sol0, sol1]
        #double[:,:] sols = np.empty((2,2), dtype=np.float64)
        double alpha, beta, depth
        double dk
        double epsilon = 1e-8
        double *mins = [0,0,0]
        double *maxs = [0,0,0]
        int *inits = [0,0,0]
        int *ends = [0,0,0]
        double *sol = [0,0,0]
    # Find bounding box
    mins[0] = INFINITY
    mins[1] = INFINITY
    mins[2] = INFINITY
    maxs[0] = -INFINITY
    maxs[1] = -INFINITY
    maxs[2] = -INFINITY
    for idx in range(8):
        ii = xq + DX[idx]
        jj = yq + DY[idx]
        kk = zq + DZ[idx]
        for d in range(3):
            P[idx][d] = v[ii, jj, kk, d]
        P[idx][0] += ii
        P[idx][1] += jj
        P[idx][2] += kk
        for d in range(3):
            if P[idx][d] < mins[d]:
                mins[d] = P[idx][d]
            if P[idx][d] > maxs[d]:
                maxs[d] = P[idx][d]

    for idx in range(3):
        inits[idx] = <int>ceil(mins[idx])
        ends[idx] = <int>floor(maxs[idx])
        if inits[idx] < 0:
            inits[idx] = 0
        if ends[idx] >= v.shape[idx]:
            ends[idx] = v.shape[idx] - 1
    for ii in range(inits[0], 1 + ends[0]):
        for jj in range(inits[1], 1 + ends[1]):
            for kk in range(inits[2], 1 + ends[2]):
                if solved[ii, jj, kk] == 1:
                    continue
                retval = _solve_trilinear_cube(P, ii, jj, kk, maxiter, sol)
                if retval > 0:
                    solved[ii, jj, kk] = 1
                    out[ii, jj, kk, 0] = xq + (1.0 - sol[0]) - ii
                    out[ii, jj, kk, 1] = yq + (1.0 - sol[1]) - jj
                    out[ii, jj, kk, 2] = zq + (1.0 - sol[2]) - kk



def invert_vf_full_box_3d(floating[:,:,:,:] dfield,
                          double[:, :] d_world2grid=None,
                          double[:] spacing=None,
                          int maxiter=5,
                          double tol=1e-2,
                          floating[:, :, :, :] start=None):
    cdef:
        ftype = np.asarray(dfield).dtype
        int nx = dfield.shape[0]
        int ny = dfield.shape[1]
        int nz = dfield.shape[2]
        int x, y, z, cnt
        double px, py, pz
        char[:,:,:] solved = np.zeros((nx, ny, nz), dtype=np.int8)
        floating[:,:,:,:] out = np.zeros((nx,ny,nz,3), dtype=ftype)
        floating[:,:,:,:] vtemp = np.empty((nx,ny,nz,3), dtype=ftype)
        double[:,:] d_grid2world = np.empty((4,4), dtype=np.double)

    with nogil:
        if d_world2grid is not None:
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        px = dfield[x,y,z, 0]
                        py = dfield[x,y,z, 1]
                        pz = dfield[x,y,z, 2]
                        vtemp[x,y,z, 0] = _apply_affine_3d_x0(px, py, pz, 0, d_world2grid)
                        vtemp[x,y,z, 1] = _apply_affine_3d_x1(px, py, pz, 0, d_world2grid)
                        vtemp[x,y,z, 2] = _apply_affine_3d_x2(px, py, pz, 0, d_world2grid)
        else:
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        vtemp[x,y,z,0] = dfield[x,y,z,0]
                        vtemp[x,y,z,1] = dfield[x,y,z,1]
                        vtemp[x,y,z,2] = dfield[x,y,z,2]

        for x in range(nx-1):
            for y in range(ny-1):
                for z in range(nz-1):
                    _mark_full_box(maxiter, vtemp, x, y, z, solved, out)

    if d_world2grid is not None:
        d_grid2world = np.linalg.inv(d_world2grid)
        with nogil:
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        px = out[x,y,z, 0]
                        py = out[x,y,z, 1]
                        pz = out[x,y,z, 2]
                        out[x,y,z, 0] = _apply_affine_3d_x0(px, py, pz, 0, d_grid2world)
                        out[x,y,z, 1] = _apply_affine_3d_x1(px, py, pz, 0, d_grid2world)
                        out[x,y,z, 2] = _apply_affine_3d_x2(px, py, pz, 0, d_grid2world)
    return out