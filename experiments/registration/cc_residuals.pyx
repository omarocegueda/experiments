#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport cython
from dipy.align.fused_types cimport floating, number

cdef inline int _int_max(int a, int b) nogil:
    r"""
    Returns the maximum of a and b
    """
    return a if a >= b else b


cdef inline int _int_min(int a, int b) nogil:
    r"""
    Returns the minimum of a and b
    """
    return a if a <= b else b

cdef enum:
    SI = 0
    SI2 = 1
    SJ = 2
    SJ2 = 3
    SIJ = 4
    CNT = 5

def compute_cc_residuals(double[:,:,:] I, double[:,:,:] J, int radius, int mask0=1):
    cdef:
        int ns = I.shape[0]
        int nr = I.shape[1]
        int nc = I.shape[2]
        double s1, s2, t1, t2, wx, p, t, ave, worst
        double det0, det1, absdet0, absdet1
        double alpha, beta
        int i, j, k, s, r, c, intersect
        int start_k, end_k, start_i, end_i, start_j, end_j
        int regression_reference

        double[:,:,:] residuals = np.zeros((ns, nr, nc))

    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):

                    #Affine fit
                    s1 = 0
                    s2 = 0
                    t1 = 0
                    t2 = 0
                    wx = 0
                    p = 0
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):
                                if mask0 != 0:
                                    intersect = (I[k, i, j]>0) * (J[k, i, j]>0)
                                    if intersect == 0:
                                        continue
                                s1 += I[k, i, j]
                                s2 += I[k, i, j] * I[k, i, j]
                                t1 += J[k, i, j]
                                t2 += J[k, i, j] * J[k, i, j]
                                wx += 1
                                p += I[k, i, j] * J[k, i, j]

                    residuals[s, r, c] = 0
                    if wx<3:
                        continue

                    det0 = s2 * wx - (s1 * s1)
                    det1 = t2 * wx - (t1 * t1)

                    absdet0 = -det0 if det0 < 0 else det0
                    absdet1 = -det1 if det1 < 0 else det1

                    if absdet0 < 1e-6 and absdet1 < 1e-6:
                        continue

                    if absdet0 > absdet1:
                        regression_reference = 0
                        beta = (t1*s2 - (s1 * p)) / det0
                        alpha = (p - beta * s1) / s2
                    else:
                        regression_reference = 1
                        beta = (s1*t2 - (t1 * p)) / det1
                        alpha = (p - beta * t1) / t2

                    #Compute residuals
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):

                                if mask0 != 0:
                                    intersect = (I[k, i, j]>0) * (J[k, i, j]>0)
                                    if intersect == 0:
                                        continue

                                if regression_reference == 0:
                                    residuals[s, r, c] += ((alpha * I[k, i, j] + beta) - J[k, i, j]) ** 2
                                else:
                                    residuals[s, r, c] += ((alpha * J[k, i, j] + beta) - I[k, i, j]) ** 2

                    residuals[s, r, c] /= wx

    return residuals


def affine_fit(double[:] x, double[:] y, int force_ref=-1):
    cdef:
        int n = x.shape[0]
        int m = y.shape[0]
        int cnt = 0
        double s1, s2, t1, t2, p, mse
        double det0, det1, absdet0, absdet1
        int i, regression_reference
        double[:] fit_x = np.ndarray(n, np.float64)
        double[:] fit_y = np.ndarray(n, np.float64)
    if n != m:
        raise ValueError("Arrays must have the same length")

    for i in range(n):
        if x[i]==0 or y[i]==0:
            continue;
        cnt += 1
        s1 += x[i]
        s2 += x[i] * x[i]
        t1 += y[i]
        t2 += y[i] * y[i]
        p += x[i] * y[i]
    n = cnt

    det0 = s2 * n - (s1 * s1)
    det1 = t2 * n - (t1 * t1)

    absdet0 = -det0 if det0 < 0 else det0
    absdet1 = -det1 if det1 < 0 else det1

    if absdet0 < 1e-6 and absdet1 < 1e-6:
        raise ValueError("Nearly constant arrays")

    if force_ref == 0 or ( force_ref != 1 and absdet0 > absdet1):
        regression_reference = 0
        beta = (t1*s2 - (s1 * p)) / det0
        alpha = (p - beta * s1) / s2
    else:
        regression_reference = 1
        beta = (s1*t2 - (t1 * p)) / det1
        alpha = (p - beta * t1) / t2

    mse = 0
    for i in range(x.shape[0]):
        if regression_reference == 0:
            fit_x[i] = x[i]
            fit_y[i] = alpha * x[i] + beta
            mse += (fit_y[i] - y[i])**2
        else:
            fit_y[i] = y[i]
            fit_x[i] = alpha * y[i] + beta
            mse += (fit_x[i] - x[i])**2
    mse /= n

    return alpha, beta, fit_x, fit_y, regression_reference, mse


def linear_fit(double[:] x, double[:] y, int force_ref=-1):
    cdef:
        int n = x.shape[0]
        int m = y.shape[0]
        double s2, t2, p, mse
        int i, regression_reference
        double[:] fit_x = np.ndarray(n, np.float64)
        double[:] fit_y = np.ndarray(n, np.float64)
    if n != m:
        raise ValueError("Arrays must have the same length")
    for i in range(n):
        s2 += x[i] * x[i]
        t2 += y[i] * y[i]
        p += x[i] * y[i]

    if s2 < 1e-6 and t2 < 1e-6:
        raise ValueError("Nearly constant arrays")

    if force_ref == 0 or ( force_ref != 1 and s2 > t2):
        regression_reference = 0
        alpha = p / s2
    else:
        regression_reference = 1
        alpha = p / t2

    mse = 0
    for i in range(n):
        if regression_reference == 0:
            fit_x[i] = x[i]
            fit_y[i] = alpha * x[i]
            mse += (fit_y[i] - y[i])**2
        else:
            fit_y[i] = y[i]
            fit_x[i] = alpha * y[i]
            mse += (fit_x[i] - x[i])**2
    mse /= n

    return alpha, fit_x, fit_y, regression_reference, mse


def compute_cc_residuals_noboundary(double[:,:,:] I, double[:,:,:] J, int radius):
    cdef:
        int ns = I.shape[0]
        int nr = I.shape[1]
        int nc = I.shape[2]
        double s1, s2, t1, t2, wx, p, t, ave, worst
        double det0, det1, absdet0, absdet1
        double alpha, beta
        int i, j, k, s, r, c, intersect
        int start_k, end_k, start_i, end_i, start_j, end_j
        int regression_reference

        double[:,:,:] residuals = np.zeros((ns, nr, nc))

    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):

                    #Affine fit
                    s1 = 0
                    s2 = 0
                    t1 = 0
                    t2 = 0
                    wx = 0
                    p = 0
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):

                                intersect = (I[k, i, j]>0) * (J[k, i, j]>0)
                                if intersect == 0:
                                    continue
                                s1 += I[k, i, j]
                                s2 += I[k, i, j] * I[k, i, j]
                                t1 += J[k, i, j]
                                t2 += J[k, i, j] * J[k, i, j]
                                wx += 1
                                p += I[k, i, j] * J[k, i, j]

                    residuals[s, r, c] = 0
                    if wx<3:
                        continue

                    det0 = s2 * wx - (s1 * s1)
                    det1 = t2 * wx - (t1 * t1)

                    absdet0 = -det0 if det0 < 0 else det0
                    absdet1 = -det1 if det1 < 0 else det1

                    if absdet0 < 1e-6 and absdet1 < 1e-6:
                        continue

                    if absdet0 > absdet1:
                        regression_reference = 0
                        beta = (t1*s2 - (s1 * p)) / det0
                        alpha = (p - beta * s1) / s2
                    else:
                        regression_reference = 1
                        beta = (s1*t2 - (t1 * p)) / det1
                        alpha = (p - beta * t1) / t2

                    #Compute residuals
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):

                                intersect = (I[k, i, j]>0) * (J[k, i, j]>0)
                                if intersect == 0:
                                    continue

                                if regression_reference == 0:
                                    residuals[s, r, c] += ((alpha * I[k, i, j] + beta) - J[k, i, j]) ** 2
                                else:
                                    residuals[s, r, c] += ((alpha * J[k, i, j] + beta) - I[k, i, j]) ** 2

                    residuals[s, r, c] /= wx

    return residuals


def compute_transfer_system_underdetermined(int[:,:,:] labels, int nlabels, double[:,:,:] x, int radius):
    cdef:
        int ns = labels.shape[0]
        int nr = labels.shape[1]
        int nc = labels.shape[2]
        int n = (2 * radius + 1) * (2 * radius + 1) * (2 * radius + 1)
        int s, r, c
        int k, i, j, start_k, end_k, start_i, end_i, start_j, end_j
        int ndiff, ii, jj
        double delta, contrib
        double sx, sx2
        int[:] K = np.ndarray((nlabels,), dtype=np.int32)
        int[:] indices = np.ndarray((nlabels,), dtype=np.int32)
        double[:] A = np.ndarray((nlabels,), dtype=np.float64)
        double[:,:] S = np.zeros((nlabels-1, nlabels-1), dtype=np.float64)

    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):
                    A[:] = 0
                    K[:] = 0
                    # Compute stats for this window
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):
                                if labels[k, i, j] == 0:
                                    continue
                                sx += x[k, i, j]
                                sx2 += x[k, i, j] * x[k, i, j]
                                A[labels[k, i, j]-1] += x[k, i, j]
                                K[labels[k, i, j]-1] += 1

                    delta = n * sx2 - sx * sx

                    if delta<1e-4:# too small
                        continue

                    # Look for present labels
                    ndiff = 0
                    for i in range(nlabels):
                        if K[i] > 0:
                            indices[ndiff] = i
                            ndiff += 1

                    # Add contribution of this window to the system
                    for i in range(ndiff):
                        ii = indices[i]
                        # Product of equal columns
                        contrib = K[ii] - (K[ii]*K[ii]*sx2 - 2*K[ii]*A[ii]*sx + n*A[ii]*A[ii]) / delta
                        S[ii, ii] += contrib

                        # Product of different columns
                        for j in range(i+1, ndiff):
                            jj = indices[j]
                            contrib = (K[ii]*K[jj]*sx2 - (K[ii]*A[jj] + K[jj]*A[ii])*sx + n*A[ii]*A[jj]) / delta
                            S[ii, jj] -= contrib
                            S[jj, ii] -= contrib
    return S


def compute_transfer_value_and_gradient(double[:] f, int[:,:,:] labels, int nlabels, double[:,:,:] x, int radius):
    cdef:
        int ns = labels.shape[0]
        int nr = labels.shape[1]
        int nc = labels.shape[2]
        int n = (2 * radius + 1) * (2 * radius + 1) * (2 * radius + 1)
        int s, r, c
        int k, i, j, start_k, end_k, start_i, end_i, start_j, end_j
        int cnt, nwindows
        double energy, num, delta, contrib
        double sx, sx2, alpha, beta, gamma
        int[:] K = np.ndarray((nlabels,), dtype=np.int32)
        double[:] A = np.ndarray((nlabels,), dtype=np.float64)
        double[:] grad = np.ndarray((nlabels,), dtype=np.float64)
        int req_mem

    with nogil:
        energy = 0
        grad[:] = 0
        nwindows = 0
        req_mem = 0
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):
                    A[:] = 0
                    K[:] = 0
                    # Compute stats for this window
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    sx = 0
                    sx2 = 0
                    cnt = 0
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):
                                #if labels[k, i, j] == 0:
                                #    continue
                                sx += x[k, i, j]
                                sx2 += x[k, i, j] * x[k, i, j]
                                A[labels[k, i, j]] += x[k, i, j]
                                K[labels[k, i, j]] += 1
                                cnt += 1
                    if cnt < n:
                        continue
                    # Compute f-stats
                    alpha = 0
                    beta = 0
                    gamma = 0
                    for i in range(0, nlabels):
                        if(K[i]>0):
                            req_mem += 2
                        alpha += A[i] * f[i]
                        beta += K[i] * f[i]
                        gamma += K[i] * f[i] * f[i]

                    delta = n * gamma - beta * beta

                    if delta<1e-4:# too small
                        continue

                    nwindows += 1

                    # Compute contribution of this window to the total energy
                    num = n*alpha*alpha - 2.0*alpha*beta*sx + gamma*sx*sx
                    energy += sx2 - num/delta

                    # Compute contribution of this window to the gradient
                    for i in range(0, nlabels):
                        contrib = 2*(n*alpha*A[i] - sx*(alpha*K[i]+beta*A[i]) + sx*sx*K[i]*f[i])*delta - 2*num*(n*K[i]*f[i] - beta*K[i])
                        grad[i] -= contrib/(delta*delta)
        energy /= nwindows
        for i in range(nlabels):
            grad[i] /= nwindows
    return energy, grad


def get_compressed_transfer(int[:,:,:] A, double[:,:,:] B):
    cdef:
        int ns = A.shape[0]
        int nr = A.shape[1]
        int nc = A.shape[2]
        int s,r,c, lab, i, l
        int[:] val_lab_map
        int[:] cnt
        double[:] means
        double[:] vars
        int[:,:,:] newA = np.empty((ns, nr, nc), dtype=np.int32)

    labels = np.unique(A)
    nlabels = len(labels)
    max_lab = np.max(A)
    val_lab_map = np.empty(1 + max_lab, dtype=np.int32)
    cnt = np.zeros(nlabels, dtype=np.int32)
    means = np.zeros(nlabels, dtype=np.float64)
    vars = np.zeros(nlabels, dtype=np.float64)

    for i,l in enumerate(labels):
        val_lab_map[l] = i

    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):
                    lab = val_lab_map[A[s,r,c]]
                    newA[s,r,c] = lab
                    means[lab] += B[s,r,c]
                    vars[lab] += B[s,r,c] * B[s,r,c]
                    cnt[lab] += 1
        for lab in range(nlabels):
            means[lab]/=cnt[lab]
            vars[lab] = vars[lab]/cnt[lab] - (means[lab]*means[lab])

    return np.array(means), np.array(vars), np.array(cnt), np.array(newA)


cdef inline int _mod(int x, int m)nogil:
    if x<0:
        return x + m
    return x


cdef _add_elementary_vector(double y, int g, double[:,:,:,:] A, int[:,:,:,:] K,
                            double[:,:,:,:] S, int ss, int rr, int cc, int weight)nogil:
    if weight == 0:
        A[ss, rr, cc, g] = y
        K[ss, rr, cc, g] = 1
        S[sss, rr, cc, 0] = y
        S[sss, rr, cc, 1] = y*y
    elif weight == 1:
        A[ss, rr, cc, g] += y
        K[ss, rr, cc, g] += 1
        S[sss, rr, cc, 0] += y
        S[sss, rr, cc, 1] += y*y
    else:
        A[ss, rr, cc, g] -= y
        K[ss, rr, cc, g] -= 1
        S[sss, rr, cc, 0] -= y
        S[sss, rr, cc, 1] -= y*y


cdef _update_factors(double[:,:,:,:] A, double[:,:,:,:] K, double[:,:,:,:] S,
                     int sss, int rr, int cc, int prev_s, int prev_r,
                     int prev_c, int weight)nogil:
    cdef:
        int nlabels = A.shape[3]
        int idx
    if weight == -1:
        S[sss, rr, cc, 0] += S[prev_ss, prev_rr, prev_cc, 0]
        S[sss, rr, cc, 1] += S[prev_ss, prev_rr, prev_cc, 1]
        for idx in range(nlabels):
            A[sss, rr, cc, idx] += A[prev_ss, prev_rr, prev_cc, idx]
            K[sss, rr, cc, idx] += K[prev_ss, prev_rr, prev_cc, idx]

    else:
        for idx in range(nlabels):
            A[sss, rr, cc, idx] += A[prev_ss, prev_rr, prev_cc, idx]
            K[sss, rr, cc, idx] += K[prev_ss, prev_rr, prev_cc, idx]


def centered_transfer_value_and_gradient(double[:] f, int[:,:,:] labels, int nlabels, double[:,:,:] y, int radius):
    """ Computes the value and gradient of the ECC energy w.r.t. the transfer function
    The execution time is Theta(N^3 * L), where N^3 is the number of voxels and L is
    the number of quantization levels

    """
    cdef:
        int ns = labels.shape[0]
        int nr = labels.shape[1]
        int nc = labels.shape[2]
        int side = 2 * radius + 1
        int n = side * side * side
        int s, r, c, sss, ss, rr, cc, sides, sider, sidec
        int k, i, j, start_k, end_k, start_i, end_i, start_j, end_j
        int firsts, firstr, firstc, lasts, lastr, lastc
        int cnt, nwindows, idx
        double energy, num, delta, contrib
        double sx, sx2, alpha, beta, gamma
        int[:,:,:,:] K = np.ndarray((2, nr, nc, nlabels,), dtype=np.int32)
        double[:,:,:,:] A = np.ndarray((2, nr, nc, nlabels,), dtype=np.float64)
        double[:,:,:,:] S = np.ndarray((2, nr, nc, 2,), dtype=np.float64)
        double[:] grad = np.ndarray((nlabels,), dtype=np.float64)

    with nogil:
        sss = 1
        for s in range(ns+radius):
            ss = _mod(s - radius, ns)
            sss = 1 - sss
            firsts = _int_max(0, ss - radius)
            lasts = _int_min(ns - 1, ss + radius)
            sides = (lasts - firsts + 1)
            for r in range(nr+radius):
                rr = _mod(r - radius, nr)
                firstr = _int_max(0, rr - radius)
                lastr = _int_min(nr - 1, rr + radius)
                sider = (lastr - firstr + 1)
                for c in range(nc+radius):
                    cc = _mod(c - radius, nc)
                    # New corner
                    _add_elementary_vector(y[s, r, c], labels[s, r, c], A, K, S, sss, rr, cc, 0)
                    # Add signed sub-volumes
                    if s>0:
                        prev_ss = 1 - sss
                        _update_factors(A, K, S, sss, rr, cc, prev_ss, rr, cc, 1)
                        if r>0:
                            prev_rr = _mod(rr-1, nr)
                            _update_factors(A, K, S, sss, rr, cc, prev_ss, prev_rr, cc, -1)
                            if c>0:
                                prev_cc = _mod(cc-1, nc)
                                _update_factors(A, K, S, sss, rr, cc, prev_ss, prev_rr, prev_cc, 1)
                        if c>0:
                            prev_cc = _mod(cc-1, nc)
                            _update_factors(A, K, S, sss, rr, cc, prev_ss, rr, prev_cc, -1)
                    if(r>0):
                        prev_rr = _mod(rr-1, nr)
                        _update_factors(A, K, S, sss, rr, cc, sss, prev_rr, cc, 1)
                        if(c>0):
                            prev_cc = _mod(cc-1, nc)
                            _update_factors(A, K, S, sss, rr, cc, sss, prev_rr, prev_cc, -1)
                    if(c>0):
                        prev_cc = _mod(cc-1, nc)
                        _update_factors(A, K, S, sss, rr, cc, sss, rr, prev_cc, 1)
                    # Add signed corners
                    if s>=side:
                        _add_elementary_vector(y[s-side, r, c], labels[s-side, r, c], A, K, S, sss, rr, cc, -1)
                        if r>=side:
                            _add_elementary_vector(y[s-side, r-side, c], labels[s-side, r-side, c], A, K, S, sss, rr, cc, 1)
                            if c>=side:
                                _add_elementary_vector(y[s-side, r-side, c-side], labels[s-side, r-side, c-side], A, K, S, sss, rr, cc, -1)
                        if c>=side:
                            _add_elementary_vector(y[s-side, r, c-side], labels[s-side, r, c-side], A, K, S, sss, rr, cc, 1)
                    if r>=side:
                        _add_elementary_vector(y[s, r-side, c], labels[s, r-side, c], A, K, S, sss, rr, cc, -1)
                        if c>=side:
                            _add_elementary_vector(y[s, r-side, c-side], labels[s, r-side, c-side], A, K, S, sss, rr, cc, 1)
                    if c>=side:
                        _add_elementary_vector(y[s, r, c-side], labels[s, r, c-side], A, K, S, sss, rr, cc, -1)

                    if s>=radius and r>=radius and c>=radius:
                        firstc = _int_max(0, cc - radius)
                        lastc = _int_min(nc - 1, cc + radius)
                        sidec = (lastc - firstc + 1)
                        cnt = sides*sider*sidec
                        # Compute contribution of this window to the total energy
                        double prod_af, muJ, ftDf, ftk

                        muJ = S[sss, rr, cc, 0]/cnt
                        prod_af = 0
                        ftDf = 0
                        ftk = 0

                        for idx in range(nlabels):
                            prod_af += (A[sss, rr, cc, idx] - muJ) * f[idx]
                            ftDf += f[idx] * f[idx] * K[sss, rr, cc, idx]
                            ftk += f[idx] * K[sss, rr, cc, idx]






                    num = n*alpha*alpha - 2.0*alpha*beta*sx + gamma*sx*sx
                    energy += sx2 - num/delta

                    # Compute contribution of this window to the gradient
                    for i in range(0, nlabels):
                        contrib = 2*(n*alpha*A[i] - sx*(alpha*K[i]+beta*A[i]) + sx*sx*K[i]*f[i])*delta - 2*num*(n*K[i]*f[i] - beta*K[i])
                        grad[i] -= contrib/(delta*delta)
        energy /= nwindows
        for i in range(nlabels):
            grad[i] /= nwindows
    return energy, grad