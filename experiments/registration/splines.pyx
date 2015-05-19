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

cdef inline int interpolate_scalar_trilinear(floating[:,:,:] volume,
                                             double dkk, double dii, double djj,
                                             floating *out) nogil:
    r"""Trilinear interpolation of a 3D scalar image

    Interpolates the 3D image at (dkk, dii, djj) and stores the
    result in out. If (dkk, dii, djj) is outside the image's domain,
    zero is written to out instead.

    Parameters
    ----------
    image : array, shape (R, C)
        the input 2D image
    dkk : floating
        the first coordinate of the interpolating position
    dii : floating
        the second coordinate of the interpolating position
    djj : floating
        the third coordinate of the interpolating position
    out : array, shape (2,)
        the array which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dkk, dii, djj) is inside the domain of the image,
        inside == 1, otherwise inside == 0
    """
    cdef:
        int ns = volume.shape[0]
        int nr = volume.shape[1]
        int nc = volume.shape[2]
        int kk, ii, jj
        double alpha, beta, calpha, cbeta, gamma, cgamma
    if not (0 <= dkk <= ns - 1 and 0 <= dii <= nr - 1 and 0 <= djj <= nc - 1):
        out[0] = 0
        return 0
    # find the top left index and the interpolation coefficients
    kk = <int>floor(dkk)
    ii = <int>floor(dii)
    jj = <int>floor(djj)
    # no one is affected
    if not ((0 <= kk < ns) and (0 <= ii < nr) and (0 <= jj < nc)):
        out[0] = 0
        return 0
    cgamma = dkk - kk
    calpha = dii - ii
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta
    gamma = 1 - cgamma
    #---top-left
    out[0] = alpha * beta * gamma * volume[kk, ii, jj]
    #---top-right
    jj += 1
    if(jj < nc):
        out[0] += alpha * cbeta * gamma * volume[kk, ii, jj]
    #---bottom-right
    ii += 1
    if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
        out[0] += calpha * cbeta * gamma * volume[kk, ii, jj]
    #---bottom-left
    jj -= 1
    if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
        out[0] += calpha * beta * gamma * volume[kk, ii, jj]
    kk += 1
    if(kk < ns):
        ii -= 1
        out[0] += alpha * beta * cgamma * volume[kk, ii, jj]
        jj += 1
        if(jj < nc):
            out[0] += alpha * cbeta * cgamma * volume[kk, ii, jj]
        #---bottom-right
        ii += 1
        if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
            out[0] += calpha * cbeta * cgamma * volume[kk, ii, jj]
        #---bottom-left
        jj -= 1
        if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
            out[0] += calpha * beta * cgamma * volume[kk, ii, jj]


cdef class Spline:
    cdef:
        int kspacing # Number of grid cells between spline knots
        int kernel_size # Number of grid cells with non-zero spline value
        int center #  = kernel_size // 2
        double* splines[3] # Precomputed kernel and derivatives

    def __cinit__(self):
        r""" Base class (contract) for all 1D splines
        Each spline must define the following (fast cdef, nogil) methods:

        1. double _kernel(self, double): the spline function
        1. double _derivative(self, double): the derivative of the spline
           function
        2. int _kernel_size(self): the number of grid cells that evaluate to a
           non-zero value, assuming the knot spacing is selg.kspacing

        Note: each derived spline (e.g. CubicSpline) should call
        self._precompute() after setting its kspacing property, it will allocate
        and precompute the spline and derivative for non-zero grid cells
        """
        self.kspacing = -1
        self.kernel_size = -1
        self.center = -1
        self.splines[0] = NULL
        self.splines[1] = NULL
        self.splines[2] = NULL


    cdef double _kernel(self, double x)nogil:
        #This function must be overriden by the derived spline classes
        return -1


    def kernel(self, double x):
        r""" Evaluates the spline kernel at x
        """
        return self._kernel(x)


    cdef double _derivative(self, double x)nogil:
        #This function must be overriden by the derived spline classes
        return -1


    def derivative(self, double x):
        r""" Evaluates the derivative of the spline kernel at x
        """
        return self._derivative(x)

    cdef double _second_derivative(self, double x)nogil:
        #This function must be overriden by the derived spline classes
        return -1


    def second_derivative(self, double x):
        r""" Evaluates the second derivative of the spline kernel at x
        """
        return self._second_derivative(x)


    cdef int _kernel_size(self, int kspacing)nogil:
        #This function must be overriden by the derived spline classes
        return -1


    def get_kernel_size(self):
        r""" Gets the number of grid cells that have non-zero kernel value
        """
        return self.kernel_size


    cdef int _num_overlapping(self)nogil:
        return 1 + 2 * ((self.kernel_size - 1) // self.kspacing)


    def num_overlapping(self):
        r""" Number of knots whose centered spline intersects any spline
        """
        return self._num_overlapping()


    cdef void _precompute(self, int kspacing):
        r""" Precompute the kernel and its derivatives at non-zero grid cells
        Prameters
        ---------
        kspacing : int
            the number of grid cells between spline knots
        """
        cdef:
            int i, j, block_size
            double x, dx, dx2

        self.kernel_size = self._kernel_size(kspacing)
        if self.kernel_size < 0: # It is probably the base class
            self.splines[0] = NULL
            self.splines[1] = NULL
            self.splines[2] = NULL
            raise ValueError("Invalid spline class.")

        block_size = self.kernel_size * sizeof(double)
        self.center = self.kernel_size // 2
        self.kspacing = kspacing

        # Precompute spline at non-zero grid cells
        for i in range(3):
            self.splines[i] = <double*>PyMem_Malloc(block_size)
            if not self.splines[i]:
                # Free previously allocated blocks
                for j in range(i):
                    PyMem_Free(self.splines[j])
                # And raise exception
                raise MemoryError()

        dx = 1.0 / kspacing
        dx2 = dx * dx
        x = -1.0 * self.center * dx
        for i in range(self.kernel_size):
            self.splines[0][i] = self._kernel(x)
            self.splines[1][i] = self._derivative(x) * dx
            self.splines[2][i] = self._second_derivative(x) * dx2
            x += dx


    cdef void _overlap_offsets(self, int idx0, int idx1, int *begin0,
                               int *begin1, int *overlap_len) nogil:
        if idx0 < idx1: # then idx0-th spline requires offset
            begin0[0] = (idx1 - idx0) * self.kspacing
            begin1[0] = 0
            overlap_len[0] = self.kernel_size - begin0[0]
        else:
            begin0[0] = 0
            begin1[0] = (idx0 - idx1) * self.kspacing
            overlap_len[0] = self.kernel_size - begin1[0]


    def overlap_offsets(self, int idx0, int idx1):
        r""" First cell in the intersection of the idx0-th and idx1-th splines

        We assume that the locations of the spline knots are all multiples of
        kspacing grid cells.

        The following is an example with a cubic spline kernel (equal to zero
        starting from 2 knots away from the center), kspacing=4, and the index
        of the splines are 2 knots (indices) apart from each other. The C
        indicates the kernel center and the k's indicate the position of
        adjacent knots.

                    overlap_len = 7

        Case 1: idx0 < idx1

                      begin0 = 8
                         v
        |0|-|-|k|-|-|-|C|-|-|-|k|-|-|-| # idx0-th Spline kernel
                        |-|-|-|k|-|-|-|C|-|-|-|k|-|-|-| # idx1-th Spline kernel
                         ^
                      begin1 = 0

        Case 2: idx1 < idx0

                      begin0 = 0
                         v
                        |-|-|-|k|-|-|-|C|-|-|-|k|-|-|-| # idx0-th Spline kernel
        |-|-|-|k|-|-|-|C|-|-|-|k|-|-|-| # idx1-th Spline kernel
                         ^
                      begin1 = 8

        Note that overlap_len <= 0 indicates no overlap.
        """
        cdef:
            int begin0, begin1, overlap_len
        self._overlap_offsets(idx0, idx1, &begin0, &begin1, &overlap_len)
        return begin0, begin1, overlap_len


    cdef int _knots_needed(self, int grid_len)nogil:
        return 2 + (grid_len - 1 + self.center) // self.kspacing


    def knots_needed(self, int grid_len):
        r""" Number of spline knots needed to cover grid_len cells

        Every knot i (located at grid cell i*kspacing) affecting the last grid
        cell (grid_len - 1) satisfies:

        (1) i * kspacing - center <= grid_len - 1

        i.e. the left-most affected cell (i*kspacing - center) must not be
        greater than the last grid cell (grid_len - 1)

        The last index i that satisfies (1) is precisely

        (2) i = (grid_len - 1 + center) // kspacing

        Therefore, we need knots at -1, 0, 1, ..., i, for a total of i + 2 knots
        """
        return self._knots_needed(grid_len)

    cdef void _grid_cells_affected(self, int index, int grid_len, int *first,
                                   int *last, int *spline_offset)nogil:
        first[0] = (index - 1) * self.kspacing - self.center
        # If the "first affected" is negative, we have an offset, else it's 0
        spline_offset[0] = -first[0]
        last[0] = (index - 1) * self.kspacing + self.center + 1
        if first[0] < 0:
            first[0] = 0;
        if last[0] > grid_len:
            last[0] = grid_len
        if spline_offset[0] < 0:
            spline_offset[0] = 0


    def grid_cells_affected(self, int index, int grid_len):
        r""" Set of grid cells where spline located at idx-th knot is not zero
        We assume that knots are located at multiples of kspacing. We assume
        that the 0-th spline is located to the left of the grid, at position
        -1*kspacing (this is the first spline that actually affects the grid),
        therefore, the center of the index-th spline is actually
        (index - 1) * kspacing. The following is an example with a cubic spline,
        kspacing=3, the C indicates the center of the kernel, and the k's
        indicate the positions of adjacent knots w.r.t. this kernel:

                    spline_offset = 0
                             v
                            |-|-|k|-|-|C|-|-|k|-|-|
        |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
         ^ ^ ^...            ^         ^         ^                     ^
         0 1 2...          first       ^       last-1               grid_len - 1
                                       ^
                           (index - 1) * kspacing

         However, if the first affected grid cell does not coincide with the
         beginning of the spline, we need to know what is the spline cell
         that coincides with this first affected cell (it is the
         "spline_offset"):

          spline_offset = 2            spline_offset = 0
                 v                             v
            |-|-|k|-|-|C|-|-|k|-|-|           |-|-|k|-|-|C|-|-|k|-|-|
                |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
                 ^               ^             ^             ^
               first           last-1        first         last-1

        Note that first >= last indicates no grid cells are affected
        """
        cdef:
            int first, last, spline_offset
        self._grid_cells_affected(index, grid_len, &first, &last, &spline_offset)


    cpdef get_inversion_matrix(self, int grid_len, double tau):
        cdef:
            int ncoef = self._knots_needed(grid_len)
            double[:,:] At = np.zeros((ncoef, grid_len), dtype=np.float64)
            double[:,:] M
            # StS is S.transpose().dot(S) where
            # S = [[2,-1,0,-1],[0,0,0,0],[0,0,0,0],[-1,0,-1,2.0]]
            # This corresponds to a least squares fit, penalizing the
            # second numeric derivative of the coefficients at the boundary.
            double * StS = [5,-2,1,-4,-2,1,0,1,1,0,1,-2,-4,1,-2,5.0]
            # P indicates the rows/columns affected by the constraints (at the
            # boundary)
            int *P = [0, 1, ncoef - 2, ncoef - 1]
            int i, j, k, first, last, spline_offset
        with nogil:
            for j in range(ncoef):
                self._grid_cells_affected(j, grid_len, &first, &last, &spline_offset)
                for i in range(first, last):
                    At[j, i] = self.splines[0][spline_offset]
                    spline_offset += 1
        M = np.dot(At, np.transpose(At))
        with nogil:
            tau *= tau
            k = 0
            for i in range(4):
                for j in range(4):
                    M[P[i], P[j]] += StS[k] * tau
                    k += 1
        M = np.linalg.inv(M).dot(At)
        return M


    def fit_to_data(self, double[:] data, double tau = 0.0001):
        M = self.get_inversion_matrix(data.shape[0], tau)
        out = np.dot(M, data)
        return out


    cdef void _evaluate(self, double[:] coef, int der, double[:] out)nogil:
        cdef:
            int grid_len = out.shape[0]
            int ncoef = coef.shape[0]
            int first, last, spline_offset
            int i, j
        out[:] = 0
        for i in range(ncoef):
            self._grid_cells_affected(i, grid_len, &first, &last, &spline_offset)
            for j in range(first, last):
                out[j] += coef[i] * self.splines[der][spline_offset]
                spline_offset += 1


    def evaluate(self, double[:] coef, int grid_len, int der=0):
        cdef:
            double[:] out = np.empty(grid_len, dtype=np.float64)
        self._evaluate(coef, der, out)
        return out

    cpdef get_kernel_grid(self, int der_order):
        cdef:
            int n = self.kernel_size
            int i
            double[:] out = np.ndarray(n, dtype=np.float64)
            double *selected = self.splines[der_order]
        with nogil:
            for i in range(n):
                out[i] = selected[i]
        return out


    def __dealloc__(self):
        cdef:
            int i
        for i in range(3):
            PyMem_Free(self.splines[i])


cdef class CubicSpline(Spline):
    def __cinit__(self, int kspacing):
        self._precompute(kspacing)


    cdef double _kernel(self, double x)nogil:
        r''' Cubic B-Spline evaluated at x
        See eq. (3) of [1].
        [1] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank, W.
            PET-CT image registration in the chest using free-form deformations.
            IEEE Transactions on Medical Imaging, 22(1), 120–8, 2003.
        '''
        cdef:
            double absx = -x if x < 0.0 else x
            double sqrx = x * x

        if absx < 1.0:
            return ( 4.0 - 6.0 * sqrx + 3.0 * sqrx * absx ) / 6.0
        elif absx < 2.0:
            return ( 8.0 - 12 * absx + 6.0 * sqrx - sqrx * absx ) / 6.0
        return 0.0


    cdef double _derivative(self, double x)nogil:
        r''' Derivative of cubic B-Spline evaluated at x
        See eq. (3) of [1].
        [1] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank, W.
            PET-CT image registration in the chest using free-form deformations.
            IEEE Transactions on Medical Imaging, 22(1), 120–8, 2003.
        '''
        cdef:
            double absx = -x if x < 0.0 else x
            double sqrx = x * x
        if absx < 1.0:
            if x >= 0.0:
                return -2.0 * x + 1.5 * x * x
            else:
                return -2.0 * x - 1.5 * x * x
        elif absx < 2.0:
            if x >= 0:
                return -2.0 + 2.0 * x - 0.5 * x * x
            else:
                return 2.0 + 2.0 * x + 0.5 * x * x
        return 0.0

    cdef double _second_derivative(self, double x) nogil:
        r''' Second derivative of cubic B-Spline evaluated at x
        See eq. (3) of [1].
        [1] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank, W.
            PET-CT image registration in the chest using free-form deformations.
            IEEE Transactions on Medical Imaging, 22(1), 120–8, 2003.
        '''
        cdef:
            double absx = -x if x < 0.0 else x
            double sqrx = x * x
        if absx < 1.0:
            return 3.0 * absx - 2.0
        elif absx < 2:
            return 2 - absx
        return 0


    cdef int _kernel_size(self, int kspacing)nogil:
        r""" Number of grid cells where spline is not zero

        The cubic splin is zero for all |x|>=2, which means that the first
        cero grid cells are exactly two knots away from the center. Beyond those
        knots, the spline is zero. Therefore, the kernel size is

        (2 * kspacing - 1) + 1 + (2 * kspacing - 1)    =   4 *  kspacing - 1
        ------------------  ---  ------------------
                 ^           ^            ^
             left side    center     right side

        The following is an example with kspacing = 4:
        first non-zero cell          last non-zero cell
                 v                           v
            |-|k|-|-|-|k|-|-|-|K|-|-|-|k|-|-|-|k|-|
                       ^       ^       ^
                      knot   center   knot
        """
        return 4 * kspacing - 1


cdef class Spline3D:
    cdef:
        Spline sx
        Spline sy
        Spline sz
        double *splines[27] # precomputed spline and partial derivatives
        int col_size, slice_size, kernel_size


    def __cinit__(self, Spline sx, Spline sy, Spline sz):
        cdef:
            int nx, ny, nz, x, y, z, pos, order
            int block_size
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.col_size = sz.kernel_size
        self.slice_size = sy.kernel_size * sz.kernel_size
        self.kernel_size = sx.kernel_size * sy.kernel_size * sz.kernel_size
        block_size = self.kernel_size * sizeof(double)

        # Precompute spline and partial derivatives
        order = 0 # Derivative order (0 to 26)
        for dx in range(3):
            for dy in range(3):
                for dz in range(3):
                    self.splines[order] = <double*>PyMem_Malloc(block_size)
                    if not self.splines[order]:
                        # Free previously allocated blocks
                        for pos in range(order):
                            PyMem_Free(self.splines[pos])
                        # And raise exception
                        raise MemoryError()
                    # Evaluate the tensor product splines
                    pos = 0
                    for x in range(sx.kernel_size):
                        for y in range(sy.kernel_size):
                            for z in range(sz.kernel_size):
                                self.splines[order][pos] = sx.splines[dx][x] *\
                                                           sy.splines[dy][y] *\
                                                           sz.splines[dz][z]
                                pos += 1
                    order += 1


    cdef double* _get_spline(self, int dx, int dy, int dz)nogil:
        return self.splines[9 * dx + 3 * dy + dz]


    cpdef _eval_spline_field(self, double[:,:,:] coefs, int[:] der,
                                 double[:,:,:] out):
        cdef:
            int len_x = out.shape[0]
            int len_y = out.shape[1]
            int len_z = out.shape[2]
            int i, j, k, ii, jj, kk, iii, jjj, kkk
            int first[3]
            int last[3]
            int offset[3]
            double *spline = self._get_spline(der[0], der[1], der[2])
            double coef
        with nogil:
            for i in range(coefs.shape[0]):
                self.sx._grid_cells_affected(i, len_x, &first[0], &last[0], &offset[0])
                for j in range(coefs.shape[1]):
                    self.sy._grid_cells_affected(j, len_y, &first[1], &last[1], &offset[1])
                    for k in range(coefs.shape[2]):
                        self.sz._grid_cells_affected(k, len_z, &first[2], &last[2], &offset[2])

                        coef = coefs[i, j, k]
                        # Add contribution of this coefficient to all neighboring voxels
                        for ii in range(first[0], last[0]):
                            iii = offset[0] + ii - first[0]
                            for jj in range(first[1], last[1]):
                                jjj = offset[1] + jj - first[1]
                                for kk in range(first[2], last[2]):
                                    kkk = offset[2] + kk - first[2]
                                    out[ii, jj, kk] += coef *\
                                        spline[iii * self.slice_size +\
                                               jjj * self.col_size + kkk]


    def evaluate(self, double[:,:,:] coefs, int[:] shape, ders=None):
        cdef:
            double[:,:,:] out = np.zeros((shape[0], shape[1], shape[2]),
                                           dtype=np.float64)
        if ders is None:
            ders = np.array([0,0,0], dtype=np.int32)
        if not np.all(ders>=0) or not np.all(ders<2) or ders.shape[0] != 3:
            raise ValueError("Invalid derivative orders")
        self._eval_spline_field(coefs, ders, out)
        return out


    cdef int _knots_needed(self, int lenx, int leny, int lenz)nogil:
        cdef:
            int prod
        prod = self.sx._knots_needed(lenx) *\
               self.sy._knots_needed(leny) *\
               self.sz._knots_needed(lenz)
        return prod

    def knots_needed(self, int lenx, int leny, int lenz):
        return self._knots_needed(lenx, leny, lenz)

    cpdef fit_axis(self, double[:,:,:] vol, int axis, double tau=0.0001):
        cdef:
            int lx = vol.shape[0]
            int ly = vol.shape[1]
            int lz = vol.shape[2]
            int x, y, z, nkx, nky, nkz
            double[:,:,:] out
            double[:,:] M
            double[:] coef
        if axis == 0:
            nkx = self.sx._knots_needed(lx)
            out = np.ndarray((nkx, ly, lz), dtype=np.float64)
            M = self.sx.get_inversion_matrix(lx, tau)
            for y in range(ly):
                for z in range(lz):
                    coef = np.dot(M, vol[:,y,z])
                    with nogil:
                        for x in range(coef.shape[0]):
                            out[x,y,z] = coef[x]
            return out
        elif axis == 1:
            nky = self.sy._knots_needed(ly)
            out = np.ndarray((lx, nky, lz), dtype=np.float64)
            M = self.sy.get_inversion_matrix(ly, tau)
            for x in range(lx):
                for z in range(lz):
                    coef = np.dot(M, vol[x,:,z])
                    with nogil:
                        for y in range(nky):
                            out[x,y,z] = coef[y]
            return out
        elif axis == 2:
            nkz = self.sz._knots_needed(lz)
            out = np.ndarray((lx, ly, nkz), dtype=np.float64)
            M = self.sz.get_inversion_matrix(lz, tau)
            for x in range(lx):
                for y in range(ly):
                    coef = np.dot(M, vol[x,y,:])
                    with nogil:
                        for z in range(nkz):
                            out[x,y,z] = coef[z]
            return out
        return None

    cpdef fit_to_data(self, double[:,:,:] vol, double tau=0.0001):
        coef = self.fit_axis(vol, 0, tau)
        coef = self.fit_axis(coef, 1, tau)
        coef = self.fit_axis(coef, 2, tau)
        return coef

    cdef void _get_kernel_grid(self, int dx, int dy, int dz, double[:,:,:] out)nogil:
        cdef:
            int nx = self.sx.kernel_size
            int ny = self.sy.kernel_size
            int nz = self.sz.kernel_size
            int x,y,z, idx
            double *spline = self._get_spline(dx, dy, dz)
        with nogil:
            idx = 0
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        out[x,y,z] = spline[idx]
                        idx += 1

    def get_kernel_grid(self, der_orders=(0,0,0)):
        cdef:
            int der_index
            int nx = self.sx.kernel_size
            int ny = self.sy.kernel_size
            int nz = self.sz.kernel_size
            double[:,:,:] out = np.ndarray((nx,ny,nz), dtype=np.float64)
        if np.min(der_orders) < 0 or np.max(der_orders) > 2:
            raise ValueError("Invalid derivative orders.")
        self._get_kernel_grid(der_orders[0], der_orders[1], der_orders[2], out)
        return out


    cdef void _get_all_autocorrelations(self, int dx, int dy, int dz, double[:] out)nogil:
        r""" Sum of products of overlapping splines
        """
        cdef:
            int nnx = self.sx._num_overlapping()
            int nny = self.sy._num_overlapping()
            int nnz = self.sz._num_overlapping()
            int nx, ny, nz
            int cx = nnx // 2
            int cy = nny // 2
            int cz = nnz // 2
            int *begin0 = [0, 0, 0]
            int *begin1 = [0, 0, 0]
            int *olen = [0, 0, 0]
            double *kernel = self._get_spline(dx, dy, dz)
            int i, j, k, idx0, idx1, out_idx
            double s

        out_idx = 0
        for nx in range(nnx):
            self.sx._overlap_offsets(nx, cx, &begin0[0], &begin1[0], &olen[0])
            for ny in range(nny):
                self.sy._overlap_offsets(ny, cy, &begin0[1], &begin1[1], &olen[1])
                for nz in range(nnz):
                    self.sz._overlap_offsets(nz, cz, &begin0[2], &begin1[2], &olen[2])

                    # sum of products of cells elements
                    s = 0
                    for i in range(olen[0]):
                        for j in range(olen[1]):
                            idx0 = (begin0[0] + i) * self.slice_size +\
                                   (begin0[1] + j) * self.col_size + begin0[2]
                            idx1 = (begin1[0] + i) * self.slice_size +\
                                   (begin1[1] + j) * self.col_size + begin1[2]
                            for k in range(olen[2]):
                                s += kernel[idx0] * kernel[idx1]
                                idx0 += 1
                                idx1 += 1

                    out[out_idx] = s
                    out_idx += 1


    cdef void _get_bending_system(self, double[:,:,:] coef, double[:] vox_size,
                                  double[:] grad, double[:] data, int[:] indices,
                                  int[:] indptr):
        cdef:
            int nnx = self.sx._num_overlapping()
            int nny = self.sy._num_overlapping()
            int nnz = self.sz._num_overlapping()
            int nx, ny, nz
            int cx = nnx // 2
            int cy = nny // 2
            int cz = nnz // 2

            int ddir1, ddir2
            int *der = [0, 0, 0]
            double[:] prods = np.ndarray(nnx*nny*nnz, dtype=np.float64)

            int ncx = coef.shape[0]
            int ncy = coef.shape[1]
            int ncz = coef.shape[2]
            int i, j, k, ii, jj, kk, cnt
            int row, col
            double mult # twice the multiplicity of the cross derivatives
            double sz_norm # normalization factor to compensate for voxel size

        with nogil:
            for ddir1 in range(3):
                for ddir2 in range(ddir1, 3):
                    der[0] = 0
                    der[1] = 0
                    der[2] = 0
                    der[ddir1] += 1
                    der[ddir2] += 1
                    self._get_all_autocorrelations(der[0], der[1], der[2], prods)
                    sz_norm = 1.0 / (vox_size[ddir1] * vox_size[ddir2])
                    if ddir1 == ddir2:
                        mult = 2.0
                    else:
                        mult = 4.0

                    # Accumulate this second order derivative
                    # Iterate over all coefficients
                    row = 0
                    cnt = 0
                    for i in range(ncx):
                        for j in range(ncy):
                            for k in range(ncz):
                                indptr[row] = cnt
                                # Look for all overlapping (neighboring) splines
                                for ii in range(i - cx, i + cx + 1):
                                    if ii < 0 or ii >= ncx:
                                        continue
                                    for jj in range(j - cy, j + cy + 1):
                                        if jj < 0 or jj >= ncy:
                                            continue
                                        for kk in range(k - cz, k + cz + 1):
                                            if kk < 0 or kk >= ncz:
                                                continue
                                            col = ii * ncy*ncz + jj * ncz + kk
                                            idx = (ii - i + cx)*nny*nnz +\
                                                  (jj - j + cy)*nnz +\
                                                  (kk - k + cz)
                                            grad[row] += mult * sz_norm * coef[ii,jj,kk] * prods[idx]

                                            data[cnt] += mult * sz_norm * prods[idx]
                                            indices[cnt] = col

                                            cnt += 1
                                row += 1
                    indptr[row] = cnt


    def get_bending_system(self, double[:,:,:] coef, double[:] vox_size):
        cdef:
            double[:] grad = np.zeros(coef.size)
            int ncoef = coef.shape[0] * coef.shape[1] * coef.shape[2]
            int n_overlaps = self.sx._num_overlapping() *\
                             self.sy._num_overlapping() *\
                             self.sz._num_overlapping()
            int max_nz = ncoef * n_overlaps
            double[:] data = np.zeros(max_nz, dtype=np.float64)
            int[:] indices = np.zeros(max_nz, dtype=np.int32)
            int[:] indptr = np.zeros(ncoef + 1, dtype=np.int32)

        self._get_bending_system(coef, vox_size, grad, data, indices, indptr)

        data = np.array(data)
        indices = np.array(indices)
        indptr = np.array(indptr)

        #hessian = sp.sparse.csr_matrix((data, indices, indptr), shape=(ncoef, ncoef))
        return grad, data, indices, indptr


    def __dealloc__(self):
        cdef:
            int i
        for i in range(27):
            PyMem_Free(self.splines[i])



cpdef wrap_scalar_field(double[:] v, int[:] sh):
    cdef:
        double[:,:,:] vol = np.ndarray((sh[0], sh[1], sh[2]), dtype=np.float64)
        int i, j, k, idx
    if sh[0]*sh[1]*sh[2] != len(v):
        raise ValueError("Wrong number of coefficients")
    with nogil:
        idx = 0
        for k in range(sh[0]):
            for i in range(sh[1]):
                for j in range(sh[2]):
                    vol[k,i,j] = v[idx]
                    idx += 1
    return vol


class CubicSplineField:
    def __init__(self, vol_shape, kspacing):
        cdef:
            int nkx, nky, nkz
        sx = CubicSpline(kspacing[0])
        sy = CubicSpline(kspacing[1])
        sz = CubicSpline(kspacing[2])
        nkx = sx._knots_needed(vol_shape[0])
        nky = sy._knots_needed(vol_shape[1])
        nkz = sz._knots_needed(vol_shape[2])

        self.spline3d = Spline3D(sx, sy, sz)
        self.grid_shape = np.array([nkx, nky, nkz], dtype=np.int32)
        self.vol_shape = vol_shape
        self.kspacing = kspacing

    def num_coefficients(self):
        return self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]

    def set_coefficients(self, coef):
        if len(coef.shape) != len(self.grid_shape):
            raise ValueError("Coefficient field dimension mismatch")
        if not np.all(coef.shape == self.grid_shape):
            raise ValueError("Coefficient field dimension mismatch")
        self.coef = coef

    def copy_coefficients(self, coef):
        if len(coef.shape) == 1: # Vectorized form
            if coef.size != self.num_coefficients():
                raise ValueError("Coefficient field dimension mismatch")
            self.coef = wrap_scalar_field(coef, self.grid_shape)
            return

        if len(coef.shape) != len(self.grid_shape):
            raise ValueError("Coefficient field dimension mismatch")
        if not np.all(coef.shape == self.grid_shape):
            raise ValueError("Coefficient field dimension mismatch")
        self.coef = coef.copy()

    def get_volume(self, der_orders = (0,0,0)):
        volume = np.zeros(tuple(self.vol_shape), dtype=np.float64)
        self.spline3d._eval_spline_field(self.coef,
                                         np.array(der_orders, dtype=np.int32),
                                         volume)
        return np.array(volume)



def regrid(floating[:,:,:]vol, double[:] factors, int[:] new_shape):
    ftype=np.asarray(vol).dtype
    cdef:
        int ns = new_shape[0]
        int nr = new_shape[1]
        int nc = new_shape[2]
        int k,i,j
        double kk, ii, jj
        floating[:,:,:] out = np.ndarray((ns, nr, nc), dtype=ftype)
    with nogil:
        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    kk = k * factors[0]
                    ii = i * factors[1]
                    jj = j * factors[2]

                    interpolate_scalar_trilinear(vol, kk, ii, jj, &out[k,i,j])
    return out
