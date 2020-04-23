import numpy as np
from mmfutils.math import bessel
import math
import scipy as sp


def prod(x):
    """Equivalent of sum but with multiplication."""
    # http://stackoverflow.com/a/595396/1088938
    return functools.reduce(operator.mul, x, 1)


def ndgrid(*v):
    """Sparse meshgrid with regular ordering.

    Examples
    --------
    >>> ndgrid([1,2])
    ([1, 2],)
    >>> ndgrid([1,2],[1,2,3])
    [array([[1],
           [2]]), array([[1, 2, 3]])]
    """
    if len(v) == 1:
        return v
    else:
        return np.meshgrid(*v, sparse=True, indexing='ij')


def get_xyz(Nxyz, Lxyz, symmetric_lattice=False):
    """Return `(x,y,z,...)` with broadcasting for a periodic lattice.

    Arguments
    ---------
    Nxyz : [int]
       Number of points in each dimension.
    Lxyz : [float]
       Size of periodic box in each dimension.
    symmetric_lattice : bool
       If `True`, then shift the grid so that the origin is in the middle
       but not on the lattice, otherwise the origin is part of the lattice,
       but the lattice will not be symmetric (if `Nxyz` is even as is
       typically the case for performance).
    """
    xyz = []
    # Special case for N = 1 should also always be centered
    _offsets = [0.5 if symmetric_lattice or _N == 1 else 0 for _N in Nxyz]
    xyz = ndgrid(*[_l/_n * (np.arange(-_n/2, _n/2) + _offset)
                   for _n, _l, _offset in zip(Nxyz, Lxyz, _offsets)])
    return xyz


def get_kxyz(Nxyz, Lxyz):
    """Return list of ks in correct order for FFT.

    Arguments
    ---------
    Nxyz : [int]
       Number of points in each dimension.
    Lxyz : [float]
       Size of periodic box in each dimension.
    """
    # Note: Do not kill the single highest momenta... this leads to bad
    # scaling of high-frequency errors.
    kxyz = ndgrid(*[2.0 * np.pi * np.fft.fftfreq(_n, _l/_n)
                    for _n, _l in zip(Nxyz, Lxyz)])
    return kxyz


######################################################################
# 1D FFTs for real functions.
def dst(f, axis=-1):
    """Return the Discrete Sine Transform (DST III) of `f`"""
    args = dict(type=3, axis=axis)
    return sp.fftpack.dst(f, **args)


def idst(F, axis=-1):
    """Return the Inverse Discrete Sine Transform (DST II) of `f`"""
    N = F.shape[-1]
    args = dict(type=2, axis=axis)
    return sp.fftpack.dst(F, **args)/(2.0*N)


class CylindricalBasis(object):
    r"""2D basis for Cylindrical coordinates via a DVR basis.

    This represents 3-dimensional problems with axial symmetry, but only has
    two dimensions (x, r).

    Parameters
    ----------
    Nxr : (Nx, Nr)
       Number of lattice points in basis.
    Lxr : (L, R)
       Size of each dimension (length of box and radius)
    twist : float
       Twist (angle) in periodic dimension.  This adds a constant offset to the
       momenta allowing one to study Bloch waves.
    boost_px : float
       Momentum of moving frame (along the x axis).  Momenta are shifted by
       this, which corresponds to working in a boosted frame with velocity
       `vx = boost_px/m`.
    axes : (int, int)
       Axes in array y which correspond to the x and r axes here.
       This is required for cases where y has additional dimensions.
       The default is the last two axes (best for performance).
    """
    _d = 2                    # Dimension of spherical part (see nu())
    
    def __init__(self, Nxr, Lxr, twist=0, boost_px=0,
                 axes=(-2, -1), symmetric_x=True):
        self.twist = twist
        self.boost_px = np.asarray(boost_px)
        self.Nxr = np.asarray(Nxr)
        self.Lxr = np.asarray(Lxr)
        self.symmetric_x = symmetric_x
        self.axes = np.asarray(axes)
        self.init()

    def init(self):
        Lx, R = self.Lxr
        x = get_xyz(Nxyz=self.Nxr, Lxyz=self.Lxr,
                    symmetric_lattice=self.symmetric_x)[0]
        kx0 = get_kxyz(Nxyz=self.Nxr, Lxyz=self.Lxr)[0]
        self.kx = (kx0 + float(self.twist) / Lx - self.boost_px)
        self._kx0 = kx0
        self._kx2 = self.kx**2

        self.y_twist = np.exp(1j*self.twist*x/Lx)

        Nx, Nr = self.Nxr

        # For large n, the roots of the bessel function are approximately
        # z[n] = (n + 0.75)*pi, so R = r_max = z_max/k_max = (N-0.25)*pi/kmax
        # This self._kmax defines the DVR basis, not self.k_max
        self._kmax = (Nr - 0.25)*np.pi/R

        # This is just the maximum momentum for diagnostics,
        # determining cutoffs etc.
        self.k_max = np.array([abs(self.kx).max(), self._kmax])

        nr = np.arange(Nr)[None, :]
        r = self._r(Nr)[None, :]  # Do this after setting _kmax
        self.xyz = [x, r]

        _lambda = np.asarray(
            [1./(self._F(_nr, _r))**2
             for _nr, _r in zip(nr.ravel(), r.ravel())])[None, :]
        self.metric = 2*np.pi * r * _lambda * (Lx / Nx)
        self.metric.setflags(write=False)
        # Get the DVR kinetic piece for radial component
        K, r1, r2, w = self._get_K()

        # We did not apply the sqrt(r) factors so at this point, K is still
        # Hermitian and we can diagonalize for later exponentiation.
        d, V = sp.linalg.eigh(K)     # K = np.dot(V*d, V.T)

        # Here we convert from the wavefunction Psi(r) to the radial
        # function u(r) = sqrt(r)*Psi(r) and back with factors of sqrt(r).
        K *= r1
        K *= r2

        self.weights = w
        self._Kr = K
        self._Kr_diag = (r1, r2, V, d)   # For use when exponentiating

        # And factor for x.
        self._Kx = self._kx2

        # Cache for K_data from apply_exp_K.
        self._K_data = []

    @property
    def Lx(self):
        return self.Lxr[0]
    
    @property
    def Nx(self):
        return self.Nxr[0]

    ######################################################################
    # IBasisMinimal: Required methods
    def laplacian(self, y, factor=1.0, exp=False, kx2=None, twist_phase_x=None):
        r"""Return the laplacian of y.

        Arguments
        ---------
        factor : float
           Additional factor (mostly used with `exp=True`).  The
           implementation must be careful to allow the factor to
           broadcast across the components.
        exp : bool
           If `True`, then compute the exponential of the laplacian.
           This is used for split evolvers.
        kx2 : array, optional
           Replacement for the default `kx2=kx**2` used when computing the
           "laplacian".  This would allow you, for example, to implement a
           modified dispersion relationship like ``1-cos(kx)`` rather than
           ``kx**2``.
        twist_phase_x : array, optional
           To implement twisted boundary conditions, one needs to remove an
           overall phase from the wavefunction rendering it periodic for use
           the the FFT.  This the the phase that should be removed.  Note: to
           compensate, the momenta should be shifted as well::
        
              -factor * twist_phase_x*ifft((k+k_twist)**2*fft(y/twist_phase_x)
        """
        if not exp:
            return self.apply_K(y=y, kx2=kx2,
                                twist_phase_x=twist_phase_x) * (-factor)
        else:
            return self.apply_exp_K(y=y, factor=-factor, kx2=kx2,
                                    twist_phase_x=twist_phase_x)
    ######################################################################

    def get_gradient(self, y):
        """Returns the gradient along the x axis."""
        kx = self.kx
        return [self.ifft(1j*kx*self.fft(y)), NotImplemented]

    def apply_Lz(self, y, hermitian=False):
        raise NotImplementedError

    def apply_Px(self, y, hermitian=False):
        r"""Apply the Pz operator to y without any px.

        Requires :attr:`_pxyz` to be defined.
        """
        return self.y_twist * self.ifft(self._kx0 * self.fft(y/self.y_twist))

    def apply_exp_K(self, y, factor, kx2=None, twist_phase_x=None):
        r"""Return `exp(K*factor)*y` or return precomputed data if
        `K_data` is `None`.
        """
        if kx2 is None:
            kx2 = self._Kx
        _K_data_max_len = 3
        ind = None
        for _i, (_f, _d) in enumerate(self._K_data):
            if np.allclose(factor, _f):
                ind = _i
        if ind is None:
            _r1, _r2, V, d = self._Kr_diag
            exp_K_r = _r1 * np.dot(V*np.exp(factor * d), V.T) * _r2
            exp_K_x = np.exp(factor * kx2)
            K_data = (exp_K_r, exp_K_x)
            self._K_data.append((factor, K_data))
            ind = -1
            while len(self._K_data) > _K_data_max_len:
                # Reduce storage
                self._K_data.pop(0)

        K_data = self._K_data[ind][1]
        exp_K_r, exp_K_x = K_data
        if twist_phase_x is None or self.twist == 0:
            tmp = self.ifft(exp_K_x * self.fft(y))
        else:
            if twist_phase_x is None:
                twist_phase_x = self.y_twist
            tmp = twist_phase_x*self.ifft(exp_K_x * self.fft(y/twist_phase_x))
        return np.einsum('...ij,...yj->...yi', exp_K_r, tmp)

    def apply_K(self, y, kx2=None, twist_phase_x=None):
        r"""Return `K*y` where `K = k**2/2`"""
        # Here is how the indices work:
        if kx2 is None:
            kx2 = self._Kx

        if twist_phase_x is None or self.twist == 0:
            yt = self.fft(y)
            yt *= kx2
            yt = self.ifft(yt)
        else:
            if twist_phase_x is None:
                twist_phase_x = self.y_twist
            yt = self.fft(y/twist_phase_x)
            yt *= kx2
            yt = self.ifft(yt)
            yt *= twist_phase_x

        # C <- alpha*B*A + beta*C    A = A^T  zSYMM or zHYMM but not supported
        # maybe cvxopt.blas?  Actually, A is not symmetric... so be careful!
        yt += np.dot(y, self._Kr.T)
        return yt

    ######################################################################
    # FFT and DVR Helper functions.
    #
    # These are specific to the basis, defining the kinetic energy
    # matrix for example.

    # We need these wrappers because the state may have additional
    # indices for components etc. in front.
    def fft(self, x):
        """Perform the fft along the x axes"""
        # Makes sure that
        axis = (self.axes % len(x.shape))[0]
        return fft(x, axis=axis)

    def ifft(self, x):
        """Perform the fft along the x axes"""
        axis = (self.axes % len(x.shape))[0]
        return ifft(x, axis=axis)

    def _get_K(self, l=0):
        r"""Return `(K, r1, r2, w)`: the DVR kinetic term for the radial function
        and the appropriate factors for converting to the radial coordinates.

        This term effects the $-d^2/dr^2 - (\nu^2 - 1/4)/r^2$ term.

        Returns
        -------
        K : array
           Operates on radial wavefunctions
        r1, r2 : array
           K*r1*r2 operators on the full wavefunction (but is no longer
           Hermitian)
        w : array
           Quadrature integration weights.
        """
        nu = self.nu(l=l)
        if l == 0:
            r = self.xyz[1].ravel()
        else:
            r = self._r(self.Nxr[1], l=l)
        z = self._kmax * r
        n = np.arange(len(z))
        i1 = (slice(None), None)
        i2 = (None, slice(None))

        # Quadrature weights
        w = 2.0 / (self._kmax * z * bessel.J(nu=nu, d=1)(z)**2)

        # DVR kinetic term for radial function:
        K = np.ma.divide(
            (-1.0)**(n[i1] - n[i2]) * 8.0 * z[i1] * z[i2],
            (z[i1]**2 - z[i2]**2)**2).filled(0)
        K[n, n] = 1.0 / 3.0 * (1.0 + 2.0*(nu**2 - 1.0)/z**2)
        K *= self._kmax**2

        # Here we convert from the wavefunction Psi(r) to the radial
        # function u(r) = sqrt(r)*Psi(r) and back with factors of
        # sqrt(wr).  This includes the integration weights (since K is
        # defined acting on the basis functions).
        # Note: this makes the matrix non-hermitian, so don't do this if you
        # want to diagonalize.
        _tmp = np.sqrt(w*r)
        r2 = _tmp[i2]
        r1 = 1./_tmp[i1]

        return K, r1, r2, w

    def nu(self, l):
        """Return `nu = l + d/2 - 1` for the centrifugal term.

        Arguments
        ---------
        l : int
           Angular quantum number.
        """
        nu = l + self._d/2 - 1
        return nu
        
    def _r(self, N, l=0):
        r"""Return the abscissa."""
        # l=0 cylindrical: nu = l + d/2 - 1
        return bessel.j_root(nu=self.nu(l=l), N=N) / self._kmax

    def _F(self, n, r, d=0):
        r"""Return the dth derivative of the n'th basis function."""
        nu = 0.0                # l=0 cylindrical: nu = l + d/2 - 1
        rn = self.xyz[1].ravel()[n]
        zn = self._kmax*rn
        z = self._kmax*r
        H = bessel.J_sqrt_pole(nu=nu, zn=zn, d=0)
        coeff = math.sqrt(2.0*self._kmax)*(-1.0)**(n + 1)/(1.0 + r/rn)
        if 0 == d:
            return coeff * H(z)
        elif 1 == d:
            dH = bessel.J_sqrt_pole(nu=nu, zn=zn, d=1)
            return coeff * (dH(z) - H(z)/(z + zn)) * self._kmax
        else:
            raise NotImplementedError

    def get_F(self, r):
        """Return a function that can extrapolate a radial
        wavefunction to a new set of abscissa (x, r)."""
        x, r0 = self.xyz
        n = np.arange(r0.size)[:, None]

        # Here is the transform matrix
        _F = self._F(n, r) / self._F(n, r0.T)

        def F(u):
            return np.dot(u, _F)

        return F

    def F(self, u, xr):
        r"""Return u evaluated on the new abscissa (Assumes x does not
        change for now)"""
        x0, r0 = self.xyz
        x, r = xr
        assert np.allclose(x, x0)

        return self.get_F(r)(u)

    def get_Psi(self, r, return_matrix=False):
        """Return a function that can extrapolate a wavefunction to a
        new set of abscissa (x, r).

        This includes the factor of $\sqrt{r}$ that converts the
        wavefunction to the radial function, then uses the basis to
        extrapolate the radial function.

        Arguments
        ---------
        r : array
           The new abscissa in the radial direction (the $x$ values
           stay the same.)
        return_matrix : bool
           If True, then return the extrapolation matrix F so that
           ``Psi = np.dot(psi, F)``
        """
        x, r0 = self.xyz
        n = np.arange(r0.size)[:, None]

        # Here is the transform matrix
        _F = (np.sqrt(r) * self._F(n, r)) / (np.sqrt(r0.T) * self._F(n, r0.T))

        if return_matrix:
            return _F

        def Psi(psi):
            return np.dot(psi, _F)

        return Psi

    def Psi(self, psi, xr):
        r"""Return psi evaluated on the new abscissa (Assumes x does not
        change for now)"""
        x0, r0 = self.xyz
        x, r = xr
        assert np.allclose(x, x0)

        return self.get_Psi(r)(psi)

    def integrate1(self, n):
        """Return the integral of n over y and z."""
        n = np.asarray(n)
        x, r = self.xyz
        x_axis, r_axis = self.axes
        bcast = [None] * len(n.shape)
        bcast[x_axis] = slice(None)
        bcast[r_axis] = slice(None)
        return ((2*np.pi*r * self.weights)[tuple(bcast)] * n).sum(axis=r_axis)

    def integrate2(self, n, y=None, Nz=100):
        """Return the integral of n over z (line-of-sight integral) at y.
        
        This is an Abel transform, and is used to compute the 1D
        line-of-sight integral as would be seen by a photographic
        image through an axial cloud.

        Arguments
        ---------
        n : array
           (Nx, Nr) array of the function to be integrated tabulated
           on the abscissa.  Note: the extrapolation assumes that `n =
           abs(psi)**2` where `psi` is well represented in the basis.
        y : array, None
           Ny points at which the resulting integral should be
           returned.  If not provided, then the function will be
           tabulated at the radial abscissa.
        Nz : int
           Number of points to use in z integral.
        """
        n = np.asarray(n)
        x, r = self.xyz
        if y is None:
            y = r

        y = y.ravel()
        Ny = len(y)

        x_axis, r_axis = self.axes
        y_axis = r_axis
        bcast_y = [None] * len(n.shape)
        bcast_z = [None] * len(n.shape)
        bcast_y[y_axis] = slice(None)
        bcast_y.append(None)
        bcast_z.append(slice(None))

        bcast_y, bcast_z = tuple(bcast_y), tuple(bcast_z)

        z = np.linspace(0, r.max(), Nz)
        shape_xyz = n.shape[:-1] + (Ny, Nz)
        rs = np.sqrt(y.ravel()[bcast_y]**2 + z[bcast_z]**2)
        n_xyz = (abs(self.Psi(np.sqrt(n),
                              (x, rs.ravel())))**2).reshape(shape_xyz)
        n_2D = 2 * np.trapz(n_xyz, z, axis=-1)
        return n_2D

if __name__ == "__main__":
    eps = np.finfo(float).eps
    hbar = m = w = 1
    a0 = np.sqrt(hbar/m/w)
    R = np.sqrt(-2*a0**2*np.log(eps))
    k_max = np.sqrt(-np.log(eps)/a0**2)
    Nr = int(np.ceil(k_max * 2*R / np.pi))

    basis = CylindricalBasis(Nxr=(1, Nr), Lxr=(1.0, R))

    def get_V(r):
        return m*w**2*r**2/2

    l = 0
    l0 = 0
    nu0 = basis.nu(l=l0)
    nu = basis.nu(l=l)

    r = basis._r(Nr, l=l0)
    K = basis._get_K(l=l0)[0]   # Without factors of sqrt(r)

    V = get_V(r) + (nu**2 - nu0**2)/r**2 * hbar**2/2/m
    H = K/2 + np.diag(V)
    assert np.allclose(H, H.T.conj())
    E = np.linalg.eigvalsh(H)