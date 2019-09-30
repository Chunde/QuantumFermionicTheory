from mmfutils.math import bessel
import numpy as np


class Basis(object):
    pass


class CylindricalBasis(Basis):
    eps = 7./3 - 4./3 -1  # machine precision
    m = hbar = 1

    def __init__(self, N_root=None, R_max=None, K_max=None, a0=None, nu=0, dim=3, **args):
        """
        Parameters
        --------------
        N_root: int
            number of roots
        R_max: float
            max radius range
        K_max: float
            momentum cutoff
        a0: float
            wavefunction position scale
        nu: int
            angular momentum quantum number
        dim: int
            dimensionality
        """
        if N_root is None or R_max is None or K_max is None:
            self._init(a0=a0)
        else:
            self.N_root = N_root
            self.R_max = R_max
            self.K_max = K_max
        self._align_K_max()
        self.dim = dim
        self.nu = nu
        self.zs = self.get_zs(nu=nu)
        self.rs = self.get_rs(zs=self.zs)
        self.K = self.get_K(zs=self.zs, nu=nu)
        self.rs_scale = self._rs_scaling_factor(zs=self.zs)

    def _init(self, a0=None):
        """evaluate R_max and K_max using Gaussian wavefunction"""
        if a0 is None:
            a0 = 1
        self.R_max = np.sqrt(-2*a0**2*np.log(self.eps))
        self.K_max = np.sqrt(-np.log(self.eps)/a0**2)
        self.N_root = int(np.ceil(self.K_max*2*self.R_max/np.pi))

    def _align_K_max(self):
        """
        For large n, the roots of the bessel function are approximately
        z[n] = (n + 0.75)*pi, so R = R_max = z_max/K_max = (N-0.25)*pi/K_max
        """
        self.K_max = (self.N_root - 0.25)*np.pi/self.R_max
    
    def get_zs(self, nu=None):
        """
        return roots for order $\nu$
        """
        if nu is None:
            nu = self.nu
        zs = bessel.j_root(nu=nu, N=self.N_root)
        return zs

    def get_rs(self, zs=None, nu=None):
        """
        return cooridnate in postition space
        """
        if nu is None:
            nu = self.nu
        if zs is None:
            zs = self.get_zs(nu=nu)
        return zs/self.K_max

    def _rs_scaling_factor(self, zs=None):
        if zs is None:
            zs = self.zs
        rs = self.get_rs(zs=zs)
        rs_ = rs**((self.dim - 1)/2.0)
        return rs_

    def get_u(self, psi):
        """convert psi to u"""
        rs_ = self._rs_scaling_factor()
        u = psi*rs_
        return u
    
    def get_psi(self, u):
        """convert u to psi"""
        rs_ = self._rs_scaling_factor()
        psi = u/rs_
        return psi

    def get_nu(self, nu=None):
        """
         `nu + d/2 - 1` for the centrifugal term
         Note:
            the naming convention use \nu as angular momentum quantum number
            but it's also being used as the order number of bessel function
         """
        if nu is None:
            nu = self.nu
        return nu + self.dim/2.0 - 1

    def get_K(self, zs=None, nu=None):
        """
        return the kinetic matrix for a given nu
        Note: the centrifugal potential is already include
        """
        if nu is None:
            nu = self.nu
        if zs is None:
            zs = self.get_zs(nu=nu)
        zi = np.array(list(range(len(zs)))) + 1
        xx, yy = np.meshgrid(zi, zi, sparse=False, indexing='ij')
        zx, zy = np.meshgrid(zs, zs, sparse=False, indexing='ij')
        nu = self.get_nu(nu)  # see get_nu(...)
        K_diag = (1+2*(nu**2 - 1)/zs**2)/3.0  # diagonal terms
        K_off = 8*(-1)**(abs(xx - yy))*zx*zy/(zx**2 - zy**2 + self.eps)**2
        np.fill_diagonal(K_off, K_diag)
        K = self.K_max**2*K_off/2.0  # factor of 1/2 include
        return K

    def get_transform_matrix(self, basis_target):
        """
        return the transform matrix that will map current data
        to its representation in the target basis
        """
        assert basis_target.dim == self.dim

    def get_V_correction(self, nu):
        """
            if nu is not the same as the basis, a piece of correction
            should be made to the centrifugal potential
        """
        return (nu**2 - self.nu**2)*self.hbar**2/2.0/self.rs**2

    def get_V_mean_field(self, nu):
        return 0
