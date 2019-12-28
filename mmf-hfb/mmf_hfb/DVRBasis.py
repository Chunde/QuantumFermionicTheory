from mmfutils.math import bessel
import numpy as np


def nan0(data):
    """convert nan to zero"""
    return np.nan_to_num(data, 0)


class CylindricalBasis(object):
    eps = 7./3 - 4./3 -1  # machine precision
    m = hbar = 1
    N_root_max = 128

    def __init__(self, R_max=None, N_root=None, K_max=None, a0=None, nu=0, **args):
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
        self.dim = 2
        self.nu = nu
        self.R_max = R_max
        self.K_max = K_max
        self.N_root = N_root
        self.a0 = a0
        self.init()
        
    def init(self):
        # if N_root is None or R_max is None or K_max is None:
        self.get_N_K_R(a0=self.a0)
        self.zs = self.get_zs(nu=self.nu)
        self.rs = self.get_rs(zs=self.zs)
        self.K = self.get_K(zs=self.zs, nu=self.nu)
        self.zero = np.zeros_like(self.zs)
        self.rs_scale = self._rs_scaling_factor(zs=self.zs)
        self.ws = self.get_F_rs()/self.rs_scale  # weight

    def get_N_K_R(self, a0=None):
        """evaluate R_max and K_max using Gaussian wavefunction"""
        if a0 is None:
            a0 = 1
        if self.R_max is None:
            self.R_max = np.sqrt(-2*a0**2*np.log(self.eps))
        if self.K_max is None:
            self.K_max = np.sqrt(-np.log(self.eps)/a0**2)
        # if self.N_root is None:
        #     self.N_root = int(np.ceil(self.K_max*2*self.R_max/np.pi))
        if self.N_root is None:
            n_ = 0
            zs = self.get_zs(N=self.N_root_max)
            for z in zs:
                if z <= self.K_max*self.R_max:
                    n_ += 1
                else:
                    break
            self.N_root = n_
        else:  # if N_root is specified, we need to change k_max and keep R_max unchanged
            self._align_K_max()

    def _align_K_max(self):
        """
        For large n, the roots of the bessel function are approximately
        z[n] = (n + 0.75)*pi, so R = R_max = z_max/K_max = (N-0.25)*pi/K_max
        """
        self.K_max = (self.N_root - 0.25)*np.pi/self.R_max
    
    def get_zs(self, nu=None, N=None):
        """
        return roots for order $\nu$
        """
        if nu is None:
            nu = self.nu
        if N is None:
            N = self.N_root
        zs = bessel.j_root(nu=nu, N=N)
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

    def get_F(self, nu=None, n=0, rs=None):
        """return the nth basis function for nu"""
        if nu is None:
            nu = self.nu
        if rs is None:
            rs = self.rs
        zs = self.get_zs(nu=nu)
        F = (-1)**(n+1)*self.K_max*zs[n]*np.sqrt(2*rs)/(
            self.K_max**2*rs**2-zs[n]**2)*bessel.J(nu, 0)(self.K_max*rs)
        F=nan0(F)
        return F

    def get_F_rs(self, zs=None, nu=None):
        """
        return the basis function values at abscissa r_n
        for the nth basis function. Since each basis function
        have non-zero value only at its own r_n, we just need to
        compute that value, all other values are simply zero

        """
        if nu is None:
            nu = self.nu
        if zs is None:
            rs = self.rs
            zs = self.zs
        else:
            rs = zs/self.K_max
        Fs = [(-1)**(n+1)*self.K_max*np.sqrt(2*rs[n]*zs[n])/(2*zs[n])*bessel.J_sqrt_pole(
            nu=nu, zn=zs[n])(zs[n]) for n in range(len(rs))]
        return Fs

    def _rs_scaling_factor(self, zs=None):
        """
        the dimension dependent scaling factor used to
        convert from u(r) to psi(r)=u(r)/rs_, or u(r)=psi(r)*rs_
        """
        if zs is None:
            zs = self.zs
        rs = self.get_rs(zs=zs)
        rs_ = rs**((self.dim - 1)/2.0)
        return rs_
        
    def _get_psi(self, u):
        """
        apply weight on the u(v) to get the actual radial wave-function
        ----------
        NOTE: divided by a factor of $\sqrt(2\pi)$ because that $2 \pi$ is
            from the angular integration in 2D
        """
        return u*self.ws/((2*np.pi)**0.5)  # the normalization factor sqrt(2pi)

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

    def get_V_correction(self, nu):
        """
            if nu is not the same as the basis, a piece of correction
            should be made to the centrifugal potential
        """
        return (nu**2 - self.nu**2)*self.hbar**2/2.0/self.rs**2


class HarmonicDVR(CylindricalBasis):
    m=hbar=w=1
    eps = 7./3 - 4./3 -1  # machine accuracy

    def __init__(self, w=1, nu=0, dim=2, **args):
        CylindricalBasis.__init__(self, nu=nu, dim=dim, **args)
        self.w = w

    def get_V(self):
        """return the external potential"""
        r2 = (self.rs)**2
        return self.w**2*r2/2

    def get_H(self, nu=None):
        if nu is None:
            nu = self.nu
        K = self.K
        V = self.get_V()
        V_corr = self.get_V_correction(nu=nu)
        H = K + np.diag(V + V_corr)
        return H


if __name__ == "__main__":
    dvr0 = CylindricalBasis(nu=0, R_max=9, N_root=49)
    dvr1 = CylindricalBasis(nu=1, R_max=9, N_root=48)
    z0 = dvr0.zs
    z1 = dvr1.zs
    a = np.sin(z1)/np.sqrt(z1)
    b = -np.cos(z0)/np.sqrt(z0)
    U10 = []
    for j in range(len(z0)):
        for i in range(len(z1)):
            v = 2*b[j]*(z0[j]*z1[i])**0.5/a[i]/(z1[i]**2 - z0[j]**2)
            U10.append(v)
    U10 = np.array(U10).reshape(len(z0), len(z1))
    print(U10)