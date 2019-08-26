#  from scipy.integrate import quad
from mmfutils.math import bessel
import numpy as np
import matplotlib.pyplot as plt


class Basis(object):
    pass


class CylindricalBasis(Basis):
    eps = 7./3 - 4./3 -1

    def __init__(self, N_root=None, R_max=None, K_max=None, a0=None):
        if N_root is None or R_max is None or K_max is None:
            self._init(a0=a0)
        else:
            self.N_root = N_root
            self.R_max = R_max
            self.K_max = K_max

    def _init(self, a0=None):
        if a0 is None:
            a0 = 1
        self.R_max = np.sqrt(-2*a0**2*np.log(self.eps))
        self.K_max = np.sqrt(-np.log(self.eps)/a0**2)
        self.N_root = int(np.ceil(self.K_max*2*self.R_max/np.pi))
        self.K_max = (self.N_root - 0.25)*np.pi/self.R_max

    def get_zs(self, nu=0):
        zs = bessel.j_root(nu=nu, N=self.N_root)
        return zs

    def get_rs(self, zs=None):
        if zs is None:
            zs = self.get_zs()
        return zs/self.K_max


class HarmonicDVR(object):
    # set m=hbar=1
    m=hbar=w=1
    eps = 7./3 - 4./3 -1  # machine accuracy

    def __init__(self, N_root=500, R_max=10.0, N_nu=1, K_max=12, w=1, dim=2):
        """
        w: float
            angular frequency of the external potential
        N_nu: int
            number of angular momentum used for calculation
        N_root: int
            max number of zero roots
        R_max: float
            range of the system
        K_max: float
            momentum cutoff
        """
        self.N_nu = N_nu
        self.N_root = N_root
        self.R_max = R_max
        self.K_max = K_max
        self.w = w
        a0 = np.sqrt(self.hbar/self.m/self.w)
        self.R_max = np.sqrt(-2*a0**2*np.log(self.eps))
        self.K_max = np.sqrt(-np.log(self.eps)/a0**2)
        self.N_root = int(np.ceil(self.K_max*2*self.R_max/np.pi))
        self.K_max = (self.N_root - 0.25)*np.pi/self.R_max

    def get_V(self, zs):
        """return the external potential"""
        r2 = (zs/self.K_max)**2
        return self.w**2*r2/2

    def get_zs(self, nu=0):
        """return the zero root for a given bessel function"""
        zs = bessel.j_root(nu=nu, N=self.N_root)
        return zs
    
    def get_K(self, zs, nu=0):
        """return the kinetic matrix for a given nu"""
        if zs is None:
            zs = self.get_zs(nu=nu)
        zi = np.array(list(range(len(zs)))) + 1
        xx, yy = np.meshgrid(zi, zi, sparse=False, indexing='ij')
        zx, zy = np.meshgrid(zs, zs, sparse=False, indexing='ij')
        K_diag = (1+2*(nu**2 - 1)/zs**2)/3.0
        K_off = 8*(-1)**(abs(xx - yy))*zx*zy/(zx**2 - zy**2)**2+self.eps
        np.fill_diagonal(K_off, K_diag)
        K = self.K_max**2*K_off/2.0
        return K

    def get_H(self, nu=0):
        zs = self.get_zs(nu=nu)
        K = self.get_K(zs=zs, nu=nu)
        r = zs/self.K_max
        r2 = r**2
        V = self.get_V(zs=zs)
        #  centrifugal potential ?
        nu0 = nu % 2
        V_ = (nu*(nu + 1) - nu0*(nu + 1))/r2/2.0
        H = K + np.diag(V_ + V)
        return H


if __name__ == "__main__":
    
    h = HarmonicDVR()
    H = h.get_H()
    Es, phis = np.linalg.eigh(H)
    plt.plot(phis[0])
    plt.show()
    print(Es)
