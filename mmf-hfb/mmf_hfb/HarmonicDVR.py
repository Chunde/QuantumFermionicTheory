#  from scipy.integrate import quad
from mmfutils.math import bessel
import numpy as np
import matplotlib.pyplot as plt


class HarmonicDVR(object):
    eps = 7./3 - 4./3 -1  # machine accuracy

    def __init__(self, N_max=100, R_max=10.0, N_nu=1, K_max=12, omega=1, dim=2):
        """
        omega: float
            angular frequency of the external potential
        """
        self.N_nu = N_nu
        self.N_max = N_max
        self.R_max = R_max
        self.K_max = K_max
        self.omega = omega

    def get_V(self, zs):
        """return the external potential"""
        return self.omega*zs**2

    def get_zs(self, nu=0):
        """return the zero root for a given bessel function"""
        zs = bessel.j_root(nu=nu, N=self.N_max)
        z_max = self.K_max*self.R_max
        for i in range(len(zs)):
            if zs[i] > z_max:
                return zs[:i]
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
        V = self.get_V(zs=zs)
        r2 = (zs/self.K_max)**2
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

    