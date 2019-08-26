from mmfutils.math import bessel
import numpy as np
from mmf_hfb.CylindricalDVRBasis import CylindricalBasis


class HarmonicDVR(CylindricalBasis):
    m=hbar=w=1
    eps = 7./3 - 4./3 -1  # machine accuracy

    def __init__(self, w=1, nu=0, dim=2):
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
        CylindricalBasis.__init__(self, nu=nu, dim=dim)
        self.w = w

    def get_V(self, zs):
        """return the external potential"""
        r2 = (zs/self.K_max)**2
        return self.w**2*r2/2


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
    import matplotlib.pyplot as plt

    h = HarmonicDVR(nu=1)
    H = h.get_H()
    Es, phis = np.linalg.eigh(H)
    plt.plot(phis[0])
    plt.show()
    print(Es)
