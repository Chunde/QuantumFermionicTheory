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

    def get_V(self):
        """return the external potential"""
        r2 = (self.rs)**2
        return self.w**2*r2/2

    def get_H(self):
        K = self.K
        V = self.get_V()
        H = K + np.diag(V)
        return H


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    h = HarmonicDVR(nu=0, dim=3)
    H = h.get_H()
    Es, phis = np.linalg.eigh(H)
    print(Es)
    plt.plot(phis[0]/h.rs_scale)
    plt.show()
