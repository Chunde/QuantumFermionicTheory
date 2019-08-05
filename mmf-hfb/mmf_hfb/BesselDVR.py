import numpy as np
from enum import Enum


class DVRBasisType(Enum):
    """Types of DVR basis"""
    POLYNOMIAL = 0
    BESSEL = 1
    SINC = 2
    AIRY = 3


class BesselDVR(object):
    """SLDA with Bessel DVR"""
    def __init__(self, N_c=100, R_c=9.0, T=0.025, **args):
        self.N_c = N_c
        self.R_c = R_c
        self.E_c = 0.95*(N_c + 1.5)
        self.k_c = np.sqrt(2*N_c + 3) + 3
        self.xi = 0.42
        self.eta=0.504
        self.alpha=1.14
        self.beta = -0.55269
        self.gamma = -1/0.090585
        self.eps = 2.2204e-16
    
    def beta_bar(self):
        """compute beta bar equation 20, PRA 76, 040502(R)(2007)"""
        return self.beta - self.eta**2*(3*np.pi**2)**(2.0/3)/self.gamma/6.0
    
    def get_zeros(self):
        N_max = 500
        nn = np.array(list(range(0, N_max, 1))) + 1
        Z0 = nn*np.pi
        Z1 = Z0 + 0.5*np.pi
        for _ in range(20):
            Z1 = nn*np.pi + np.arctan(Z1)
        
        z_c = self.R_c*self.k_c
        i0, i1 = 0, 0
        for i in range(len(Z0)):
            if Z0[i] > z_c:
                i0 = i
                break
        for i in range(len(Z1)):
            if Z1[i] > z_c:
                i1 = i
                break
        
        return (Z0[:i0], Z1[:i1])

    def _get_K(self, nu, zeros):
        """return kinetic matrix for a given angular momentum $\nu$"""
        zi = np.array(list(range(len(zeros)))) + 1
        xx, yy = np.meshgrid(zi, zi, sparse=False, indexing='ij')
        zx, zy = np.meshgrid(zeros, zeros, sparse=False, indexing='ij')
        K_diag = (1+2*(nu**2 - 1)/zeros**2)/3.0
        K_off = 8*(-1)**(abs(xx - yy))*zx*zy/(zx**2 - zy**2)**2+self.eps
        np.fill_diagonal(K_off, K_diag)
        T = self.k_c**2*K_off/2.0*self.alpha
        return T

    def get_Ks(self):
        """return kinetic matrix for different angular momentums"""
        Z0, Z1 = self.get_zeros()
        K0 = self._get_K(nu=0.5, zeros=Z0)
        K1 = self._get_K(nu=1.5, zeros=Z1)
        return (K0, K1)


if __name__ == "__main__":
    dvr = BesselDVR()
    dvr.get_Ks()
    


