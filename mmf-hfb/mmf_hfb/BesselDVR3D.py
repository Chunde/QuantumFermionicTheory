import numpy as np


def nan0(data):
    """convert nan to zero"""
    return np.nan_to_num(data, 0)


class BesselDVR3D(object):
    """SLDA with Bessel DVR"""
    def __init__(self, N_c=100, R_c=9.0, T=0.025, **args):
        self.N_c = N_c
        self.R_c = R_c
        self.T = T
        self.k_c = np.sqrt(2*N_c + 3) + 3
        self.eps = 2.2204e-16

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

    def _get_K(self, nu, zs):
        """return kinetic matrix for a given angular momentum $\nu$"""
        zi = np.array(list(range(len(zs)))) + 1
        xx, yy = np.meshgrid(zi, zi, sparse=False, indexing='ij')
        zx, zy = np.meshgrid(zs, zs, sparse=False, indexing='ij')
        K_diag = (1+2*(nu**2 - 1)/zs**2)/3.0
        K_off = 8*(-1)**(abs(xx - yy))*zx*zy/(zx**2 - zy**2)**2+self.eps
        np.fill_diagonal(K_off, K_diag)
        T = self.k_c**2*K_off/2.0
        return T

    def get_Ks(self, zs=None):
        """return kinetic matrix for different angular momentums"""
        if zs is None:
            zs = self.get_zeros()
        z0, z1 = zs
        K0 = self._get_K(nu=0.5, zs=z0)
        K1 = self._get_K(nu=1.5, zs=z1)
        return (K0, K1)
 
    def get_H(self, zs=None, Ts=None, l=0):
        """return the Hamiltonian"""
        if zs is None:
            zs = self.get_zeros()
        if Ts is None:
            Ts = self.get_Ks(zs=zs)

        # the correction term for centrifugal potential if l !=\nu
        # But seem it should be l^2 - l0^2, so the follow code may
        # only be accurate when l = l0 ?
        l0 = l % 2
        ll = (l*(l + 1) - l0*(l0 + 1))/2.0
        r2 = (zs[l0]/self.k_c)**2  #  HO r2/2, here we got k_c as
        V_corr = ll/r2
        V_harm = r2/2
        H = Ts[l0] + np.diag(V_corr + V_harm)
        return H


if __name__ == "__main__":
    b = BesselDVR3D()
    H = b.get_H(l=0)
    eigen, phi = np.linalg.eigh(H)
    print(eigen)