from mmf_hfb.CylindricalDVRBasis import CylindricalBasis
from mmf_hfb.utils import block
import numpy as np


class VortexDVR(object):
    """
    A 2D and 3D vortex class without external potential
    """
    def __init__(self, bases_N=2, T=0, l_max=100, **args):
        """
        Construct and cache some information of bases

        """
        self.bases = [CylindricalBasis(dim=2, nu=nu, **args) for nu in range(bases_N)]
        self.l_max = max(l_max, 1)  # the angular momentum cut_off
        assert T==0

    def basis_match_rule(self, l):
        """
            Assign different bases to different angular momentum \nu
            it assign 0 to even \nu and 1 to odd \nu
        Note:
            inherit a child class to override this function
        """
        assert len(self.bases) > 1  # make sure the number of bases is at least two
        return l % 2

    def get_H(self, mus, delta, l=0):
        """
        return the full Hamiltonian(with pairing field)
        """
        basis_index = self.basis_match_rule(l)
        T = self.bases[basis_index].K
        Delta = np.diag(delta)
        mu_a, mu_b = mus
        # l0 = l % 2
        # LL = self.alpha*(l*(l + 1) - l0*(l0 + 1))/2.0
        # r2 = (zs[l0]/self.k_c)**2
        # V_ = LL /r2
        # V_eff = V_ + V + r2/2
        V_eff = 0
        H_a = T + np.diag(V_eff - mu_b)
        H_b = T + np.diag(V_eff - mu_a)
        H = block(H_a, Delta, Delta.conj(), -H_b)
        return H

    def _get_den(self, H):

        
    def get_densities(self, mus, delta):
        """"""
        H = self.get_H(mus=mus, delta=delta)
        
