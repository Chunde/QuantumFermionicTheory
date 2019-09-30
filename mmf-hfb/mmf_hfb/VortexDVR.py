from mmf_hfb.CylindricalDVRBasis import CylindricalBasis
from mmf_hfb.utils import block
from mmf_hfb import homogeneous
import numpy as np


class VortexDVR(object):
    """
    A 2D and 3D vortex class without external potential
    """
    def __init__(self, bases_N=2, mu=1, dmu=0, delta=1, T=0, l_max=100, **args):
        """
        Construct and cache some information of bases

        """
        self.bases = [CylindricalBasis(dim=2, nu=nu, **args) for nu in range(bases_N)]
        self.l_max = max(l_max, 1)  # the angular momentum cut_off
        assert T==0
        self.T=T
        self.g = self.get_g(mu=mu, delta=delta)
        self.mus = (mu + dmu, mu - dmu)

    def f(self, E, T=0):
        if T is None:
            T = self.T
        if T == 0:
            if E < 0:
                return 1
            return 0
        else:
            return 1./(1+np.exp(E/T))

    def basis_match_rule(self, nu):
        """
            Assign different bases to different angular momentum \nu
            it assign 0 to even \nu and 1 to odd \nu
        Note:
            inherit a child class to override this function
        """
        assert len(self.bases) > 1  # make sure the number of bases is at least two
        return nu % 2

    def get_H(self, mus, delta, nu=0):
        """
        return the full Hamiltonian(with pairing field)
        """
        basis = self.bases[self.basis_match_rule(nu)]
        T = basis.K
        Delta = np.diag(delta)
        mu_a, mu_b = mus
        V_corr = basis.get_V_correction(nu=nu)
        V_mean_field = basis.get_V_mean_field(nu=nu)
        V_eff = V_corr + V_mean_field
        H_a = T + np.diag(V_eff - mu_b)
        H_b = T + np.diag(V_eff - mu_a)
        H = block(H_a, Delta, Delta.conj(), -H_b)
        return H

    def get_g(self, mu=1.0, delta=0.2):
        """
        the interaction strength
        """
        h = homogeneous.Homogeneous(dim=3) 
        res = h.get_densities(mus_eff=(mu, mu), delta=delta)
        g = delta/res.nu.n
        return g

    def _get_den(self, H):
        """
        return the densities for a given H
        """
        es, phis = np.linalg.eigh(H)
        phis = phis.T
        offset = phis.shape[0] // 2
        den = 0
        for i in range(len(es)):
            E, uv = es[i], phis[i]
            u, v = uv[: offset], uv[offset:]
            fe = self.f(E=E)
            n_a = (1 - fe)*v**2
            n_b = fe*u**2
            kappa = (1 - 2*fe)*u*v
            den = den + np.array([n_a, n_b, kappa])
        return den

    def get_densities(self, mus, delta):
        """
        return the particle number density and anomalous density
        """
        dens = 0
        for nu in range(self.l_max):
            H = self.get_H(mus=mus, delta=delta, nu=nu)
            dens = dens + self._get_den(H)
        n_a, n_b, kappa = dens
