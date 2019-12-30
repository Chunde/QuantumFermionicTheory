from mmf_hfb.DVRBasis import CylindricalBasis
from mmf_hfb.utils import block
from mmf_hfb import homogeneous
import numpy as np
import sys
from mmfutils.math.special import mstep


class bdg_dvr(object):
    """
    A 2D and 3D vortex class without external potential
    """
    def __init__(
            self, bases_N=2, mu=1, dmu=0, delta=1,
                E_c=None, T=0, l_max=100, g=None, **args):
        """
        Construct and cache some information of bases

        """
        self.bases = [CylindricalBasis(nu=nu, **args) for nu in range(bases_N)]
        self.l_max = max(l_max, 1)  # the angular momentum cut_off
        assert T==0
        self.T=T
        self.g = self.get_g(mu=mu, delta=np.mean(delta)) if g is None else g
        self.mus = (mu + dmu, mu - dmu)
        self.E_c = sys.maxsize if E_c is None else E_c

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

    def get_Vext(self, rs):
        """return external potential"""
        return 0

    def get_H(self, mus, delta, lz=0, nu=0):
        """
        return the full Hamiltonian(with pairing field)
        """
        basis = self.bases[self.basis_match_rule(nu)]
        T = basis.K
        Delta = np.diag(basis.zero + delta)
        mu_a, mu_b = mus
        V_ext = self.get_Vext(rs=basis.rs)
        V_corr = basis.get_V_correction(nu=nu)
        V_eff = V_ext + V_corr
        # The additional term should be double-checked
        # it seems l(l+1) works better than l^2
        lz2 = lz**2/basis.rs**2/2
        # lz2 = lz*(lz-1)/basis.rs**2/2
        H_a = T + np.diag(V_eff - mu_a + lz2)
        H_b = T + np.diag(V_eff - mu_b + lz2)
        H = block(H_a, Delta, Delta.conj(), -H_b)
        return H

    def get_g(self, mu=1.0, delta=0.2):
        """
        the interaction strength
        """
        # [Check] will be dim = 3 when integrate over z
        h = homogeneous.Homogeneous(dim=2)
        res = h.get_densities(mus_eff=(mu, mu), delta=delta)
        g = 0 if res.nu == 0 else delta/res.nu
        return g

    def get_psi(self, nu, u, uid=None):
        """
        apply weight on the u(v) to get the actual radial wave-function
        -------------
        uid: the index of the basis in the bases array
        """
        if uid is None:
            uid = self.basis_match_rule(nu)
        assert uid >=0 and uid < len(self.bases)
        b = self.bases[uid]
        return b._get_psi(u=u)
    
    def transform(self, nu_s, nu_t, us_s):
        """
        Transform us from a DVR basis to another one
        --------------
        NOTE: very slow
        """
        if nu_s == nu_t:
            return us_s
        dvr_s = self.bases[self.basis_match_rule(nu_s)]
        dvr_t = self.bases[self.basis_match_rule(nu_t)]
        def f(r):
            fs = [us_s[n]*dvr_s.get_F(n=n, rs=r) for n in range(len(us_s))]
            return sum(fs)
        psi = [f(r) for r in dvr_t.rs]
        Fs = dvr_t.get_F_rs()
        us_t = np.array(psi)/np.array(Fs)
        return us_t
    
    def _get_den(self, H, nu):
        """
        return the densities for a given H
        """
        es, phis = np.linalg.eigh(H)
        phis = phis.T
        offset = phis.shape[0] // 2
        den = 0
        for i in range(len(es)):
            E, uv = es[i], phis[i]
            if abs(E) > self.E_c:
                continue
            
            u, v = uv[: offset], uv[offset:]
            
            # u = self.transform(nu_s=nu, nu_t=0, us_s=u)
            # v = self.transform(nu_s=nu, nu_t=0, us_s=v)
            u = self.get_psi(nu=nu, u=u, uid=0)
            v = self.get_psi(nu=nu, u=v, uid=0)
            
            f_p, f_m = self.f(E=E), self.f(E=-E)
            n_a = u*u.conj()*f_p
            n_b = v*v.conj()*f_m
            kappa = u*v.conj()*(f_p - f_m)/2
            den = den + np.array([n_a, n_b, kappa])
        return den

    def get_densities(self, mus, delta, lz=0):
        """
        return the particle number density and anomalous density
        Note: Here the anomalous density is represented as kappa
        instead of \nu so it's not that confusing when \nu has
        been used as angular momentum quantum number.
        """
        dens = self._get_den(self.get_H(mus=mus, delta=delta, nu=0, lz=lz), nu=0)
        for nu in range(1, self.l_max):  # sum over angular momentum
            H = self.get_H(mus=mus, delta=delta, nu=nu, lz=lz)
            dens = dens + 2*self._get_den(H, nu=nu)  # double-degenerate
        n_a, n_b, kappa = dens
        return (n_a, n_b, kappa)


class bdg_dvr_ho(bdg_dvr):
    """a 2D DVR with harmonic potential class"""
    def get_Vext(self, rs):
        return rs**2/2


class dvr_vortex(bdg_dvr):
    """BCS Vortex"""
    barrier_width = 0.2
    barrier_height = 100.0
    
    def get_Vext(self, rs):
        self.R = 5
        R0 = self.barrier_width * self.R
        V = self.barrier_height * mstep(rs-self.R+R0, R0)
        return V

    