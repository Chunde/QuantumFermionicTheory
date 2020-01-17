from mmf_hfb.DVRBasis import CylindricalBasis
from mmf_hfb.utils import block
from mmf_hfb import homogeneous
import numpy as np
import sys
from mmfutils.math.special import mstep
from collections import namedtuple
from mmf_hfb.bcs import BCS


Densities = namedtuple('Densities', ['n_a', 'n_b', 'tau_a', 'tau_b', 'nu', 'j_a', 'j_b'])


def get_transform_matrix(dvr_s, dvr_t):
    rs_s = dvr_s.rs
    rs_t = dvr_t.rs
    ws = dvr_t.get_F_rs()
    return np.array(
        [[dvr_s.get_F(
            n=i, rs=rs_t[j])/ws[j] for i in range(
                len(rs_s))] for j in range(len(rs_t))])


class dvr_basis_set(object):
    pass


class dvr_odd_even_set(dvr_basis_set):
    
    def __init__(self, **args):
        self.N_basis = 2
        self.bases = [CylindricalBasis(nu=nu, **args) for nu in range(self.N_basis)]
        self.Us =[None, get_transform_matrix(self.bases[1], self.bases[0])]

    def basis_match_rule(self, nu):
        """
            Assign different bases to different angular momentum \nu
            it assign 0 to even \nu and 1 to odd \nu
        Note:
            inherit a child class to override this function
        """
        assert len(self.bases) > 1
        return nu % 2

    @property
    def zero(self):
        return self.bases[0].zero

    def get_rs(self):
        return self.bases[0].rs

    def get_basis(self, nu):
        return self.bases[self.basis_match_rule(nu=nu)]

    def get_psi(self, nu, u):
        U_matrix = self.Us[nu % 2]
        if U_matrix is not None:
            u = U_matrix.dot(u)
        b = self.bases[0]
        return b._get_psi(u=u)


class bdg_dvr(object):
    """
    A 2D and 3D vortex class without external potential
    """
    def __init__(
            self, bases_N=2, mu=1, dmu=0, delta=1, lz=0,
            E_c=None, T=0, l_max=100, g=None, **args):
        """
        Construct and cache some information of bases

        """
        self.bases = dvr_odd_even_set(**args)
        self.l_max = max(l_max, 1)  # the angular momentum cut_off
        assert T==0
        self.T=T
        self.lz=lz
        self.g = self.get_g(mu=mu, delta=np.mean(delta)) if g is None else g
        self.mus = (mu + dmu, mu - dmu)
        self.E_c = sys.maxsize if E_c is None else E_c
        # self.U10 = get_transform_matrix(self.bases[1], self.bases[0]) #-----------------
        self.rs = self.bases.get_rs()
        
    def f(self, E, T=0):
        if T is None:
            T = self.T
        if T == 0:
            if E < 0:
                return 1
            return 0
        else:
            return 1./(1+np.exp(E/T))

    def get_Vext(self, rs):
        """return external potential"""
        return 0
    
    def get_lz_term(self, lz):
        return lz**2
    
    def get_H(self, mus, delta, lz=0, nu=0):
        """
        return the full Hamiltonian(with pairing field)
        """
        basis = self.bases.get_basis(nu=nu)
        T = basis.K
        Delta = np.diag(basis.zero + delta)
        mu_a, mu_b = mus
        V_ext = self.get_Vext(rs=basis.rs)
        V_corr = basis.get_V_correction(nu=nu)
        V_eff = V_ext + V_corr
        lz2 = self.get_lz_term(lz)/basis.rs**2/2
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

    def get_psi(self, nu, u):
        """
        apply weight on the u(v) to get the actual radial wave-function
        """
        return self.bases.get_psi(nu=nu, u=u)
    
    # def transform(self, nu_s, nu_t, us_s):
    #     if nu_s == nu_t:
    #         return us_s
    #     dvr_s = self.bases[self.basis_match_rule(nu_s)]
    #     dvr_t = self.bases[self.basis_match_rule(nu_t)]

    #     def f(r):
    #         fs = [us_s[n]*dvr_s.get_F(n=n, rs=r) for n in range(len(us_s))]
    #         return sum(fs)
    #     psi = [f(r) for r in dvr_t.rs]
    #     Fs = dvr_t.get_F_rs()
    #     us_t = np.array(psi)/np.array(Fs)
    #     return us_t

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
            u = self.get_psi(nu=nu, u=u)
            v = self.get_psi(nu=nu, u=v)
            
            f_p, f_m = self.f(E=E), self.f(E=-E)
            n_a = u*u.conj()*f_p
            n_b = v*v.conj()*f_m
            j_a = -n_a*self.lz/self.rs
            j_b = -n_b*self.lz/self.rs
            kappa = u*v.conj()*(f_p - f_m)/2
            den = den + np.array([n_a, n_b, kappa, j_a, j_b])
        return den
        
    def get_densities(self, mus, delta, lz=None):
        """
        return the particle number density and anomalous density
        Note: Here the anomalous density is represented as kappa
        instead of \nu so it's not that confusing when \nu has
        been used as angular momentum quantum number.
        """
        if lz is None:
            lz = self.lz
        else:
            self.lz = lz
        
        dens = self._get_den(self.get_H(mus=mus, delta=delta, nu=0, lz=lz), nu=0)
        for nu in range(1, self.l_max):  # sum over angular momentum
            H = self.get_H(mus=mus, delta=delta, nu=nu, lz=lz)
            dens = dens + 2*self._get_den(H, nu=nu)  # double-degenerate
        n_a, n_b, kappa, j_a, j_b = dens
        return Densities(
            n_a=n_a, n_b=n_b,
            tau_a=None, tau_b=None,
            nu=kappa,
            j_a=j_a, j_b=j_b)

       
class BCS_vortex(BCS):
    """BCS Vortex"""
    barrier_width = 0.2
    barrier_height = 100.0
    
    def __init__(self, delta, mus_eff, **args):
        BCS.__init__(self, **args)
        h = homogeneous.Homogeneous(Nxyz=self.Nxyz, Lxyz=self.Lxyz) 
        res = h.get_densities(mus_eff=mus_eff, delta=delta)
        self.g = delta/res.nu.n
        
    def get_v_ext(self, **kw):
        self.R = min(self.Lxyz)/2
        r = np.sqrt(sum([_x**2 for _x in self.xyz[:2]]))
        R0 = self.barrier_width * self.R
        V = self.barrier_height * mstep(r-self.R+R0, R0)
        return (V, V)


class dvr_vortex(bdg_dvr):
    """BCS Vortex"""
    barrier_width = 0.2
    barrier_height = 100.0
    
    def get_Vext(self, rs):
        self.R = 5
        R0 = self.barrier_width * self.R
        V = self.barrier_height * mstep(rs-self.R+R0, R0)
        return V


class bdg_dvr_ho(bdg_dvr):
    """a 2D DVR with harmonic potential class"""
    def get_Vext(self, rs):
        return rs**2/2
