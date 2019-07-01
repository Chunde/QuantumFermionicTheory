"""ASLDA class
This module provides a ASLDA method for solving the polarized
two-species Fermi gas with short-range interaction.
"""
from mmf_hfb.Functionals import FunctionalBdG, FunctionalSLDA
from mmf_hfb.Functionals import FunctionalASLDA
from mmf_hfb.homogeneous import Homogeneous
from scipy.optimize import brentq
import scipy.optimize
import numpy as np
import numpy


class BDG(Homogeneous, FunctionalBdG):
    
    def __init__(
        self, mu_eff, dmu_eff, delta=1, m=1, T=0,
            hbar=1, k_c=None, C=None, dim=3):
        kcs=[1000, 1000, 100]
        if k_c is None:
            k_c = kcs[dim - 1]
        self.m = m
        self._tf_args = dict(m_a=m, m_b=m, dim=dim, hbar=hbar, T=T, k_c=k_c)
        Homogeneous.__init__(self, **self._tf_args)
        self.C = C
        self.mus_eff = (mu_eff, dmu_eff)
        self.delta = delta
        self.k_c = k_c
     
    def get_v_ext(self):
        """
            return the external potential
        """
        return np.array([0, 0])

    def get_Vs(self, delta=0, ns=None, taus=None, nu=None, **args):
        """
            return the modified V functional terms
        """
        if ns is None or taus is None:
            return self.get_v_ext()
        U_a, U_b = self.get_v_ext()  # external trap
        tau_a, tau_b = taus
        tau_p, tau_m = tau_a + tau_b, tau_a - tau_b

        alpha_p = sum(self.get_alphas(ns=ns))/2.0
        dalpha_p_dn_a, dalpha_p_dn_b, dalpha_m_dn_a, dalpha_m_dn_b=self.get_alphas(ns=ns, d=1)
        dC_dn_a, dC_dn_b = self.get_C(ns=ns, d=1)
        dD_dn_a, dD_dn_b = self.get_D(ns=ns, d=1)
       
        C0_ = self.hbar**2/self.m
        C1_ = C0_/2.0
        C2_ = tau_p*C1_ + np.conj(delta).T*nu/alpha_p
        C3_ = abs(delta)**2/alpha_p
        V_a = dalpha_m_dn_a*tau_m*C1_ + dalpha_p_dn_a*C2_ + dC_dn_a*C3_ + C0_*dD_dn_a + U_a
        V_b = dalpha_m_dn_b*tau_m*C1_ + dalpha_p_dn_b*C2_ + dC_dn_b*C3_ + C0_*dD_dn_b + U_b
        return np.array([V_a, V_b])
    
    def get_C(self, ns, d=0):
        """override the C functional to support fixed C value"""
        if d==0:
            if self.C is None:
                return FunctionalBdG.get_C(self, ns=ns)
            return self.C

        if d==1:
            if self.C is None:
                return FunctionalBdG.get_C(self, ns=ns, d=1)
            return (0, 0)

    def solve(self, mus, delta, use_Broyden=True):
        """use the Broyden solver may be much faster"""
        mu, dmu = mus
        mu_a, mu_b = mu + dmu, mu - dmu
        mu_a_eff, mu_b_eff =np.array([mu_a, mu_b]) + self.get_Vs(delta=delta)
        if use_Broyden:

            def fun(x):
                mu_a_eff, mu_b_eff, delta = x
                res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta)
                ns, taus, nu = (res.n_a.n, res.n_b.n), (res.tau_a.n, res.tau_b.n), res.nu.n
                mu_a_eff_, mu_b_eff_ = (
                    np.array([mu_a, mu_b])
                    + self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu))
                g_eff = self._g_eff(
                    mus_eff=(mu_a_eff_, mu_b_eff_), ns=ns,
                    dim=self.dim, E_c=self.k_c**2/2/self.m)
                delta_ = g_eff*nu
                print(
                    f"mu_a_eff={mu_a_eff},\tmu_b_eff={mu_b_eff},\tdelta={delta}"
                    +f"\tC={self.C},\tg={g_eff},\tn={ns[0]},\ttau={taus[0]},\tnu={nu}")
                x_ = np.array([mu_a_eff_, mu_b_eff_, delta_])
                return x - x_

            x0 = np.array([mu_a_eff, mu_b_eff, delta])  # initial guess
            x = scipy.optimize.broyden1(fun, x0, maxiter=100, f_tol=1e-4)
            mu_a_eff, mu_b_eff, delta = x
            res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta)
            ns, taus, nu = (res.n_a.n, res.n_b.n), (res.tau_a.n, res.tau_b.n), res.nu.n
            g_eff = self._g_eff(
                mus_eff=(mu_a_eff, mu_b_eff), ns=ns,
                dim=self.dim, k_c=self.k_c, E_c=self.k_c**2/2/self.m)
        else:
            while(True):
                res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta)
                ns, taus, nu = (res.n_a.n, res.n_b.n), (res.tau_a.n, res.tau_b.n), res.nu.n
                mu_a_eff_, mu_b_eff_ = (
                    np.array([mu_a, mu_b])
                    + self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu))
                g_eff = self._g_eff(
                    mus_eff=(
                        mu_a_eff_, mu_b_eff_), ns=ns,
                        dim=self.dim, E_c=self.k_c**2/2/self.m)
                delta_ = g_eff*nu
                if np.allclose((mu_a_eff_, mu_b_eff_, delta_), (mu_a_eff, mu_b_eff, delta), rtol=1e-8):
                    break
                delta, mu_a_eff, mu_b_eff = delta_, mu_a_eff_, mu_b_eff_
                print(f"mu_a_eff={mu_a_eff},\tmu_b_eff={mu_b_eff},\tdelta={delta}"
                      +f"\tC={self.C},\tg={g_eff},\tn={ns[0]},\ttau={taus[0]},\tnu={nu}")
        return (ns, taus, nu, g_eff, delta, mu_a_eff, mu_b_eff)

    def get_ns_e_p(self, mus, delta, update_C, use_Broyden=False, **args):
        """
            compute then energy density for BdG, equation(77) in page 39
            Note:
                the return value also include the pressure and densities
            -------------
            mus = (mu, dmu)
        """
        if delta is None:
            delta = self.delta
        mu, dmu = mus
        mu_a, mu_b = mu + dmu, mu - dmu
        ns, taus, nu, g_eff, delta, mu_a_eff, mu_b_eff = self.solve(
            mus=mus, delta=delta, use_Broyden=use_Broyden)
        alpha_a, alpha_b = self.get_alphas(ns=ns)
        D = self.get_D(ns=ns)
        energy_density = taus[0]/2.0 + taus[1]/2.0 + g_eff*abs(nu)**2
        if self.T !=0:
            energy_density = (
                energy_density
                + self.T*self.get_entropy(mus_eff=(mu_a_eff, mu_b_eff), delta=delta).n)
        energy_density = energy_density - D
        pressure = ns[0]*mu_a + ns[1]*mu_b - energy_density
        if update_C:
            self.C = self.get_C(ns)
        return (ns, energy_density, pressure)
    

class SLDA(BDG, FunctionalSLDA):
    pass

    # def get_alphas(self, ns, d=0):
    #     dx = 9
    #     if d==0:
    #         return (1+dx, 1+dx)
    #     elif d==1:
    #         return (0, 0, 0, 0)
    
    # def get_D(self, ns, d=0):
    #    if d==0:
    #        return 1
    #    if d==1:
    #        return (0,0)


class ASLDA(SLDA, FunctionalASLDA):
    pass
