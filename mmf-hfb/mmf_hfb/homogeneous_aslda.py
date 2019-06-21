"""ASLDA class
This module provides a ASLDA method for solving the polarized
two-species Fermi gas with short-range interaction.
"""
from mmf_hfb.Functionals import IFunctional, FunctionalASLDA, FunctionalBdG, FunctionalSLDA
from mmf_hfb.homogeneous import Homogeneous
from mmf_hfb import tf_completion as tf
import numpy as np
import numpy
from scipy.optimize import brentq


class BDG(Homogeneous, FunctionalBdG):
    
    def __init__(
        self, mu_eff, dmu_eff, delta=1, q=0, dq=0,
            m=1, T=0, hbar=1, k_c=None, C=None, dim=3):
        FunctionalBdG.__init__(self)
        kcs=[np.inf, 1000, 50]
        if k_c is None:
            k_c = kcs[dim - 1]
        self.C = C
        Homogeneous.__init__(self, dim=dim, k_c=k_c)
        self.T = T
        self.mus_eff = (mu_eff, dmu_eff)
        self.m = m
        self.delta = delta
        self.hbar = hbar
        self.k_c = k_c
        self._tf_args = dict(m_a=1, m_b=1, dim=dim, hbar=hbar, T=T, k_c=k_c)
     
    def get_v_ext(self, **args):
        """
            return the modified V functional terms
        """
        return (0, 0)

    def get_Vs(self, delta=0, ns=None, taus=None, nu=None, **args):
        """
            return the modified V functional terms
        """
        if ns is None or taus is None:
            return self.get_v_ext()
        U_a, U_b = self.get_v_ext()  # external trap
        tau_a, tau_b = taus
        tau_p, tau_m = tau_a + tau_b, tau_a - tau_b

        alpha_p = sum(self.get_alphas(ns))/2.0
        dalpha_p_dn_a, dalpha_p_dn_b, dalpha_m_dn_a, dalpha_m_dn_b=self.get_alphas(ns=ns, d=1)
        dC_dn_a, dC_dn_b = self.get_C(ns=ns, d=1)
        dD_dn_a, dD_dn_b = self.get_D(ns=ns, d=1)
       
        C0_ = self.hbar**2/self.m
        C1_ = C0_/2
        C2_ = tau_p*C1_ - np.conj(delta).T*nu/alpha_p
        V_a = dalpha_m_dn_a*tau_m*C1_ + dalpha_p_dn_a*C2_ + dC_dn_a + C0_*dD_dn_a + U_a
        V_b = dalpha_m_dn_b*tau_m*C1_ + dalpha_p_dn_b*C2_ + dC_dn_b + C0_*dD_dn_b + U_b
        return (V_a, V_b)
    
    def get_C(self, ns, d=0):
        if d==0:
            if self.C is None:
                return FunctionalBdG.get_C(self, ns=ns)
            return self.C

        if d==1:
            if self.C is None:
                return FunctionalBdG.get_C(self, ns=ns, d=1)
            return (0, 0)

    def get_ns_e_p(self, mus, delta, update_C, **args):
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
        mu_a_eff, mu_b_eff =np.array([mu_a, mu_b]) + self.get_Vs(delta=delta)

        while(True):
            args.update(self._tf_args, mu_a=mu_a_eff, mu_b=mu_b_eff, delta=delta)
            res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta)
            ns, taus, nu = (res.n_a.n, res.n_b.n), (res.tau_a.n, res.tau_b.n), res.nu.n
            print(ns, taus, nu)
            mu_a_eff_, mu_b_eff_ = np.array([mu_a, mu_b]) + self.get_Vs(ns=ns, taus=taus, nu=nu)
            g_eff = self._g_eff(mus_eff=(mu_a_eff_, mu_b_eff_), ns=ns, dim=self.dim, k_c=self.k_c, E_c=self.k_c**2/2/self.m)
            delta_ = g_eff*nu
            if np.allclose((mu_a_eff_, mu_b_eff_, delta_), (mu_a_eff, mu_b_eff, delta), rtol=1e-5):
                break
            delta, mu_a_eff, mu_b_eff = delta_, mu_a_eff_, mu_b_eff_
            print(f"mu_a_eff={mu_a_eff}, mu_b_eff={mu_b_eff}, delta={delta}")
        
        alpha_a, alpha_b = self.get_alphas(ns=ns)
        D = self.get_D(ns=ns)
        nu = tf.integrate_q(tf.nu_integrand, **args)
        energy_density = taus[0]*alpha_a/2.0 + taus[1]*alpha_b/2.0 + g_eff*abs(nu)**2
        energy_density = energy_density + D
        if self.dim == 1:  # [check] may be wrong
            energy_density = energy_density + g_eff*np.prod(ns)
        pressure = ns[0]*mu_a + ns[1]*mu_b - energy_density
        if update_C:
            self.C = self.get_C(ns)
        return (ns, energy_density.n, pressure.n)
    

class SLDA(BDG, FunctionalSLDA):

    def get_alphas(self, ns, d=0):
        if d==0:
            return (1, 1)
        elif d==1:
            return (0, 0, 0, 0)

    
    # def get_D(self, ns, d=0):
    #     if d==0:
    #         return 0
    #     if d==1:
    #         return (0, 0)

    
class ASLDA(SLDA, FunctionalASLDA):
    pass
