"""ASLDA class
This module provides a ASLDA method for solving the polarized
two-species Fermi gas with short-range interaction.
"""
from mmf_hfb.functionals import FunctionalBdG, FunctionalSLDA
from mmf_hfb.functionals import FunctionalASLDA
from mmf_hfb.homogeneous import Homogeneous
import scipy.optimize
import numpy as np
import numpy


class BDG(Homogeneous, FunctionalBdG):
    
    def __init__(
        self, mu_eff, dmu_eff, delta=1, m=1, T=0,
            hbar=1, k_c=None, C=None, dim=3, fix_C=False):
        kcs=[1000, 1000, 50]
        if k_c is None:
            k_c = kcs[dim - 1]
        self.m = m
        self._tf_args = dict(m_a=m, m_b=m, dim=dim, hbar=hbar, T=T, k_c=k_c)
        Homogeneous.__init__(self, **self._tf_args)
        self.C = C
        self.mus_eff = (mu_eff, dmu_eff)
        self.delta = delta
        self.k_c = k_c
     
    def get_Vext(self, **args):
        """
            return the external potential
        """
        return np.array([0, 0])
 
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

    def solve(self, mus, delta, use_solver=True):
        """
        use the Broyden solver may be much faster
        -------------
        mus = (mu_a, mu_b)
        """
        mu_a, mu_b = mus
        mu_a_eff, mu_b_eff =np.array([mu_a, mu_b]) - self.get_Vs(delta=delta)
        if use_solver:

            def fun(x):
                mu_a_eff, mu_b_eff, delta = x
                res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta)
                ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
                mu_a_eff_, mu_b_eff_ = (
                    np.array([mu_a, mu_b] - self.get_Vs(
                        delta=delta, ns=ns, taus=taus, nu=nu)))
                g_eff = self.get_effective_g(
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
            ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
            g_eff = self.get_effective_g(
                mus_eff=(mu_a_eff, mu_b_eff), ns=ns,
                dim=self.dim, k_c=self.k_c, E_c=self.k_c**2/2/self.m)
        else:
            while(True):
                res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta)
                ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
                mu_a_eff_, mu_b_eff_ = (
                    np.array([mu_a, mu_b])
                    - self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu))
                g_eff = self.get_effective_g(
                    mus_eff=(mu_a_eff_, mu_b_eff_), ns=ns,
                                dim=self.dim, E_c=self.k_c**2/2/self.m)
                delta_ = g_eff*nu
                if np.allclose((mu_a_eff_, mu_b_eff_, delta_), (mu_a_eff, mu_b_eff, delta), rtol=1e-8):
                    break
                delta, mu_a_eff, mu_b_eff = delta_, mu_a_eff_, mu_b_eff_
                print(f"mu_a_eff={mu_a_eff},\tmu_b_eff={mu_b_eff},\tdelta={delta}"
                      +f"\tC={self.C},\tg={g_eff},\tn={ns[0]},\ttau={taus[0]},\tnu={nu}")
        return (ns, taus, nu, g_eff, delta, mu_a_eff, mu_b_eff)

    def get_ns_e_p(self, mus, delta, update_C, use_solver=False, **args):
        """
            compute then energy density for BdG, equation(77) in page 39
            Note:
                the return value also include the pressure and densities
            -------------
            mus = (mu_a, mu_b)
        """
        if delta is None:
            delta = self.delta
        mu_a, mu_b = mus
        ns, taus, nu, g_eff, delta, mu_a_eff, mu_b_eff = self.solve(
            mus=mus, delta=delta, use_solver=use_solver)
        alpha_a, alpha_b = self.get_alphas(ns=ns)
        D = self.get_D(ns=ns)
        energy_density = alpha_a*taus[0]/2.0 + alpha_b*taus[1]/2.0 + g_eff*abs(nu)**2
        if self.T !=0:
            energy_density = (
                energy_density
                + self.T*self.get_entropy(mus_eff=(mu_a_eff, mu_b_eff), delta=delta).n)
        energy_density = energy_density + D
        pressure = ns[0]*mu_a + ns[1]*mu_b - energy_density
        if update_C:
            self.C = self.get_C(ns)
        return (ns, energy_density, pressure)
    

class SLDA(BDG, FunctionalSLDA):
    # pass
    def get_alphas(self, ns, d=0):
        if d==0:
            return (1.0, 1.0)
        elif d==1:
            return (0, 0, 0, 0)
   

class ASLDA(SLDA, FunctionalASLDA):
    pass
