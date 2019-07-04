"""ASLDA class
This module provides a ASLDA method for solving the polarized
two-species Fermi gas with short-range interaction.
"""
from mmf_hfb.homogeneous import Homogeneous
import numpy as np
import scipy.optimize


class KernelHomogeneous(Homogeneous):
    
    def __init__(
        self, mu_eff, dmu_eff, delta=1, m=1, T=0,
            hbar=1, k_c=None, C=None, dim=3, fix_C=False, **args):
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
     
    def get_v_ext(self, **args):
        """
            return the external potential
        """
        return np.array([0, 0])

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
                mu_a_eff_, mu_b_eff_ = (np.array([mu_a, mu_b])
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
                    mus_eff=(mu_a_eff_, mu_b_eff_), ns=ns,
                                dim=self.dim, E_c=self.k_c**2/2/self.m)
                delta_ = g_eff*nu
                if np.allclose((mu_a_eff_, mu_b_eff_, delta_), (mu_a_eff, mu_b_eff, delta), rtol=1e-8):
                    break
                delta, mu_a_eff, mu_b_eff = delta_, mu_a_eff_, mu_b_eff_
                print(f"mu_a_eff={mu_a_eff},\tmu_b_eff={mu_b_eff},\tdelta={delta}"
                      +f"\tC={self.C},\tg={g_eff},\tn={ns[0]},\ttau={taus[0]},\tnu={nu}")
        return (ns, taus, nu, g_eff, delta, mu_a_eff, mu_b_eff)