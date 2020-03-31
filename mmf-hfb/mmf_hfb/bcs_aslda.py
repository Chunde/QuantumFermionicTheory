"""ASLDA class
This module provides a ASLDA method for solving the polarized
two-species Fermi gas with short-range interaction.
Note: This class can 
"""
import numpy as np

from .functionals import FunctionalASLDA, FunctionalBdG, FunctionalSLDA
from .bcs_kernel import bcs_kernel


class BDG(FunctionalBdG, bcs_kernel):
    """????"""
    def __init__(self, Nxyz, Lxyz, dx=None, T=0, C=None, fix_C=False, **args):
       bcs_kernel.__init__(self, Nxyz=Nxyz, Lxyz=Lxyz, dx=dx, T=T, **args)

    def solve(self, mus, delta, use_solver=True, rtol=1e-12, **args):
        """use the Broyden solver may be much faster"""
        mu_a, mu_b = mus
        V_a, V_b = self.get_Vs()
        mu_a_eff, mu_b_eff = mu_a + V_a, mu_b + V_b
        args.update(dim=self.dim, k_c=self.k_c, E_c=self.E_c)

        def _fun(x):
            mu_a_eff, mu_b_eff, delta=x
            res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta, **args)
            ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
            args.update(ns=ns)
            V_a, V_b = self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu)
            mu_a_eff_, mu_b_eff_ = mu_a - V_a, mu_b - V_b
            g_eff = self.get_effective_g(mus_eff=(mu_a_eff_, mu_b_eff_), **args)
            delta_ = g_eff*nu
            print(
                f"mu_a_eff={mu_a_eff_.flat[0].real},\tmu_b_eff={mu_b_eff_.flat[0].real},\tdelta={delta_.flat[0].real}"
                +f"\tC={self.C if (self.C is None or (len(self.C)==1)) else self.C.flat[0]},\tg={g_eff.flat[0].real},"
                +f"\tn={ns[0].flat[0].real},\ttau={taus[0].flat[0].real},\tnu={nu.flat[0].real}")
            return np.array([mu_a_eff_, mu_b_eff_, delta_])

        if use_solver:
            def fun(x):
                return _fun(x) - x
            
            x0 = np.array([mu_a_eff, mu_b_eff, delta*np.ones_like(sum(self.xyz))])
            mu_a_eff, mu_b_eff, delta = solver(fun, x0)
        else:
            while(True):  # use simple iteration if no solver is specified
                mu_a_eff_, mu_b_eff_, delta_ = _fun((mu_a_eff, mu_b_eff, delta))
                if (np.allclose(
                    mu_a_eff_, mu_a_eff, rtol=rtol) and np.allclose(
                        mu_b_eff_, mu_b_eff, rtol=rtol) and np.allclose(
                            delta, delta_, rtol=rtol)):
                    break
                delta, mu_a_eff, mu_b_eff = delta_, mu_a_eff_, mu_b_eff_
        res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta, **args)
        ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
        args.update(ns=ns)
        g_eff = self.get_effective_g(mus_eff=(mu_a_eff, mu_b_eff), **args)
        return (ns, taus, nu, g_eff, delta, mu_a_eff, mu_b_eff)


    def get_ns_e_p(self, mus, delta, update_C, use_solver=False, **args):
        """
            compute then energy density for BdG, equation(77) in page 39
            Note: the return value also include the pressure and densities
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
        energy_density = taus[0]/2.0 + taus[1]/2.0 + g_eff*abs(nu)**2
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

    def get_alphas(self, ns, d=0):
        if d==0:
            return (1.0, 1.0)
        elif d==1:
            return (0, 0, 0, 0)


class ASLDA(SLDA, FunctionalASLDA):
    pass
