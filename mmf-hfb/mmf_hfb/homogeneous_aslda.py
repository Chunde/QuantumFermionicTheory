"""ASLDA class
This module provides a ASLDA method for solving the polarized
two-species Fermi gas with short-range interaction.
"""
from mmf_hfb.Functionals import IFunctional, FunctionalASLDA, FunctionalBdG, FunctionalSLDA
from mmf_hfb import tf_completion as tf
import numpy as np
import numpy
from scipy.optimize import brentq


class BDG(FunctionalBdG):
    
    def __init__(
        self, mu_eff, dmu_eff, delta=1, q=0, dq=0,
            m=1, T=0, hbar=1, k_c=None, dim=3):
        FunctionalBdG.__init__(self)
        kcs=[np.inf, 1000, 50]
        if k_c is None:
            k_c = kcs[dim - 1]
        self.dim = dim
        self.T = T
        self.mus_eff = (mu_eff, dmu_eff)
        self.m = m
        self.delta = delta
        self.hbar = hbar
        self.k_c = k_c
        self._tf_args = dict(m_a=1, m_b=1, dim=dim, hbar=hbar, T=T, k_c=k_c)
        self._g = self.get_g(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta, q=q, dq=dq)

    def get_g(self, delta, mu_eff=None, dmu_eff=None, q=0, dq=0, **kw):
        assert (mu_eff is None) == (dmu_eff is None)
        if mu_eff is None:
            mu_eff, dmu_eff = self.mus_eff
        args = dict(self._tf_args, q=q, dq=dq, delta=delta)
        args.update(kw, mu_a=mu_eff + dmu_eff, mu_b=mu_eff - dmu_eff)
        nu_delta = tf.integrate_q(tf.nu_delta_integrand, **args)
        g = 1./nu_delta.n
        return g

    def get_densities(self, mu_eff, dmu_eff, delta, q=0, dq=0, k_c=None):
        """return the densities of two the components"""
        if delta is None:
            delta = self.solve(mu_eff=mu_eff, dmu_eff=dmu_eff,
                               a=self.delta * 0.8, b=self.delta * 1.2)
        args = dict(self._tf_args, mu_a=mu_eff + dmu_eff,
                    mu_b=mu_eff - dmu_eff, delta=delta, q=q, dq=dq)
        if k_c is not None:
            args['k_c'] = k_c
        n_p = tf.integrate_q(tf.n_p_integrand, **args)
        n_m = tf.integrate_q(tf.n_m_integrand, **args)
        tau_p = tf.integrate_q(tf.tau_p_integrand, **args)
        tau_m = tf.integrate_q(tf.tau_m_integrand, **args)
        kappa = tf.integrate_q(tf.kappa_integrand, **args)
        js = tf.compute_current(**args)
        n_a, n_b = (n_p + n_m)/2, (n_p - n_m)/2
        tau_a, tau_b = (tau_m + tau_p)/2, (tau_p - tau_m)/2
        return ((n_a, n_b), (tau_a, tau_b), js, kappa)

    def _get_bare_mus(self, mus_eff, ns=None, delta=None, q=0, dq=0):
        if self.dim != 1:
            return mus_eff
        if delta is None:
            delta = self.delta
        mu_eff, dmu_eff = mus_eff
        mu_a_eff, mu_b_eff = mu_eff + dmu_eff, mu_eff - dmu_eff
        args = dict(self._tf_args, mu_a=mu_a_eff, mu_b=mu_b_eff, delta=delta, q=q, dq=dq)
        if ns is None:
            n_p = tf.integrate_q(tf.n_p_integrand, **args).n
            n_m = tf.integrate_q(tf.n_m_integrand, **args).n
            n_a, n_b = (n_p + n_m)/2, (n_p - n_m)/2
        else:
            n_a, n_b = ns
        mu_a, mu_b = mu_a_eff + self._g*n_b, mu_b_eff + self._g*n_a
        return (mu_a + mu_b)/2, (mu_a - mu_b)/2

    def f(self, mu_eff, dmu_eff, delta, q=0, dq=0, **kw):
        args = dict(self._tf_args)
        args.update(kw)
        return self._g - self.get_g(mu_eff=mu_eff, dmu_eff=dmu_eff, q=q, dq=dq, delta=delta, **args)

    def solve(self, mu_eff=None, dmu_eff=None, q=0, dq=0,
                a=None, b=None, throwException=False, **args):
        """
        On problem with brentq is that it requires very smooth function with a 
        and b having different sign of values, this can fail frequently if our
        integration is not with high accuracy. Should be solved in the future.
        """
        assert (mu_eff is None) == (dmu_eff is None)
        if mu_eff is None:
            mu_eff, dmu_eff = self.mus_eff

        if a is None:
            a = self.delta * 0.1
        if b is None:
            b = self.delta * 2

        def f(delta):
            return self.f(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta, q=q, dq=dq)

        self._delta = None  # a another possible solution
        if throwException:
            delta = brentq(f, a, b)
        else:
            try:
                delta = brentq(f, a, b)
            except ValueError:  # It's important to deal with specific exception.
                offset = 0
                if not np.allclose(abs(dmu_eff), 0):
                    offset = min(abs(dq/dmu_eff), 100)
                ds = np.linspace(0, max(a, b) * (2 + offset), min(100, int((2 + offset)*10)))

                assert len(ds) <=100
                f0 = f(ds[-1])
                index0 = -1
                delta = 0
                for i in reversed(range(0, len(ds)-1)):
                    f_ = f(ds[i])
                    if f0 * f_ < 0:
                        delta = brentq(f, ds[index0], ds[i])
                        if f_ * f(ds[0]) < 0:  # another solution
                            delta_ = brentq(f, ds[0], ds[i])
                            self._delta = delta_
                            print(f"Another solution delta={delta_} was found, return the one with higher pressure")
                        break
                    else:
                        f0 = f_
                        index0 = i
                if delta == 0 and (f(0.999 * self.delta) * f(1.001 * self.delta) < 0):
                    delta = brentq(f, 0.999 *self.delta, 1.001 *self.delta)
        return delta

    def get_ns_e_p(self, mus_eff, delta, use_kappa=True, **args):
        """
            compute then energy density for BdG, equation(76) in page 39
            Note:
                the return value also include the pressure and densities
        """

        mu_eff, dmu_eff = mus_eff
        if delta is None:
            delta = self.solve(mu_eff=mu_eff, dmu_eff=dmu_eff,
                               a=self.delta * 0.8, b=self.delta * 1.2)
        ns, taus, _, kappa = self.get_densities(
            mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta)
        args.update(self._tf_args, mu_a=mu_eff + dmu_eff,
                    mu_b=mu_eff - dmu_eff, delta=delta)
        if use_kappa:
            energy_density = kappa
        else:
            nu = tf.integrate_q(tf.nu_integrand, **args)
            energy_density = sum(taus)/2.0 + self._g * abs(nu)**2
        if self.dim == 1:
            energy_density = energy_density + self._g * np.prod(ns)
        mu, dmu = self._get_bare_mus(mus_eff, ns=ns, delta=delta)
        mu_a, mu_b = mu + dmu, mu - dmu
        pressure = ns[0]*mu_a + ns[1]*mu_b - energy_density
        return (ns, energy_density, pressure)


class SLDA(BDG, FunctionalSLDA):

    def get_ns_e_p(self, mus_eff, delta, use_kappa=False, **args):
        """
            compute then energy density for BdG, equation(77) in page 39
            Note:
                the return value also include the pressure and densities
        """

        mu_eff, dmu_eff = mus_eff
        if delta is None:
            delta = self.solve(mu_eff=mu_eff, dmu_eff=dmu_eff,
                               a=self.delta * 0.8, b=self.delta * 1.2)
        args.update(self._tf_args, mu_a=mu_eff + dmu_eff,
                    mu_b=mu_eff - dmu_eff, delta=delta)

        ns, taus, _, kappa = self.get_densities(
            mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta)
        alpha_a, alpha_b = self.get_alphas(ns=ns)
        D = self.get_D(ns=ns)
        assert np.allclose(alpha_a, alpha_b)
        if use_kappa:
            energy_density = kappa
        else:
            nu = tf.integrate_q(tf.nu_integrand, **args)
            energy_density = sum(taus)*alpha_a/2.0 + self._g * abs(nu)**2
        energy_density = energy_density + D
        if self.dim == 1:
            energy_density = energy_density + self._g * np.prod(ns)
        mus = self._get_bare_mus(mus_eff, ns=ns, delta=delta)
        pressure = ns[0] * mus[0] + ns[1]*mus[1] - energy_density

        return (ns, energy_density, pressure)


class ASLDA(SLDA, FunctionalASLDA):
    pass
    