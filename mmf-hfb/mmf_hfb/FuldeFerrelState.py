import os
import numpy as np
from mmf_hfb import tf_completion as tf
from scipy.optimize import brentq
from multiprocessing import Pool
import json
from json import dumps
from functools import partial
import itertools
import time
import warnings
tf.MAX_DIVISION = 500
MAX_ITERATION = 20
    
class FFState(object):
    def __init__(self, mu, dmu, delta=1, q=0, dq=0, m=1, T=0, hbar=1, k_c=100,
                 dim=2, fix_g=False,  bStateSentinel=False):
        """
        Arguments
        ---------
        fix_g : bool
           If fix_g is False, the class will fix C_tilde when
           performing the non-linear iterations, otherwise the
           coupling constant g_c will be fixed.  Note: this g_c will
           depend on the cutoff k_c.
        bStateSentinel: bool
           if bStateSentinel is True, the solve function will check
           if the resulted delta is zero or not, if it's different
           from the initial delta, assert will be failed. It's used
           to make sure the state is always in superfluid state or
           normal state when using the instance to compute densities,
           pressures etc.
        """
        self.fix_g = fix_g
        self.dim = dim
        self.T = T
        self.mu = mu
        self.dmu = dmu
        self.m = m
        self.delta = delta
        self.hbar = hbar
        self.k_c = k_c
        self.bStateSentinel = bStateSentinel
        self._tf_args = dict(m_a=1, m_b=1, dim=dim, hbar=hbar, T=T, k_c=k_c)
        if fix_g:
            self._g = self.get_g(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)
        else:
            self._C = tf.compute_C(mu_a=mu, mu_b=mu, 
                                   delta=delta, q=q, dq=dq, **self._tf_args).n
        self.bSuperfluidity = delta > 0
        
    def f(self, mu, dmu, delta, q=0, dq=0, **kw):
        args = dict(self._tf_args)
        args.update(kw)
        if self.fix_g:
            return self.get_g(mu=mu, dmu=dmu, q=q, dq=dq, delta=delta, **args) - self._g

        return tf.compute_C(mu_a=mu+dmu, mu_b=mu-dmu, delta=delta, q=q, dq=dq, **args).n - self._C

    def get_g(self, delta, mu=None, dmu=None, q=0, dq=0, **kw):
        args = dict(self._tf_args, q=q, dq=dq, delta=delta)
        if mu is None:
            mu = self.mu
        if dmu is None:
            dmu = self.dmu
        args.update(kw, mu_a=mu+dmu, mu_b=mu-dmu)
        nu_delta = tf.integrate_q(tf.nu_delta_integrand, **args)
        g = 1./nu_delta.n
        return g

    def get_densities(self, mu, dmu, q=0, dq=0, delta=None, k_c=None):
        """return the densities of two the components"""
        if delta is None:
            delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq, 
                               a=self.delta * 0.8, b=self.delta * 1.2)

        args = dict(self._tf_args, mu_a=mu + dmu, mu_b=mu - dmu, delta=delta,
                    q=q, dq=dq)
        if k_c is not None:
            args['k_c'] = k_c
            
        n_p = tf.integrate_q(tf.n_p_integrand, **args)
        n_m = tf.integrate_q(tf.n_m_integrand, **args)
        n_a, n_b = (n_p + n_m)/2, (n_p - n_m)/2
        return n_a, n_b

    def get_ns_p_e_mus_1d(self, mu, dmu, mus_eff=None, delta=None,
                         q=0, dq=0, k_c=None, rtol=1e-5, update_g=False):
        """
        return the particle densities, pressure, energy density,
            effective mus for 1d using self-consistent method
        Arguments
        ---------
        mu: The bare mu (mu_a + mu_b)/2
        dmu: The bare chemical potential difference (mu_a - mu_b)/2
        delta: Nonable
            if delta is None, its value will be solved using mu and dmu
        mus_eff :(mu, dmu) 
            Effective mu, dmu that can be used to evaluate
            the gap equation in the beginning of the iteration if
            it's not None.
        update_g: bool
            Indicate if the g_c should be updated in the iteration
            By default its value is False, means the g_c will be fixed
        """

        self.dim = 1
        self._tf_args.update(dim=1)
        update_delta = False
        if delta is None:
            if mus_eff is not None:
                delta = self.solve(mu=mus_eff[0], dmu=mus_eff[1], q=q, dq=dq, 
                               a=self.delta * 0.8, b=self.delta * 1.2)
            else:
                delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq, 
                               a=self.delta * 0.8, b=self.delta * 1.2)
            update_delta = True

        args = dict(self._tf_args, mu_a=mu + dmu, mu_b=mu - dmu, delta=delta,
                    q=q, dq=dq)
        if k_c is not None:
            args['k_c'] = k_c
            
        n_p = tf.integrate_q(tf.n_p_integrand, **args)
        n_m = tf.integrate_q(tf.n_m_integrand, **args)
        n_a, n_b = (n_p + n_m)/2, (n_p - n_m)/2
        # for 1d case, we need to solve the densities self-consistently
        error = 1.0
        mu_a0 = mu + dmu
        mu_b0 = mu - dmu
        itr = 0
        while(error > rtol):
            itr = itr + 1
            mu_a = mu_a0 +  self._g * n_b.n
            mu_b = mu_b0 +  self._g * n_a.n
            mu = (mu_a + mu_b) / 2
            dmu = (mu_a - mu_b) /2 
            if update_g:
                self._g = self.get_g(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)
            if update_delta:
                delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq,
                                  a=delta * 0.8, b=delta * 1.2)
                print(f"Delta={delta}")

            args.update(mu_a=mu_a, mu_b=mu_b, delta=delta)
            n_p = tf.integrate_q(tf.n_p_integrand, **args)
            n_m = tf.integrate_q(tf.n_m_integrand, **args)
            n_a_, n_b_ = (n_p + n_m)/2, (n_p - n_m)/2
            error = np.sqrt((n_a.n - n_a_.n)**2 + (n_b.n - n_b_.n)**2)/n_p.n
            n_a = n_a_
            n_b = n_b_
            # print(error, n_a.n, n_b.n, self._g)
            if itr > MAX_ITERATION:
                warnings.warn("""Reach max iteration without converging
                                 to desired accuracy""")
                break
        mu_a = mu_a0 + self._g * n_b.n
        mu_b = mu_b0 + self._g * n_a.n
        print(f"mu={(mu_a + mu_b)/2}, dmu={(mu_a - mu_b)/2}, n_a={n_a.n}, n_b={n_b.n}, g_c={self._g}")

        args = dict(self._tf_args, mu_a=mu + dmu, mu_b=mu - dmu, delta=delta,
                    q=q, dq=dq)
        kappa = tf.integrate_q(tf.kappa_integrand, **args)
        e = kappa - self._g * n_a * n_b /2
        p = mu_a * n_a + mu_b * n_b - e
        return (n_a.n, n_b.n, e.n, p.n, ((mu_a + mu_b)/2, (mu_a - mu_b)/2))

    def get_current(self, mu, dmu, q=0, dq=0, delta=None, k_c=None):
        """return overall current"""
        n_a, n_b = self.get_densities(mu=mu, dmu=dmu, q=q, dq=dq, delta=delta, k_c=k_c)
        return n_a * (q + dq) + n_b * (q - dq)

    def get_FFG_energy_density(self, mu, dmu, q=0, dq=0):
        """return the Free Fermi Gas(FFG) energy density(delta=0)"""
        mu_a, mu_b = mu + dmu, mu - dmu
        if self.dim == 1:
            energy_density = np.sqrt(2)/np.pi *( mu_a**1.5 + mu_b**1.5)/3.0
            na, nb=np.sqrt(2 *mu_a)/np.pi,np.sqrt(2 * mu_b)/np.pi
            n_p = na + nb
        elif self.dim == 2:
            energy_density = (mu_a**2 + mu_b**2)/4.0/np.pi
            na, nb =mu_a/np.pi/2.0,mu_b/np.pi/2.0
            n_p = mu/np.pi
        elif self.dim == 3:
            energy_density = (mu_a**2.5 + mu_b**2.5)*2.0**1.5/10.0/np.pi**2
            na, nb = ((2.0 *mu_a)**1.5)/6.0/np.pi**2,((2.0 * mu_b)**1.5)/6.0/np.pi**2
            n_p = ((2.0 *mu_a)**1.5 + (2.0 * mu_b)**1.5)/6.0/np.pi**2
        return energy_density

    def get_energy_density(self, mu, dmu, q=0, dq=0, delta=None,
                           n_a=None, n_b=None, k_c=None):
        if delta is None:
            delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq, 
                               a=self.delta * 0.8, b=self.delta * 1.2)

        if n_a is None:
            n_a, n_b = self.get_densities(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq, k_c=k_c)

        args = dict(self._tf_args, mu_a=mu + dmu, mu_b=mu - dmu, delta=delta,
                    q=q, dq=dq)
        kappa = tf.integrate_q(tf.kappa_integrand, **args)
        if self.fix_g:
            g_c = self._g
        else:
            g_c = 1./self._C
         
        #if self.dim == 1: # will fail for 1d if add the [- g_c * n_a * n_b /2] term
        #    return kappa - g_c * n_a * n_b /2
        return kappa
    
    def get_pressure(self, mu=None, dmu=None, q=0, dq=0, delta=None):
        """return the pressure"""
        if mu is None:
            mu = self.mu
        if dmu is None:
            dmu = self.dmu
        if delta is None:
            delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq, 
                               a=self.delta * 0.8, b=self.delta * 1.2)
        n_a, n_b = self.get_densities(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)
        energy_density = self.get_energy_density(
            mu=mu, dmu=dmu, delta=delta, q=q, dq=dq,
            n_a=n_a, n_b=n_b)
        mu_a, mu_b = mu + dmu, mu - dmu
        pressure = mu_a * n_a + mu_b * n_b - energy_density
        return pressure

    def check_superfluidity(self, mu=None, dmu=None, q=0, dq=0):
        """
        Check if a configuration will yield superfluid state.
        May yield wrong results as the solve routine not 
        always works properly
        """
        oldFlag = self.bStateSentinel
        self.bStateSentinel = False
        delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq,
                           a=self.delta * 0.8, b=self.delta * 1.2)
        self.bStateSentinel = oldFlag
        return delta > 0

    def check_polarization(self, mu=None, dmu=None, q=0, dq=0):
        """Check if a configuration will yield polarized state"""
        oldFlag = self.bStateSentinel
        self.bStateSentinel = False
        n_a, n_b = self.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
        self.bStateSentinel = oldFlag
        return not np.allclose(n_a.n, n_b.n)

    def solve(self, mu=None, dmu=None, q=0, dq=0, a=None, b=None, throwException=False):
        """
        On problem with brentq is that it requires very smooth function with a 
        and b having different sign of values, this can fail frequently if our
        integration is not with high accuracy. Should be solved in the future.
        """
        if a is None:
            a = a=self.delta * 0.1
        if b is None:
            b = b=self.delta * 2
        args = dict(self._tf_args, q=q, dq=dq) 
        def f(delta):
            if self.fix_g:
                return self._g - self.get_g(delta=delta, mu=mu, dmu=dmu, q=q, dq=dq)
            return self._C - tf.compute_C(delta=delta,mu_a=mu+dmu, mu_b=mu-dmu, **args).n
        if  throwException:
             delta = brentq(f, a, b)
        else:
            try:
                delta = brentq(f, a, b)
            except ValueError: # It's important to deal with specific exception.
                delta = 0

        if self.bStateSentinel:
                assert self.bSuperfluidity == (delta > 0)
        return delta
