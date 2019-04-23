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
MAX_ITERATION = 50
    
class FFState(object):
    def __init__(self, mu, dmu, delta=1, q=0, dq=0, m=1, T=0, hbar=1, k_c=100,
                 dim=2, fix_g=True,  bStateSentinel=False):
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
        self.mus = (mu,dmu)
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
        assert (mu is None) == (dmu is None)
        if mu is None:
            mu, dmu = self.mus
        args = dict(self._tf_args, q=q, dq=dq, delta=delta)
        args.update(kw, mu_a=mu+dmu, mu_b=mu-dmu)
        nu_delta = tf.integrate_q(tf.nu_delta_integrand, **args)
        g = 1./nu_delta.n
        return g
    def get_current(self, mu=None, dmu=None, q=0, dq=0, delta=None, k_c=None):
        """return the densities of two the components"""
        assert (mu is None) == (dmu is None)
        if mu is None:
            mu, dmu = self.mus

        if delta is None:
            delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq, 
                               a=self.delta * 0.8, b=self.delta * 1.2)
        mu_a, mu_b = mu + dmu, mu - dmu
        args = dict(self._tf_args)
        return tf.compute_current(mu_a=mu_a, mu_b=mu_b, delta=delta, **args)
        
    def get_densities(self, mu=None, dmu=None, q=0, dq=0, delta=None, k_c=None):
        """return the densities of two the components"""
        assert (mu is None) == (dmu is None)
        if mu is None:
            mu, dmu = self.mus
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

    def _get_effetive_mus(self, mu, dmu, mus_eff=None, delta=None, q=0, dq=0
                          , k_c=np.inf, rtol=1e-6, update_g=False):
        """
        return effective mu, dmu
        ---------
        update_g: bool
            if it's true, the g_c will be update as the iteration goes
            this can be done just once time for each instance, but
            can be multiple times as needed
        """
        if self.dim != 1:
            return (mu, dmu)
        k_c = np.inf 
        update_delta = False
        if delta is None:
            if mus_eff is not None:
                delta = self.solve(mu=mus_eff[0], dmu=mus_eff[1], q=q, dq=dq, 
                               a=self.delta * 0.8, b=self.delta * 1.2)
                mu_a, mu_b = mus_eff[0] + mus_eff[1], mus_eff[0] - mus_eff[1]
            else:
                delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq, 
                               a=self.delta * 0.8, b=self.delta * 1.2)
            update_delta = True     
        args = dict(self._tf_args, mu_a=mu + dmu, mu_b=mu - dmu, delta=delta,
                    q=q, dq=dq)
        if k_c is not None:
            args['k_c'] = k_c
            
        n_p = tf.integrate_q(tf.n_p_integrand, **args).n
        n_m = tf.integrate_q(tf.n_m_integrand, **args).n
        n_a, n_b = (n_p + n_m)/2, (n_p - n_m)/2
        # for 1d case, we need to solve the densities self-consistently
        error = 1.0
        mu_a0, mu_b0 = mu + dmu, mu - dmu
        itr = 0
        while(error > rtol):
            itr = itr + 1
            mu_a_eff = mu_a0 -  self._g * n_b
            mu_b_eff = mu_b0 -  self._g * n_a
            args.update(mu_a=mu_a_eff, mu_b=mu_b_eff, delta=delta)
            if update_g:
                self._g =delta/tf.integrate_q(tf.nu_integrand, **args).n
            n_p = tf.integrate_q(tf.n_p_integrand, **args).n
            n_m = tf.integrate_q(tf.n_m_integrand, **args).n
            n_a, n_b = (n_p + n_m)/2, (n_p - n_m)/2
            mu_a, mu_b = mu_a_eff + self._g * n_b,  mu_b_eff + self._g * n_a
            mu_ = (mu_a + mu_b) / 2
            error = np.abs(mu_ - mu)/mu
            mu_eff = (mu_a_eff + mu_b_eff)/2
            dmu_eff = (mu_a_eff - mu_b_eff)/2 
            if update_delta:
                self.delta = self.solve(mu=mu_eff, dmu=dmu_eff, q=q, dq=dq,
                                  a=delta * 0.8, b=delta * 1.2)
            if itr > MAX_ITERATION:
                warnings.warn("""Reach max iteration without converging
                                 to desired accuracy""")
                break
        mu_a_eff = mu_a - self._g * n_b
        mu_b_eff = mu_b - self._g * n_a
        return ((mu_a_eff + mu_b_eff)/2, (mu_a_eff - mu_b_eff)/2)

    def get_ns_p_e_mus_1d(self, mu, dmu, mus_eff=None, delta=None,
                         q=0, dq=0, k_c=3000, update_g=False):
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
        assert self.dim == 1
        mu_eff, dmu_eff = self._get_effetive_mus(mu=mu, dmu=dmu, mus_eff=mus_eff, 
                                   delta=delta, q=q, dq=dq, k_c=k_c, update_g=update_g)
        if delta is None:
            delta = self.delta
        mu_a_eff = mu_eff + dmu_eff
        mu_b_eff = mu_eff - dmu_eff
        args = dict(self._tf_args, mu_a=mu_a_eff, mu_b=mu_b_eff, delta=delta,
                    q=q, dq=dq)
        n_p = tf.integrate_q(tf.n_p_integrand, **args).n
        n_m = tf.integrate_q(tf.n_m_integrand, **args).n
        n_a, n_b = (n_p + n_m)/2, (n_p - n_m)/2
        print(f"mu_eff={mu_eff}, dmu_eff={dmu_eff}, n_a={n_a}, n_b={n_b}, g_c={self._g}")
        kappa = tf.integrate_q(tf.kappa_integrand, **args).n
        e = kappa + self._g * n_a * n_b
        p = (mu+dmu) * n_a + (mu-dmu) * n_b - e
        return (n_a, n_b, e, p, ((mu_a_eff + mu_b_eff)/2, (mu_a_eff - mu_b_eff)/2))

    def get_FFG_energy_density(self, mu, dmu, q=0, dq=0):
        """return the Free Fermi Gas(FFG) energy density(delta=0)"""
        mu_a, mu_b = mu + dmu, mu - dmu
        if self.dim == 1:
            energy_density = np.sqrt(2)/np.pi *( mu_a**1.5 + mu_b**1.5)/3.0
            na, nb=np.sqrt(2 *mu_a)/np.pi,np.sqrt(2 * mu_b)/np.pi
        elif self.dim == 2:
            energy_density = (mu_a**2 + mu_b**2)/4.0/np.pi
            na, nb =mu_a/np.pi/2.0,mu_b/np.pi/2.0
        elif self.dim == 3:
            energy_density = (mu_a**2.5 + mu_b**2.5)*2.0**1.5/10.0/np.pi**2
            na, nb = ((2.0 *mu_a)**1.5)/6.0/np.pi**2,((2.0 * mu_b)**1.5)/6.0/np.pi**2
        return energy_density

    def get_energy_density(self, mu=None, dmu=None, q=0, dq=0, delta=None,
                           n_a=None, n_b=None, k_c=None):
        assert (mu is None) == (dmu is None)
        if mu is None:
            mu, dmu = self.mus
        if delta is None:
            delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq, 
                               a=self.delta * 0.8, b=self.delta * 1.2)
        if n_a is None:
            n_a, n_b = self.get_densities(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq, k_c=k_c)

        args = dict(self._tf_args, mu_a=mu + dmu, mu_b=mu - dmu, delta=delta, q=q, dq=dq)
        kappa = tf.integrate_q(tf.kappa_integrand, **args)
        if self.fix_g:
            g_c = self._g
        else:
            g_c = 1./self._C         
        if self.dim == 1:
            return kappa + g_c * n_a * n_b
        return kappa
    
    def get_pressure(self, mu=None, dmu=None, mu_eff=None, dmu_eff=None, q=0, dq=0, delta=None):
        """return the pressure"""
        assert (mu is None) == (dmu is None)
        assert (mu_eff is None) == (dmu_eff is None)
        if mu is None:
            mu, dmu = self.mus 
        if mu_eff is None:
            mu_eff, dmu_eff = self._get_effetive_mus(mu=mu, dmu=dmu, q=q, dq=dq)
        if delta is None:
            delta = self.solve(mu=mu_eff, dmu=dmu_eff, q=q, dq=dq, 
                               a=self.delta * 0.8, b=self.delta * 1.2)
        n_a, n_b = self.get_densities(mu=mu_eff, dmu=dmu_eff, delta=delta, q=q, dq=dq)
        energy_density = self.get_energy_density(
            mu=mu_eff, dmu=dmu_eff, delta=delta, q=q, dq=dq,
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
        assert (mu is None) == (dmu is None)
        if mu is None:
            mu, dmu = self.mus
        oldFlag = self.bStateSentinel
        self.bStateSentinel = False
        delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq,
                           a=self.delta * 0.8, b=self.delta * 1.2)
        self.bStateSentinel = oldFlag
        return delta > 0

    def solve(self, mu=None, dmu=None, q=0, dq=0, a=None, b=None, throwException=False):
        """
        On problem with brentq is that it requires very smooth function with a 
        and b having different sign of values, this can fail frequently if our
        integration is not with high accuracy. Should be solved in the future.
        """
        assert (mu is None) == (dmu is None)
        if mu is None:
            mu, dmu = self.mus

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
