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

tf.MAX_DIVISION = 500

    
class FFState(object):
    def __init__(self, mu, dmu, delta=1, q=0, dq=0, m=1, T=0,hbar=1, k_c=100,
                   dim=2, fix_g=False,  bStateSentinel=False):
        """
        Arguments
        ---------
        fix_g : bool
           If fix_g is False, the class will fix C_tilde when
           performing the non-linear iterations, otherwise the
           coupling constant g_c will be fixed.  Note: this g_c will
           depend on the cutoff k_c.
        """
        self.fix_g = fix_g
        self.dim = dim
        self.T = T
        self.mu = mu
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
        self.bSuperfluidity = self.check_superfluidity(mu=mu, dmu=dmu, q=q, dq=dq)
        self.bPolorized = self.check_polarization(mu=mu, dmu=dmu, q=q, dq=dq)
        
    def f(self, mu, dmu, delta, q=0, dq=0, **kw):
        args = dict(self._tf_args)
        args.update(kw)
        if self.fix_g:
            return self.get_g(mu=mu, dmu=dmu, q=q, dq=dq, delta=delta, **args) - self._g

        return tf.compute_C(mu_a=mu+dmu, mu_b=mu-dmu, delta=delta, q=q, dq=dq, **args).n - self._C

    def get_g(self, delta, mu=None, dmu=None, q=0, dq=0, **kw):
        args = dict(self._tf_args, q=q, dq=dq, delta=delta)
        if mu is not None:
            args.update(mu_a=mu+dmu, mu_b=mu-dmu)
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
        if self.bStateSentinel:
             assert np.allclose(n_a.n, n_b.n) != self.bPolorized

        return n_a, n_b

     def get_current(self, mu, dmu, q=0, dq=0, delta=None, k_c=None):
        """return oveerall current"""
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
        return kappa  # - g_c * n_a * n_b /2
    
    def get_pressure(self, mu, dmu, q=0, dq=0, delta=None, return_ns = False):
        if delta is None:
            delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq, 
                               a=self.delta * 0.8, b=self.delta * 1.2)
        n_a, n_b = self.get_densities(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)
        energy_density = self.get_energy_density(
            mu=mu, dmu=dmu, delta=delta, q=q, dq=dq,
            n_a=n_a, n_b=n_b)
        mu_a, mu_b = mu + dmu, mu - dmu
        pressure = mu_a * n_a + mu_b * n_b - energy_density
        if return_ns:
            return (pressure, n_a, n_b)
        return pressure

    def check_superfluidity(self, mu=None, dmu=None, q=0, dq=0):
        """Check if a configuration will yield superfluid state"""
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

    def solve(self, mu=None, dmu=None, q=0, dq=0, a=None, b=None):
        if a is None:
            a = a=self.delta * 0.1
        if b is None:
            b = b=self.delta * 2
        args = dict(self._tf_args, q=q, dq=dq) # put only the max common set of varible in the dict
        def f(delta):
            if self.fix_g:
                return self._g - self.get_g(delta=delta, mu=mu, dmu=dmu, q=q, dq=dq)
            return self._C - tf.compute_C(delta=delta,mu_a=mu+dmu, mu_b=mu-dmu, **args).n
        try:
            delta = brentq(f, a, b)
        except ValueError: # It's important to deal with specific exception.
            delta = 0
        if self.bStateSentinel and self.bSuperfluidity is not None:
                assert self.bSuperfluidity == (delta > 0)
        return delta

if __name__ == "__main__":
    #generate_phase_diagram()
    mu = 10
    dmu = 0.4
    delta = 5
    k_c = 200
    dim = 1
    ff = FFState(mu=mu, dmu=dmu, delta=delta, dim=dim, k_c=k_c,fix_g=True)
    rs = np.linspace(0.1, 0.5, 20)
    rs = np.linspace(.25, 0.5, 20)
    qs = 1.0/rs