import os
import numpy as np
from mmf_hfb import tf_completion as tf
from scipy.optimize import brentq
from multiprocessing import Pool
import json
from json import dumps

tf.set_max_iteration(200)

class FFState(object):
    def __init__(self, mu=10, dmu=0.4, delta=1,
                 m=1, T=0, hbar=1, k_c=100, d=2,fix_g=False):
        """Compute a double integral.

        Note: The order of arguments is not the same as dblquad.  They are
        func(x, y) here.
    
        Arguments
        ---------
        fix_g : bool
           If fix_g is False, the class will fix C_tilde when
           performing the non-linear iterations, otherwise the
           coupling constant g_c will be fixed.  Note: this g_c will
           depend on the cutoff k_c.
        """
        self.fix_g = fix_g
        self.d = d
        self.T = T
        self.mu = mu
        self.m = m
        self.dmu = dmu
        self.delta = delta
        self.hbar = hbar
        self.k_c = k_c
        self._tf_args = dict(m_a=1, m_b=1, d=d, hbar=hbar, T=T, k_c=k_c)

        if fix_g:
            self._g = self.get_g(mu_a=mu, mu_b=mu, delta=delta, r=np.inf)
        else:
            self._C = tf.compute_C(mu_a=mu, mu_b=mu, delta=delta, q=0,
                                   **self._tf_args).n
            
        self._tf_args.update(mu_a=mu + dmu, mu_b=mu - dmu)
        
    def f(self, delta, r, **kw):
        args = dict(self._tf_args)
        args.update(kw)
        q = 1/r
        return tf.compute_C(delta=delta, q=q, **args).n - self._C

    def get_g(self, r, delta, mu_a=None, mu_b=None):
        q = 1/r
        args = dict(self._tf_args, q=q, delta=delta)
        if mu_a is not None:
            args.update(mu_a=mu_a, mu_b=mu_b)
        nu_delta = tf.integrate_q(tf.nu_delta_integrand, **args)
        g = 1./nu_delta.n
        return g

    def get_densities(self, mu_a, mu_b, r, delta=None):
        q = 1/r
        if delta is None:
            delta = self.solve(r=r, mu_a=mu_a, mu_b=mu_b)
        args = dict(self._tf_args, mu_a=mu_a, mu_b=mu_b, delta=delta, q=q)
        n_p = tf.integrate_q(tf.n_p_integrand, **args)
        n_m = tf.integrate_q(tf.n_m_integrand, **args)
        n_a, n_b = (n_p + n_m)/2, (n_p - n_m)/2
        return n_a, n_b
    
    def get_energy_density(self, mu_a, mu_b, r,delta=None,  n_a=None, n_b=None):
        q = 1/r
        if delta is None:
            delta = self.solve(r=r, mu_a=mu_a, mu_b=mu_b)
        mu_p, mu_m = (mu_a + mu_b)/2, (mu_a - mu_b)/2
        args = dict(self._tf_args, mu_a=mu_a, mu_b=mu_b, delta=delta, q=q)
        if n_a is None:
            n_a, n_b = self.get_densities(mu_a=mu_a, mu_b=mu_b, delta=delta, r=r)
        kappa = tf.integrate_q(tf.kappa_integrand, **args)
        g_c = 1/self._C
        return kappa #  - 0*g_c * n_a * n_b 
    
    def get_pressure(self, mu_a, mu_b, r, delta=None):
        q = 1/r
        
        if delta is None:
            delta = self.solve(r=r, mu_a=mu_a, mu_b=mu_b)
            
        args = dict(self._tf_args, mu_a=mu_a, mu_b=mu_b, delta=delta, q=q)
        n_a, n_b = self.get_densities(mu_a=mu_a, mu_b=mu_b, delta=delta, r=r)
        energy_density = self.get_energy_density(
            n_a=n_a, n_b=n_b, mu_a=mu_a, mu_b=mu_b, delta=delta, r=r)
        pressure = mu_a * n_a + mu_b * n_b - energy_density
        return pressure

    def solve(self, r, mu_a=None, mu_b=None, a=0.8, b=1.2):
        q = 1/r
        args = dict(self._tf_args, q=q)
        if mu_a is not None:
            args.update(mu_a=mu_a, mu_b=mu_b)
        
        def f(delta):
            if self.fix_g:
                return self._g - self.get_g(r=r, delta=delta, mu_a=mu_a, mu_b=mu_b)
            return self._C - tf.compute_C(delta=delta, **args).n
        try:
            delta = brentq(f, a, b)
        except:
            delta = 0
        return delta


def min_index(fs):
    min_value = fs[0]
    min_index = 0
    for i in range(1,len(fs)):
        if fs[i] < min_value:
            min_value = fs[i]
            min_index = i
    return min_index,min_value

def compute_delta_ns(r, d ,mu=10, dmu=0.4):
    ff = FFState(dmu=dmu, mu=mu, d=d, fix_g=False)
    ds = np.linspace(0.1,1.5,10)
    fs = [ff.f(delta=delta, r=r, mu_a=mu+dmu, mu_b=mu-dmu) for delta in ds]
    index, value = min_index(fs)
    delta = 0
    if value < 0:
        delta = ff.solve(r=r,a= ds[index], mu_a=mu+dmu, mu_b=mu-dmu)
        if fs[0] > 0:
            smaller_delta = ff.solve(r=r,a=ds[0],b=ds[index], mu_a=mu+dmu, mu_b=mu-dmu)
            print(f"a smaller delta={smaller_delta} is found for r={r}")
            p1 = ff.get_pressure(delta=delta,r=r, mu_a=mu+dmu, mu_b=mu-dmu)
            p2 = ff.get_pressure(delta=smaller_delta, r=r, mu_a=mu+dmu, mu_b=mu-dmu)
            if(p2 > p1):
                delta = smaller_delta
    na,nb = ff.get_densities(delta=delta, r=r, mu_a=mu+dmu, mu_b=mu-dmu)
    return (delta, na, nb)

def worker_thread(r):
    return compute_delta_ns(r, d=2)

def compute_ff_delta_ns_2d():
    """Compute 2d FF State Delta, densities"""
    deltas2 = []
    na2 = []
    nb2 = []
    rs2 = np.linspace(0.1,10,100).tolist() #np.append(np.linspace(0.1,1,10),[np.linspace(2,4,20),np.linspace(4.1,8,20)]).tolist()#
    logic_cpu_count = os.cpu_count() - 1
    logic_cpu_count = 1 if logic_cpu_count < 1 else logic_cpu_count
    with Pool(logic_cpu_count) as Pools:
        rets = Pools.map(worker_thread,rs2)
        for ret in rets:
            deltas2.append(ret[0])
            na2.append(ret[1].n)
            nb2.append(ret[2].n)
        outputs =[rs2,deltas2,na2,nb2]
        print(outputs)
        with open("delta_ns.txt",'w',encoding ='utf-8') as wf:
            json.dump(outputs,wf, ensure_ascii=False)


def simple_test():
    mu=10
    dmu=0.4
    delta= 0
    m_a=m_b=1
    T=0
    q=0.2222222222222222
    d=2
    k_c=100
    tf.compute_C(mu_a = mu + dmu, mu_b = mu - dmu, delta=delta, m_a=m_a, m_b=m_b, d=d, k_c=k_c, T=T, q = q)

if __name__ == "__main__":
    compute_ff_delta_ns_2d() #generate 2d data
    # compute_delta_ns(5, d=2) #produce division by zero error
    #compute_delta_ns(0.1, d=2) #produce warnings
