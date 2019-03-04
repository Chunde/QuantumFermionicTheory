import os
import numpy as np
from mmf_hfb import tf_completion as tf
from scipy.optimize import brentq
from multiprocessing import Pool
import json
from json import dumps
from functools import partial
import itertools

tf.MAX_ITERATION = 200


class FFState(object):
    def __init__(self, mu=10, dmu=0.4, delta=1,
                 m=1, T=0, hbar=1, k_c=100, d=2, fix_g=False):
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
            self._g = self.get_g(mu=mu, dmu=0, delta=delta, r=np.inf)
        else:
            self._C = tf.compute_C(mu_a=mu, mu_b=mu, delta=delta, q=0,
                                   **self._tf_args).n
        
    def f(self, mu, dmu, delta, q=0, dq=0, **kw):
        args = dict(self._tf_args)
        args.update(kw)

        if self.fix_g:
            return self.get_g(mu=mu, dmu=dmu, q=q, dq=dq, delta=delta, **args) - self._g

        return tf.compute_C(delta=delta, q=q, dq=dq **args).n - self._C

    def get_g(self, delta, mu=None, dmu=None, q=0, dq=0, **kw):
        args = dict(self._tf_args, q=q, dq=dq, delta=delta)
        if mu is not None:
            args.update(mu_a=mu+dmu, mu_b=mu-dmu)
        nu_delta = tf.integrate_q(tf.nu_delta_integrand, **args)
        g = 1./nu_delta.n
        return g

    def get_densities(self, mu, dmu, q=0, dq=0, delta=None, k_c=None):
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
    
    def get_energy_density(self, mu, dmu, q=0, dq=0, delta=None,
                           n_a=None, n_b=None):
        if delta is None:
            delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq, 
                               a=self.delta * 0.8, b=self.delta * 1.2)
        if n_a is None:
            n_a, n_b = self.get_densities(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)

        args = dict(self._tf_args, mu_a=mu + dmu, mu_b=mu - dmu, delta=delta,
                    q=q, dq=dq)
            
        kappa = tf.integrate_q(tf.kappa_integrand, **args)
        if self.fix_g:
            g_c = self._g
        else:
            g_c = 1./self._C
        return kappa #  - g_c * n_a * n_b /2
    
    def get_pressure(self, mu, dmu, q=0, dq=0, delta=None, return_ns = False):
        if delta is None:
            delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq, 
                               a=self.delta * 0.1, b=self.delta * 2)
        #print(f"dq={dq}\tdelta={delta}")   
        n_a, n_b = self.get_densities(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)
        energy_density = self.get_energy_density(
            mu=mu, dmu=dmu, delta=delta, q=q, dq=dq,
            n_a=n_a, n_b=n_b)
        mu_a, mu_b = mu + dmu, mu - dmu
        pressure = mu_a * n_a + mu_b * n_b - energy_density
        if return_ns:
            return (pressure, n_a, n_b)
        return pressure

    def solve(self, mu=None, dmu=None, q=0, dq=0, a=0.8, b=1.2):
        args = dict(self._tf_args, q=q, dq=dq)
        if mu is not None:
            args.update(mu=mu, dmu=dmu, mu_a=mu+dmu, mu_b=mu-dmu)
        def f(delta):
            if self.fix_g:
                return self._g - self.get_g(delta=delta, mu=mu, dmu=dmu, q=q, dq=dq)
            return self._C - tf.compute_C(delta=delta, **args).n
        try:
            delta = brentq(f, a, b)
        except:
            delta = 0
        return delta

class FFStatePhaseMapper(object):

    def get_dmus(mu, delta0):
        """return the sample values for dmu"""
        dmus = np.linspace(0, 2*mu, 10)
        return dmus

    def get_dqs(mu, delta0):
        """return sample values for dq"""
        return np.linspace(0, delta0, 5)


    def find_delta_pressure(delta0, mu, dmu, dq, id):
        """compute detla and pressure"""
        ff = FFState(fix_g=True, mu=mu, dmu=dmu, delta=delta0, d=2, k_c=500, m=0, T=0)
        g = ff._g
        ds = np.linspace(0.1 * delta0, 2* delta0, 10)
        fs = [ff.f(mu=mu, dmu=dmu, delta=d, dq=dq) for d in ds]
        if(fs[0] * fs[-1] > 0):
            for i in range(len(fs)):
                if fs[0] * fs[i] < 0: #two solutions
                    d1 = ff.solve(mu=mu, dmu=dmu, dq= dq,a=ds[0],b = ds[i])
                    p1, na1, nb1 = ff.get_pressure(mu=mu, dmu=dmu, dq = dq, delta=d1 ,return_ns = True)
                    d2 = ff.solve(mu=mu, dmu=dmu, dq= dq,a=ds[i],b = ds[-1])
                    p2, na2, nb2 = ff.get_pressure(mu=mu, dmu=dmu, dq = dq, delta=d2, return_ns = True)
                    print(f"p1={p1:10.7}\tp2={p2:10.7}")
                    if(p2 > p1):
                        return (g, d2, p2.n, na2.n, nb2.n)
                    return (g, d1, p1.n, na1.n, nb1.n)
            return (g, 0, 0, 0, 0)
        else:
            d = ff.solve(mu=mu, dmu=dmu, dq= dq, a=ds[0], b=ds[-1])
            if d > 0:
                p, na, nb = ff.get_pressure(mu=mu, dmu=dmu, dq = dq, delta=d, return_ns = True)
                return (g, d, p.n, na.n, nb.n)
            return (g, d, 0, 0, 0)

    def compute_2d_phase_map(mu_delta_id):
        """compute press, density for given mu and delta"""
        print(f"-------------------------{mu_delta_id}-------------------------")
        mu, delta0, id = mu_delta_id
        dmus = FFStatePhaseMapper.get_dmus(mu, delta0)
        dqs = FFStatePhaseMapper.get_dqs(mu, delta0)
        output = dict(mu=mu, delta=delta0)
        data=[]
        for dmu in dmus:
            press0 = -np.inf
            for dq in dqs:
                g, delta, press, na, nb = FFStatePhaseMapper.find_delta_pressure(delta0=delta0, mu=mu, dmu=dmu, dq=dq, id=id)
                if press == 0:
                    break
                print(f"{id}\tdelta0={delta0:15.7}\tmu={mu:10.7}\tdmu={dmu:10.7}\tdq={dq:10.7}:\tg={g:10.7}\tdelta={delta:10.7}\tP={press:10.7}")
                if press > press0:
                    data.append((dmu, dq, g, delta, press, na, nb))
                    press0 = press
                else:
                    break
        output["data"]=data
        print(output)
        return output

    def compute_2d_phase_diagram():
        """using multple pools to compute 2d phase diagram"""
        kF = 1
        m = 1
        T = 0
        eF=kF**2/2/m
        mu0 = 0.5 * eF
        delta0 = np.sqrt(2.0) * eF
        mus = np.linspace(0,1.0,10) * mu0
        deltas = np.linspace(.0001,2,10) * delta0
        args = list(itertools.product(mus,deltas))
        for i in range(len(args)):
            args[i]=args[i] + (i,)

        logic_cpu_count = os.cpu_count() - 1
        logic_cpu_count = 1 if logic_cpu_count < 1 else logic_cpu_count
        with Pool(logic_cpu_count) as Pools:
            rets = Pools.map(FFStatePhaseMapper.compute_2d_phase_map,args)
            with open("2d_phase_map_data.txt",'w',encoding ='utf-8') as wf:
                json.dump(rets,wf, ensure_ascii=False)

class DeltaNSGenerator(object):
    def min_index(fs):
        min_value = fs[0]
        min_index = 0
        for i in range(1,len(fs)):
            if fs[i] < min_value:
                min_value = fs[i]
                min_index = i
        return min_index,min_value

    def compute_delta_ns(r, d ,mu=10, dmu=2, delta=1):
        ff = FFState(dmu=dmu, mu=mu, d=d, delta=delta, fix_g=False)
        b = 2*delta
        ds = np.linspace(0.01,2*delta,10)
        fs = [ff.f(delta=delta, dq=1.0/r, mu=mu, dmu=dmu) for delta in ds]
        index, value = DeltaNSGenerator.min_index(fs)
        delta = 0
        if value < 0:
            delta = ff.solve(dq=1.0/r, a=ds[index], b=b, mu=mu, dmu=dmu)
            if fs[0] > 0:
                smaller_delta = ff.solve(r=r,a=ds[0],b=ds[index], mu_a=mu+dmu, dmu=dmu)
                print(f"a smaller delta={smaller_delta} is found for r={r}")
                p1 = ff.get_pressure(delta=delta,dq=1.0/r, mu=mu, dmu=dmu)
                p2 = ff.get_pressure(delta=smaller_delta, dq=1.0/r, mu=mu ,dmu=dmu)
                if(p2 > p1):
                    delta = smaller_delta
        na,nb = ff.get_densities(delta=delta, dq=1.0/r, mu=mu, dmu=dmu)
        return (delta, na, nb)

    def worker_thread(r, delta):
        return DeltaNSGenerator.compute_delta_ns(r, d=2, delta=delta)

    def compute_ff_delta_ns_2d(delta):
        """Compute 2d FF State Delta, densities"""
        deltas2 = []
        na2 = []
        nb2 = []
        rs2 = np.linspace(0.001, 10, 100).tolist()


        logic_cpu_count = os.cpu_count() - 1
        logic_cpu_count = 1 if logic_cpu_count < 1 else logic_cpu_count
        with Pool(logic_cpu_count) as Pools:
            worker_thread_partial = partial(DeltaNSGenerator.worker_thread, delta=delta)
            rets = Pools.map(worker_thread_partial,rs2)
            for ret in rets:
                deltas2.append(ret[0])
                na2.append(ret[1].n)
                nb2.append(ret[2].n)
            outputs =[rs2,deltas2,na2,nb2]
            print(outputs)
            with open("delta_ns.txt",'w',encoding ='utf-8') as wf:
                json.dump(outputs,wf, ensure_ascii=False)

def generate_2d_phase_diagram():
    #FFStatePhaseMapper.compute_2d_phase_map(mu=mu, delta0=delta0)
    FFStatePhaseMapper.compute_2d_phase_diagram()

if __name__ == "__main__":
    # DeltaNSGenerator.compute_ff_delta_ns_2d(delta=5) #generate 2d data
    generate_2d_phase_diagram()