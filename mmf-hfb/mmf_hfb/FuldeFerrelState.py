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

class LogManager(object):
    def __init__(self, verbose=1):
        self.verbose=verbose
    
class FFState(object):
    def __init__(self, mu=10, delta=1,
                 m=1, T=0, hbar=1, k_c=100, dim=2, fix_g=False, verbose=1):
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
        self.verbose = verbose
        self._tf_args = dict(m_a=1, m_b=1, dim=dim, hbar=hbar, T=T, k_c=k_c)

        if fix_g:
            self._g = self.get_g(mu=mu, dmu=0, delta=delta)
        else:
            self._C = tf.compute_C(mu_a=mu, mu_b=mu, delta=delta,
                                   **self._tf_args).n
        
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
    def get_normal_energy_density(self, mu, dmu, q=0, dq=0):
        """return energy density when delta=0"""
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
        return kappa  # - g_c * n_a * n_b /2
    
    def get_pressure(self, mu, dmu, q=0, dq=0, delta=None, return_ns = False):
        if delta is None:
            delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq, 
                               a=self.delta * 0.1, b=self.delta * 2)
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
        args = dict(self._tf_args, q=q, dq=dq) # put only the max common set of varible in the dict
        def f(delta):
            if self.fix_g:
                return self._g - self.get_g(delta=delta, mu=mu, dmu=dmu, q=q, dq=dq)
            return self._C - tf.compute_C(delta=delta,mu_a=mu+dmu, mu_b=mu-dmu, **args).n
        try:
            delta = brentq(f, a, b)
        except ValueError: # It's important to deal with specific exception.
            delta = 0
        return delta

class FFStatePhaseMapper(object):

    def get_dmus(mu, delta0):
        """return the sample values for dmu"""
        dmus = np.linspace(0.5*mu, mu, 5)
        return dmus

    def get_dqs(mu, delta0):
        """return sample values for dq"""
        return np.linspace(0, 0.1 * delta0, 5)


    def find_delta_pressure(ff, delta0, mu, dmu, q, dq, id, dim=2):
        """compute detla and pressure"""
        ds = np.linspace(0.1 * delta0, 2* delta0, 10)
        fs = [ff.f(mu=mu, dmu=dmu, delta=d, q=q, dq=dq) for d in ds]
        if(fs[0] * fs[-1] > 0):
            for i in range(len(fs)):
                if fs[0] * fs[i] < 0: #two solutions
                    d1 = ff.solve(mu=mu, dmu=dmu, q=q, dq= dq,a=ds[0], b=ds[i])
                    p1 = ff.get_pressure(mu=mu, dmu=dmu, q=q, dq=dq, delta=d1)
                    d2 = ff.solve(mu=mu, dmu=dmu, q=q, dq= dq,a=ds[i], b = ds[-1])
                    p2= ff.get_pressure(mu=mu, dmu=dmu, q=q, dq=dq, delta=d2)
                    #print(f"{id}:p1={p1.n:10.7}\tp2={p2.n:10.7}")
                    if(p2 > p1):
                        return (d2, p2.n)
                    return (d1, p1.n)
            return ( 0, 0)
        else:
            d = ff.solve(mu=mu, dmu=dmu, q=q, dq=dq, a=ds[0], b=ds[-1])
            if d > 0:
                p = ff.get_pressure(mu=mu, dmu=dmu, q=q, dq=dq, delta=d)
                return (d, p.n)
            return (d, 0)

    def phase_map_worker_thread(id_q_mu_delta):
        """compute press, density for given mu and delta"""
        print(f"-------------------------{id_q_mu_delta}-------------------------")
        id, dim, q, mu, delta0 = id_q_mu_delta
        dmus = FFStatePhaseMapper.get_dmus(mu, delta0)
        dqs = FFStatePhaseMapper.get_dqs(mu, delta0)
        data=[]
        ff = FFState(fix_g=True, mu=mu, delta=delta0, dim=dim, k_c=100, m=0, T=0)
        na, nb = ff.get_densities(mu=mu, dmu=0, delta=delta0)
        output = dict(mu=mu, delta=delta0, q=q, g=ff._g, na=na.n, nb=nb.n)
        for dmu in dmus:

            delta, press = FFStatePhaseMapper.find_delta_pressure(ff=ff, delta0=delta0, mu=mu, dmu=dmu, q=q, dq=0, id=id, dim=dim)
            if delta > 0:
                continue
            max_press = press
            max_dq = 0
            max_delta = delta

            for dq in dqs:
                if dq == 0:
                    continue
                delta, press = FFStatePhaseMapper.find_delta_pressure(ff=ff, delta0=delta0, mu=mu, dmu=dmu, q=q, dq=dq, id=id, dim=dim)
                if delta == 0:
                    continue
                #if press == 0:
                #    break
                # print(f"{id}\tdelta0={delta0:15.7}\tmu={mu:10.7}\tdmu={dmu:10.7}\tdq={dq:10.7}:\tg={g:10.7}\tdelta={delta:10.7}\tP={press:10.7}")
                if press > max_press:
                    max_press = press
                    max_dq = dq
                    max_delta = delta
                    print(f"{id}:delta={delta}\tpress={press}")
                
            if max_press > -np.inf and max_delta > 0:
                data.append((dmu, max_dq, max_delta, max_press))
            if dmu > 0 and max_dq > 0 and max_delta > 0:
                print(f"{id}:Find one FF State{data[-1]}")

        output["data"]=data
        #print(output)
        return output

    def compute_phase_diagram(q=0.5, dim=2):
        """using multple pools to compute 2d phase diagram"""
        kF = 5.0
        m = 1
        T = 0
        eF=kF**2/2/m
        file_name = f"{dim}d_phase_map_data_" + time.strftime("%Y%m%d%H%M%S_")
        if dim == 1:
            mu0 = 0.28223521359741266 * eF
            delta0 = 0.41172622996179004 * eF
        elif dim == 2:
            mu0 = 0.5 * eF
            delta0 = np.sqrt(2.0) * eF
        else:
            mu0 = 0.59060550703283853378393810185221521748413488992993 * eF
            delta0 = 1.162200561790012570995259741628790656202543181557689 * mu0

        mus = np.linspace(1,2,10) * mu0
        deltas = np.linspace(0.001,2, 10) * delta0
        args = list(itertools.product(mus,deltas))
        for i in range(len(args)):
            args[i]=(i, dim, q,) + args[i]

        logic_cpu_count = os.cpu_count() - 1
        logic_cpu_count = 1 if logic_cpu_count < 1 else logic_cpu_count
        with Pool(logic_cpu_count) as Pools:
            rets = Pools.map(FFStatePhaseMapper.phase_map_worker_thread,args)
            file_name = file_name + time.strftime("%Y%m%d%H%M%S.txt")
            with open(file_name,'w',encoding ='utf-8') as wf:
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

    def compute_delta_ns(r, dim ,mu=10, dmu=2, delta=1):
        ff = FFState(dmu=dmu, mu=mu, dim=dim, delta=delta, fix_g=False)
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
        return DeltaNSGenerator.compute_delta_ns(r, dim=2, delta=delta)

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


def generate_phase_diagram():
    #FFStatePhaseMapper.compute_2d_phase_map(mu=mu, delta0=delta0)
    qs = np.linspace(0,1,10)
    for q in qs:
        FFStatePhaseMapper.compute_phase_diagram(q=q, dim=1)

if __name__ == "__main__":
    # DeltaNSGenerator.compute_ff_delta_ns_2d(delta=5) #generate 2d data
    generate_phase_diagram()
    #         id, dim, q, mu, delta0 = id_q_mu_delta

    #FFStatePhaseMapper.phase_map_worker_thread((0,1,0,3.5279401699676582, 1.1482587080045477))
