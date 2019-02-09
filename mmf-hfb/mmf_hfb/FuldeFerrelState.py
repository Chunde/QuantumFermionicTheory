import os
import numpy as np
from mmf_hfb import homogeneous
from mmf_hfb import tf_completion as tf
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mmf_hfb.homogeneous import Homogeneous1D,Homogeneous3D
from multiprocessing import Pool
import json
from json import dumps

class FF(object):
    def __init__(self, mu=10, dmu=0.4, delta=1,
                 m=1, T=0, hbar=1, k_c=100, d=2):
        self.d = d
        self.T = T
        self.mu = mu
        self.m = m
        self.dmu = dmu
        self.delta = delta
        self.hbar = hbar
        self._tf_args = dict(m_a=1, m_b=1, d=d, hbar=hbar, T=T, k_c=k_c)
        self.C = tf.compute_C(mu_a=mu, mu_b=mu, delta=delta, q=0, **self._tf_args).n
        self._tf_args.update(mu_a=mu + dmu, mu_b=mu - dmu)
        
    def f(self, delta, r, **kw):
        args = dict(self._tf_args)
        args.update(kw)
        q = 1/r
        return tf.compute_C(delta=delta, q=q, **args).n - self.C
    
    def get_densities(self, delta, r):
        q = 1/r
        mu_a, mu_b = self.mu + self.dmu, self.mu - self.dmu
        mu_p,mu_m = (mu_a + mu_b)/2, (mu_a - mu_b)/2
        args = dict(mu_a=mu_a, mu_b=mu_b, delta=delta, hbar=self.hbar,
                    m_a=self.m, m_b=self.m, q=q, T=self.T)
        n_p = tf.integrate_q(tf.n_p_integrand,d=self.d, **args)
        n_m = tf.integrate_q(tf.n_m_integrand,d=self.d, **args)
        return (n_p + n_m)/2, (n_p - n_m)/2

    def get_pressure(self, delta, r):
        q = 1/r
        mu_a, mu_b = self.mu + self.dmu, self.mu - self.dmu
        mu_p,mu_m = (mu_a + mu_b)/2, (mu_a - mu_b)/2
        args = dict(mu_a=mu_a, mu_b=mu_b, delta=delta, hbar=self.hbar,
                    m_a=self.m, m_b=self.m, q=q, T=self.T)
  
        n_p = tf.integrate_q(tf.n_p_integrand,d=self.d, **args)
        n_m = tf.integrate_q(tf.n_m_integrand,d=self.d, **args)
        kappa = tf.integrate_q(tf.kappa_integrand,d=3,k_c=10.0, **args)
        pressure = mu_p * n_p + mu_m * n_m - kappa
        return pressure

        
    def solve(self, r, a=0.8, b=1.2):
        q = 1/r
        def f(delta):
            return self.C - tf.compute_C(delta=delta, q=q, **self._tf_args).n
        try:
            delta = brentq(f,a,b)
            return delta
        except:
            return 0


def min_index(fs):
    min_value = fs[0]
    min_index = 0
    for i in range(1,len(fs)):
        if fs[i] < min_value:
            min_value = fs[i]
            min_index = i
    return min_index,min_value


def compute_delta_n(r, d=1 ,mu=10, dmu=0.4):
    # return (1,2,3) # for quick debug
    ff = FF(dmu=dmu, mu=mu, d=d)
    ds = np.linspace(0,1.5,10)
    fs = [ff.f(delta=delta, r=r, mu_a=mu+dmu, mu_b=mu-dmu) for delta in ds]
    index, value = min_index(fs)
    delta = 0
    if value < 0:
        delta = ff.solve(r=r,a= ds[index])
        if fs[0] > 0:
            smaller_delta = ff.solve(r=r,a=ds[0],b=ds[index])
            print(f"a smaller delta={smaller_delta} is found for r={r}")
            p1 = ff.get_pressure(delta=delta,r=r)
            p2 = ff.get_pressure(delta=smaller_delta, r=r)
            if(p2 > p1):
                delta = smaller_delta
    na,nb = ff.get_densities(delta=delta, r=r)
    return (delta, na, nb)

def test_thermodynamic_relations():
    """Not work"""
    np.random.seed(1)
    m, hbar, kF = 1 + np.random.random(3)
    eF = (hbar*kF)**2/2/m
    nF = kF**3/3/np.pi**2
    mu = 0.59060550703283853378393810185221521748413488992993*eF
    q = 0
    delta = None# 0.68640205206984016444108204356564421137062514068346*eF
    delta = get_delta(mu_a= mu, mu_b=mu, m_a=m, m_b=m, q=q)
    args = dict( d=3, delta=delta, m_a=m, m_b=m, hbar=hbar,q=q, T=0.0)
    n_p = tf.integrate_q(tf.n_p_integrand, mu_a=mu,mu_b = mu,**args)
    n_m = tf.integrate_q(tf.n_m_integrand, mu_a=mu,mu_b = mu,**args)
    n_a = (n_p + n_m)/2
    n_b = (n_p - n_m)/2
    dmu = 1e-10
    args['delta'] = None
    n0 = get_pressure(mu_a=mu, mu_b=mu, **args)
    na1 = get_pressure(mu_a = mu - dmu,mu_b=mu, **args)
    na2 = get_pressure(mu_a = mu + dmu,mu_b=mu, **args)
    nb1 = get_pressure(mu_a = mu,mu_b=mu - dmu, **args)
    nb2 = get_pressure(mu_a = mu,mu_b=mu + dmu, **args)
    n_a_ = (na2 - na1)/2/dmu
    n_b_ = (nb2 - nb1)/2/dmu
    print((n_a,n_b),(n_a_,n_b_))
    assert np.allclose(n_a,n_a_)
    assert np.allclose(n_b,n_b_)

def plot_pressure():
    """Plot how pressure changes with q"""
    np.random.seed(1)
    m, hbar, kF = 1 + np.random.random(3)
    eF = (hbar*kF)**2/2/m
    nF = kF**3/3/np.pi**2
    mu = 0.59060550703283853378393810185221521748413488992993*eF
    delta = 0.68640205206984016444108204356564421137062514068346*eF
    args = dict(mu_a=mu, mu_b=mu, delta=delta, m_a=m, m_b=m, hbar=hbar, T=0.0)
    #p0 = get_pressure(mu_a = mu,mu_b=mu,delta=delta,m=m,T=0,q=1)
    qs = np.linspace(0,2,10)
    dmu = 0.4 * delta
    ps = [get_pressure(mu_a = mu +  q*dmu /2, mu_b = mu - q*dmu/2, delta=delta,m_a=m, m_b=m, T=0, q=0).n for q in qs]
    plt.plot(qs,ps)
    print(f'Delta={delta} mu={mu} ')
    plt.show()



def work_thread(r):
    return compute_delta_n(r, d=2)

def compute_ff_delta_ns_2d():
    """Compute 2d FF State Delta, densities"""
    deltas2 = []
    na2 = []
    nb2 = []
    rs2 = np.append(np.linspace(0.1,1,10),[np.linspace(2,4,20),np.linspace(4.1,8,20)]).tolist()#np.linspace(1,3,3).tolist() #
    logic_cpu_count = os.cpu_count()
    logic_cpu_count = 1 if logic_cpu_count < 1 else logic_cpu_count
    with Pool(logic_cpu_count) as Pools:
        rets = Pools.map(work_thread,rs2)
        for ret in rets:
            deltas2.append(ret[0])
            na2.append(ret[1].n)
            nb2.append(ret[2].n)
        outputs =[rs2,deltas2,na2,nb2]
        print(outputs)
        with open("delta_ns.txt",'w',encoding ='utf-8') as wf:
            json.dump(outputs,wf, ensure_ascii=False)

if __name__ == "__main__":
    #test_thermodynamic_relations()
    compute_ff_delta_ns_2d()
