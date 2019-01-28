import numpy as np
from scipy.optimize import brentq
import homogeneous
import tf_completion as tf
import matplotlib.pyplot as plt
import ParallellelHelpers
from ParallellelHelpers import ParallelAgent
from homogeneous import Homogeneous1D,Homogeneous3D
from multiprocessing import Pool

def special_momenta(kz, kp,q, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    ka2 = (kz+q)**2 + kp**2
    kb2 = (kz-q)**2 + kp**2
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    return e_m**2 - e_p**2 - delta**2 # this will be zero at the boundary of interested

def get_n_p(args,q):
    n_p = tf.integrate_q(tf.n_p_integrand, d=3, q=q,**args)
    return  n_p

def get_n_m(args,q):
    n_m = tf.integrate_q(tf.n_m_integrand, d=3, q=q,**args)
    return n_m

def get_kappa(args,q):
    kappa = tf.integrate_q(tf.kappa_integrand,d=3,q=q,k_c=10.0,**args)
    return kappa

def get_inv_scattering_len(mu_a,mu_b,delta,m=1, T=0,hbar=1, q=0,Ec=1000):
    args_ = dict(mu_a=mu, mu_b=mu, delta=delta, m=m, hbar=hbar, T=0.0)
    C_tilde = tf_completion.compute_C(d=3, q=q, **args_)
    a_inv = (dnu - m * kc/ hbar**2 /2/np.pi**2*(1-0.5*k0_kc*np.log((1+k0_kc)/(1-k0_kc)))) * 4 * np.pi * hbar**2/m
    return a_inv

def get_pressure(mu_a,mu_b,delta,m=1,hbar=1, T=0, q=0):
    args = dict(mu_a=mu_a, mu_b=mu_b, delta=delta,hbar=hbar,q=q,T=0.0)
    mu_p,mu_m = mu_a + mu_b, mu_a - mu_b
   # a_inv = get_inv_scattering_len(**args)
    args['m_a']=m
    args['m_b']=m
    n_p = tf.integrate_q(tf.n_p_integrand,d=3, **args)
    n_m = tf.integrate_q(tf.n_m_integrand,d=3, **args)
    kappa = tf.integrate_q(tf.kappa_integrand,d=3,k_c=10.0, **args)
    pressure = mu_p * n_p + mu_m * n_m - kappa
    return pressure

def scan_phase_map():
    N = 100
    deltas = 1
    def f(x):
        return x*x
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))
    
if __name__ == "__main__":
    np.random.seed(1)
    m, hbar, kF = 1 + np.random.random(3)
    eF = (hbar*kF)**2/2/m
    nF = kF**3/3/np.pi**2
    mu = 1# 0.59060550703283853378393810185221521748413488992993*eF
    delta = 1# 0.68640205206984016444108204356564421137062514068346*eF
    args = dict(mu_a=mu, mu_b=mu, delta=delta, m_a=m, m_b=m, hbar=hbar, T=0.0)
    #p0 = get_pressure(mu_a = mu,mu_b=mu,delta=delta,m=m,T=0,q=1)
    qs = np.linspace(20,25,10)
    ps = [get_pressure(mu_a = mu,mu_b=mu,delta=delta,m=m,T=0,q=q).n for q in qs]
    plt.plot(qs,ps)
    print(f'Delta={delta} mu={mu} ')
    plt.show()
