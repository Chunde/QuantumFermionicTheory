import numpy as np
from scipy.optimize import brentq
import homogeneous
import tf_completion as tf
import matplotlib.pyplot as plt

from homogeneous import Homogeneous1D,Homogeneous3D

def special_momenta(kz, kp,q, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    ka2 = (kz+q)**2 + kp**2
    kb2 = (kz-q)**2 + kp**2
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    return e_m**2 - e_p**2 - delta**2 # this will be zero at the boundary of interested


def get_inv_scattering_len(mu_a,mu_b,delta,m=1, T=0,hbar=1, q=0,Ec=1000):
    args = dict(mu_a=mu_a, mu_b=mu_b, delta=delta, m_a=m, m_b=m,hbar=hbar, q=q, T=0.0)
    dnu = tf.integrate(tf.dnu_integrand,d=3, **args)
    mu_p = (mu_a + mu_b)/2 
    mu_eff = mu_p - hbar**2 * q**2 /2/m
    kc = np.sqrt(2 * m * (Ec - mu_eff))/hbar
    k0_kc = np.sqrt(mu_eff/(Ec + mu_eff ))
    a_inv = (dnu - m * kc/ hbar**2 /2/np.pi**2*(1-0.5*k0_kc*np.log((1+k0_kc)/(1-k0_kc)))) * 4 * np.pi * hbar**2/m
    return a_inv

def get_pressure(mu_a,mu_b,delta,m=1,hbar=1, T=0, q=0):
    args = dict(mu_a=mu_a, mu_b=mu_b, delta=delta,hbar=hbar,q=q,T=0.0)
    mu_p,mu_m = mu_a + mu_b, mu_a - mu_b
    a_inv = get_inv_scattering_len(**args)
    args['m_a']=m
    args['m_b']=m
    n_p = tf.integrate(tf.n_p_integrand,d=3, **args)
    n_m = tf.integrate(tf.n_m_integrand,d=3, **args)
    kappa = tf.integrate(tf.dkappa_integrand,d=3, **args)
    pressure = mu_p * n_p + mu_m * n_m - kappa
    return pressure

if __name__ == "__main__":
    np.random.seed(1)
    m, hbar, kF = 1 + np.random.random(3)
    eF = (hbar*kF)**2/2/m
    nF = kF**3/3/np.pi**2
    mu = 0.59060550703283853378393810185221521748413488992993*eF
    delta = 0.68640205206984016444108204356564421137062514068346*eF
    args = dict(mu_a=mu, mu_b=mu, delta=delta, m_a=m, m_b=m, hbar=hbar, T=0.0)
    qs = np.linspace(0,10,10)
    ps = [get_pressure(mu_a = mu,mu_b=mu,delta=delta,m=m,T=0,q=q) for q in qs]
    plt.plot(qs,ps)
    plt.show()
