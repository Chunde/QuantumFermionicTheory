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



def get_delta(mu_a,mu_b,m=1,T=0,hbar=1, q=0):
    def get_C_tilde(delta):
        args_ = dict(mu_a=mu_a, mu_b=mu_b, delta=delta, m=m, hbar=hbar, T=T)
        C_tilde = tf_completion.compute_C(d=3, q=q, **args_)
    delta = brentq(get_C_tilde, 0, 0.8)
    return delta

def get_pressure(mu_a,mu_b,delta,m=1,hbar=1, T=0, q=0):
    args = dict(mu_a=mu_a, mu_b=mu_b, delta=delta,hbar=hbar,q=q,T=0.0)
    mu_p,mu_m = (mu_a + mu_b)/2, (mu_a - mu_b)/2
    args['m_a']=m
    args['m_b']=m
    n_p = tf.integrate_q(tf.n_p_integrand,d=3, **args)
    n_m = tf.integrate_q(tf.n_m_integrand,d=3, **args)
    kappa = tf.integrate_q(tf.kappa_integrand,d=3,k_c=10.0, **args)
    pressure = mu_p * n_p + mu_m * n_m - kappa
    return pressure



def test_thermodynamic_relations():
    np.random.seed(1)
    m, hbar, kF = 1 + np.random.random(3)
    eF = (hbar*kF)**2/2/m
    nF = kF**3/3/np.pi**2
    mu = 0.59060550703283853378393810185221521748413488992993*eF
    delta = 0.68640205206984016444108204356564421137062514068346*eF
    args = dict(mu_a=mu, mu_b=mu, delta=delta, m_a=m, m_b=m, hbar=hbar, T=0.0)
    n_p = 0#tf.integrate_q(tf.n_p_integrand, d=3, q=0,**args)
    n_m = 0#tf.integrate_q(tf.n_m_integrand, d=3, q=0,**args)
    n_a = (n_p + n_m)/2
    n_b = (n_p - n_m)/2
    dmu = 1e-6
    q = 0
    n_a_ = (get_pressure(mu_a = mu + dmu,mu_b=mu,delta=delta,m=m,T=0,q=q) - get_pressure(mu_a = mu-dmu,mu_b=mu,delta=delta,m=m,T=0,q=q))/2/dmu
    n_b_ = (get_pressure(mu_a = mu,mu_b=mu + dmu,delta=delta,m=m,T=0,q=q) - get_pressure(mu_a = mu,mu_b=mu-dmu,delta=delta,m=m,T=0,q=q))/2/dmu
    print((n_a,n_b),(n_a_,n_b_))
    assert np.allclose(n_a,n_a_)
    assert np.allclose(n_b,n_b_)

if __name__ == "__main__":
    np.random.seed(1)
    m, hbar, kF = 1 + np.random.random(3)
    eF = (hbar*kF)**2/2/m
    nF = kF**3/3/np.pi**2
    mu =  0.59060550703283853378393810185221521748413488992993*eF
    delta = 0.68640205206984016444108204356564421137062514068346*eF
    dmu = 0.9 * delta
    #p0 = get_pressure(mu_a = mu,mu_b=mu,delta=delta,m=m,T=0,q=1)
    qs = np.linspace(0,2,10)
   
    ps = [get_pressure(mu_a = mu +  q * dmu /2,mu_b=mu - q * dmu /2,delta=delta,m=m,T=0,q=0).n for q in qs]
    plt.plot(qs,ps)
    print(f'Delta={delta} mu={mu} ')
    plt.show()
