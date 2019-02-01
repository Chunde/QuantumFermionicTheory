import numpy as np
from mmf_hfb import homogeneous
from mmf_hfb import tf_completion as tf
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mmf_hfb.homogeneous import Homogeneous1D,Homogeneous3D
from multiprocessing import Pool


def special_momenta(kz, kp, q, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    """Return the condition at the boundary of integration regions."""
    ka2 = (kz+q)**2 + kp**2
    kb2 = (kz-q)**2 + kp**2
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    return e_m**2 - e_p**2 - delta**2


def get_delta(mu_a, mu_b, m_a, m_b, T=0, hbar = 1, k_c = 10, q=0):
    def f(delta):        
        C = tf.compute_C(mu_a = mu_a,mu_b = mu_b,delta = delta,m_a = m_a,  m_b=m_b,d=3, hbar= hbar,T=T, q=q, k_c= k_c)
        return C.n
    return brentq(f, 0.5, 1.0)

def get_pressure(mu_a, mu_b, delta=None, d=3, m_a=1,m_b=1, hbar=1, T=0, q=0):
    if delta is None:
        delta = get_delta(mu_a= mu_a, mu_b=mu_b, m_a=m_a, m_b=m_b,T=T, q=q)
    print(delta)
    args = dict(mu_a=mu_a, mu_b=mu_b, m_a = m_a, m_b=m_b,delta=delta,hbar=hbar,q=q,T=0.0)
    mu_p,mu_m = (mu_a + mu_b)/2, (mu_a - mu_b)/2
    n_p = tf.integrate_q(tf.n_p_integrand,d=d, **args)
    n_m = tf.integrate_q(tf.n_m_integrand,d=d, **args)
    kappa = tf.integrate_q(tf.kappa_integrand,d=3,k_c=10.0, **args)
    pressure = mu_p * n_p + mu_m * n_m - kappa
    return pressure



def test_thermodynamic_relations():
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


if __name__ == "__main__":
    #test_thermodynamic_relations()
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
