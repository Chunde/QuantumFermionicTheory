import numpy as np
import pytest
from mmf_hfb import tf_completion as tf
from mmf_hfb import bcs, homogeneous
from scipy.optimize import brentq

from mmf_hfb.FuldeFerrelState import FFState as FF
tf.MAX_DIVISION = 200


@pytest.fixture(params=[1.0, 2.0, 3])
def delta(request):
    return request.param

@pytest.fixture(params=[1, 2, 3])
def dim(request):
    return request.param


@pytest.fixture(params=[0, 0.2, 0.6])
def q_delta(request):
    return request.param


@pytest.fixture(params=[0])
def dq_delta(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def mu_delta(request):
    return request.param


@pytest.fixture(params=[0.4])
def dmu_delta(request):
    return request.param

@pytest.fixture(params=[200])
def k_c(request):
    return request.param

def test_Thermodynamic(delta, mu_delta, dmu_delta, q_delta, dq_delta, dim, k_c=100):
    mu = mu_delta * delta
    dmu = dmu_delta * delta
    q = q_delta * delta
    dq = dq_delta * delta
    Thermodynamic(mu=mu, dmu=dmu, k_c=k_c, q=q, dq=dq, dim=dim, delta0=delta)

def get_e_n(mu, dmu, q=0, dq=0, dim=1):
    """"return the analytical energy and particle density"""
    if dim == 1:
        def f(e_F): #energy density
            return np.sqrt(2)/np.pi * e_F**1.5/3.0
        def g(e_F): #particle density
            return np.sqrt(2 * e_F)/np.pi
    elif dim == 2:
        def f(e_F):
            return  (e_F**2)/4.0/np.pi
        def g(e_F):
            return e_F/np.pi/2.0
    elif dim == 3:
        def f(e_F):
            return  (e_F**2.5)*2.0**1.5/10.0/np.pi**2
        def g(e_F):
            return  ((2.0 * e_F)**1.5)/6.0/np.pi**2
        
    kF_a, kF_b = np.sqrt(2.0 * (mu+dmu)),np.sqrt(2.0 * (mu-dmu))
    mu_a1, mu_b1, mu_a2, mu_b2 = (q+dq)**2/2.0, (q-dq)**2/2.0, (kF_a)**2/2.0, (kF_b)**2/2.0
    E_a, E_b = f(mu_a2) - f(mu_a1), f(mu_b2) - f(mu_b1)
    n_a, n_b = g(mu_a2) - g(mu_a1), g(mu_b2) - g(mu_b1)
    energy_density = E_a + E_b
    return energy_density, (n_a, n_b)

def get_dE_dn(mu, dmu, dim, q=0, dq=0):
    """compute the dE/dn for free Fermi Gas"""
    dx = 1e-6
    e1, n1 = get_e_n(mu=mu + dx, dmu=dmu, dim=dim, q=q, dq=dq)
    e2, n2 = get_e_n(mu=mu - dx, dmu=dmu, dim=dim, q=q, dq=dq)
    return (e1-e2)/(sum(n1)-sum(n2))

def Thermodynamic(mu, dmu,delta0=1, dim=1, k_c=100, q=0, dq=0, T=0.0, dx=1e-3):
    #print(f"mu={mu}\tdmu={dmu}\tkc={k_c}\tq={q}\tdq={dq}\tdim={dim}")    
    ff = FF(mu=mu, delta=delta0, dim=dim, k_c=k_c, T=T, fix_g=True)
   
    def get_P(mu, dmu):
        delta = ff.solve(mu=mu, dmu=dmu, q=q, dq=dq, a=0.8*delta0, b=1.2*delta0)
        return ff.get_pressure(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)

    def get_E_n(mu, dmu):
        E = ff.get_energy_density(mu=mu, dmu=dmu, q=q, dq=dq)
        na, nb = ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
        return E, na, nb

    def get_ns(mu, dmu):
        return ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
    
    
    def f_ns_dmu(dx, n):
        na, nb = ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
        dn = (na - nb).n
        def f(dmu_):
            na_, nb_ = ff.get_densities(mu=mu+dx, dmu=dmu_, q=q, dq=dq)
            if n == 0: # fix n_-
                dn_ = (na_ - nb_).n
                return dn - dn_
            elif n == 1: # fix n_b
                return (nb - nb_).n
            elif n == 2: # fix n_a
                return (na - na_).n
        return brentq(f, -3*dmu , 3*dmu)
    

    dmu1 = f_ns_dmu(dx, 0)
    dmu2 = f_ns_dmu(-dx, 0)
    E1, na1, nb1 = get_E_n(mu=mu+dx, dmu=dmu1)
    E0, na0, nb0 = get_E_n(mu=mu-dx, dmu=dmu2)
    n1, n0 = (na1 + nb1).n, (na0 + nb0).n
    mu_ = ((E1-E0)/(n1-n0)).n
    print(f"Expected mu={mu}\tNumerical mu={mu_}")
    assert np.allclose(mu,mu_)
    dmu1 = f_ns_dmu(dx, 1)
    dmu2 = f_ns_dmu(-dx, 1)
    E1, na1, nb1 = get_E_n(mu=mu+dx, dmu=dmu1)
    E0, na0, nb0 = get_E_n(mu=mu-dx, dmu=dmu2)
    print(f"na1={na1}\tnb1={nb1}\tna0={na0}\tnb0{nb0}")

    n1, n0 = (na1 + nb1).n, (na0 + nb0).n
    mu_ = ((E1-E0)/(n1-n0)).n
    print(f"Expected mu_a={mu + dmu}\tNumerical mu_a={mu_}")
    assert np.allclose(mu + dmu,mu_)
    dmu1 = f_ns_dmu(dx, 2)
    dmu2 = f_ns_dmu(-dx, 2)
    E1, na1, nb1 = get_E_n(mu=mu+dx, dmu=dmu1)
    E0, na0, nb0 = get_E_n(mu=mu-dx, dmu=dmu2)
    n1, n0 = (na1 + nb1).n, (na0 + nb0).n
    mu_ = ((E1-E0)/(n1-n0)).n
    print(f"Expected mu_b={mu - dmu}\tNumerical mu_b={mu_}")
    assert np.allclose(mu - dmu,mu_)
    
    n_a, n_b = get_ns(mu, dmu)
    n_a_ = (get_P(mu+dx/2, dmu+dx/2) - get_P(mu-dx/2, dmu - dx/2))/2/dx
    n_b_ = (get_P(mu+dx/2, dmu-dx/2) - get_P(mu-dx/2, dmu + dx/2))/2/dx
    print(f"Expected n_a={n_a.n}\tNumerical n_a={n_a_.n}")
    print(f"Expected n_b={n_b.n}\tNumerical n_b={n_b_.n}")
    assert np.allclose(n_a.n, n_a_.n)
    assert np.allclose(n_b.n, n_b_.n)


if __name__ == "__main__":
    # test_Thermodynamic(mu=5, dmu=0.0, k_c=200, q=0.0, dq=0.0, dim=1)
    Thermodynamic(mu=5, dmu=.5, k_c=200, q=0, dq=.0, dim=1)