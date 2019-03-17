import numpy as np
import pytest
from mmf_hfb import tf_completion as tf
from mmf_hfb import bcs, homogeneous

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

def Thermodynamic(mu, dmu, delta0=1, dim=1, k_c=100, q=0, dq=0, T=0.0, dx=1e-6):
    print(f"mu={mu}\tdmu={dmu}\tkc={k_c}\tq={q}\tdq={dq}\tdim={dim}")    
    ff = FF(mu=mu, delta=delta0, dim=dim, k_c=k_c, T=T, fix_g=True)

    def get_P(mu, dmu):
        delta = ff.solve(mu=mu, dmu=dmu, q=q, dq=dq, a=0.8*delta0, b=1.2*delta0)
        return ff.get_pressure(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)

    def get_E_n(mu, dmu):
        E = ff.get_energy_density(mu=mu, dmu=dmu, q=q, dq=dq)
        n = sum(ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq))
        return E, n

    def get_ns(mu, dmu):
        return ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
    energy_density, (na, nb) = get_e_n(mu=mu, dmu=dmu, dim=dim)        
    print(f"{dim}D: E={energy_density}\tn_a={na}\tn_b={nb}\tn_p={na+nb}")    
    E1, n1 = get_E_n(mu=mu+dx, dmu=dmu)
    E0, n0 = get_E_n(mu=mu-dx, dmu=dmu)
    print(f"E1={E1.n}\tE0={E0.n}\tn1={n1.n}\tn0={n0.n}")
    n_p = (get_P(mu+dx, dmu) - get_P(mu-dx, dmu))/2/dx
    n_a, n_b = get_ns(mu, dmu)
    n_a_ = (get_P(mu+dx/2, dmu+dx/2) - get_P(mu-dx/2, dmu - dx/2))/2/dx
    n_b_ = (get_P(mu+dx/2, dmu-dx/2) - get_P(mu-dx/2, dmu + dx/2))/2/dx
    print(f"n_a={n_a.n}\tNumerical  n_a={n_a_.n}")
    print(f"n_b={n_b.n}\tNumerical  n_b={n_b_.n}")
    print(f"n_p={n_a.n+n_b.n}\tNumerical  n_p={n_p.n}")
    print(f"mu={mu}\tNumerical mu={((E1-E0)/(n1-n0)).n}")
    if dim == 1:
        rtol = 1e-5
    else:
        rtol = 1e-2 # For 2d and 3d, the accurary is not high, I need to figure out the reason!
    assert np.allclose(n_a.n, n_a_.n, rtol=rtol)
    assert np.allclose(n_b.n, n_b_.n, rtol=rtol)
    assert np.allclose(n_p.n, (n_a+n_b).n, rtol=rtol)
    if not np.allclose(((E1-E0)/(n1-n0)).n, get_dE_dn(mu=mu,dmu=dmu,dim=dim, q=q, dq=dq)):# if is superfluid(Delta !=0)
        assert np.allclose(mu,((E1-E0)/(n1-n0)).n, rtol=rtol)


if __name__ == "__main__":
    # test_Thermodynamic(mu=5, dmu=0.0, k_c=200, q=0.0, dq=0.0, dim=1)
    Thermodynamic(mu=5, dmu=2.5, k_c=200, q=0, dq=.0, dim=1)