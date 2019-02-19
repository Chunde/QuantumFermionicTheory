import numpy as np
import pytest
from mmf_hfb import tf_completion as tf

from mmf_hfb.FuldeFerrelState import FFState as FF
tf.set_max_iteration(200)

@pytest.fixture(params=[1,2])
def d(request):
    return request.param

@pytest.fixture(params=[np.inf, 3,10])
def r(request):
    return request.param

def Thermodynamic(mu, dmu, d, k_c, r):
    print(f"mu={mu}\tdmu={dmu}\tkc={k_c}\tr={r}\td={d}")
    delta0 = 1

    ff = FF(dmu=dmu, mu=mu, delta=delta0, d=d, k_c=k_c, fix_g=True) 
    def get_P(mu, dmu):
        mu_a = mu + dmu
        mu_b = mu - dmu
        delta = ff.solve(r=r, mu_a=mu_a, mu_b=mu_b)
        #print(delta)
        return ff.get_pressure(mu_a=mu_a, mu_b=mu_b, delta=delta, r=r)    

    def get_E_n(mu, dmu=0):
        mu_a = mu + dmu
        mu_b = mu - dmu
        E = ff.get_energy_density(mu_a=mu_a, mu_b=mu_b, r=r)
        n = sum(ff.get_densities(mu_a=mu_a, mu_b=mu_b, r=r))
        return E, n

    def get_ns(mu, dmu):
        mu_a = mu + dmu
        mu_b = mu - dmu
        return ff.get_densities(mu_a=mu_a, mu_b=mu_b, r=r)   
    dx = 1e-3
    E1, n1 = get_E_n(mu+dx)
    E0, n0 = get_E_n(mu-dx)

    n_p = (get_P(mu+dx, dmu) - get_P(mu-dx, dmu))/2/dx
    n_a, n_b = get_ns(mu, dmu)

    n_a_ = (get_P(mu+dx/2, dmu+dx/2) - get_P(mu-dx/2, dmu - dx/2))/2/dx
    n_b_ = (get_P(mu+dx/2, dmu-dx/2) - get_P(mu-dx/2, dmu + dx/2))/2/dx
    print(f"mu={mu}\tNumerical mu={(E1-E0)/(n1-n0)}")
    print(f"n_a={n_a.n}\tNumerical  n_a={n_a_.n}")
    print(f"n_b={n_b.n}\tNumerical  n_b={n_b_.n}")
    print(f"n_p={n_a.n+n_b.n}\tNumerical  n_p={n_p.n}")
       
    assert np.allclose(n_p.n, (n_a+n_b).n)
    assert np.allclose(n_a.n, n_a_.n)
    assert np.allclose(n_b.n, n_b_.n)


#@pytest.mark.bench()
def test_thermodynamic_relations(d, r=np.inf, k_c=100):
    mus = [2,5,10]
    dmus = [0.4,0.5,0.64]
    for mu in mus:
        for dmu in dmus:
            Thermodynamic(mu=mu, dmu=dmu, d = d, k_c=k_c, r=r)


if __name__ == "__main__":
    """Failure case:
    1) mu=10   dmu=0.64    d=2    kc=100
    """
    Thermodynamic(mu=10, dmu=0.64, d= 2, r=10.0, k_c=100) # will fail
    ks = [500]
    d = 2
    r = 10.0
    print(f"Performing {d}-dimension, r={r} test...")
    for kc in ks:
        test_thermodynamic_relations(d=d, r=r, k_c=kc)
       