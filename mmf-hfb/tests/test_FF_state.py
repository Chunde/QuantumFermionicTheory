import numpy as np
import pytest
from mmf_hfb import tf_completion as tf

from mmf_hfb.FuldeFerrelState import FFState as FF
tf.MAX_ITERATION = 200


@pytest.fixture(params=[1, 2])
def d(request):
    return request.param


@pytest.fixture(params=[0])
def q(request):
    return request.param


@pytest.fixture(params=[0])
def dq(request):
    return request.param


@pytest.fixture(params=[5, 10])
def mu(request):
    return request.param


@pytest.fixture(params=[0.4, 0.64])
def dmu(request):
    return request.param


@pytest.fixture(params=[100])
def k_c(request):
    return request.param


#@pytest.mark.bench()
def test_Thermodynamic(mu, dmu, d, k_c, q, dq):
    print(f"mu={mu}\tdmu={dmu}\tkc={k_c}\tq={q}\tdq={dq}\td={d}")
    delta0 = 1

    ff = FF(dmu=dmu, mu=mu, delta=delta0, d=d, k_c=k_c, fix_g=True)
    
    def get_P(mu, dmu):
        delta = ff.solve(mu=mu, dmu=dmu, q=q, dq=dq)
        return ff.get_pressure(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)

    def get_E_n(mu, dmu):
        E = ff.get_energy_density(mu=mu, dmu=dmu, q=q, dq=dq)
        n = sum(ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq))
        return E, n

    def get_ns(mu, dmu):
        return ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
    
    dx = 1e-3
    E1, n1 = get_E_n(mu=mu+dx, dmu=dmu)
    E0, n0 = get_E_n(mu=mu-dx, dmu=dmu)

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


@pytest.mark.bench()
def test_thermodynamic_relations(d, q, dq, k_c=500):
    mus = [5, 10]
    dmus = [0.4, 0.64]
    for mu in mus:
        for dmu in dmus:
            test_Thermodynamic(mu=mu, dmu=dmu, d=d, k_c=k_c, q=q, dq=dq)


if __name__ == "__main__":
    test_Thermodynamic(mu=15, dmu=0.5012, d=1, q=0.0, dq=0, k_c=500)
    test_Thermodynamic(mu=15, dmu=0.5011, d=1, q=0.0, dq=0, k_c=500)
    