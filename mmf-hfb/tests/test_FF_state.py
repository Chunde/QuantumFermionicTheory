import numpy as np
import pytest
from mmf_hfb import tf_completion as tf
from mmf_hfb import bcs, homogeneous

from mmf_hfb.FuldeFerrelState import FFState as FF
tf.MAX_DIVISION = 500


@pytest.fixture(params=[2])
def dim(request):
    return request.param


@pytest.fixture(params=[0,0.5,1.5,2.5])
def q(request):
    return request.param


@pytest.fixture(params=[0, 0.5, 1.5])
def dq(request):
    return request.param


@pytest.fixture(params=[5, 10])
def mu(request):
    return request.param


@pytest.fixture(params=[0.4, 0.64, 2.5])
def dmu(request):
    return request.param

# for dim = 3, if k_c is too larger
# e.g.k_c=500, lots of test will fail
@pytest.fixture(params=[200])
def k_c(request):
    return request.param


#@pytest.mark.bench()
def test_Thermodynamic(mu, dmu, dim, k_c, q, dq,  dx = 1e-3):
    print(f"mu={mu}\tdmu={dmu}\tkc={k_c}\tq={q}\tdq={dq}\td={dim}")
    delta0 = 1

    ff = FF(dmu=dmu, mu=mu, delta=delta0, dim=dim, k_c=k_c, fix_g=True)
    
    def get_P(mu, dmu):
        delta = ff.solve(mu=mu, dmu=dmu, q=q, dq=dq)
        return ff.get_pressure(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)

    def get_E_n(mu, dmu):
        E = ff.get_energy_density(mu=mu, dmu=dmu, q=q, dq=dq)
        n = sum(ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq))
        return E, n

    def get_ns(mu, dmu):
        return ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
    #h = homogeneous.Homogeneous(dim = dim)
    #ns = h.get_ns(mus_eff=(mu+dmu, mu-dmu), delta=delta0, N_twist=12)
    #print(f"Homogeneuous:{ns[0].n, ns[1].n}")
    #assert np.allclose(sum(ns).n, (n_a+n_b).n, rtol=1e-3)
   
    E1, n1 = get_E_n(mu=mu+dx, dmu=dmu)
    E0, n0 = get_E_n(mu=mu-dx, dmu=dmu)
    #mu_a_ = (E1-E0)/(n1-n0)
    #print(f"mu_a={mu_a_}")

    n_p = (get_P(mu+dx, dmu) - get_P(mu-dx, dmu))/2/dx
    n_a, n_b = get_ns(mu, dmu)
    n_a_ = (get_P(mu+dx/2, dmu+dx/2) - get_P(mu-dx/2, dmu - dx/2))/2/dx
    n_b_ = (get_P(mu+dx/2, dmu-dx/2) - get_P(mu-dx/2, dmu + dx/2))/2/dx
    print(f"n_a={n_a.n}\tNumerical  n_a={n_a_.n}")
    print(f"n_b={n_b.n}\tNumerical  n_b={n_b_.n}")
    print(f"n_p={n_a.n+n_b.n}\tNumerical  n_p={n_p.n}")
    print(f"mu={mu}\tNumerical mu={(E1-E0)/(n1-n0)}")
    assert np.allclose(n_a.n, n_a_.n, rtol=1e-4)
    assert np.allclose(n_b.n, n_b_.n, rtol=1e-4)
    assert np.allclose(n_p.n, (n_a+n_b).n, rtol=1e-4)
    assert np.allclose(mu,((E1-E0)/(n1-n0)).n, rtol=1e-2)


if __name__ == "__main__":
    # this line will give a numerical mu =2.5 not 5, totally wrong
    # test_Thermodynamic(mu=5, dmu=2.5, k_c=200, q=2.5, dq=1.5, dim=1)
    #test_Thermodynamic(mu=5, dmu=2.5, k_c=100, q=2.5, dq=1.5, dim=3)
    #test_Thermodynamic(mu = 5, dmu = 0.64, dim = 3, k_c = 500, q = 2.5, dq = 1.5, dx = 1e-4)
    test_Thermodynamic(mu = 5, dmu = 0.64, dim = 3, k_c = 500, q = 0, dq = 0, dx = 0.001)
    #test_Thermodynamic(mu=10, dmu=0.64, dim=2, q=0, dq=0, k_c=200)

    