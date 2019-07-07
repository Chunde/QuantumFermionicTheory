from mmf_hfb.homogeneous_aslda import BDG, SLDA, ASLDA
from  mmf_hfb import bcs_aslda
import numpy as np
import pytest
import warnings
warnings.filterwarnings("ignore")


@pytest.fixture(params=[0, 0.5, 1])
def T(request):
    return request.param


@pytest.fixture(params=[-0.54])
def C(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def dim(request):
    return request.param


@pytest.fixture(params=[BDG, SLDA, ASLDA])
def Functional(request):
    return request.param


def test_homogeneous_aslda_thermodynamic(Functional, T, C, dim):
    #if Functional == BDG and C is None:
    #    return
    dx=1e-2
    delta = 1
    mu = 10
    dmu = 0
    lda = Functional(T=T, mu_eff=mu, dmu_eff=dmu, delta=delta, C=C, dim=dim)

    def get_ns_e_p(mu, dmu, delta=delta, update_C=False):
        ns, e, p = lda.get_ns_e_p(
            mus=(mu, dmu), delta=delta, update_C=update_C, use_solver=True)
        return ns, e, p

    ns, _, _ = get_ns_e_p(mu=mu, dmu=dmu, delta=None)
    ns1, e1, p1 = get_ns_e_p(mu=mu+dx, dmu=dmu, delta=None)
    print("-------------------------------------")

    ns2, e2, p2 = get_ns_e_p(mu=mu-dx, dmu=dmu, delta=None)
    print("-------------------------------------")

    n_p = (p1-p2)/2/dx
    mu_ = (e1-e2)/(sum(ns1) - sum(ns2))
    print("-------------------------------------")
    print(f"Numerical n_p={n_p}, Expected n_p={sum(ns1 + ns2)/2.0}")
    print(f"Numerical mu={mu_}, Expected mu={mu}")
    print("-------------------------------------")
    assert np.allclose(n_p, sum(ns1 + ns2)/2.0, rtol=1e-2)
    assert np.allclose(mu_, mu, rtol=1e-2)

@pytest.mark.skip(reason="Not pass yet")
def test_bcs_aslda_thermodynamic(dx=1e-3):
    L = 0.46
    N = 16
    N_twist = 1
    delta = 1.0
    mu=10
    dmu = 0
    lda = bcs_aslda.BDG(T=0, Nxyz=(N,N), Lxyz=(L,N))
    k_c = abs(np.array(lda.kxyz).max())
    #lda.E_c = 3 * (lda.hbar*k_c)**2/2/lda.m

    def get_ns_e_p(mu, dmu, update_C=False):
        ns, e, p = lda.get_ns_e_p(
            mus=(mu, dmu), delta=delta, N_twist=N_twist, Laplacian_only=True,
            update_C=update_C, max_iter=32, use_solver=False)
        return ns, e, p
    ns, e, p = get_ns_e_p(mu=mu, dmu=dmu, update_C=True)
    ns1, e1, p1 = get_ns_e_p(mu=mu+dx, dmu=dmu)
    ns2, e2, p2 = get_ns_e_p(mu=mu-dx, dmu=dmu)

    n_p = (p1-p2)/2.0/dx
    mu_ = (e1-e2)/(sum(ns1) - sum(ns2))
    print("-------------------------------------")
    print(np.max(n_p), np.max(sum(ns)))
    print(np.max(mu_), mu)
    print("-------------------------------------")
    assert np.allclose(n_p.max().real, sum(ns), rtol=1e-2)
    assert np.allclose(mu_[0].max().real, mu, rtol=1e-2)

if __name__ == "__main__":
    # test_bcs_aslda_thermodynamic()
    test_homogeneous_aslda_thermodynamic(Functional=SLDA, T=0, C=-0.54, dim=2)
