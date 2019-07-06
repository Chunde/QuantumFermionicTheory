from mmf_hfb.ClassFactory import ClassFactory, FunctionalType, KernelType, Solvers
import numpy as np
import pytest
import warnings
warnings.filterwarnings("ignore")


@pytest.fixture(params=[FunctionalType.ASLDA, FunctionalType.BDG, FunctionalType.SLDA])
def functional(request):
    return request.param

@pytest.fixture(params=[KernelType.HOM])
def kernel(request):
    return request.param

@pytest.fixture(params=[1, np.pi, 10])
def mu(request):
    return request.param


def test_class_factory(functional, kernel, mu):
    dx = 1e-3
    L = 0.46
    N = 8
    N_twist = 1
    delta = 1.0
    mu=mu
    dmu = 0
    LDA = ClassFactory(
        className="LDA",
        functionalType=functional,
        kernelType=kernel)

    lda = LDA(
        Nxyz=(N, ), Lxyz=(L,), mu_eff=mu, dmu_eff=dmu,
        delta=delta, T=0, dim=3)

    def get_ns_e_p(mu, dmu, update_C=False):
        ns, e, p = lda.get_ns_e_p(
            mus=(mu, dmu), delta=delta, N_twist=N_twist, Laplacian_only=True,
            update_C=update_C, max_iter=32, solver=Solvers.BROYDEN1)
        return ns, e, p

    lda.fix_C(mu=mu, dmu=dmu, delta=delta)
    ns, _, _ = get_ns_e_p(mu=mu+dx, dmu=dmu)
    print("-------------------------------------")
    ns1, e1, p1 = get_ns_e_p(mu=mu+dx, dmu=dmu)
    print("-------------------------------------")
    ns2, e2, p2 = get_ns_e_p(mu=mu-dx, dmu=dmu)
    n_p = (p1-p2)/2.0/dx
    mu_ = (e1-e2)/(sum(ns1) - sum(ns2))
    print(np.max(n_p), np.max(sum(ns)))
    print(np.max(mu_), mu)
    print("-------------------------------------")
    assert np.allclose(np.max(n_p).real, sum(ns), rtol=1e-2)
    assert np.allclose(np.mean(mu_).real, mu, rtol=1e-2)


if __name__ == "__main__":
    test_class_factory(functional=FunctionalType.SLDA, kernel=KernelType.HOM, mu=1)
