from mmf_hfb.class_factory import class_factory, FunctionalType, KernelType, Solvers
from mmf_hfb import homogeneous
import numpy as np
import pytest
import warnings
warnings.filterwarnings("ignore")

#  should add ASLDA here to have full test
@pytest.fixture(params=[FunctionalType.BDG, FunctionalType.SLDA])
def functional(request):
    return request.param


@pytest.fixture(params=[KernelType.HOM])
def kernel(request):
    return request.param


@pytest.fixture(params=[np.pi, 10])
def mu(request):
    return request.param


@pytest.fixture(params=[0, 1.0])
def dmu(request):
    return request.param


@pytest.fixture(params=[0, 0.])
def dq(request):
    return request.param


def create_LDA(mu, dmu, delta):
    LDA = class_factory(
        className="LDA",
        functionalType=FunctionalType.SLDA,
        kernelType=KernelType.HOM)

    lda = LDA(mu_eff=mu, dmu_eff=dmu, delta=delta, T=0, dim=3)
    return lda


def test_BDG(mu, dmu):
    delta = 1
    h = homogeneous.Homogeneous3D()
    _, ns, _, _ = h.get_BCS_v_n_e(mus_eff=(mu + dmu, mu - dmu), delta=delta)

    LDA = class_factory(
        className="LDA",
        functionalType=FunctionalType.BDG,
        kernelType=KernelType.HOM)

    lda = LDA(mu_eff=mu, dmu_eff=dmu, delta=delta, T=0, dim=3)
    lda.C = lda._get_C(mus_eff=(mu, mu), delta=delta)

    def get_ns_e_p(mu, dmu, update_C=False):
        ns, e, p = lda.get_ns_e_p(
            mus=(mu + dmu, mu - dmu), delta=delta,
            solver=Solvers.BROYDEN1, verbosity=False)
        return ns, e, p

    ns_, _, _ = get_ns_e_p(mu=mu, dmu=dmu)
    print(ns[0].n, ns_[0])
    print(ns[1].n, ns_[1])
    assert np.allclose(np.array([ns[0].n, ns[1].n]), ns_, rtol=1e-2)


def test_effective_mus(mu, dmu, dq=0):
    delta = 1
    lda = create_LDA(mu=mu, dmu=dmu, delta=delta)
    mus_eff = (mu + dmu, mu-dmu)
    res = lda.get_ns_mus_e_p(mus_eff=mus_eff, delta=delta, dq=dq)
    mus_eff_ = lda.get_mus_eff(
        mus=res[1],
        delta=delta, dq=dq, verbosity=False)
    print(mus_eff, mus_eff_)
    assert np.allclose(np.array(mus_eff), np.array(mus_eff_))


def test_effective_mus_BdG(mu, dmu, dq=0):
    """For BDG, effective mus should be the same as bare mus"""
    delta = 1
    LDA = class_factory(
        className="LDA",
        functionalType=FunctionalType.BDG,
        kernelType=KernelType.HOM)

    lda = LDA(mu_eff=mu, dmu_eff=dmu, delta=delta, T=0, dim=3)
    lda.C = lda._get_C(mus_eff=(mu, mu), delta=delta)
    mus_eff = (mu + dmu, mu-dmu)
    res = lda.get_ns_mus_e_p(mus_eff=mus_eff, delta=delta, dq=dq)
    mus_eff_ = lda.get_mus_eff(
        mus=res[1],
        delta=delta, dq=dq, verbosity=False)
    print(mus_eff, mus_eff_)
    assert np.allclose(np.array(mus_eff), np.array(mus_eff_))
    assert np.allclose(np.array(mus_eff), np.array(res[1]))


def test_effective_mus_thermodynamic(mu):
    # the thermodynamic is not to high accuracy
    # because the bare mu is not picked so numerical
    # results are not guaranteed to be in the middle
    delta = 1
    dx = 1e-3
    lda = create_LDA(mu=mu, dmu=0, delta=delta)
    lda.C = lda._get_C(mus_eff=(mu, mu), delta=delta)
    mus_eff = (mu, mu)
    res = lda.get_ns_mus_e_p(mus_eff=mus_eff, delta=delta)
    mus_eff = (mu + dx, mu + dx)
    res1 = lda.get_ns_mus_e_p(mus_eff=mus_eff, delta=None)
    mus_eff = (mu - dx, mu - dx)
    res2 = lda.get_ns_mus_e_p(mus_eff=mus_eff, delta=None)
    e2, e1, _ = res2[2], res1[2], res[2]
    p2, p1, _ = res2[3], res1[3], res[3]
    n2, n1, n = sum(res2[0]), sum(res1[0]), sum(res[0])
    mu2, mu1, mu = sum(res2[1])/2.0, sum(res1[1])/2.0, sum(res[1])/2.0
    mu_ = (e2 - e1)/(n2 - n1)
    np_ = (p2 - p1)/(mu2 - mu1)
    print(res1)
    print(res2)
    print(mu_, mu)
    print(n, np_)
    assert np.allclose(mu_, mu, rtol=1e-2)
    assert np.allclose(np_, n, rtol=1e-2)


def test_bare_mus(mu, dmu):
    """test the method get_mus_bare"""
    delta = 1
    lda = create_LDA(mu=mu, dmu=0, delta=delta)
    mus_eff = lda.get_mus_eff(mus=(mu + dmu, mu - dmu), delta=delta)
    mu_a, mu_b = lda.get_mus_bare(mus_eff=mus_eff, delta=delta)
    print(mu_a, mu + dmu)
    print(mu_b, mu - dmu)
    assert np.allclose(mu_a, mu + dmu)
    assert np.allclose(mu_b, mu - dmu)


def test_class_factory(functional, kernel, mu, dmu=1, dim=3):
    dx = 1e-3
    L = 0.46
    N = 8
    N_twist = 1
    delta = 1.0
    mu=mu
    dmu = 0
    LDA = class_factory(
        className="LDA",
        functionalType=functional,
        kernelType=kernel)

    lda = LDA(
        Nxyz=(N, ), Lxyz=(L,), mu_eff=mu, dmu_eff=dmu,
        delta=delta, T=0, dim=dim)

    def get_ns_e_p(mu, dmu, update_C=False, **args):
        ns, e, p = lda.get_ns_e_p(
            mus=(mu + dmu, mu - dmu), delta=delta, N_twist=N_twist, Laplacian_only=True,
            update_C=update_C, max_iter=32, solver=Solvers.BROYDEN1,
            verbosity=True, **args)
        return ns, e, p
 
    # lda.C = lda._get_C(mus=(mu_a, mu_b), delta=0.75)
    ns, _, _ = get_ns_e_p(mu=mu, dmu=dmu, update_C=True)
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


def test_C():
    """
    when use BDG functional, the C should reproduce
    the result from a BdG code
    """
    mu_eff = 10
    dmu_eff = 0
    delta = 1
    args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
        T=0, dim=3, k_c=50, verbosity=False)
    lda = class_factory(
        "LDA",
        functionalType=FunctionalType.BDG,
        kernelType=KernelType.HOM, args=args)
    C = lda._get_C(mus_eff=(mu_eff, mu_eff), delta=delta)
    lda.fix_C_BdG(mu=mu_eff, dmu=0, delta=delta)
    assert np.allclose(lda.C, C)


if __name__ == "__main__":
    # test_bare_mus(mu=np.pi, dmu=0.5)
    test_effective_mus_thermodynamic(mu=np.pi)
    test_effective_mus(mu=np.pi, dmu=0.3, dq=0)
    test_BDG(mu=5, dmu=1)
    test_class_factory(
        functional=FunctionalType.ASLDA, kernel=KernelType.HOM, mu=np.pi, dim=3)
    test_C()
    test_effective_mus_BdG(mu=10, dmu=1)
