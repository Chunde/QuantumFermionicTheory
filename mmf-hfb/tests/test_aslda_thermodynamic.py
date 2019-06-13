from mmf_hfb import bcs_aslda, homogeneous_aslda
import numpy as np
import pytest


@pytest.mark.skip(reason="Not pass yet")
def test_bcs_aslda_thermodynamic(dx=1e-3):
    # Failed  case: mu=5, dmu=2, N=16, N_twist=32, 2D
    xi = 0.44
    N0 = 30
    E0 = (3*N0)**(4/3)/4*xi**0.5
    mu = (3*N0)**(1/3)*xi**0.5
    
    L = 0.46
    N = 16
    N_twist = 1
    delta = 1.0
    mu=10
    dmu = 0
    lda = bcs_aslda.SLDA(T=0, Nxyz=(N,), Lxyz=(L,))
    k_c = abs(np.array(lda.kxyz).max())
    lda.E_c = 3 * (lda.hbar*k_c)**2/2/lda.m

    def get_ns_e_p(mu, dmu):
        ns, e, p = lda.get_ns_e_p(mus_eff=(mu+dmu, mu-dmu), delta=delta, N_twist=N_twist, Laplacian_only=True, max_iter=32)
        return ns, e, p
    ns, e, p = get_ns_e_p(mu=mu, dmu=dmu)
    ns1, e1, p1 = get_ns_e_p(mu=mu+dx, dmu=dmu)
    ns2, e2, p2 = get_ns_e_p(mu=mu-dx, dmu=dmu)
    n_p = (p1-p2)/2/dx
    mu_ = (e1-e2)/(sum(ns1) - sum(ns2))
    print("-------------------------------------")
    print(n_p.max().real, sum(ns).max())
    print(mu_[0].max().real, mu)
    print("-------------------------------------")
    assert np.allclose(n_p.max().real, sum(ns), rtol=1e-2)
    assert np.allclose(mu_[0].max().real, mu, rtol=1e-2)

@pytest.mark.skip(reason="Not pass yet")
def test_homogeneous_aslda_thermodynamic(dx=1e-3):
    # Failed  case: mu=5, dmu=2, N=16, N_twist=32, 2D
    """ xi = 0.44
    N0 = 30
    E0 = (3*N0)**(4/3)/4*xi**0.5
    mu = (3*N0)**(1/3)*xi**0.5
     """
    delta = 1.0
    mu = 6
    dmu = 0
    lda = homogeneous_aslda.SLDA(T=0, mu_eff=mu, dmu_eff=dmu, delta=delta, dim=3)

    def get_ns_e_p(mu, dmu, delta=delta):
        ns, e, p = lda.get_ns_e_p(mus_eff=(mu, dmu), delta=delta, max_iter=32)
        return ns, e, p
    ns, e, p = get_ns_e_p(mu=mu, dmu=dmu, delta=delta)
    ns1, e1, p1 = get_ns_e_p(mu=mu+dx, dmu=dmu, delta=None)
    ns2, e2, p2 = get_ns_e_p(mu=mu-dx, dmu=dmu, delta=None)
    n_p = (p1-p2)/2/dx
    mu_ = (e1-e2)/(sum(ns1) - sum(ns2))
    print("-------------------------------------")
    print(n_p.n, sum(ns).n)
    print(mu_.n, mu)
    print("-------------------------------------")
    assert np.allclose(n_p.n, sum(ns).n, rtol=1e-2)
    assert np.allclose(mu_.n, mu, rtol=1e-2)


if __name__ == "__main__":
    test_homogeneous_aslda_thermodynamic()
