from mmf_hfb import vortex_aslda
from mmf_hfb.xp import xp
from mmf_hfb import homogeneous
import pytest
from mmf_hfb.xp import xp   



@pytest.mark.skip(reason="Not pass yet")
def test_aslda_thermodynamic(dx=1e-3):
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
    lda = vortex_aslda.SLDA(T=0, Nxyz=[N,], Lxyz=[L,])
    k_c = abs(xp.array(lda.kxyz).max())
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
    assert xp.allclose(n_p.max().real, sum(ns), rtol=1e-2)
    assert xp.allclose(mu_[0].max().real, mu, rtol=1e-2)

if __name__ == "__main__":
    test_aslda_thermodynamic()
