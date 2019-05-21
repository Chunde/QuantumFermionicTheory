from mmf_hfb import vortex_aslda
from mmf_hfb.xp import xp
from mmf_hfb import homogeneous
import pytest
from mmf_hfb.xp import xp   

#@pytest.mark.skip(reason="Not pass yet")
def test_aslda_homogenous():
    L = 0.46
    N = 16
    N_twist = 64
    delta = 1.0
    mu_eff = 10.0
    v_0, n, mu, e_0 = homogeneous.Homogeneous1D().get_BCS_v_n_e(
        delta=delta, mus_eff=(mu_eff, mu_eff))
    n_ = xp.ones((N),)*(n[0].n+n[1].n)
    print("Test 1d lattice with homogeneous system")
    b = vortex_aslda.ASLDA(T=0, Nxyz=(N,), Lxyz=(L,))
    k_c = abs(xp.array(b.kxyz).max())
    E_c = (b.hbar*k_c)**2/2/b.m
    b.E_c = E_c
    R = b.get_R(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    na = xp.diag(R)[:N]/b.dV
    nb = (1 - xp.diag(R)[N:])/b.dV
    kappa = xp.diag(R[:N, N:])/b.dV
    print((sum(n).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))
    assert xp.allclose(n_, na.real + nb.real, rtol=0.01)
    assert xp.allclose(delta, -v_0.n*kappa[0].real, rtol=0.01)
    print("Test 1d lattice with homogeneous system")
    ns, taus, js, kappa = b.get_densities(
        mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist, struct=False)
    na, nb = ns
    print((sum(n).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))

    print("Test 1d lattice with homogeneous system with high precision(1e-12)")
    # Test against homogeneous class
    h = homogeneous.Homogeneous(Nxyz=(N,), Lxyz=(L), dim=1)
    ret = h.get_densities(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    print(f"{(ret.n_a + ret.n_b).n}, {sum(ns)[0]}")
    print(f"{ret.nu.n}, {kappa[0].real}")
    assert xp.allclose((ret.n_a + ret.n_b).n, sum(ns), rtol=1e-12)
    assert xp.allclose(ret.nu.n, kappa, rtol=1e-12)


@pytest.mark.skip(reason="Not pass yet")
def test_aslda_thermodynamic(dx=1e-3):
    L = 0.46
    N = 16
    N_twist = 5
    delta = 1.0
    mu=10
    dmu = 2
    v_0, n, _, e_0 = homogeneous.Homogeneous1D().get_BCS_v_n_e(
        delta=delta, mus_eff=(mu+dmu, mu-dmu))
    b = vortex_aslda.ASLDA(T=0, Nxyz=[N,N], Lxyz=[L,L])
    k_c = abs(xp.array(b.kxyz).max())
    b.E_c = (b.hbar*k_c)**2/2/b.m
    def get_ns_e_p(mu, dmu):
        ns, e, p = b.get_ns_e_p(mus_eff=(mu+dmu, mu-dmu), delta=delta, N_twist=N_twist, Laplacian_only=True)
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
