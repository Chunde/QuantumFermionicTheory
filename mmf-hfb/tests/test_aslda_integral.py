
from mmf_hfb import vortex_aslda
from mmf_hfb.xp import xp
from mmf_hfb import homogeneous
import pytest
from mmf_hfb.xp import xp
    

@pytest.mark.skip(reason="Not pass yet")
def test_aslda_homogenous():
    L = 0.46
    N = 16
    N_twist = 64
    delta = 1.0
    mu_eff = 10.0
    b = vortex_aslda.ASLDA(T=0, Nxyz=(N,), Lxyz=(L,))


    print("Test 1d lattice with homogeneous system with high precision(1e-12)")
    ns, taus, js, kappa = b.get_densities(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist, struct=False)
    h = homogeneous.Homogeneous(Nxyz=(N,), Lxyz=(L), dim=1)
    ret = h.get_densities(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    print(f"{(ret.n_a + ret.n_b).n}, {sum(ns)[0]}")
    print(f"{ret.nu.n}, {kappa[0].real}")
    assert xp.allclose((ret.n_a + ret.n_b).n, sum(ns), rtol=1e-12)
    assert xp.allclose(ret.nu.n, kappa, rtol=1e-12)

    k_c = abs(xp.array(b.kxyz).max())
    b.E_c = (b.hbar*k_c)**2/2/b.m
    v_0, n, mu, e_0 = homogeneous.Homogeneous2D().get_BCS_v_n_e(delta=delta, mus_eff=(mu_eff, mu_eff), k_inf=k_c)

    h = homogeneous.Homogeneous(Nxyz=(N, N), Lxyz=(L, L), dim=2)
    ret = h.get_densities(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    print("Test 1d lattice plus 1d integral over perpendicular dimension")
    ns, taus, js, kappa = b.get_dens_integral(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist, k_c=k_c)
    na, nb = ns
    print(((ret.n_a + ret.n_b).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))
    assert xp.allclose((ret.n_a + ret.n_b).n, na.real + nb.real, rtol=0.001)
    assert xp.allclose(delta, -v_0.n*kappa[0].real, rtol=0.01)



if __name__ == "__main__":
    test_aslda_homogenous()
