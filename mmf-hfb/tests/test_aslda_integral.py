
from mmf_hfb import bcs_aslda
from mmf_hfb.np import np
from mmf_hfb import homogeneous
import pytest
from mmf_hfb.np import np
    

@pytest.mark.skip(reason="Not pass yet")
def test_aslda_integral_2d():
    """
        test 2d lattice BCS with 1d lattice + 1d integral
        ------------------------
        As 1d case is very fast, so we can have higher twisting
        number to get good result.
    """
    L = 0.46
    N = 8
    N_twist = 32
    delta = 1.0
    mu_eff = 10.0
    b = vortex_aslda.ASLDA(T=0, Nxyz=(N,), Lxyz=(L,))


    print("Test 1d lattice with homogeneous system with high precision(1e-12)")
    ns, taus, js, kappa = b.get_densities(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist, struct=False)
    h = homogeneous.Homogeneous(Nxyz=(N,), Lxyz=(L), dim=1)
    ret = h.get_densities(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    print(f"{(ret.n_a + ret.n_b).n}, {sum(ns)[0]}")
    print(f"{ret.nu.n}, {kappa[0].real}")
    assert np.allclose((ret.n_a + ret.n_b).n, sum(ns), rtol=1e-12)
    assert np.allclose(ret.nu.n, kappa, rtol=1e-12)

    k_c = abs(np.array(b.kxyz).max())
    b.E_c = (b.hbar*k_c)**2/2/b.m
    v_0, n, mu, e_0 = homogeneous.Homogeneous2D().get_BCS_v_n_e(delta=delta, mus_eff=(mu_eff, mu_eff), k_inf=k_c)

    h = homogeneous.Homogeneous(Nxyz=(N, N), Lxyz=(L, L), dim=2)
    ret = h.get_densities(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    print("Test 1d lattice plus 1d integral over perpendicular dimension")
    ns, taus, js, kappa = b.get_dens_integral(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist, k_c=k_c)
    na, nb = ns
    print(((ret.n_a + ret.n_b).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))
    assert np.allclose((ret.n_a + ret.n_b).n, na.real + nb.real, rtol=0.001)
    assert np.allclose(delta, -v_0.n*kappa[0].real, rtol=0.01)

def test_aslda_integral_3d():
    """
        test the 3d lattice again 2d lattice + 1d integral
        ------------------
        Due to computational time, we pick a very small lattice size=4
        with a twisting number that make sure it's enough to yield reasonable 
        results
    """
    L = 0.46
    N = 4
    N_twist = 16
    delta = 1.0
    mu_eff = 10.0
    b = vortex_aslda.ASLDA(T=0, Nxyz=(N, N), Lxyz=(L,L))
    k_c = abs(np.array(b.kxyz).max())
    b.E_c = (b.hbar*k_c)**2/2/b.m
    v_0, n, mu, e_0 = homogeneous.Homogeneous3D().get_BCS_v_n_e(delta=delta, mus_eff=(mu_eff, mu_eff), k_inf=k_c)
    h = homogeneous.Homogeneous(Nxyz=(N, N, N), Lxyz=(L, L, L), dim=3)

    ret = h.get_densities(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    print("Test 2d lattice plus 1d integral over perpendicular dimension")
    ns, taus, js, kappa = b.get_dens_integral(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist, k_c=k_c)
    na, nb = ns
    print(((ret.n_a + ret.n_b).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))
    assert np.allclose((ret.n_a + ret.n_b).n, na.real + nb.real, rtol=0.01)
    assert np.allclose(delta, -v_0.n*kappa[0].real, rtol=0.05)

if __name__ == "__main__":
    test_aslda_integral_2d()
    test_aslda_integral_3d()
