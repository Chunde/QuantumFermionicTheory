import numpy as np
from mmf_hfb import homogeneous
from mmf_hfb import bcs_aslda
hbar = m = 1


def test_aslda_homogenous():
    L = 0.46
    N = 32
    N_twist = 32
    delta = 1.0
    mu_eff = 1.0
    v_0, n, mu, e_0 = homogeneous.Homogeneous1D().get_BCS_v_n_e(
        delta=delta, mus_eff=(mu_eff, mu_eff))
    n_ = np.ones((N),)*(n[0].n+n[1].n)
    print("Test 1d lattice with homogeneous system")
    b = bcs_aslda.ASLDA(T=0, Nxyz=(N,), Lxyz=(L,))
    k_c = abs(b.kxyz[0]).max()
    E_c = (b.hbar*k_c)**2/2/b.m
    b.E_c = E_c
    R = b.get_R(
        mus_eff=(mu_eff *np.ones((N),), mu_eff*np.ones((N),)),
        delta=delta*np.ones((N),), N_twist=N_twist)
    na = np.diag(R)[:N]/np.prod(b.dxyz)
    nb = (1 - np.diag(R)[N:])/np.prod(b.dxyz)
    kappa = np.diag(R[:N, N:])/np.prod(b.dxyz)
    print((sum(n).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))
    assert np.allclose(n_, na.real + nb.real, rtol=0.001)
    assert np.allclose(delta, -v_0.n*kappa[0].real, rtol=0.01)
    print("Test 1d lattice with homogeneous system")
    ns, taus, js, kappa = b.get_densities(
        mus_eff=(mu_eff*np.ones((N),), mu_eff*np.ones((N),)),
        delta=delta*np.ones((N),), N_twist=N_twist, struct=False)
    na, nb = ns
    print((sum(n).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))
    assert np.allclose(n_, na.real + nb.real, rtol=0.001)
    assert np.allclose(delta, -v_0.n*kappa[0].real, atol=0.01)
