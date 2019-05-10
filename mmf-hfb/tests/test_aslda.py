from mmf_hfb import vortex_aslda
import numpy as np
from mmf_hfb import homogeneous


class ASLDA_(vortex_aslda.ASLDA):
    # a modified class from ASLDA with const alphas
    def _get_alphas(self, ns=None):
        if ns is None:
            return (None, None, None)
        dim = sum(self.xyz)
        alpha_a, alpha_b, alpha_p =np.ones_like(dim), np.ones_like(dim), np.ones_like(dim)
        return (alpha_a, alpha_b, alpha_p)      


def test_aslda_homogenous():
    L = 0.46
    N = 32
    N_twist = 256
    delta = 1.0
    mu_eff = 10.0
    v_0, n, mu, e_0 = homogeneous.Homogeneous1D().get_BCS_v_n_e(
        delta=delta, mus_eff=(mu_eff, mu_eff))
    n_ = np.ones((N),)*(n[0].n+n[1].n)
    print("Test 1d lattice with homogeneous system")
    b = ASLDA_(T=0, Nxyz=(N,), Lxyz=(L,))
    k_c = abs(np.array(b.kxyz).max())
    E_c = (b.hbar*k_c)**2/2/b.m
    b.E_c = E_c
    R = b.get_R(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    na = np.diag(R)[:N]/b.dV
    nb = (1 - np.diag(R)[N:])/b.dV
    kappa = np.diag(R[:N, N:])/b.dV
    print((sum(n).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))
    assert np.allclose(n_, na.real + nb.real)
    assert np.allclose(delta, -v_0.n*kappa[0].real, rtol=0.01)
    print("Test 1d lattice with homogeneous system")
    ns, taus, js, kappa = b.get_densities(
        mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist, struct=False)
    na, nb = ns
    print((sum(n).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))

    print("Test 1d lattice with homogeneous system with high precision(1e-12")
    # Test against homogeneous class
    h = homogeneous.Homogeneous(Nxyz=(N,), Lxyz=(L), dim=1)
    ret = h.get_densities(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    print(f"{(ret.n_a + ret.n_b).n}, {sum(ns)[0]}")
    print(f"{ret.nu.n}, {kappa[0].real}")
    assert np.allclose((ret.n_a + ret.n_b).n, sum(ns), rtol=1e-12)
    assert np.allclose(ret.nu.n, kappa, rtol=1e-12)
    return
    print("Test 1d lattice plus 1d integral over y with homogeneous system")
    ns, taus, js, kappa = b.get_dens_integral(
        mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    na, nb = ns
    print((sum(n).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))
    assert np.allclose(n_, na.real + nb.real, rtol=0.001)
    assert np.allclose(delta, -v_0.n*kappa[0].real, rtol=0.01)

def test_bcs_thermodynamic(dx=1e-5):
    L = 0.46
    N = 32
    N_twist = 256
    delta = 1.0
    mu=10
    dmu = 0
    v_0, n, _, e_0 = homogeneous.Homogeneous1D().get_BCS_v_n_e(
        delta=delta, mus_eff=(mu+dmu, mu-dmu))
    n_ = np.ones((N),)*(n[0].n+n[1].n)
    print("Test 1d lattice with homogeneous system")
    b = ASLDA_(T=0, Nxyz=(N,), Lxyz=(L,))
    args = dict(N_twist=N_twist)
    def get_ns_e_p(mu, dmu):
        ns, e, p = b.get_ns_e_p(mus_eff=(mu+dmu, mu-dmu), delta=delta, **args)
        return ns, e, p

    ns, e, p = get_ns_e_p(mu=mu, dmu=dmu)
    assert np.allclose(sum(ns)[0], (n[0] + n[1]).n)
    ns1, e1, p1 = get_ns_e_p(mu=mu+dx, dmu=dmu)
    ns2, e2, p2 = get_ns_e_p(mu=mu-dx, dmu=dmu)
    n_p = (p1-p2)/2/dx
    mu_ = (e1-e2)/(sum(ns1) - sum(ns2))
    assert np.allclose(mu_[0], mu, rtol=1e-2)
    assert np.allclose(n_p, sum(ns), rtol=1e-2)

if __name__ == "__main__":
    test_bcs_thermodynamic()
