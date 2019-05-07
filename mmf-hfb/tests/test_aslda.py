from mmf_hfb import vortex_aslda
import numpy as np
from mmf_hfb import homogeneous
from mmf_hfb.bcs import BCS


class ASLDA_(vortex_aslda.ASLDA):
    # a modified class from ASLDA with different alphas which are constant, so their derivatives are zero
    def _get_alphas(self, ns=None):
        dim = sum(self.xyz)
        alpha_a, alpha_b, alpha_p =np.ones_like(dim), np.ones_like(dim), np.ones_like(dim)
        return (alpha_a, alpha_b, alpha_p)       


def test_aslda_homogenous():
    L = 0.46
    N = 32
    N_twist = 32
    delta = 1.0
    mu_eff = 1.0
    v_0, n, mu, e_0 = homogeneous.Homogeneous1D().get_BCS_v_n_e(
        delta=delta, mus_eff=(mu_eff,mu_eff))
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
    assert np.allclose(n_, na.real + nb.real, rtol=0.001)
    assert np.allclose(delta, -v_0.n*kappa[0].real, rtol=0.01)
    print("Test 1d lattice with homogeneous system")
    ns, taus, js, kappa = b.get_dens_twisting(
        mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    na, nb = ns
    print((sum(n).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))
    print("Test 1d lattice plus 1d integral over y with homogeneous system")
    return
    ns, taus, js, kappa = b.get_dens_integral(
        mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    na, nb = ns
    print((sum(n).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))
    assert np.allclose(n_, na.real + nb.real, rtol=0.001)
    assert np.allclose(delta, -v_0.n*kappa[0].real, rtol=0.01)
    
    
if __name__ == "__main__":
    test_aslda_homogenous()
    # a = vortex_aslda.ASLDA(Nxyz=(32,), Lxyz=(1,))
    # a.get_densities()
    # print(a.dxyz)
