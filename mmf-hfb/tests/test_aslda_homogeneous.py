import numpy as np
from mmf_hfb import homogeneous
from mmf_hfb.bcs import BCS
from mmf_hfb import bcs_aslda
hbar = 1
m = 1


class Lattice(BCS):
    """Adds optical lattice potential to species a with depth V0."""
    cells = 1.0
    power = 4
    V0 = -10.5

    def __init__(
        self, cells=1, N=2**5, L=10.0,
        mu_a=1.0, mu_b=1.0, v0=0.1, V0=-10.5,
        power=2, **kw):
        self.power = power
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.v0 = v0
        self.V0 = V0
        self.cells = cells
        BCS.__init__(self, L=cells*L, N=cells*N, **kw)
    
    def get_v_ext(self):
        v_a =  (-self.V0*(1 - ((1 + np.cos(2*np.pi*self.cells*self.x/self.L))/2)**self.power))
        v_b = 0*self.x
        return v_a, v_b


class ASLDA_(bcs_aslda.ASLDA):
    # a modified class from ASLDA with different 
    # alphas which are constant, so their derivatives are zero
    def _get_alphas(self, ns=None):
        alpha_a, alpha_b, alpha_p =np.ones(self.Nx), np.ones(self.Nx), np.ones(self.Nx)
        return (alpha_a, alpha_b, alpha_p)       

def test_aslda_homogenous():
    L = 0.46
    N = 32
    N_twist = 32
    delta = 1.0
    mu_eff = 1.0
    v_0, n, mu, e_0 = homogeneous.Homogeneous1D().get_BCS_v_n_e(delta=delta, mus_eff=(mu_eff,mu_eff))
    n_ = np.ones((N),)*(n[0].n+n[1].n)
    print("Test 1d lattice with homogeneous system")
    b = ASLDA_(T=0, Nxyz=(N,), Lxyz=(L,))
    k_c = abs(b.kxyz[0]).max()
    E_c = (b.hbar*k_c)**2/2/b.m 
    b.E_c = E_c
    R = b.get_R(mus_eff=(mu_eff *np.ones((N),), mu_eff*np.ones((N),)),
        delta=delta*np.ones((N),), N_twist=N_twist)
    na = np.diag(R)[:N]/np.prod(b.dxyz)
    nb = (1 - np.diag(R)[N:])/np.prod(b.dxyz)
    kappa = np.diag(R[:N, N:])/np.prod(b.dxyz)
    print((sum(n).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))
    assert np.allclose(n_, na.real + nb.real, rtol=0.001)
    assert np.allclose(delta, -v_0.n*kappa[0].real, rtol=0.01)
    print("Test 1d lattice with homogeneous system")
    ns, taus, js, kappa = b.get_densities(mus_eff=(mu_eff*np.ones((N),),
                                                mu_eff*np.ones((N),)), delta=delta*np.ones((N),), N_twist=N_twist, struct=False)
    na, nb = ns
    print((sum(n).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))
    assert np.allclose(n_, na.real + nb.real, rtol=0.001)
    assert np.allclose(delta, -v_0.n*kappa[0].real, atol=0.01)
 

if __name__ == '__main__':
    test_aslda_homogenous()