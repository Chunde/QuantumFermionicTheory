import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from uncertainties import ufloat
from importlib import reload  # Python 3.4+
import numpy as np
from mmf_hfb import homogeneous;reload(homogeneous)
from mmf_hfb import bcs;reload(bcs)
from mmf_hfb.bcs import BCS
from mmf_hfb import vortex_1d_aslda;reload(vortex_1d_aslda)
import itertools  
import matplotlib.pyplot as plt

import time
hbar = 1
m = 1
innerTest = False
plt.autoscale(enable=True, axis='both', tight=None)
plt.ion()
fig = None

class Lattice(BCS):
    """Adds optical lattice potential to species a with depth V0."""
    cells = 1.0
    t = 0.0007018621290128983
    E0 = -0.312433127299677
    power = 4
    V0 = -10.5
    def __init__(self, cells=1, N=2**5, L=10.0,  mu_a=1.0, mu_b=1.0, v0=0.1, V0=-10.5, power=2,**kw):
        self.power = power
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.v0 = v0
        self.V0 = V0
        self.cells = cells
        BCS.__init__(self, L=cells*L, N=cells*N, **kw)
    
    def get_v_ext(self):
        v_a =  (-self.V0 * (1-((1+np.cos(2*np.pi * self.cells*self.x/self.L))/2)**self.power))
        v_b = 0 * self.x
        return v_a, v_b
class ASLDA_(vortex_1d_aslda.ASLDA):
# a modified class from ASLDA with different alphas which are constant, so their derivatives are zero
    def get_alphas(self, ns = None):
        alpha_a,alpha_b,alpha_p =np.ones(self.Nx),np.ones(self.Nx),np.ones(self.Nx)
        return (alpha_a,alpha_b,alpha_p)       
        return super().get_alphas(ns)
    #def _dp_dn(self,ns):
    #    return (ns[0] * 0, ns[1]*0)
    #def f(self, E, E_c=None):
        #return 1
    #def get_Ks_Vs(self, delta, mus=(0,0), ns=None, taus=None, kappa=0, ky=0, kz=0, twist=0):
    #    alphas = self.get_alphas(ns)
    #    return (self.get_Ks(twist=twist), self.get_modified_Vs(delta=delta,ns=ns,taus=taus,kappa=kappa,alphas=alphas))

def test_ASLDA_Homogenous():
    L = 0.46
    N = 16
    N_twist = 128
    delta = 1.0
    mu_eff = 1.0
    v_0, n, mu, e_0 = homogeneous.Homogeneous1D().get_BCS_v_n_e(delta=delta, mus_eff=(mu_eff,mu_eff))
    n_ = np.ones((N),)*(n[0].n+n[1].n)
    print("Test 1d lattice with homogeneous system")
    b = ASLDA_(T=0,Nx=N,Lx = L)
    k_c = abs(b.kx).max()
    E_c = (b.hbar*k_c)**2/2/b.m 
    b.E_c = E_c
    R = b.get_R(mus=(mu_eff  * np.ones((N),), mu_eff  * np.ones((N),)), delta=delta * np.ones((N),), N_twist=N_twist)
    na = np.diag(R)[:N]/b.dx
    nb = (1 - np.diag(R)[N:])/b.dx
    kappa = np.diag(R[:N, N:])/b.dx
    print((n, na[0].real + nb[0].real), (delta, -v_0*kappa[0].real))
    assert np.allclose(n_, na.real + nb.real,atol=0.001)
    assert np.allclose(delta, -v_0.n*kappa[0].real,atol=0.01)
    print("Test 1d lattice with homogeneous system")
    ns,taus,kappa = b.get_ns_taus_kappa_average_1d(mus=(mu_eff  * np.ones((N),), mu_eff  * np.ones((N),)), delta=delta * np.ones((N),), N_twist=N_twist)
    na,nb = ns
    print((n, na[0].real + nb[0].real), (delta, -v_0*kappa[0].real))
    assert np.allclose(n_, na.real + nb.real,atol=0.001)
    assert np.allclose(delta, -v_0.n*kappa[0].real,atol=0.01)
    print("Test 1d lattice plus 1d integral over y with homogeneous system")
    ns,taus,kappa = b.get_ns_taus_kappa_average_2d(mus=(mu_eff  * np.ones((N),), mu_eff  * np.ones((N),)), delta=delta * np.ones((N),), N_twist=N_twist)
    na,nb = ns
    print((n, na[0].real + nb[0].real), (delta, -v_0*kappa[0].real))
    assert np.allclose(n_, na.real + nb.real,atol=0.001)
    assert np.allclose(delta, -v_0.n*kappa[0].real,atol=0.01)
    print("Test 1d lattice plus 2d integrals over y and  z with homogeneous system")
    ns,taus,kappa = b.get_ns_taus_kappa_average_3d(mus=(mu_eff  * np.ones((N),), mu_eff  * np.ones((N),)), delta=delta * np.ones((N),), N_twist=N_twist)
    na,nb = ns
    print((n, na[0].real + nb[0].real), (delta, -v_0*kappa[0].real))
    assert np.allclose(n_, na.real + nb.real,atol=0.01)
    #assert np.allclose(delta, v_0*kappa[0].real,atol=0.001)



if __name__ == '__main__':
    innerTest = True
    test_ASLDA_Homogenous()