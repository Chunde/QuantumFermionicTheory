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
    def _dp_dn(self,ns):
        return (ns[0] * 0, ns[1]*0)
    #def f(self, E, E_c=None):
        #return 1
    #def get_Ks_Vs(self, delta, mus=(0,0), ns=None, taus=None, kappa=0, ky=0, kz=0, twist=0):
    #    alphas = self.get_alphas(ns)
    #    return (self.get_Ks(twist=twist), self.get_modified_Vs(delta=delta,ns=ns,taus=taus,kappa=kappa,alphas=alphas))

def test_aslda_twisting():
    """[pass]"""
    L = 0.46
    N = 128
    delta = 1.0
    N_twist = 5
    mu_eff = 1.0
    v_0, n, mu, e_0 = homogeneous._get_BCS_v_n_e(delta=delta, mu_eff=mu_eff)

    print("Test twisting validation...")
    b1 = ASLDA_(T=0,Nx=N,Lx = L)
    b2 = ASLDA_(T=0,Nx=N*2,Lx = L*2)
    for i in range(1,10):
        N_twist = i
        ns1,taus1,kappa1 = b1.get_ns_taus_kappa_average_1d(mus=(mu_eff  * np.exp(np.ones((N)),), mu_eff  * np.ones((N),)), delta=delta * np.ones((N),), N_twist=N_twist * 2)
        ns2,taus2,kappa2 = b2.get_ns_taus_kappa_average_1d(mus=(mu_eff  * np.exp( np.ones((b2.Nx)),), mu_eff  * np.ones((b2.Nx),)), delta=delta * np.ones((b2.Nx),), N_twist=N_twist)
        assert np.allclose(np.concatenate((ns1[0],ns1[0])), ns2[0])
        assert np.allclose(np.concatenate((ns1[1],ns1[1])), ns2[1])
        print("Twisting Number = %d pass"%(i))

if __name__ == "__main__":
    test_aslda_twisting()
