import numpy as np
from importlib import reload  # Python 3.4+
import numpy as np
from mmf_hfb import homogeneous;reload(homogeneous)
from mmf_hfb import bcs;reload(bcs)
from mmf_hfb.bcs import BCS
from mmf_hfb import bcs_aslda
import time
from collections import namedtuple
import warnings
import scipy.integrate
import scipy as sp
hbar = 1
m = 1


def _quad(f, kF=None, k_0=0, k_inf=np.inf, limit=1000):
    """Wrapper for quad that deals with singularities
    at the Fermi surface.
    """
    if kF is None:
        res, err = sp.integrate.quad(f, k_0, k_inf)
    else:
        # One might think that `points=[kF]` could be used here, but
        # this does not work with infinite limits.
        res0, err0 = sp.integrate.quad(f, k_0, kF)
        res1, err1 = sp.integrate.quad(f, kF, k_inf, limit=limit)
        res = res0 + res1
        err = max(err0, err1)

    if abs(err) > 1e-6 and abs(err/res) > 1e-6:
        warnings.warn("Gap integral did not converge: res, err = %g, %g" % (res, err))
    return 2*res   # Accounts for integral from -inf to inf

def get_BCS_v_n_e(delta, mu_eff):
    m = hbar = 1.0
    kF = np.sqrt(2*m*max(0, mu_eff))/hbar

    def gap_integrand(k):
        e_p = (hbar*k)**2/2.0/m - mu_eff
        return 1./np.sqrt(e_p**2 + abs(delta)**2)

    v_0 = 4*np.pi / _quad(gap_integrand, kF)

    def n_integrand(k):
        """Density"""
        e_p = (hbar*k)**2/2.0/m - mu_eff
        denom = np.sqrt(e_p**2 + abs(delta)**2)
        return (denom - e_p)/denom

    n = _quad(n_integrand, kF) / 2/np.pi

    def e_integrand(k):
        """Energy"""
        e_p = (hbar*k)**2/2.0/m - mu_eff
        denom = np.sqrt(e_p**2 + abs(delta)**2)
        return (hbar*k)**2/2.0/m * (denom - e_p)/denom
        """Where this formula comes from?"""
    e = _quad(e_integrand, kF) / 2/np.pi - v_0*n**2/4.0 - abs(delta)**2/v_0
    mu = mu_eff - n*v_0/2

    return namedtuple('BCS_Results', ['v_0', 'n', 'mu', 'e'])(v_0, n, mu, e)

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
        BCS.__init__(self, Lxyz=[cells*L,], Nxyz=[cells*N,], **kw)
    
    def get_v_ext(self):
        v_a =  (-self.V0 * (1-((1+np.cos(2*np.pi * self.cells*self.xyz[0]/self.Lxyz[0]))/2)**self.power))
        v_b = 0 * self.xyz[0]
        return v_a, v_b
class ASLDA_(bcs_aslda.ASLDA):
    # a modified class from ASLDA with different alphas which are constant, so their derivatives are zero
    def _get_alphas(self, ns = None):
        alpha_a,alpha_b,alpha_p =np.ones(self.Nx), np.ones(self.Nx), np.ones(self.Nx)
        return (alpha_a,alpha_b)       
        return super().get_alphas(ns)
    def _dp_dn(self,ns):
        return (ns[0] * 0, ns[1]*0)

def test_aslda_unitary():
    """"test the unitary case, but seems not close"""
    L = 0.46
    N = 32
    N_twist = 32
    b = ASLDA_(T=0, Nxyz=(N,), Lxyz=(L,))
    k_c = abs(b.kxyz[0]).max()
    E_c = (b.hbar*k_c)**2/2/b.m 
    mu_eff = 1.0
    n = 1.0
    k_F = np.sqrt(2*m*E_c)
    n_F = k_F**3/3/np.pi**2
    E_FG = 2./3*n_F*E_c
    delta = 1  #0.68640205206984016444108204356564421137062514068346 * E_c
    mu_eff = 1 # 0.59060550703283853378393810185221521748413488992993*E_c
    v_0, n, mu, e_0 = get_BCS_v_n_e(delta=delta, mu_eff=mu_eff)

    l = Lattice(T=0.0, N=N, L=L, v0=v_0, V0=0)
    R = l.get_Rs(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)[0]
    na = np.diag(R)[:N]/l.dV
    nb = (1 - np.diag(R)[N:])/l.dV
    kappa = np.diag(R[:N, N:])/l.dV
    print((n, na[0].real + nb[0].real), (delta, -v_0*kappa[0].real))
    assert np.allclose(n, na[0].real + nb[0].real, rtol=0.001)
    assert np.allclose(delta, -v_0*kappa[0].real, rtol=0.01)
    print("Test 1d lattice with homogeneous system")
    R = b.get_R(mus_eff=(mu_eff*np.ones((N),), mu_eff*np.ones((N),)), delta=delta*np.ones((N),), N_twist=N_twist)
    na = np.diag(R)[:N]/np.prod(b.dxyz)
    nb = (1 - np.diag(R)[N:])/np.prod(b.dxyz)
    kappa = np.diag(R[:N, N:])/np.prod(b.dxyz)
    print((n, na[0].real + nb[0].real), (delta, -v_0*kappa[0].real))
    assert np.allclose(n, na[0].real + nb[0].real, rtol=0.001)
    assert np.allclose(delta, -v_0*kappa[0].real, rtol=0.01)
    return # not pass yet
    print("Test 1d lattice with homogeneous system")
    ns, taus, kappa = b.get_ns_taus_kappa_average_1d(mus=(mu_eff*np.ones((N),), mu_eff*np.ones((N),)), delta=delta*np.ones((N),), N_twist=N_twist)
    na, nb = ns
    print((n, na[0].real + nb[0].real), (delta, -v_0*kappa[0].real))
    assert np.allclose(n, na[0].real + nb[0].real, rtol=0.001)
    assert np.allclose(delta, -v_0*kappa[0].real, rtol=0.01)
    print("Test 1d lattice plus 1d integral over y with homogeneous system")
    k_c = abs(b.kx).max()
    E_c = (b.hbar*k_c)**2/2/b.m # 3 dimension, so E_c should have a factor of 3
    b.E_c = E_c
    ns, taus, kappa = b.get_ns_taus_kappa_average_2d(mus=(mu_eff*np.ones((N),), mu_eff*np.ones((N),)), delta=delta*np.ones((N),), N_twist=N_twist)
    na, nb = ns
    print((n, na[0].real + nb[0].real), (delta, -v_0*kappa[0].real))
    assert np.allclose(n, na[0].real + nb[0].real, rtol=0.001)
    assert np.allclose(delta, -v_0*kappa[0].real, rtol=0.01)
    print("Test 1d lattice plus 2d integrals over y and  z with homogeneous system")
    ns, taus, kappa = b.get_ns_taus_kappa_average_3d(mus=(mu_eff*np.ones((N),), mu_eff*np.ones((N),)), delta=delta*np.ones((N),), N_twist=N_twist)
    na, nb = ns
    print((n, na[0].real + nb[0].real), (delta, -v_0*kappa[0].real))

if __name__ == '__main__':
    test_aslda_unitary()
