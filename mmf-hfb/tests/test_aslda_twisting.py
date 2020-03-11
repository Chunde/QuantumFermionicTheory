import numpy as np
from mmf_hfb import bcs_aslda
from collections import namedtuple
import warnings
import scipy.integrate
import scipy as sp


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


class ASLDA_(bcs_aslda.ASLDA):
    # a modified class from ASLDA with different
    # alphas which are constant, so their derivatives are zero
    def _get_alphas(self, ns=None):
        alpha_a, alpha_b = np.ones(self.Nx), np.ones(self.Nx)
        return (alpha_a, alpha_b)
    
    def _dp_dn(self, ns):
        return (ns[0] * 0, ns[1]*0)


def test_aslda_twisting():
    """[pass]"""
    L = 0.46
    N = 128
    delta = 1.0
    N_twist = 5
    mu_eff = 1.0
    print("Test twisting validation...")
    b1 = ASLDA_(T=0, Nxyz=(N,), Lxyz=(L,))
    b2 = ASLDA_(T=0, Nxyz=(N*2,), Lxyz=(L*2,))
    for i in range(1, 10):
        N_twist = i
        ns1, _, _, _ = b1.get_densities(
            mus_eff=(
                mu_eff*np.exp(np.ones((N)),), mu_eff*np.ones((N),)),
            delta=delta*np.ones((N),),
            N_twist=N_twist*2, struct=False)
        ns2, _, _, _ = b2.get_densities(
            mus_eff=(
                mu_eff*np.exp(np.ones((b2.Nxyz)),),
                mu_eff*np.ones((b2.Nxyz),)),
            delta=delta*np.ones((b2.Nxyz),), N_twist=N_twist, struct=False)
        assert np.allclose(np.concatenate((ns1[0], ns1[0])), ns2[0])
        assert np.allclose(np.concatenate((ns1[1], ns1[1])), ns2[1])
        print(f"Twisting Number={i+1} pass")
