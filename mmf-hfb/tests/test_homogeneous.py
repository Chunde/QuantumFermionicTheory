"""Test the homogeneous code."""
import numpy as np
from scipy.optimize import brentq
import pytest
from mmfutils.testing import allclose
from mmf_hfb import homogeneous
from mmf_hfb.homogeneous import Homogeneous1D, Homogeneous3D


@pytest.fixture(params=[1, 2, 3])
def dim(request):
    return request.param

def BCS(mu_eff, delta=1.0):
    m = hbar = 1.0
    """Return `(E_N_E_2, lam)` for comparing with the exact Gaudin
    solution.

    Arguments
    ---------
    delta : float
       Pairing gap.  This is the gap in the energy spectrum.
    mu_eff : float
       Effective chemical potential including both the bare chemical
       potential and the self-energy correction arising from the
       Hartree term.

    Returns
    -------
    E_N_E_2 : float
       Energy per particle divided by the two-body binding energy
       abs(energy per particle) for 2 particles.
    lam : float
       Dimensionless interaction strength.
    """
    h = Homogeneous1D()
    v_0, ns, mu, e = h.get_BCS_v_n_e(mus_eff=(mu_eff,)*2, delta=delta)
    n = sum(ns)
    lam = m*v_0/n/hbar**2

    # Energy per-particle
    E_N = e/n

    # Energy per-particle for 2 particles
    E_2 = -m*v_0**2/4.0 / 2.0
    E_N_E_2 = E_N/abs(E_2)
    return E_N_E_2.n, lam.n


class TestIntegration(object):
    """Test the integrators in homogeneous"""
    # Normalized Gaussian in d=1,2,3 dimensions
    f = {1: lambda k: 2*np.sqrt(np.pi) * np.exp(-k**2),
         2: lambda k: 4*np.pi * np.exp(-k**2),
         3: lambda k: 8*np.sqrt(np.pi)**3 * np.exp(-k**2)}

    def test_quad_k(self, dim):
        f = self.f[dim]
        assert np.allclose(homogeneous.quad_k(f, dim=dim).n, 1)
    
    def test_quad_l(self, dim):
        f = self.f[dim]
        Nxyz = (64,)*dim
        Lxyz = (10.0,)*dim
        assert np.allclose(homogeneous.quad_l(f, Nxyz=Nxyz, Lxyz=Lxyz).n, 1)


class TestHomogeneous(object):
    """Test Homogeneous classes."""
    def test_1D_T0(self):
        """Test the Homogeneous1D class for a known solution."""
        np.random.seed(2)
        hbar, m, kF = 1.0 + np.random.random(3)
        nF = kF/np.pi
        eF = (hbar*kF)**2/2/m
        # E_FG = 2*nF*eF/3
        C_unit = m/hbar**2/kF

        mu = 0.28223521359748843*eF
        delta = 0.411726229961806*eF
        h = homogeneous.Homogeneous1D(m=m, hbar=hbar)
        res1 = h.get_BCS_v_n_e(mus_eff=(mu,)*2, delta=delta)
        assert allclose(res1.v_0, 1./C_unit)
        assert allclose(sum(res1.ns), nF)

    def test_1D_Quick(self, lam_inv=0.5):
        """Test a few values from Table I of Quick:1993."""
        m = hbar = 1.0
        np.random.seed(2)
        delta = np.random.random(1)
        lam = 1./lam_inv

        h = homogeneous.Homogeneous1D(m=m, hbar=hbar)
        
        def _lam(mu_eff):
            E_N_E_2, _lam = BCS(mu_eff=mu_eff, delta=delta)
            return _lam - lam

        mu_eff = brentq(_lam, 0.6, 0.8, xtol=1e-5)
        v_0, n, mu, e = h.get_BCS_v_n_e(mus_eff=(mu_eff, mu_eff), delta=delta)
        E_N_E_2, lam = BCS(mu_eff=mu_eff, delta=delta)
        mu_tilde = (hbar**2/m/v_0**2)*mu
        assert allclose(lam, 1./0.5)
        assert allclose(mu_tilde, 0.0864, atol=0.0005)
        assert allclose(E_N_E_2, -0.3037, atol=0.0005)
        
    def test_2D_T0(self):
        """Test the Homogeneous1D class for a known solution."""
        np.random.seed(2)
        hbar, m, kF = 1.0 + np.random.random(3)
        nF = kF**2/2/np.pi
        eF = (hbar*kF)**2/2/m
        # E_FG = nF*eF/2

        mu = 0.5 * eF
        delta = np.sqrt(2) * eF

        h = homogeneous.Homogeneous2D(m=m, hbar=hbar)
        res = h.get_densities(mus_eff=(mu,)*2, delta=delta)
        assert allclose(res.n_a+res.n_b, nF)

    def test_3D_T0(self):
        """Test the Homogeneous1D class for a known solution."""
        np.random.seed(2)
        hbar, m, kF = 1.0 + np.random.random(3)
        xi = 0.59060550703283853378393810185221521748413488992993
        nF = kF**3/3/np.pi**2
        eF = (hbar*kF)**2/2/m
        # E_FG = 3*nF*eF/5

        mu = xi * eF
        delta = 0.68640205206984016444108204356564421137062514068346 * eF

        h = homogeneous.Homogeneous3D(m=m, hbar=hbar)
        res = h.get_densities(mus_eff=(mu,)*2, delta=delta)
        assert allclose(res.n_a+res.n_b, nF)



def _test_Homogeneous3D_T0_Unitary():
    """Compare the Homogeneous1D class with get_BCS_v_n_e for T=0."""
    h3 = Homogeneous3D(T=0)
    delta = 3.4
    res = h3.get_BCS_v_n_e(mus_eff=(1.2,)*2, delta=delta,unitary = True)
    #1.1622005617900125710mu_+
    mu_p = res.mus.sum() / 2.0
    assert allclose(delta, mu_p * 1.1622005617900125710)

    
def _test_Homogeneous3D_scattering_length():
    """Compare the Homogeneous1D class with get_BCS_v_n_e for T=0."""
    kc = 10000.0
    h3 = Homogeneous3D(T=0)
    res0 = h3.get_scattering_length(mus_eff=(1.2,)*2, delta=3.4,k_c=kc)
    res1 = h3.get_scattering_length(mus_eff=(1.2,)*2, delta=3.4,k_c=2.0 * kc)
    print(res0, res1)
    assert allclose(res0, res1,atol=0.0005)    
 # to debug in Visual Studio
if __name__ == '__main__':
    t = TestHomogeneous()
    t.test_3D_T0()
