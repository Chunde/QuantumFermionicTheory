"""Test the homogeneous code."""
import hfb_dir_init

import numpy as np

from scipy.optimize import brentq

import homogeneous_mmf
from homogeneous_mmf import Homogeneous


def test_Homogeneous_3D():
    """Test known results at unitarity."""
    h = Homogeneous(T=0, dim=3)
    e_F = 1.0
    k_F = np.sqrt(2*h.m*e_F)/h.hbar
    n_F = k_F**3/3/np.pi**2
    E_FG = 3./5*n_F*e_F
    xi = 0.59060550703283853378393810185221521748413488992993*e_F
    eta = 0.68640205206984016444108204356564421137062514068346
    mu = xi*e_F
    delta = eta*e_F
    C0 = h.get_C0(mus=(mu, mu), delta=delta)
    n_p = h.get_n_p(mus=(mu, mu), delta=delta)
    n_m = h.get_n_m(mus=(mu, mu), delta=delta)
    kappa = h.get_kappa(mus=(mu, mu), delta=delta)
    
    assert np.allclose(C0.n, 0)
    assert abs(C0.n) <= C0.s
    
    assert np.allclose(n_m.n, 0)
    assert abs(n_m.n) <= n_m.s

    n_p_ = k_F**3/3/np.pi**2
    assert np.allclose(n_p.n, n_p_)
    assert abs(n_p.n - n_p_) <= n_p.s

    assert np.allclose(kappa.n, xi*E_FG)
    assert abs(kappa.n) <= kappa.s


def test_Homogeneous_2D():
    """Test known results at unitarity."""
    h = Homogeneous(T=0, dim=2)
    e_F = 1.0
    k_F = np.sqrt(2*h.m*e_F)/h.hbar
    n_F = k_F**3/3/np.pi**2
    E_FG = 2./3*n_F*e_F
    mu = 0.59060550703283853378393810185221521748413488992993*e_F
    delta = 0.68640205206984016444108204356564421137062514068346*e_F
    C0 = h.get_C0(mus=(mu, mu), delta=delta)
    n_p = h.get_n_p(mus=(mu, mu), delta=delta)
    n_m = h.get_n_m(mus=(mu, mu), delta=delta)

    C0_ = -0.1052014998191428
    assert np.allclose(C0.n, C0_)
    assert abs(C0.n-C0_) <= C0.s
    
    assert np.allclose(n_m.n, 0)
    assert abs(n_m.n) <= n_m.s

    n_p_ = 0.23811543349241718
    assert np.allclose(n_p.n, n_p_)
    assert abs(n_p.n - n_p_) <= n_p.s


def test_Homogeneous_1D(lam_inv=0.5):
    """Test a few values from Table I of Quick:1993."""
    h = Homogeneous(T=0, dim=1)
    np.random.seed(2)    
    delta = np.random.random(1)
    lam = 1./lam_inv
    def _lam(mu_eff):
        E_N_E_2, _lam = homogeneous.BCS(mu_eff=mu_eff,  delta=delta)
        return _lam - lam

    mu_eff = brentq(_lam, 0.6, 0.8)
    v_0, n, mu, e = homogeneous.get_BCS_v_n_e(mu_eff=mu_eff, delta=delta)
    E_N_E_2, lam = homogeneous.BCS(mu_eff=mu_eff,  delta=delta)
    mu_tilde = (hbar**2/m/v_0**2)*mu
    assert np.allclose(lam, 1./0.5)
    assert np.allclose(mu_tilde, 0.0864, atol=0.0005)    
    assert np.allclose(E_N_E_2, -0.3037, atol=0.0005)


def test_Homogeneous_1D():
    """Test known 1D results."""
    h = Homogeneous(T=0, dim=1)
    e_F = 1.0
    k_F = np.sqrt(2*h.m*e_F)/h.hbar
    n_F = k_F**3/3/np.pi**2
    E_FG = 2./3*n_F*e_F
    mu = 0.59060550703283853378393810185221521748413488992993*e_F
    delta = 0.68640205206984016444108204356564421137062514068346*e_F
    C0 = h.get_C0(mus=(mu, mu), delta=delta)
    n_p = h.get_n_p(mus=(mu, mu), delta=delta)
    n_m = h.get_n_m(mus=(mu, mu), delta=delta)

    C0_ = -0.1052014998191428
    assert np.allclose(C0.n, C0_)
    assert abs(C0.n-C0_) <= C0.s
    
    assert np.allclose(n_m.n, 0)
    assert abs(n_m.n) <= n_m.s

    n_p_ = 0.23811543349241718
    assert np.allclose(n_p.n, n_p_)
    assert abs(n_p.n - n_p_) <= n_p.s
    
