"""Test the homogeneous code."""
import numpy as np

from scipy.optimize import brentq

import homogeneous
from homogeneous import Homogeneous3D

def test_BCS_1D(lam_inv=0.5):
    """Test a few values from Table I of Quick:1993."""
    m = hbar = 1.0
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


def test_Homogeneous1D_T0():
    """Compare the Homogeneous1D class with get_BCS_v_n_e for T=0."""
    np.random.seed(2)    
    mu_eff, delta = np.random.random(2)
    res0 = homogeneous.get_BCS_v_n_e(mu_eff=mu_eff, delta=delta)
    for T in [0, 0.001]:
        res1 = homogeneous.Homogeneous1D(T=T).get_BCS_v_n_e(
            mus_eff=(mu_eff,)*2, delta=delta)
        assert np.allclose(res0.v_0, res1.v_0)
        assert np.allclose(res0.n, res1.ns.sum())
        assert np.allclose(res0.mu, res1.mus)    


def test_Homogeneous3D():
    """Compare the Homogeneous1D class with get_BCS_v_n_e for T=0."""
    res0 = homogeneous.get_BCS_v_n_e(mu_eff=1.2, delta=3.4)

    h3 = Homogeneous3D(T=10.0)
    res1 = h3.get_BCS_v_n_e(mus_eff=(1.2,)*2, delta=3.4)
    (res0, res1)

 # to debug in Visual Studio
if __name__ == '__main__':
    test_Homogeneous3D()