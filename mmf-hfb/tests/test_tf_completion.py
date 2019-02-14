import numpy as np
from scipy.optimize import brentq
import py.test

from mmf_hfb import tf_completion, homogeneous


def test_3D():
    """Test the 3D UFG solution."""
    np.random.seed(1)
    m, hbar, kF = 1 + np.random.random(3)
    eF = (hbar*kF)**2/2/m
    nF = kF**3/3/np.pi**2
    E_FG = 3./5*nF*eF
    xi = 0.59060550703283853378393810185221521748413488992993
    mu = xi*eF
    delta = 0.68640205206984016444108204356564421137062514068346*eF
    args = dict(mu_a=mu, mu_b=mu, delta=delta, m_a=m, m_b=m, hbar=hbar, T=0.0)
    n_p = tf_completion.integrate(tf_completion.n_p_integrand, d=3, **args)
    assert np.allclose(n_p.n, nF)

    E = tf_completion.integrate(tf_completion.kappa_integrand, d=3,
                                k_c=100.0, **args)
    assert np.allclose(E.n, xi*E_FG, rtol=1e-2)
    
    args_ = dict(mu_a=mu, mu_b=mu, delta=delta, m_a=m, m_b=m, hbar=hbar, T=0.0)
    C_tilde = tf_completion.compute_C(d=3, **args_)
    assert np.allclose(C_tilde.n, 0)
    
    n_p = tf_completion.integrate_q(tf_completion.n_p_integrand, d=3, q=0, **args)
    assert np.allclose(n_p.n, nF, rtol=0.0002)
    C_tilde = tf_completion.compute_C(d=3, q=0.00001, **args_)
    assert np.allclose(C_tilde.n, 0)

    
def test_2D():
    """Test the 2D UFG solution."""
    np.random.seed(1)
    m, hbar, kF = 1 + np.random.random(3)
    eF = (hbar*kF)**2/2/m
    nF = kF**2/2/np.pi
    mu = 0.5*eF
    delta = np.sqrt(2)*eF
    args = dict(mu_a=mu, mu_b=mu, delta=delta, m_a=m, m_b=m, hbar=hbar, T=0.0)
    n_p = tf_completion.integrate(tf_completion.n_p_integrand, d=2, **args)
    assert np.allclose(n_p.n, nF)

    n_p = tf_completion.integrate_q(tf_completion.n_p_integrand, d=2, q=0, **args)
    assert np.allclose(n_p.n, nF)
    

def test_1D():
    """Test a few values from Table I of Quick:1993."""
    lam_inv = 0.5
    np.random.seed(1)
    m, hbar, delta = 0.1 + np.random.random(3)
    lam = 1./lam_inv

    def _lam(mu_eff):
        E_N_E_2, _lam = homogeneous.BCS(mu_eff=mu_eff, delta=delta)
        return _lam - lam

    mu_eff = brentq(_lam, 0.1, 20)

    args = dict(mu_a=mu_eff, mu_b=mu_eff, delta=delta, m_a=m, m_b=m,
                hbar=hbar, T=0.0)
    
    n_p = tf_completion.integrate(tf_completion.n_p_integrand, d=1, **args)
    nu = tf_completion.integrate(tf_completion.nu_integrand, d=1, **args)
    v_0 = delta/nu.n
    mu = mu_eff - n_p.n*v_0/2
    lam = m*v_0/n_p.n/hbar**2
    
    #v_0, n, mu, e = homogeneous.get_BCS_v_n_e(mu_eff=mu_eff, delta=delta)
    #E_N_E_2, lam = homogeneous.BCS(mu_eff=mu_eff,  delta=delta)
    mu_tilde = (hbar**2/m/v_0**2)*mu
    assert np.allclose(lam, 1./lam_inv)
    assert np.allclose(mu_tilde, 0.0864, atol=0.0005)    
    #assert np.allclose(E_N_E_2, -0.3037, atol=0.0005)

    n_p = tf_completion.integrate_q(tf_completion.n_p_integrand, d=1, q=0, **args)
    nu = tf_completion.integrate_q(tf_completion.nu_integrand, d=1, q=0, **args)
    v_0 = delta/nu.n
    mu = mu_eff - n_p.n*v_0/2
    lam = m*v_0/n_p.n/hbar**2
    
    mu_tilde = (hbar**2/m/v_0**2)*mu
    assert np.allclose(lam, 1./lam_inv)
    assert np.allclose(mu_tilde, 0.0864, atol=0.0005)    
    
if __name__ == "__main__":

    test_1D()
    test_2D()
    test_3D()
