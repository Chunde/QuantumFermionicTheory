import numpy as np
from scipy.optimize import brentq
import py.test
import warnings
from mmf_hfb import tf_completion, homogeneous

def test_ws():
    np.random.seed(1)
    ks = np.random.random(3) * 1000
    mu_a, mu_b, m = 1 + np.random.random(3)
    mu_p = (mu_a + mu_b)/2
    mu_m = (mu_a - mu_b)/2

    for k in ks:
        dq, q = np.random.random(2) * 2
        k2_a = (k + q)**2
        k2_b = (k - q)**2
        em0, ep0, _, _, _ = tf_completion.get_ws(k2_a, k2_b,  mu_a, mu_b, 1, m, m, 1, 0)
        assert np.allclose(ep0, k**2/2/m - (mu_p - q**2/2/m))
        assert np.allclose(em0, q*k/m - mu_m)

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
    n_p = tf_completion.integrate(tf_completion.n_p_integrand, dim=3, **args)
    assert np.allclose(n_p.n, nF)

    E = tf_completion.integrate(tf_completion.kappa_integrand, dim=3,
                                k_c=100.0, **args)
    assert np.allclose(E.n, xi*E_FG, rtol=1e-2)
    
    args_ = dict(mu_a=mu, mu_b=mu, delta=delta, m_a=m, m_b=m, hbar=hbar, T=0.0)
    C_tilde = tf_completion.compute_C(dim=3, **args_)
    assert np.allclose(C_tilde.n, 0)
    
    n_p = tf_completion.integrate_q(tf_completion.n_p_integrand, dim=3, dq=0, **args)
    assert np.allclose(n_p.n, nF, rtol=0.0002)
    C_tilde = tf_completion.compute_C(dim=3, dq=0.00001, **args_)
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
    n_p = tf_completion.integrate(tf_completion.n_p_integrand, dim=2, **args)
    assert np.allclose(n_p.n, nF)

    n_p = tf_completion.integrate_q(tf_completion.n_p_integrand, dim=2, dq=0, **args)
    assert np.allclose(n_p.n, nF)
    
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
    h = homogeneous.Homogeneous1D()
    v_0, ns, mu, e = h.get_BCS_v_n_e(mus_eff=(mu_eff,)*2, delta=delta)
    n = sum(ns)
    lam = m*v_0/n/hbar**2

    # Energy per-particle
    E_N = e/n

    # Energy per-particle for 2 particles
    E_2 = -m*v_0**2/4.0 / 2.0
    E_N_E_2 = E_N/abs(E_2)
    return E_N_E_2.n, lam.n


def test_1D():
    """Test a few values from Table I of Quick:1993."""
    lam_inv = 0.5
    np.random.seed(1)
    m, hbar, delta = 0.1 + np.random.random(3)
    lam = 1./lam_inv

    def _lam(mu_eff):
        E_N_E_2, _lam = BCS(mu_eff=mu_eff, delta=delta)
        return _lam - lam

    mu_eff = brentq(_lam, 0.1, 20)

    args = dict(mu_a=mu_eff, mu_b=mu_eff, delta=delta, m_a=m, m_b=m,
                hbar=hbar, T=0.0)
    
    n_p = tf_completion.integrate(tf_completion.n_p_integrand, dim=1, **args)
    nu = tf_completion.integrate(tf_completion.nu_integrand, dim=1, **args)
    v_0 = -delta/nu.n
    mu = mu_eff - n_p.n*v_0/2
    lam = m*v_0/n_p.n/hbar**2
    
    #v_0, n, mu, e = homogeneous.get_BCS_v_n_e(mu_eff=mu_eff, delta=delta)
    E_N_E_2, lam = BCS(mu_eff=mu_eff,  delta=delta)
    mu_tilde = (hbar**2/m/v_0**2)*mu
    assert np.allclose(lam, 1./lam_inv)
    assert np.allclose(mu_tilde, 0.0864, atol=0.0005)
    assert np.allclose(E_N_E_2, -0.3037, atol=0.0005)

    n_p = tf_completion.integrate_q(tf_completion.n_p_integrand, dim=1, dq=0, **args)
    nu = tf_completion.integrate_q(tf_completion.nu_integrand, dim=1, dq=0, **args)
    v_0 = -delta/nu.n
    mu = mu_eff - n_p.n*v_0/2
    lam = m*v_0/n_p.n/hbar**2
    
    mu_tilde = (hbar**2/m/v_0**2)*mu
    assert np.allclose(lam, 1./lam_inv)
    assert np.allclose(mu_tilde, 0.0864, atol=0.0005)


if __name__ == "__main__":
    test_1D()