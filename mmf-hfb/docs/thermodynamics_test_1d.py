# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import * 
from mmf_hfb import tf_completion as tf
from mmf_hfb.FuldeFerrellState import FFState as FF
from mmf_hfb import bcs, homogeneous
from scipy.optimize import brentq
from mmfutils.plot import imcontourf
plt.figure(figsize(10,4))
clear_output()


# # Test Thermodynamic in 1D
# * This is the self-consistent method, but not success yet.

# ## Check the effective 
# * check if the effective mu is same as given in the homogeneous case

# +
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


# -

def test_efftive_mus():
    """Test a few values from Table I of Quick:1993."""
    lam_invs = [0.5]#,  1.5
    mu_tilde_s = [0.0864]#,  2.0259
    E_N_E_2_s =  [-0.3037]#,  4.4021
    np.random.seed(1)
    for i in range(len(lam_invs)):
        lam_inv = lam_invs[i]
        mu_tilde_ = mu_tilde_s[i]
        E_N_E_2_ = E_N_E_2_s[i]
        m = hbar=delta=1
        delta = (0.1 + np.random.random(1))[0]
        print(f"m={m}, hbar={hbar}, delta={delta}")
        lam = 1./lam_inv

        def _lam(mu_eff):
            E_N_E_2, _lam = BCS(mu_eff=mu_eff, delta=delta)
            return _lam - lam

        mu_eff = brentq(_lam, 0.1, 20)

        args = dict(mu_a=mu_eff, mu_b=mu_eff, delta=delta, m_a=m, m_b=m,
                    hbar=hbar, T=0.0)
    
        n_p = tf.integrate(tf.n_p_integrand, dim=1, **args)
        nu = tf.integrate(tf.nu_integrand, dim=1, **args)
        v_0 = -delta/nu.n
        mu = mu_eff - n_p.n*v_0/2
        E_N_E_2, lam = BCS(mu_eff=mu_eff,  delta=delta)
        mu_tilde = (hbar**2/m/v_0**2)*mu
        assert np.allclose(lam, 1./lam_inv)
        assert np.allclose(mu_tilde,mu_tilde_, atol=0.0005)
        assert np.allclose(E_N_E_2, E_N_E_2_, atol=0.0005)
        ff = FF(mu=mu, dmu=0, delta=delta, dim=1, k_c=np.inf, fix_g=True)
        mus_eff = ff._get_effective_mus(mu=mu, dmu=0, delta=delta, update_g=True)
        ns = ff.get_densities(mu=mus_eff[0], dmu = mus_eff[1],delta=delta)
        assert np.allclose(n_p.n, (ns[0] + ns[1]).n)
        assert np.allclose(mus_eff[0], mu_eff)
        assert np.allclose(v_0, -ff._g)


test_efftive_mus()


# ## Check Thermodynamics

def test_Thermodynamic_1d(mu, dmu, delta0=1, k_c=1000, q=0, dq=0,
                 T=0.0,a=0.8, b=1.2, dx=1e-2):
    """test id case"""

    ff = FF(mu=mu, dmu=dmu, delta=delta0, q=q, dq=dq, dim=1, k_c=k_c, T=T, 
            fix_g=True, bStateSentinel=True)
    n_a, n_b, e, p, mus_eff = ff.get_ns_p_e_mus_1d(mu=mu, dmu=dmu, delta=delta0, q=q, dq=dq, k_c=k_c, update_g=True)

    n_a_1, n_b_1, e1, p1, mus1 = ff.get_ns_p_e_mus_1d(mu=mu+dx, dmu=dmu, mus_eff=mus_eff, q=q, dq=dq, k_c=k_c, update_g=False)
    n_a_2, n_b_2, e2, p2, mus2 = ff.get_ns_p_e_mus_1d(mu=mu-dx, dmu=dmu, mus_eff=mus_eff, q=q, dq=dq, k_c=k_c, update_g=False)
    n_p_ = (p1 - p2)/2/dx
    print(f"Expected n_p={n_a + n_b}\tNumerical n_p={n_p_}")

    # Fixed mu_b by changing mu and dmu with same value , as mu_b = mu - dmu
    # Then dP / dx = n_a
    n_a_1, n_b_1, e1, p1, mus1 = ff.get_ns_p_e_mus_1d(mu=mu+dx/2, dmu=dmu+dx/2, mus_eff=mus_eff, q=q, dq=dq, k_c=k_c, update_g=False)
    n_a_2, n_b_2, e2, p2, mus2 = ff.get_ns_p_e_mus_1d(mu=mu-dx/2, dmu=dmu-dx/2, mus_eff=mus_eff,  q=q, dq=dq, k_c=k_c, update_g=False)
    n_a_ = (p1 - p2)/2/dx
    print(f"Expected n_a={n_a}\tNumerical n_a={n_a_}")

    # Fixed mu_a by changing mu and dmu with opposite values , as mu_a = mu + dmu
    # Then dP / dx = n_b
    n_a_3, n_b_3, e3, p3, mus3 = ff.get_ns_p_e_mus_1d(mu=mu+dx/2, dmu=dmu-dx/2, mus_eff=mus_eff, q=q, dq=dq, k_c=k_c, update_g=False)
    n_a_4, n_b_4, e4, p4, mus4 = ff.get_ns_p_e_mus_1d(mu=mu-dx/2, dmu=dmu+dx/2, mus_eff=mus_eff, q=q, dq=dq, k_c=k_c, update_g=False)
    n_b_ = (p3 - p4)/2/dx

    print(f"Expected n_b={n_b}\tNumerical n_b={n_b_}")
    assert np.allclose(n_a, n_a_)
    assert np.allclose(n_b, n_b_)


test_Thermodynamic_1d(mu=3, dmu=0.5, delta0=1, k_c=100, q=0, dq=0, T=0.0,a=0.8, b=1.2, dx=1e-3)


# # Old Version
# * In the old version code, we do not have the Hartree term in the energy density

def test_Thermodynamic_1d(delta, mu_delta, dmu_delta, q_dmu, dq_dmu, dx=1e-3):
    """test id case"""
    mu = mu_delta * delta
    dmu = dmu_delta * delta
    q = q_dmu * mu
    dq = dq_dmu * mu
    ff = FF(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq, dim=1, fix_g=True, bStateSentinel=True)
    n_a, n_b, e, p, mus_eff = ff.get_ns_p_e_mus_1d(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq, update_g=True)

    n_a_1, n_b_1, e1, p1, mus1 = ff.get_ns_p_e_mus_1d(mu=mu+dx, dmu=dmu, mus_eff=mus_eff, q=q, dq=dq, update_g=False)
    n_a_2, n_b_2, e2, p2, mus2 = ff.get_ns_p_e_mus_1d(mu=mu-dx, dmu=dmu, mus_eff=mus_eff, q=q, dq=dq, update_g=False)
    n_p_ = (p1 - p2)/2/dx
    print(f"Expected n_p={n_a + n_b}\tNumerical n_p={n_p_}")
    assert np.allclose(n_a + n_b, n_p_, rtol=1e-2)

    if False: # skip for speed
        # Fixed mu_b by changing mu and dmu with same value , as mu_b = mu - dmu
        # Then dP / dx = n_a
        n_a_1, n_b_1, e1, p1, mus1 = ff.get_ns_p_e_mus_1d(mu=mu+dx/2, dmu=dmu+dx/2, mus_eff=mus_eff, q=q, dq=dq, update_g=False)
        n_a_2, n_b_2, e2, p2, mus2 = ff.get_ns_p_e_mus_1d(mu=mu-dx/2, dmu=dmu-dx/2, mus_eff=mus_eff,  q=q, dq=dq, update_g=False)
        n_a_ = (p1 - p2)/2/dx
        print(f"Expected n_a={n_a}\tNumerical n_a={n_a_}")
        assert np.allclose(n_a, n_a_)
        # Fixed mu_a by changing mu and dmu with opposite values , as mu_a = mu + dmu
        # Then dP / dx = n_b
        n_a_3, n_b_3, e3, p3, mus3 = ff.get_ns_p_e_mus_1d(mu=mu+dx/2, dmu=dmu-dx/2, mus_eff=mus_eff, q=q, dq=dq, update_g=False)
        n_a_4, n_b_4, e4, p4, mus4 = ff.get_ns_p_e_mus_1d(mu=mu-dx/2, dmu=dmu+dx/2, mus_eff=mus_eff, q=q, dq=dq, update_g=False)
        n_b_ = (p3 - p4)/2/dx

        print(f"Expected n_b={n_b}\tNumerical n_b={n_b_}")
        assert np.allclose(n_b, n_b_)



test_Thermodynamic_1d(delta = 1.0, mu_delta = 3, dmu_delta = 0.5, q_dmu = 0, dq_dmu = 0, dx = 0.001)


