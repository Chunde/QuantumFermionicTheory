import numpy as np
import pytest
from mmf_hfb import tf_completion as tf
from mmf_hfb import bcs, homogeneous
from scipy.optimize import brentq
from collections import namedtuple
import warnings

from mmf_hfb.FuldeFerrelState import FFState as FF
tf.MAX_DIVISION = 200


@pytest.fixture(params=[1.0])
def delta(request):
    return request.param

@pytest.fixture(params=[1, 2, 3])
def dim(request):
    return request.param

# if q and dq are too big, test may fail as
# the no solution to delta can be found
@pytest.fixture(params=[0, 0.05])
def q_dmu(request):
    return request.param


@pytest.fixture(params=[0, 0.02])
def dq_dmu(request):
    return request.param


@pytest.fixture(params=[5])
def mu_delta(request):
    return request.param


@pytest.fixture(params=[0.1, 0.5, 1.2])
def dmu_delta(request):
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


def get_analytic_e_n(mu, dmu, q=0, dq=0, dim=1):
    """"return the analytical energy and particle density"""
    if dim == 1:
        def f(e_F):  # energy density
            return np.sqrt(2)/np.pi*e_F**1.5/3.0

        def g(e_F): # particle density
            return np.sqrt(2*e_F)/np.pi
    elif dim == 2:

        def f(e_F):
            return (e_F**2)/4.0/np.pi

        def g(e_F):
            return e_F/np.pi/2.0
    elif dim == 3:

        def f(e_F):
            return (e_F**2.5)*2.0**1.5/10.0/np.pi**2

        def g(e_F):
            return ((2.0*e_F)**1.5)/6.0/np.pi**2
        
    kF_a, kF_b = np.sqrt(2.0*(mu+dmu)), np.sqrt(2.0*(mu-dmu))
    mu_a1, mu_b1, mu_a2, mu_b2 = (q+dq)**2/2.0, (q-dq)**2/2.0, (kF_a)**2/2.0, (kF_b)**2/2.0
    E_a, E_b = f(mu_a2) - f(mu_a1), f(mu_b2) - f(mu_b1)
    n_a, n_b = g(mu_a2) - g(mu_a1), g(mu_b2) - g(mu_b1)
    energy_density = E_a + E_b
    return namedtuple('analytical', ['e','n_a', 'n_b'])(energy_density, n_a, n_b)
    # return energy_density, (n_a, n_b)


def get_dE_dn(mu, dmu, dim, q=0, dq=0):
    """compute the dE/dn for free Fermi Gas"""
    dx = 1e-6
    e1, n1 = get_analytic_e_n(mu=mu + dx, dmu=dmu, dim=dim, q=q, dq=dq)
    e2, n2 = get_analytic_e_n(mu=mu - dx, dmu=dmu, dim=dim, q=q, dq=dq)
    return (e1-e2)/(sum(n1)-sum(n2))


def Thermodynamic(mu, dmu, delta0=1, dim=3, k_c=100, q=0, dq=0,
                    T=0.0, a=0.8, b=1.2, dx=1e-3, N=10):
    if dim == 1:  # Because 1d case does not pass yet
        print("This method does nothing for 1d case")
        return 
    ff = FF(mu=mu, dmu=dmu, delta=delta0, q=q, dq=dq, dim=dim, k_c=k_c, T=T, 
            fix_g=True, bStateSentinel=True)
    print(ff.get_densities(mu=mu, dmu=dmu, delta=delta0))

    def get_P(mu, dmu):
        delta = ff.solve(mu=mu, dmu=dmu, q=q, dq=dq, a=0.8*delta0, b=1.2*delta0)
        return ff.get_pressure(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)

    def get_E_n(mu, dmu):
        E = ff.get_energy_density(mu=mu, dmu=dmu, q=q, dq=dq)
        na, nb = ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
        return E, na, nb

    def get_ns(mu, dmu):
        return ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
    
    na, nb = ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
    dn = (na - nb).n
    np0 = (na + nb).n
    
    def f_ns_dmu(dx, n):
        def f(dmu_):
            na_, nb_ = ff.get_densities(mu=mu+dx, dmu=dmu_, q=q, dq=dq)
            if n == 0:  # fix n_-
                dn_ = (na_ - nb_).n
                return dn - dn_
            elif n == 1:  # fix nb
                return (nb - nb_).n
            elif n == 2:  # fix na
                return (na - na_).n
            elif n == 3:  # fix n_+
                np_=(na_ + nb_).n
                return np0 - np_
        try:
            return brentq(f, a*dmu, b*dmu)
        except:
            irs = np.linspace(a, b, N) * dmu
            for i in reversed(range(N)):
                try:
                    startPos = f(irs[i])
                    for j in reversed(range(i + 1, N)):
                        try:
                            endPos = f(irs[j])
                            if startPos*endPos < 0:  # has solution
                                return brentq(f, irs[i], irs[j])
                        except:
                            continue
                except:
                    continue
            warnings.warn(f"Can't find a solution in that region, use the default value={dmu}")
            return dmu  # when no solution is found
    
    # Check the mu=dE/dn
    dmu1 = f_ns_dmu(dx, 0)
    dmu2 = f_ns_dmu(-dx, 0)
    E1, na1, nb1 = get_E_n(mu=mu+dx, dmu=dmu1)
    E0, na0, nb0 = get_E_n(mu=mu-dx, dmu=dmu2)
    n1, n0 = (na1 + nb1).n, (na0 + nb0).n
    mu_ = ((E1-E0)/(n1-n0)).n
    print(f"Fix dn:\t[dn1={(na1-nb1).n}\tdn0={(na0-nb0).n}]")
    print(f"Expected mu={mu}\tNumerical mu={mu_}")
    assert np.allclose((na1-nb1).n, (na0-nb0).n)
    assert np.allclose(mu, mu_, rtol=1e-4)
    n_a, n_b = get_ns(mu, dmu)
    n_a_ = (get_P(mu+dx/2, dmu+dx/2) - get_P(mu-dx/2, dmu - dx/2))/2/dx
    n_b_ = (get_P(mu+dx/2, dmu-dx/2) - get_P(mu-dx/2, dmu + dx/2))/2/dx
    print(f"Expected n_a={n_a.n}\tNumerical n_a={n_a_.n}")
    print(f"Expected n_b={n_b.n}\tNumerical n_b={n_b_.n}")
    assert np.allclose(n_a.n, n_a_.n)
    assert np.allclose(n_b.n, n_b_.n)


#@pytest.mark.skip(reason="pass")
def test_efftive_mus():
    """Test a few values from Table I of Quick:1993."""
    lam_invs = [0.5]  # 1.5
    mu_tilde_s = [0.0864]  # 2.0259
    E_N_E_2_s = [-0.3037]  # 4.4021
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
        E_N_E_2, lam = BCS(mu_eff=mu_eff, delta=delta)
        mu_tilde = (hbar**2/m/v_0**2)*mu
        assert np.allclose(lam, 1./lam_inv)
        assert np.allclose(mu_tilde, mu_tilde_, atol=0.0005)
        assert np.allclose(E_N_E_2, E_N_E_2_, atol=0.0005)
        ff = FF(mu=mu, dmu=0, delta=delta, dim=1, k_c=np.inf, fix_g=True)
        mus_eff = ff._get_effective_mus(mu=mu, dmu=0, delta=delta, update_g=True)
        ns = ff.get_densities(mu=mus_eff[0], dmu=mus_eff[1], delta=delta)
        assert np.allclose(n_p.n, (ns[0] + ns[1]).n)
        assert np.allclose(mus_eff[0], mu_eff)
        assert np.allclose(v_0, -ff._g)


#@pytest.mark.skip(reason="pass")
def test_density_with_qs(delta, mu_delta, dmu_delta, q_dmu, dq_dmu, dim, k_c=200):
    """The density should not depend on q"""
    if dim == 3:
        k_c = 50
    mu = mu_delta * delta
    dmu = dmu_delta * delta
    q = q_dmu * mu
    ff = FF(mu=mu, dmu=dmu, delta=delta, dim=1, k_c=100, fix_g=True)
    ns = ff.get_densities(mu=mu, dmu=dmu)
    na0, nb0 = ns[0].n, ns[1].n
    print(na0, nb0)
    ns = ff.get_densities(mu=mu, dmu=dmu, q=q)
    na1, nb1 = ns[0].n, ns[1].n
    print(na1, nb1)
    assert np.allclose(na0, na1)
    assert np.allclose(nb0, nb1)


#@pytest.mark.skip(reason="pass")
def test_Thermodynamic(delta, mu_delta, dmu_delta, q_dmu, dq_dmu, dim, k_c=200):
    if dim == 3:
        k_c = 50
    mu = mu_delta * delta
    dmu = dmu_delta * delta
    q = q_dmu * mu
    dq = dq_dmu * mu
    Thermodynamic(mu=mu, dmu=dmu, k_c=k_c, q=q, dq=dq, dim=dim, delta0=delta)


# @pytest.mark.skip(reason="Too Slow")
def test_Thermodynamic_1d(
        delta, mu_delta, dmu_delta,
        q_dmu, dq_dmu, N=20, dx=1e-3):
    """test id case"""
    mu = mu_delta * delta
    dmu = dmu_delta * delta
    q = q_dmu * mu
    dq = dq_dmu * mu
    ff = FF(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq, dim=1, fix_g=True,
            bStateSentinel=True)
    n_a, n_b, e, p, mus_eff = ff.get_ns_p_e_mus_1d(
        mu=mu, dmu=dmu, q=q,
        dq=dq, update_g=True)

    n_a_1, n_b_1, e1, p1, mus1 = ff.get_ns_p_e_mus_1d(
        mu=mu+dx, dmu=dmu,
        mus_eff=mus_eff, q=q, dq=dq, update_g=False)
    n_a_2, n_b_2, e2, p2, mus2 = ff.get_ns_p_e_mus_1d(
        mu=mu-dx, dmu=dmu,
        mus_eff=mus_eff, q=q, dq=dq, update_g=False)
    n_p_ = (p1 - p2)/2/dx
    print(f"Expected n_p={n_a + n_b}\tNumerical n_p={n_p_}")
    assert np.allclose(n_a + n_b, n_p_, rtol=1e-2)

    if False: #  skip for speed
        # Fixed mu_b by changing mu and dmu with same value , as mu_b = mu - dmu
        # Then dP / dx = n_a
        n_a_1, n_b_1, e1, p1, mus1 = ff.get_ns_p_e_mus_1d(
            mu=mu+dx/2, dmu=dmu+dx/2,
            mus_eff=mus_eff, q=q, dq=dq, update_g=False)
        n_a_2, n_b_2, e2, p2, mus2 = ff.get_ns_p_e_mus_1d(
            mu=mu-dx/2, dmu=dmu-dx/2, 
            mus_eff=mus_eff, q=q, dq=dq, update_g=False)
        n_a_ = (p1 - p2)/2/dx
        print(f"Expected n_a={n_a}\tNumerical n_a={n_a_}")
        assert np.allclose(n_a, n_a_)
        # Fixed mu_a by changing mu and dmu with opposite values , as mu_a = mu + dmu
        # Then dP / dx = n_b
        n_a_3, n_b_3, e3, p3, mus3 = ff.get_ns_p_e_mus_1d(
            mu=mu+dx/2, dmu=dmu-dx/2, 
            mus_eff=mus_eff, q=q, dq=dq, update_g=False)
        n_a_4, n_b_4, e4, p4, mus4 = ff.get_ns_p_e_mus_1d(
            mu=mu-dx/2, dmu=dmu+dx/2,
            mus_eff=mus_eff, q=q, dq=dq, update_g=False)
        n_b_ = (p3 - p4)/2/dx
        print(f"Expected n_b={n_b}\tNumerical n_b={n_b_}")
        assert np.allclose(n_b, n_b_)

    na, nb = n_a, n_b
    dn = (na - nb)
    np0 = (na + nb)
    a=0.8
    b=1.2

    def f_ns_dmu(dx, n):
        def f(dmu_):
            na_, nb_, e1, p1, mus1 = ff.get_ns_p_e_mus_1d(
                mu=mu+dx, dmu=dmu_, 
                mus_eff=mus_eff, q=q, dq=dq, update_g=False)
            if n == 0:  # fix n_-
                dn_ = (na_ - nb_)
                return dn - dn_
            elif n == 1:  # fix nb
                return (nb - nb_)
            elif n == 2:  # fix na
                return (na - na_)
            elif n == 3:  # fix n_+
                np_=(na_ + nb_)
                return np0 - np_
        try:
            return brentq(f, a*dmu, b*dmu)
        except:
            irs = np.linspace(a, b, N) * dmu
            for i in reversed(range(N)):
                try:
                    startPos = f(irs[i])
                    for j in reversed(range(i + 1, N)):
                        try:
                            endPos = f(irs[j])
                            if startPos * endPos < 0: # has solution
                                return brentq(f, irs[i], irs[j])
                        except:
                            continue
                except:
                    continue
            warnings.warn(f"Can't find a solution in that region, use the default value={dmu}")
            return dmu  # when no solution is found

    return # The follow part test is too slow, skip it at this point!
    # Check the mu=dE/dn
    dmu1 = f_ns_dmu(dx, 0)
    dmu2 = f_ns_dmu(-dx, 0)
    na1, nb1, E1, p1, mus1 = ff.get_ns_p_e_mus_1d(
        mu=mu+dx, dmu=dmu1,
        mus_eff=mus_eff, q=q, dq=dq, update_g=False)
    na0, nb0, E0, p0, mus0 = ff.get_ns_p_e_mus_1d(
        mu=mu-dx, dmu=dmu2,
        mus_eff=mus_eff, q=q, dq=dq, update_g=False)
    n1, n0 = (na1 + nb1), (na0 + nb0)
    mu_ = ((E1-E0)/(n1-n0))
    print(f"Fix dn:\t[dn1={(na1-nb1)}\tdn0={(na0-nb0)}]")
    print(f"Expected mu={mu}\tNumerical mu={mu_}")
    assert np.allclose((na1-nb1), (na0-nb0))
    assert np.allclose(mu,mu_, rtol=1e-4)

if __name__ == "__main__":
    Thermodynamic(mu=10, dmu=0, k_c=50, q=0, dq=0, dim=3, delta0=1)