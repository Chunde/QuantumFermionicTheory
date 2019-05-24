"""Test the bcs code."""
from mmf_hfb.xp import xp
from scipy.optimize import brentq
import pytest
import numpy
from mmfutils.testing import allclose
from mmf_hfb import bcs, homogeneous

@pytest.fixture(params=[1, 2, 3])
def dim(request):
    return request.param


@pytest.fixture(params=[0, 0.1])
def T(request):
    return request.param


@pytest.fixture(params=[1, 2])
def N_twist(request):
    return request.param


@pytest.fixture(params=[(4, 10.0, None), (4, None, 1.1), (None, 5.0, 1.1)])
def NLx(request):
    return request.param


def test_BCS(dim, NLx, T, N_twist):
    """Compare the BCS lattice class with the homogeneous results."""
    xp.random.seed(1)
    hbar, m, kF = 1 + xp.random.random(3)
    eF = (hbar*kF)**2/2/m
    # nF = kF/xp.pi
    # E_FG = 2*nF*eF/3
    # C_unit = m/hbar**2/kF

    mu = 0.28223521359748843*eF
    delta = 0.411726229961806*eF

    N, L, dx = NLx
    if dx is None:
        args = dict(Nxyz=(N,)*dim, Lxyz=(L,)*dim)
    elif L is None:
        args = dict(Nxyz=(N,)*dim, dx=dx)
    else:
        args = dict(Lxyz=(L,)*dim, dx=dx)

    args.update(T=T)

    h = homogeneous.Homogeneous(**args)
    b = bcs.BCS(**args)

    res_h = h.get_densities((mu, mu), delta, N_twist=N_twist)
    res_b = b.get_densities((mu, mu), delta, N_twist=N_twist)

    assert xp.allclose(res_h.n_a.n, res_b.n_a.mean())
    assert xp.allclose(res_h.n_b.n, res_b.n_b.mean())
    assert xp.allclose(res_h.nu.n, res_b.nu.mean())

    
def test_twist_average(dim, T):
    """Check twist averaging."""
    xp.random.seed(1)
    hbar, m, kF = 1 + xp.random.random(3)
    eF = (hbar*kF)**2/2/m
    # nF = kF/xp.pi
    # E_FG = 2*nF*eF/3
    # C_unit = m/hbar**2/kF

    mu = 0.28223521359748843*eF
    delta = 0.411726229961806*eF

    N = 8
    dx = 0.01
    args = dict(Nxyz=(N,)*dim, dx=dx, T=T)
    h = homogeneous.Homogeneous(dim=dim, T=T)
    b = bcs.BCS(**args)

    if dim == 1:
        res_h = h.get_densities((mu, mu), delta)
        res_b = b.get_densities((mu, mu), delta, N_twist=xp.inf)

        assert xp.allclose(res_h.n_a.n, res_b.n_a.mean())
        assert xp.allclose(res_h.n_b.n, res_b.n_b.mean())
        assert xp.allclose(res_h.nu.n, res_b.nu.mean(), rtol=0.002)

    else:
        with pytest.raises(NotImplementedError):
            res_b = b.get_densities((mu, mu), delta, N_twist=xp.inf)


def test_BCS_get_densities(dim, NLx, T, N_twist):
    """Compare the two get_densities methods."""
    xp.random.seed(1)
    hbar, m, kF = 1 + numpy.random.random(3)
    eF = (hbar*kF)**2/2/m
    # nF = kF/xp.pi
    # E_FG = 2*nF*eF/3
    # C_unit = m/hbar**2/kF

    mu = 0.28223521359748843*eF
    delta = 0.411726229961806*eF

    N, L, dx = NLx
    if dx is None:
        args = dict(Nxyz=(N,)*dim, Lxyz=(L,)*dim)
    elif L is None:
        args = dict(Nxyz=(N,)*dim, dx=dx)
    else:
        args = dict(Lxyz=(L,)*dim, dx=dx)

    args.update(T=T)

    b = bcs.BCS(**args)

    delta = xp.exp(1j*b.xyz[0])
    res = b.get_densities((mu, mu), delta, N_twist=N_twist)
    res_R = b.get_densities_R((mu, mu), delta, N_twist=N_twist)

    assert xp.allclose(res.n_a, res_R.n_a)
    assert xp.allclose(res.n_b, res_R.n_b)
    if xp == numpy: # complex128 type not works as expected
        assert numpy.allclose(res.nu, res_R.nu)
    else:
        assert numpy.allclose(res.nu.get(), res_R.nu.get())

def test_BCS_get_currents_1d(dim, NLx, T, N_twist):
    if dim != 1:
        return
    xp.random.seed(1)
    hbar, m, kF = 1 + xp.random.random(3)
    eF = (hbar*kF)**2/2/m

    mu = 0.28223521359748843*eF
    delta = 0.411726229961806*eF

    N, L, dx = NLx
    if dx is None:
        args = dict(Nxyz=(N,)*dim, Lxyz=(L,)*dim)
    elif L is None:
        args = dict(Nxyz=(N,)*dim, dx=dx)
    else:
        args = dict(Lxyz=(L,)*dim, dx=dx)

    args.update(T=T)

    b = bcs.BCS(**args)

    delta = xp.exp(1j*b.xyz[0])
    res = b.get_densities((mu, mu), delta, N_twist=N_twist)
    j_a, j_b = b.get_1d_currents((mu, mu), delta, N_twist=N_twist)

    assert xp.allclose(res.j_a[0], j_a)
    assert xp.allclose(res.j_b[0], j_b)  

def test_bcs_integral_2d():
    """
        test 2d lattice BCS with 1d lattice + 1d integral
        ------------------------
        As 1d case is very fast, so we can have higher twisting
        number to get good result.
    """
    L = 0.46
    N = 4
    N_twist = 32
    delta = 1.0
    mu_eff = 10.0
    b = bcs.BCS(T=0, Nxyz=(N,), Lxyz=(L,))


    print("Test 1d lattice with homogeneous system with high precision(1e-12)")
    ns, taus, js, kappa = b.get_densities(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist, struct=False)
    h = homogeneous.Homogeneous(Nxyz=(N,), Lxyz=(L), dim=1)
    ret = h.get_densities(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    print(f"{(ret.n_a + ret.n_b).n}, {sum(ns)[0]}")
    print(f"{ret.nu.n}, {kappa[0].real}")
    assert xp.allclose((ret.n_a + ret.n_b).n, sum(ns), rtol=1e-12)
    assert xp.allclose(ret.nu.n, kappa, rtol=1e-12)

    k_c = abs(xp.array(b.kxyz).max())
    b.E_c = (b.hbar*k_c)**2/2/b.m
    v_0, n, mu, e_0 = homogeneous.Homogeneous2D().get_BCS_v_n_e(delta=delta, mus_eff=(mu_eff, mu_eff), k_inf=k_c)

    h = homogeneous.Homogeneous(Nxyz=(N, N), Lxyz=(L, L), dim=2)
    ret = h.get_densities(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    print("Test 1d lattice plus 1d integral over perpendicular dimension")
    ns, taus, js, kappa = b.get_dens_integral(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist, k_c=k_c)
    na, nb = ns
    print(((ret.n_a + ret.n_b).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))
    assert xp.allclose((ret.n_a + ret.n_b).n, na.real + nb.real, rtol=0.001)
    assert xp.allclose(delta, -v_0.n*kappa[0].real, rtol=0.01)

@pytest.mark.skip(reason="too slow")
def test_bcs_integral_3d():
    """
        test 2d lattice BCS with 1d lattice + 1d integral
        ------------------------
        As 1d case is very fast, so we can have higher twisting
        number to get good result.
    """
    L = 0.46
    N = 4
    N_twist = 16
    delta = 1.0
    mu_eff = 10.0
    b = bcs.BCS(T=0, Nxyz=(N, N), Lxyz=(L,L))
    k_c = abs(xp.array(b.kxyz).max())
    b.E_c = (b.hbar*k_c)**2/2/b.m
    v_0, n, mu, e_0 = homogeneous.Homogeneous3D().get_BCS_v_n_e(delta=delta, mus_eff=(mu_eff, mu_eff), k_inf=k_c)
    h = homogeneous.Homogeneous(Nxyz=(N, N, N), Lxyz=(L, L, L), dim=3)

    ret = h.get_densities(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    print("Test 2d lattice plus 1d integral over perpendicular dimension")
    ns, taus, js, kappa = b.get_dens_integral(mus_eff=(mu_eff, mu_eff), delta=delta, N_twist=N_twist, k_c=k_c)
    na, nb = ns
    print(((ret.n_a + ret.n_b).n, na[0].real + nb[0].real), (delta, -v_0.n*kappa[0].real))
    assert xp.allclose((ret.n_a + ret.n_b).n, na.real + nb.real, rtol=0.01)
    assert xp.allclose(delta, -v_0.n*kappa[0].real, rtol=0.05)

