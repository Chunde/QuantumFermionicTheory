"""Test the bcs code."""
import pytest
import numpy as np
import numpy
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
    np.random.seed(1)
    hbar, m, kF = 1 + np.random.random(3)
    eF = (hbar*kF)**2/2/m
    # nF = kF/np.pi
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

    assert np.allclose(res_h.n_a.n, res_b.n_a.mean())
    assert np.allclose(res_h.n_b.n, res_b.n_b.mean())
    assert np.allclose(res_h.nu.n, res_b.nu.mean())

    
def test_twist_average(dim, T):
    """Check twist averaging."""
    np.random.seed(1)
    hbar, m, kF = 1 + np.random.random(3)
    eF = (hbar*kF)**2/2/m
    # nF = kF/np.pi
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
        res_b = b.get_densities((mu, mu), delta, N_twist=np.inf)

        assert np.allclose(res_h.n_a, res_b.n_a.mean())
        assert np.allclose(res_h.n_b, res_b.n_b.mean())
        assert np.allclose(res_h.nu, res_b.nu.mean(), rtol=0.002)

    else:
        with pytest.raises(NotImplementedError):
            res_b = b.get_densities((mu, mu), delta, N_twist=np.inf)


def test_BCS_get_densities(dim, NLx, T, N_twist):
    """Compare the two get_densities methods."""
    np.random.seed(1)
    hbar, m, kF = 1 + numpy.random.random(3)
    eF = (hbar*kF)**2/2/m
    # nF = kF/np.pi
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

    delta = np.exp(1j*b.xyz[0])
    res = b.get_densities((mu, mu), delta, N_twist=N_twist)
    res_R = b.get_densities_R((mu, mu), delta, N_twist=N_twist)

    assert np.allclose(res.n_a, res_R.n_a)
    assert np.allclose(res.n_b, res_R.n_b)
    if np == numpy:  # complex128 type not works as expected
        assert numpy.allclose(res.nu, res_R.nu)
    else:
        assert numpy.allclose(res.nu.get(), res_R.nu.get())


def test_BCS_get_currents_1d(dim, NLx, T, N_twist):
    if dim != 1:
        return
    np.random.seed(1)
    hbar, m, kF = 1 + np.random.random(3)
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

    delta = np.exp(1j*b.xyz[0])
    res = b.get_densities((mu, mu), delta, N_twist=N_twist)
    j_a, j_b = b.get_1d_currents((mu, mu), delta, N_twist=N_twist)

    assert np.allclose(res.j_a[0], j_a)
    assert np.allclose(res.j_b[0], j_b)  


if __name__ == "__main__":
    test_twist_average(dim=1, T=0)

    # test_BCS(dim=2, NLx=(4, 10.0, None), T=0, N_twist=2)