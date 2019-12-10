"""Test the bcs code."""
import pytest
import numpy as np
import numpy
from mmf_hfb import bcs, homogeneous
import itertools


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


def get_1d_currents(self, mus_eff, delta, N_twist=1):
    """return current for 1d only"""
    twistss = itertools.product(
        *(np.arange(0, N_twist)*2.0*np.pi/N_twist,)*self.dim)
    j_a = 0
    j_b = 0
    
    def df(k, f):
        return np.fft.ifft(1j*k*np.fft.fft(f))

    for twists in twistss:
        ks_bloch = np.divide(twists, self.Lxyz)
        k = [_k + _kb for _k, _kb in zip(self.kxyz, ks_bloch)][0]

        H = self.get_H(mus_eff=mus_eff, delta=delta, twists=twists)
        N = self.Nxyz[0]
        d, psi = np.linalg.eigh(H)
        us, vs = psi.reshape(2, N, N * 2)
        us, vs = us.T, vs.T
        j_a_ = -0.5j*sum(
            (us[i].conj()*df(k, us[i])
                -us[i]*df(k, us[i]).conj())*self.f(d[i]) for i in range(len(us)))
        j_b_ = -0.5j*sum(
            (vs[i]*df(k, vs[i]).conj()
                -vs[i].conj()*df(k, vs[i]))*self.f(-d[i]) for i in range(len(vs)))
        j_a = j_a + j_a_
        j_b = j_b + j_b_
    return (j_a/N_twist/np.prod(self.dxyz), j_b/N_twist/np.prod(self.dxyz))


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
    j_a, j_b = get_1d_currents(b, mus_eff=(mu, mu), delta=delta, N_twist=N_twist)

    assert np.allclose(res.j_a[0], j_a)
    assert np.allclose(res.j_b[0], j_b)  


if __name__ == "__main__":
    test_BCS_get_currents_1d(dim=1, T=0)

    # test_BCS(dim=2, NLx=(4, 10.0, None), T=0, N_twist=2)