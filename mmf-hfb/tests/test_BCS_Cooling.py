from mmf_hfb.BCSCooling import BCSCooling
import numpy as np
import pytest


def Normalize(psi):
    return psi/psi.dot(psi.conj())**0.5


def Prob(psi):
    return np.abs(psi)**2


@pytest.fixture(params=[1, 2, 3])
def n(request):
    return request.param


@pytest.fixture(params=[0, 2, 3])
def da(request):
    return request.param


@pytest.fixture(params=[0, 2, 3])
def db(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def N_state(request):
    return request.param


def test_derivative_cooling(n, da, db):
    """
    Test that for free particle hamiltonian, when da=da, Vc is zero
    """
    args = dict(N=128, dx=0.1, beta_0=1, V=0, beta_K=0, beta_V=0, beta_D=1)
    b = BCSCooling(**args)
    k0 = 2*np.pi/b.L
    x = b.xyz[0]
    H0 = b._get_H(mu_eff=0, V=0)
    U0, E0 = b.get_U_E(H0, transpose=True)
    psi_1 = Normalize(np.cos(k0*x))
    assert np.allclose(Prob(psi_1), Prob(U0[1]))
    assert np.allclose(E0[1], k0**2/2.0)
    psi = np.exp(1j*n*(k0*x))
    E =n**2*k0**2/2
    # compute d^n \psi / d^n x
    psi_a = b.Del(psi, n=da)
    # d[d^n \psi / d^n x] / dt
    Hpsi = np.array(b.apply_H([psi]))[0]/(1j)
    Hpsi_a = b.Del(Hpsi, n=da)

    if da == db:
        psi_b = psi_a
        Hpsi_b = Hpsi_a
    else:
        psi_b = b.Del(psi, n=db)
        Hpsi_b = b.Del(Hpsi, n=db)
    Vc = psi_a*Hpsi_b.conj() + Hpsi_a*psi_b.conj()

    assert np.allclose(Vc, 0, atol=1e-7)
    assert np.allclose(
        psi_a*Hpsi_b.conj(),
        1j*(n*k0)**(da+db)*E*psi*psi.conj()*((-1j)**db)*(1j)**da)
    assert np.allclose(
        Hpsi_a*psi_b.conj(),
        (-1j)*(n*k0)**(da+db)*E*psi*psi.conj()*((-1j)**db)*(1j)**da)


def test_apply_Vs(N_state=2):
    """
    check the equivalence of apply_H, apply_V, and apply_K for computing Vc and Kc
    """
    args = dict(N=128, dx=0.1, beta_0=1, divs=(1, 1), beta_K=0, beta_V=0, beta_D=1)
    b = BCSCooling(**args)
    x = b.xyz[0]
    V = x**2/2
    H0 = b._get_H(mu_eff=0, V=0)
    U0, E0 = b.get_U_E(H0, transpose=True)
    psis = U0[:N_state]

    def get_kc(psis, fun):
        N = b.get_N(psis)
        Kc = 0
        # can also apply_H, result is unchanged
        Hpsis = fun(psis)  # [check]apply_V or apply_H
        for i, psi in enumerate(psis):
            psi_k = np.fft.fft(psi)*b.dV
            Vpsi_k = np.fft.fft(Hpsis[i])*b.dV
            Kc = Kc + 2*(psi_k.conj()*Vpsi_k).imag/N*b.dV/np.prod(b.Lxyz)
        return Kc

    def get_Vc(psis, fun):
        N = b.get_N(psis)
        Vc = 0
        Hpsis = fun(psis)
        for i, psi in enumerate(psis):
            Vc = Vc + 2*(psi.conj()*Hpsis[i]).imag/N
        return Vc

    kc1 = get_kc(psis, b.apply_V)
    kc2 = get_kc(psis, b.apply_H)
    assert np.allclose(kc1, kc2)
    vc1 = get_Vc(psis, b.apply_K)
    vc2 = get_Vc(psis, b.apply_H)
    assert np.allclose(vc1, vc2)


if __name__ == "__main__":
    test_apply_Vs()