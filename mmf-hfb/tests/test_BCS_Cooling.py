from mmf_hfb.BCSCooling import BCSCooling
import matplotlib.pyplot as plt
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
def d(request):
    return request.param

def test_derivative_cooling(n, d):
    """
    Test that for free particle hamiltonian, when da=da, Vc is zero
    """
    args = dict(N=128, dx=0.1, beta_0=1, divs=(1, 1), beta_K=0, beta_V=0, beta_D=1)
    b = BCSCooling(**args)
    k0 = 2*np.pi/b.L
    x = b.xyz[0]
    V = 0   # x**2/2
    H0 = b._get_H(mu_eff=0, V=0)
    U0, E0 = b.get_U_E(H0, transpose=True)
    psi_1 = Normalize(np.cos(k0*x))
    assert np.allclose(Prob(psi_1), Prob(U0[1]))
    assert np.allclose(E0[1], k0**2/2.0)
    psi = np.exp(1j*n*(k0*x))
    E =n**2*k0**2/2
    da=db=d
    # compute d^n \psi / d^n x
    psi_a = b.Del(psi, n=da)
    # d[d^n \psi / d^n x] / dt
    Hpsi = np.array(b.apply_H([psi], V=V))[0]
    Hpsi_a = b.Del(Hpsi, n=da)

    if da == db:
        psi_b = psi_a
        Hpsi_b = Hpsi_a
    else:
        psi_b = b.Del(psi, n=db)
        Hpsi_b = b.Del(Hpsi, n=db)
    Vc = psi_a*Hpsi_b.conj()/(1j)-Hpsi_a*psi_b.conj()/(1j)

    assert np.allclose(Vc, 0, atol=1e-7)
   

if __name__ == "__main__":
    test_derivative_cooling(n=3, d=3)
