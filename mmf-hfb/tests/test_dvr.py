import numpy as np
import pytest
from mmf_hfb.DVRBasis import HarmonicDVR
from mmf_hfb.VortexDVR import CylindricalDVR


@pytest.fixture(params=[2, 10, 32])
def N_root(request):
    return request.param


def Normalize(psi):
    """Normalize a wave function"""
    return psi/(psi.conj().dot(psi))**0.5


def HO_psi(n, m, rs):
    """
    n = E -1
        e.g if E=1, to select the corresponding
        wavefunction, use n=E-1=0, and m = 0
    m is used to pick the degerated wavefunction
    m <=n
    """
    assert n < 4 and n >=0
    assert m <=n
    P=1
    if n ==1:
        P = rs
    elif n == 2:
        P=rs**2
        if m == 1:
            P=P-1
    elif n == 3:
        P = rs**3
        if m == 1 or m==2:
            P = P - rs/2
    return P*np.exp(-rs**2/2)


def test_wave_function_reconstruction():
    # test  basis for even angular momentum
    h = HarmonicDVR(nu=0, dim=2, w=1)
    H = h.get_H()
    _, us = np.linalg.eigh(H)
    rs = np.linspace(0.01, 5, 200)
    wf =sum([u*h.get_F(n=i, rs=rs) for (i, u) in enumerate(us.T[1])])
    psi_rec = -Normalize(wf/rs**0.5)
    psi_ana = Normalize(HO_psi(n=2, m=1, rs=rs))
    assert np.allclose(psi_rec, psi_ana, atol=1e-6)

    # test basis for odd angular momentum
    h = HarmonicDVR(nu=1, dim=2, w=1)
    H = h.get_H()
    _, us = np.linalg.eigh(H)
    wf =sum([u*h.get_F(n=i, rs=rs) for (i, u) in enumerate(us.T[0])])
    psi_rec = -Normalize(wf/rs**0.5)
    psi_ana = Normalize(HO_psi(n=0, m=0, rs=rs))
    assert np.allclose(psi_rec, psi_ana, atol=1e-6)


def test_vortex_hamiltonian(N_root=32):
    # test the T in different bases
    # since the bases are different
    # The T with the same lz should not
    # be the same, but when N_root is small
    # they should be close.
    # ---------------------------------------
    # However, we can shift the basis so
    # in any case we still use the same basis
    # then the resulted Hamiltonian piece
    # should be exactly the same.
    mu, dmu, delta=5, 0, 1
    g=-1
    winding = 1
    d = CylindricalDVR(
        mu=mu, dmu=dmu, delta=delta, g=g, E_c=100,
        bases=None, wz=winding, verbosity=0,
        N_root=N_root, R_max=5, l_max=100)
    mus = (5, 5)
    delta = 2
    assert mus[0]==mus[1]
    H1 = d.get_H(mus=mus, delta=delta, lz=0)
    H2 = d.get_H(mus=mus, delta=delta, lz=1, lz_offset=1)
    H1_ = H1[0:N_root, 0:N_root]
    H2_ = -1*H2[N_root:, N_root:]
    assert np.allclose(H1_, H2_)
