import numpy as np
from mmf_hfb.DVRBasis import HarmonicDVR


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
            P=P - rs/2
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
    assert np.allclose(psi_rec, psi_ana)

    # test basis for odd angular momentum
    h = HarmonicDVR(nu=1, dim=2, w=1)
    H = h.get_H()
    _, us = np.linalg.eigh(H)
    wf =sum([u*h.get_F(n=i, rs=rs) for (i, u) in enumerate(us.T[0])])
    psi_rec = -Normalize(wf/rs**0.5)
    psi_ana = Normalize(HO_psi(n=1, m=1, rs=rs))
    assert np.allclose(psi_rec, psi_ana)
