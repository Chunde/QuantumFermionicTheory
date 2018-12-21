"""Test the homogeneous code."""
from importlib import reload  # Python 3.4+
import numpy as np
from scipy.optimize import brentq
import homogeneous;reload(homogeneous)
from homogeneous import get_BCS_v_n_e, Homogeneous3D
import vortex_2d;reload(vortex_2d)

if __name__ == "__main__":
    Nx, Ny = 64, 64
    H = np.eye(Nx*Ny).reshape(Nx, Ny, Nx, Ny) # Hamiltanian is of 4d > 2d? I need to think about it H is of size 4096 * 4096, or 64*64*64*64

    U = np.fft.fftn(H, axes=[0,1]).reshape(Nx*Ny, Nx*Ny)
    psi = np.random.random((Nx, Ny)) # the wave function is of 2d
    np.allclose(np.fft.fftn(psi).ravel(), U.dot(psi.ravel()))

    s = vortex_2d.BCS(Nxy=(16,)*2)
    k_c = abs(s.kxy[0]).max()
    E_c = (s.hbar*k_c)**2/2/s.m
    s = vortex_2d.BCS(Nxy=(16,)*2, E_c=E_c)
    kw = dict(mus=(mu, mu), delta=delta)
    #R = s.get_R(**kw)
    H = s.get_H(**kw)
    assert np.allclose(H, H.T.conj())