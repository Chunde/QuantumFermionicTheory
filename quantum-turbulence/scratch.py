"""Test the homogeneous code."""
from importlib import reload  # Python 3.4+
import numpy as np
from scipy.optimize import brentq
import homogeneous;reload(homogeneous)
from homogeneous import get_BCS_v_n_e, Homogeneous3D

if __name__ == "__main__":
    h3 = Homogeneous3D(T=0.0)
    k0 = 1.0
    mu = k0**2/2
    eF = mu/0.5906055
    kF = np.sqrt(2*eF)
    n_p = kF**3/3/np.pi**2
    mus_eff = (mu,)*2
    delta = 1.16220056*mus_eff[0]
    k_c = 10.0
    #Lambda = h3.get_inverse_scattering_length(mus_eff=mus_eff, delta=delta, k_c=k_c)/4/np.pi
    #Lambda, -k_c/2/np.pi**2*k0/2/k_c*np.log((k_c+k0)/(k_c-k0))
    v0, ns, mus = h3.get_BCS_v_n_e(mus_eff=mus_eff, delta=delta, k_c=100)
    v0, ns, mus = h3.get_BCS_v_n_e_in_spherical(mus_eff=mus_eff, delta=delta, k_c=100)