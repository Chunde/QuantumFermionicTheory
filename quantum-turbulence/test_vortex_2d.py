# this file will eventually be changed to a test file for pytest, now just for debugging in Visual Studio
# Chunde's comment will start with a single sharp '#', to track modifications.
import numpy as np
from scipy.integrate import quad
from uncertainties import ufloat
from importlib import reload  # Python 3.4+
import numpy as np
import vortex_2d;reload(vortex_2d)



hbar = 1
m = 1

def n_integrand(k, mu, delta):
    ep = (hbar*k)**2/2/m - mu
    E = np.sqrt(ep**2 + abs(delta)**2) # what's the different between delta**2 ?
    return (1 - ep/E)

def nu_integrand(k, mu, delta):
    ep = (hbar*k)**2/2/m - mu
    E = np.sqrt(ep**2 + abs(delta)**2)
    return delta/2/E

def tau_integrand(k, mu, delta):
    ep = (hbar*k)**2/2/m - mu
    E = np.sqrt(ep**2 + abs(delta)**2)
    return k**2*(1 - ep/E)

def E_integrand(k, mu, delta):
    tau = tau_integrand(k, mu=mu, delta=delta)
    nu = nu_integrand(k, mu=mu, delta=delta)
    n = n_integrand(k, mu=mu, delta=delta)
    return hbar**2*tau/2/m - delta*nu - mu*n

def C_integrand(k, mu, delta):
    """Integrand without chemical potential shift.
    
    This works better for numerical integration with infinite limits.
    """
    ep = (hbar*k)**2/2/m - mu
    return (0.5/(ep+1e-7j) - nu_integrand(k, mu=mu, delta=delta)/delta).real

def C0_integrand(k, mu, delta):
    """Integrand without chemical potential shift.
    
    This works better for numerical integration with infinite limits.
    """
    k0 = np.sqrt(2*m*mu)/hbar # set alpha_+ = 1 at present
    ep0 = (hbar*k)**2/2/m
    return (0.5/ep0 - nu_integrand(k, mu=mu, delta=delta)/delta).real

def integrate(integrand, mu, delta, k0=0, k_c=np.inf, d=3, **kw):
    if d == 2:
        f = lambda k: integrand(k, mu=mu, delta=delta)*k/2/np.pi
    elif d == 3:
        f = lambda k: integrand(k, mu=mu, delta=delta)*k**2/2/np.pi**2
    return quad(f, k0, k_c, **kw)

def Lambda(k_c, mu, d=3):
    k_0 = np.sqrt(2*m*mu)/hbar
    if d == 3:
        return m/hbar**2 * k_c/2/np.pi**2*(
            1 - k_0/2/k_c*np.log((k_c+k_0)/(k_c-k_0)))
    elif d == 2:
        return m/hbar**2/4/np.pi * np.log((k_c/k_0)**2-1)

"""Test the homogeneous code."""
def test_BdG_homogeneous():
    e_F = 1.0
    k_F = np.sqrt(2*m*e_F)
    n_F = k_F**3/3/np.pi**2  # not k_F**3/6/np.pi**2, this is the total density of two spieces(spin up and down)?
    E_FG = 2./3*n_F*e_F
    mu = 0.59060550703283853378393810185221521748413488992993*e_F
    delta = 1.162200561790012570995259741628790656202543181557689*mu
    # delta = 0.68640205206984016444108204356564421137062514068346*e_F # equivalent to the upper line

    kw = dict(mu=mu, delta=delta)
    n_p = ufloat(*integrate(n_integrand, **kw))
    assert np.allclose(n_p.n, k_F**3/3/np.pi**2)

    C0 = ufloat(*integrate(C0_integrand, **kw))
    assert np.allclose(C0.n, 0)

    ## 2D values
    kw.update(d=2)
    n2_p = ufloat(*integrate(n_integrand, **kw))
    C2 = ufloat(*integrate(C_integrand, **kw))
    n2_p, C2

    print(n_p, C0, n2_p, C2)
    # visulization of 4d data structure for 2*2*2*2, resharp from 4 *4 unitary matrix
    # 10 01
    # 00 00
    # 00 00
    # 10 01
    # in the code, only dft is applied only to the fisrt and second dimensions i.e. applied to the four 2d lattices(which are  2d-matrix elements)
    # |1|0|, |0|1|,|0|0|, |0|0|
    # |0|0|, |0|0|,|1|0|, |0|1|

def test_BdG_lattice_2d():
    e_F = 1.0
    k_F = np.sqrt(2*m*e_F)
    n_F = k_F**3/3/np.pi**2
    E_FG = 2./3*n_F*e_F
    mu = 0.59060550703283853378393810185221521748413488992993*e_F
    delta = 0.68640205206984016444108204356564421137062514068346*e_F

    Nx, Ny = 64, 64
    H = np.eye(Nx*Ny).reshape(Nx, Ny, Nx, Ny) # apply 2d dft to the first and second dimensions only
    U = np.fft.fftn(H, axes=[0,1]).reshape(Nx*Ny, Nx*Ny)
    psi = np.random.random((Nx, Ny)) # the wave function is 2d
    np.allclose(np.fft.fftn(psi).ravel(), U.dot(psi.ravel())) # the relation means : dft(H) . psi = dft(psi) ???

    s = vortex_2d.BCS(Nxy=(2,)*2)
    k_c = abs(s.kxy[0]).max()
    E_c = (s.hbar*k_c)**2/2/s.m
    s = vortex_2d.BCS(Nxy=(2,)*2, E_c=E_c)
    kw = dict(mus=(mu, mu), delta=delta)
    #R = s.get_R(**kw)
    H = s.get_H(**kw)
    assert np.allclose(H, H.T.conj())

if __name__ == '__main__':
    test_BdG_lattice_2d()