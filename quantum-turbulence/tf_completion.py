"""Numba functions for fast integration of homogeneous matter."""

import math

import numpy as np
import numba

from scipy.integrate import quad
import scipy as sp

from uncertainties import ufloat


@numba.jit(nopython=True)
def step(t, t1):
    r"""Smooth step function that goes from 0 at time ``t=0`` to 1 at time
    ``t=t1``.  This step function is $C_\infty$:
    """
    alpha = 3.0
    if t < 0.0:
        return 0.0
    elif t < t1:
        return (1 + math.tanh(alpha*math.tan(math.pi*(2*t/t1-1)/2)))/2
    else:
        return 1.0


@numba.jit(nopython=True)
def f(E, T):
    """Fermi distribution function."""
    if T == 0:
        return (1 - np.sign(E))/2
    else:
        return 1./(1+np.exp(E/T))


def dquad(f, kF=None, k_0=0, k_inf=np.inf, limit=1000):
    res, err = sp.integrate.dblquad(f, k_0, k_inf,lambda x: k_0, lambda x: k_inf)
    if kF is None:
        res, err = sp.integrate.dblquad(f, k_0, k_inf,lambda x: k_0, lambda x: k_inf)
    else:
        res0, err0 = sp.integrate.dblquad(f, k_0, kF,lambda x: k_0, lambda x: k_inf)
        res1, err1 = sp.integrate.dblquad(f, kF, np.inf,lambda x: k_0, lambda x: k_inf)
        res = res0 + res1
        err = max(err0, err1)
    return (res,err)

######################################################################
# These *_integrand functions do not have the integration measure
# factors, so they can be used for any dimension (but need an
# appropriate wrapper).
    
@numba.jit(nopython=True)
def n_p_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    E = np.sqrt(e_p**2 + abs(delta)**2)
    w_m, w_p = e_m - E, e_m + E
    f_nu = (f(w_m, T) - f(w_p, T))
    f_p = 1 - e_p/E*f_nu
    return f_p


@numba.jit(nopython=True)
def n_m_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    E = np.sqrt(e_p**2 + abs(delta)**2)
    w_m, w_p = e_m - E, e_m + E
    f_m = f(w_p, T) - f(-w_m, T)
    return f_m


@numba.jit(nopython=True)
def tau_p_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    E = np.sqrt(e_p**2 + abs(delta)**2)
    w_m, w_p = e_m - E, e_m + E
    f_nu = (f(w_m, T) - f(w_p, T))
    f_p = 1 - e_p/E*f_nu
    #k2 = kp^22 + (kz+q)^2
    return ka2*f_p


@numba.jit(nopython=True)
def tau_m_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    E = np.sqrt(e_p**2 + abs(delta)**2)
    w_m, w_p = e_m - E, e_m + E
    f_m = f(w_p, T) - f(-w_m, T)
    #k2 = kp^22 + (kz-q)^2
    #return k2*f_m
    return kb2*f_m


@numba.jit(nopython=True)
def nu_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    E = np.sqrt(e_p**2 + abs(delta)**2)
    w_m, w_p = e_m - E, e_m + E
    f_nu = (f(w_m, T) - f(w_p, T))
    return delta/2/E*f_nu


@numba.jit(nopython=True)
def kappa_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    E = np.sqrt(e_p**2 + abs(delta)**2)
    w_m, w_p = e_m - E, e_m + E
    f_nu = (f(w_m, T) - f(w_p, T))
    f_p = 1 - e_p/E*f_nu
    m_p_inv = (1/m_a + 1/m_b)/2
    return e*m_p_inv*f_p + abs(delta)**2/2/E*f_nu


@numba.jit(nopython=True)
def C_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    E = np.sqrt(e_p**2 + abs(delta)**2)
    w_m, w_p = e_m - E, e_m + E
    f_nu = (f(w_m, T) - f(w_p, T))
    return f_nu/2*(1/e_p - 1/E)

@numba.jit(nopython=True)
def dnu_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    E = np.sqrt(e_p**2 + abs(delta)**2)
    w_m, w_p = e_m - E, e_m + E
    f_nu = (f(w_m, T) - f(w_p, T))
    return 1/2/E*f_nu

@numba.jit(nopython=True)
def dkappa_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    return tau_p_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T) * hbar**2 /2 - delta * nu_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T)



def integrate(f, mu_a, mu_b, delta, m_a, m_b, d=3, q=0.0, hbar=1.0, T=0.0):
    args = (mu_a, mu_b, delta, m_a, m_b, hbar, T)
    if d == 1:
        def integrand(k):
            k2_a = (k+q)**2
            k2_b = (k-q)**2
            return f(k2_a, k2_b, *args)/np.pi
    elif d == 2:
        def integrand(kz,kp):
            k2_a = (kz+q)**2 + kp**2
            k2_b = (kz-q)**2 + kp**2
            return f(k2_a, k2_b, *args)/np.pi**2
    elif d == 3:
        def integrand(kz,kp):
            k2_a = (kz+q)**2 + kp**2
            k2_b = (kz-q)**2 + kp**2
            return f(k2_a, k2_b, *args)*kp/2/np.pi**2
    else:
        raise ValueError(f"Only d=1, 2, or 3 supported (got d={d})")
    

    mu = (mu_a + mu_b)/2
    minv = (1/m_a + 1/m_b)/2
    kF = math.sqrt(2*mu/minv)/hbar
    points = [kF]
    if d == 1:
        integrand = numba.cfunc(numba.float64(numba.float64))(integrand)
        integrand = sp.LowLevelCallable(integrand.ctypes)

        return (ufloat(*quad(integrand, 0, max(points), points=points))
            +ufloat(*quad(integrand, max(points), np.inf)))
    #integrand = numba.cfunc(numba.float64(numba.float64,numba.float64))(integrand)
    #integrand = sp.LowLevelCallable(integrand.ctypes)
    return (ufloat(*dquad(f=integrand,kF=kF,k_0=0, k_inf=np.inf)))
           
    
