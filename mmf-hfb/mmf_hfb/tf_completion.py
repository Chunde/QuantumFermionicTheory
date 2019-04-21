"""Numba functions for fast integration of homogeneous matter."""
import math
import cmath

import numpy as np
import numba
import warnings
import scipy as sp

from .integrate import quad, dquad


MAX_DIVISION = 65


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
    f_m = f(w_p, T) - f(-w_m, T)
    f_a = (f_p + f_m)/2
    f_b = (f_p - f_m)/2
    return ka2*f_a + kb2*f_b


@numba.jit(nopython=True)
def tau_m_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    E = np.sqrt(e_p**2 + abs(delta)**2)
    w_m, w_p = e_m - E, e_m + E
    f_nu = (f(w_m, T) - f(w_p, T))
    f_p = 1 - e_p/E*f_nu
    f_m = f(w_p, T) - f(-w_m, T)
    f_a = (f_p + f_m)/2
    f_b = (f_p - f_m)/2
    return ka2*f_a - kb2*f_b


@numba.jit(nopython=True)
def nu_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    E = np.sqrt(e_p**2 + abs(delta)**2)
    w_m, w_p = e_m - E, e_m + E
    f_nu = (f(w_m, T) - f(w_p, T))
    return -0.5*delta/E*f_nu


@numba.jit(nopython=True)
def nu_delta_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    E = np.sqrt(e_p**2 + abs(delta)**2)
    w_m, w_p = e_m - E, e_m + E
    f_nu = (f(w_m, T) - f(w_p, T))
    return -0.5/E*f_nu


@numba.jit(nopython=True)
def kappa_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    m_p_inv = (1/m_a + 1/m_b)/2
    tau_p = tau_p_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T)
    nu_delta = nu_delta_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T)
    return (tau_p * m_p_inv * hbar**2 / 2 + abs(delta)**2 * nu_delta)


@numba.jit(nopython=True)
def pressure_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    mu_p, mu_m = (mu_a + mu_b)/2, (mu_a-mu_b)/2
    n_p = n_p_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T)
    n_m = n_m_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T)
    kappa = kappa_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T)
    return mu_p*n_p + mu_m*n_m - kappa


@numba.jit(nopython=True)
def C_integrand(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
    e = hbar**2/2
    e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
    e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
    E = np.sqrt(e_p**2 + abs(delta)**2)
    w_m, w_p = e_m - E, e_m + E
    f_nu = (f(w_m, T) - f(w_p, T))
    return 0.5*(1/e_p - f_nu/E)

@numba.jit(nopython=True)
def f_p_m(ka2, kb2, mu_a, mu_b, delta, m_a, m_b, hbar, T):
        e = hbar**2/2
        e_a, e_b = e*ka2/m_a - mu_a, e*kb2/m_b - mu_b
        e_m, e_p = (e_a - e_b)/2, (e_a + e_b)/2
        E = np.sqrt(e_p**2 + abs(delta)**2)
        w_m, w_p = e_m - E, e_m + E
        f_nu = (f(w_m, T) - f(w_p, T))
        f_p = 1 - e_p/E*f_nu
        f_m = f(w_p, T) - f(-w_m, T)
        return (f_p,f_m)

class Series(object):
    """Series expansions for T=0 and large enough k_c."""
            
    
def Lambda(m, mu, hbar, dim, E_c=None, k_c=None):
    """Return the cutoff function Lambda assuming equal masses.

    To include the effects of the Fulde Ferrell q, shift mu -> mu_q as
    appropriate.

    This function assumes that k_c >> q.

    Arguments
    ---------
    mu : float
       Average chemical potential.  Must include any shift from the
       Fulde-Ferrell momentum q.
    E_c : float
       Optional alternative cutoff:
       
       E_c = (hbar*k_c)**2/2/m - mu
    """
    k_0 = np.sqrt(2*m*mu/hbar**2 + 0.j)
    if k_c is None:
        k_c = np.sqrt(2*m*(E_c+mu)/hbar**2)
    if dim == 1:
        res = m/hbar**2/2/np.pi/k_0*np.log((k_c-k_0)/(k_c+k_0))
    elif dim == 2:
        res = m/hbar**2/4/np.pi * np.log((k_c/k_0)**2-1)
    elif dim == 3:
        res = m/hbar**2 * k_c/2/np.pi**2 * (
            1 - k_0/2/k_c*np.log((k_c+k_0)/(k_c-k_0)))
    else:
        raise ValueError(f"Only dim=1, 2, or 3 supported (got dim={dim})")
    # assert np.allclose(res.imag, 0)
    # [Check] this will go false at some point when q is large
    return res.real


def compute_C(mu_a, mu_b, delta, m_a, m_b, dim=3, hbar=1.0, T=0.0,
              q=0, dq=0, k_c=None, debug=False):
    # Note: code only works for m_a == m_b
    # ERROR: Check with new q and dq relations...
    
    args = dict(mu_a=mu_a, mu_b=mu_b, delta=delta, m_a=m_a, m_b=m_b,
                dim=dim, hbar=hbar, T=T)

    m = 2*m_a*m_b/(m_a+m_b)
    mu = (mu_a + mu_b)/2
    mu_q = mu - (hbar*dq)**2/2/m
    if k_c is None:
        k_F = np.sqrt(2*m*mu)/hbar
        k_c = 100*k_F
    
    Lambda_c = Lambda(m=m, mu=mu_q, hbar=hbar, dim=dim, k_c=k_c)
    nu_c_delta = integrate_q(f=nu_delta_integrand, k_c=k_c, q=q, dq=dq, **args)
    C_corr = integrate_q(f=C_integrand, k_0=k_c, q=q, dq=dq, **args)
    C_c = nu_c_delta + Lambda_c
    C = C_c + C_corr
    return C
    
def do_integration(integrand, mu_a, mu_b, delta, m_a, m_b, dim=3,
                q=0, dq=0.0, hbar=1.0, k_0=0, k_inf=None, limit=None):
    mu = (mu_a + mu_b)/2
    dmu = (mu_a - mu_b)/2
    minv = (1/m_a + 1/m_b)/2
    mu_q = mu - dq**2/2*minv
    assert m_a == m_b   # Need to re-derive for different masses
    m = 1./minv
    if limit is None:
        limit = MAX_DIVISION
    #kF = math.sqrt(2*mu/minv)/hbar

    # The following conditions were derived from w(kx, kp) = 0 and
    # similar conditions which are now w(kx+q, kp) = 0.  If the old
    # solutions were kp(kx) and kx, then new solutions should be
    # kp(kx+q) and kx - q.
    p_x_special = (np.ma.divide(m*(dmu - np.array([delta, -delta])),
                                dq).filled(np.nan)
                    - q).tolist()

    # Quartic polynomial for special points.  See Docs/Integrate.ipynb
    P = [1, 0, -4*(m*mu_q + dq**2),
            8*m*dq*dmu, 4*m**2*(delta**2 + mu_q**2 - dmu**2)]
    p_x_special.extend([p.real - q for p in np.roots(P)])
    points = sorted(set([x/hbar for x in p_x_special if not math.isnan(x)]))
    if dim == 1:
        integrand = numba.cfunc(numba.float64(numba.float64))(integrand)
        integrand = sp.LowLevelCallable(integrand.ctypes)
        return quad(func=integrand, a=-k_inf, b=k_inf, points=points, limit=limit)
        #return quad(func=integrand, a=k_0, b=k_inf, points=points, limit=limit)

    def kp0(kx):
        # k**2 = kx**2 + kp**2 > k_0**2
        # kp**2 > k_0**2 - kx**2
        kx2 = kx**2
        k_02 = k_0**2
        if kx2 < k_02:
            return math.sqrt(k_02 - kx2)
        else:
            return 0.0

    def kp1(kx):
        # k**2 = kx**2 + kp**2 < k_inf**2
        # kp**2 < k_inf**2 - kx**2
        # This will always work because kx < k_inf
        return math.sqrt(k_inf**2 - kx**2)

    def kp_special(kx):
        px = hbar*(kx + q)
        D = (dq*px/m - dmu)**2 - delta**2
        A = 2*m*mu_q - px**2
        return (cmath.sqrt(A + 2*m*cmath.sqrt(D)).real/hbar,
                cmath.sqrt(A - 2*m*cmath.sqrt(D)).real/hbar)
    return dquad(func=integrand,
                x0=-k_inf, x1=k_inf,
                y0_x=kp0, y1_x=kp1,
                points_x=points,
                points_y_x=kp_special,
                limit=limit)

def compute_current(mu_a, mu_b, delta, m_a=1, m_b=1, dim=3, hbar=1.0, T=0.0,
                    q=0, dq=0, k_0=0, k_c=None):
    """compute the overall current"""
    k_inf = np.inf if k_c is None else k_c
    

    if dim == 1:
        def integrand(k):
            k2_a = (k + q + dq)**2
            k2_b = (k + q - dq)**2
            f_p, f_m = f_p_m(k2_a, k2_b, mu_a, mu_b, delta, m_a, m_b, hbar, T)
            return (k * f_p + dq * f_m) / 2 / np.pi
    elif dim == 2:
        def integrand(kx, kp):
            # print(kx, kp)
            k2_a = (kx + q + dq)**2 + kp**2
            k2_b = (kx + q - dq)**2 + kp**2
            f_p, f_m = f_p_m(k2_a, k2_b, mu_a, mu_b, delta, m_a, m_b, hbar, T)
            return (kx * f_p + dq * f_m) / (2*np.pi**2)
    elif dim == 3:
        def integrand(kx, kp):
            # print(kx, kp)
            k2_a = (kx + q + dq)**2 + kp**2
            k2_b = (kx + q - dq)**2 + kp**2
            f_p, f_m = f_p_m(k2_a, k2_b, mu_a, mu_b, delta, m_a, m_b, hbar, T)
            return (kx * f_p + dq * f_m) * (kp/4/np.pi**2)
    else:
        raise ValueError(f"Only dim=1, 2, or 3 supported (got dim={dim})")
        
    return do_integration(integrand, delta=delta, mu_a=mu_a, mu_b=mu_b, m_a=m_a, m_b=m_b, 
                   dim=dim, q=q, dq=dq, hbar=hbar, k_0=k_0, k_inf=k_inf, limit=None)

def integrate(f, mu_a, mu_b, delta, m_a, m_b, dim=3, hbar=1.0, T=0.0,
              k_0=0.0, k_c=np.inf):
    """Integrate the function f from k=k_0 to k=k_c.

    Assume that k_0 = 0 or k_0 >> k_F etc.
    Assumes k_c >> k_F etc.
    """
    args = (mu_a, mu_b, delta, m_a, m_b, hbar, T)

    # We can do spherical integration
    if dim == 1:
        def integrand(k):
            k2 = k**2
            return f(k2, k2, *args) / np.pi
    elif dim == 2:
        def integrand(k):
            k2 = k**2
            return f(k2, k2, *args) * (k/2/np.pi)
    elif dim == 3:
        def integrand(k):
            k2 = k**2
            return f(k2, k2, *args) * (k2/(2*np.pi**2))
    else:
        raise ValueError(f"Only dim=1, 2, or 3 supported (got dim={dim})")

    integrand = numba.cfunc(numba.float64(numba.float64))(integrand)
    integrand = sp.LowLevelCallable(integrand.ctypes)

    mu = (mu_a + mu_b)/2
    minv = (1/m_a + 1/m_b)/2
    kF = math.sqrt(2*mu/minv)/hbar
    points = [kF]

    return quad(integrand, a=k_0, b=k_c, points=points)


def integrate_q(f, mu_a, mu_b, delta, m_a, m_b, dim=3,
                q=0, dq=0.0, hbar=1.0, T=0.0, k_0=0, k_c=None, limit=None):
    """Return the integral of f() over dim-dimensional momentum space.
 
    Arguments
    ---------
    f : function
       One of the base tf_completion functions called as f(k2_a, k2_b, ...).
    
    """
    if k_0 != 0:
        warnings.warn(f"""This integration routine does not assume it's symmetric in k,
                       since your k_0 is nonzero, result may be wrong""")
    k_inf = np.inf if k_c is None else k_c
    
    if False and q == 0 and dq == 0:
        return integrate(f, dim=dim, k_0=k_0, k_c=k_inf,
                         mu_a=mu_a, mu_b=mu_b, delta=delta,
                         m_a=m_a, m_b=m_b, hbar=hbar, T=T)

        
    delta = abs(delta)
    args = (mu_a, mu_b, delta, m_a, m_b, hbar, T)
    
    if dim == 1:
        def integrand(k):
            k2_a = (k + q + dq)**2
            k2_b = (k + q - dq)**2
            return f(k2_a, k2_b, *args) / 2 / np.pi
    elif dim == 2:
        def integrand(kx, kp):
            # print(kx, kp)
            k2_a = (kx + q + dq)**2 + kp**2
            k2_b = (kx + q - dq)**2 + kp**2
            assert(kp>=0)
            return f(k2_a, k2_b, *args) / (2*np.pi**2)
    elif dim == 3:
        def integrand(kx, kp):
            # print(kx, kp)
            k2_a = (kx + q + dq)**2 + kp**2
            k2_b = (kx + q - dq)**2 + kp**2
            assert(kp>=0)
            return f(k2_a, k2_b, *args) * (kp/4/np.pi**2)
    else:
        raise ValueError(f"Only dim=1, 2, or 3 supported (got dim={dim})")

    return do_integration(integrand, delta=delta, mu_a=mu_a, mu_b=mu_b, m_a=m_a, m_b=m_b, 
                   dim=dim, q=q, dq=dq, hbar=hbar, k_0=k_0, k_inf=k_inf, limit=limit)
