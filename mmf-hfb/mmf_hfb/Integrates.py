import math
import numpy as np
from scipy.integrate import quad
from uncertainties import ufloat
"""
Implement customized integrate functions here
"""
import warnings

def _infunc(x, func, gfun, hfun, more_args, limit=50):
    """
    Compute a definite integral.
    Integrate func from `a` to `b` (possibly infinite interval) 
    More options can be packed in more_args
    """
    if callable(gfun):
        a = gfun(x)
    else:
        a=gfun
    if callable(hfun):
        b = hfun(x)
    else:
        b = hfun
    args = (x,) + more_args
    #with warnings.catch_warnings():
    #    warnings.filterwarnings('error')
    #    try:
    #        return quad(func=func, a=a, b=b, limit=limit, args=args)[0]
    #    except Warning as e:
    #        print(f"Warning {e}")

    return quad(func=func, a=a, b=b, limit=limit, args=args)[0]
"""

"""

def _dquad(func, a, b, gfun, hfun, limit,args=(), epsabs=1.49e-8, 
                   epsrel=1.49e-8, maxp1=50):
    """
    Compute 2d integral for f(x,y)
    x from a to b
    y from func(x) to gfun(x)
    Support limit options
    """
    return quad(_infunc, a, b, (func, gfun, hfun, args, limit), 
                          epsabs=epsabs, epsrel=epsrel, maxp1=maxp1, limit=limit)

def dquad_kF(f, kF=None, k_0=0, k_inf=np.inf, limit = 50):
    """Return ufloat(res, err) for 2D integral of f(kz, kp) over the
    entire plane.    
        k_0**2 < kz**2 + kp**2 < k_inf**2
        sqrt(k_0**2 - kz**2) < kp < sqrt(k_inf**2 - kz**2)
    Assumes k_F << k_inf, k_0
    """

    def kp_0(kz):
        D = k_0**2 - kz**2
        if D < 0:
            return 0
        else:
            return math.sqrt(D)
        
    def kp_inf(kz):
        return math.sqrt(k_inf**2 - kz**2)

    if np.isinf(k_inf):
        kp_inf = k_inf

    if k_0 == 0:
        kp_0 == 0

    if kF is None:
        if k_0 == 0:
            res = ufloat(*_dquad(f,
                                  -k_inf, k_inf,   # kz
                                  kp_0, kp_inf,limit))   # kp
        else:
            res = (
                ufloat(*_dquad(f,
                                -k_inf, -k_0,  # kz
                                kp_0, kp_inf,limit)) # kp
                +
                ufloat(*_dquad(f,
                                k_0, k_inf,
                                kp_0, kp_inf,limit)))
    else:
        if k_0 == 0:
            res = (
                ufloat(*_dquad(f,
                                -k_inf, -kF,
                                kp_0, kp_inf,limit))
                +
                ufloat(*_dquad(f,
                                -kF, kF,
                                kp_0, kp_inf,limit))
                +
                ufloat(*_dquad(f,
                                kF, k_inf,
                                kp_0, kp_inf,limit))
            )
        else:
            res = (
                ufloat(*_dquad(f,
                                -k_inf, -k_0,
                                kp_0, kp_inf,limit))
                +
                ufloat(*_dquad(f,
                                k_0, k_inf,
                                kp_0, kp_inf,limit))
            )
    # Factor of 2 here to complete symmetric integral over kp.
    return 2*res



def _kpfunc(x, func, more_args, limit=50):
    a = 0
    b = np.inf
    m = 1 # need to change later
    """
    Compute a definite integral.
    Integrate func from `a` to `b` (possibly infinite interval) 
    More options can be packed in more_args
    """
    mu, dmu, delta, q = more_args
    args = (x,) # + more_args
    px = x
    #e_a = ((px+q)**2+p_perp**2)/2/m - mu_a
    #e_b = ((px-q)**2 + p_perp**2)/2/m - mu_b
    #e_p, e_m = (e_a + e_b)/2.0, (e_a-e_b)/2.0
    #E = np.sqrt(e_p**2 + delta**2)
    #w_p, w_m = e_m + E, e_m - E
    mu_q = mu - q**2/2/m
    p1 = np.sqrt(2*m*(mu_q + np.sqrt((q*px/m - dmu)**2 - delta**2)) - px**2)
    p2 = np.sqrt(2*m*(mu_q - np.sqrt((q*px/m - dmu)**2 - delta**2)) - px**2)
    #with warnings.catch_warnings():
    #    warnings.filterwarnings('error')
    #    try:
    #        return quad(func=func, a=a, b=b, limit=limit, args=args)[0]
    #    except Warning as e:
    #        print(f"Warning {e}")
    points = []
    if not math.isnan(p1) and p1.imag == 0:
        points.append(p1)
    if not math.isnan(p2)  and p2.imag == 0:
        points.append(p2)
    return quad(func=func, a=a, b=b, points=points, limit=limit, args=args)[0]

def dquad_q(func, mu_a, mu_b, delta, q, hbar=1, m = 1, k_0=0, k_inf=np.inf, limit=50):
    mu = (mu_a + mu_b)/2
    dmu = (mu_a - mu_b)/2
    

    args=(mu, dmu, delta, q)

    return quad(_kpfunc, k_0, k_inf, (func, args, limit), limit=limit)