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
    mu, dmu, m_a, m_b,  delta, hbar, q = more_args
    minv = (1/m_a + 1/m_b)/2
    m = m_a #1/minv # now only support m_a=m_b
    px = x #x=kx may need to multiply the hbar [check]
    mu_q = mu - q**2/2/m
    k1 = k2 = 0 
    sqrt0 = (q*px/m - dmu)**2 - delta**2
    if sqrt0 >=0:
        sqrt1 =np.sqrt(sqrt0)
        sqrt2 = 2*m*(mu_q + sqrt1) - px**2
        sqrt3 = 2*m*(mu_q - sqrt1) - px**2
        if sqrt2 > 0:
            k1 = np.sqrt(sqrt2)/hbar
        if sqrt3 > 0:
            k2 = np.sqrt(sqrt3)/hbar

    
    if callable(gfun):
        a = gfun(x)
    else:
        a=gfun
    if callable(hfun):
        b = hfun(x)
    else:
        b = hfun
    args = (x,)
    #with warnings.catch_warnings():
    #    warnings.filterwarnings('error')
    #    try:
    #        return quad(func=func, a=a, b=b, limit=limit, args=args)[0]
    #    except Warning as e:
    #        print(f"Warning {e}")
    if k1 > k2:
        k1,k2=k2,k1
    if k1 > 0 and k2 > 0: 
        res1 = quad(func=func, a=a, b=k1 - 1e-10, limit=limit, args=args)[0]
        res2 = quad(func=func, a=k2 + 1e-10, b=b, limit=limit, args=args)[0]
        res3 = quad(func=func, a=k1, b=k2, limit=limit, args=args)[0]
        return res1 + res2 + res3
    if k1 > 0:
        res1 = quad(func=func, a=a, b=k1, limit=limit, args=args)[0]
        res2 = quad(func=func, a=k1, b=b, limit=limit, args=args)[0]
        return res1 + res2
    if k2 > 0:
        res1 = quad(func=func, a=a, b=k2, limit=limit, args=args)[0]
        res2 = quad(func=func, a=k2, b=b, limit=limit, args=args)[0]
        return res1 + res2

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

def dquad_kF(f,  mu_a, mu_b, delta, q, hbar=1,
           m_a = 1, m_b=1, kF=None, k_0=0, k_inf=np.inf, limit = 50):
    """Return ufloat(res, err) for 2D integral of f(kz, kp) over the
    entire plane.    
        k_0**2 < kz**2 + kp**2 < k_inf**2
        sqrt(k_0**2 - kz**2) < kp < sqrt(k_inf**2 - kz**2)
    Assumes k_F << k_inf, k_0
    """
    mu = (mu_a + mu_b)/2
    dmu = (mu_a - mu_b)/2
    args=(mu, dmu, m_a, m_b, delta, hbar, q)
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
            res = ufloat(*_dquad(f, -k_inf, k_inf, kp_0, kp_inf,limit, args=args))
        else:
            res = (ufloat(*_dquad(f, -k_inf, -k_0,  kp_0, kp_inf,limit, args=args))  + ufloat(*_dquad(f, k_0, k_inf, kp_0, kp_inf,limit,args=args)))
    else:
        if k_0 == 0:
            res = (ufloat(*_dquad(f, -k_inf, -kF,kp_0, kp_inf,limit,args=args)) + ufloat(*_dquad(f,-kF, kF,kp_0, kp_inf,limit,args=args)) + ufloat(*_dquad(f, kF, k_inf,kp_0, kp_inf,limit,args=args)))
        else:
            res = (ufloat(*_dquad(f, -k_inf, -k_0,kp_0, kp_inf,limit,args=args))+ufloat(*_dquad(f,k_0, k_inf,kp_0, kp_inf,limit,args=args)))
    # Factor of 2 here to complete symmetric integral over kp.
    return 2*res



def _infunc_q(x, func, a, b, more_args, limit=50):
 
    def f(y):
        return func(x,y)

    mu, dmu, m_a, m_b,  delta, hbar, q = more_args
    minv = (1/m_a + 1/m_b)/2
    m = m_a #1/minv # now only support m_a=m_b
    args = () # + more_args
    px = x #x=kx may need to multiply the hbar [check]
    mu_q = mu - q**2/2/m
    k1 = k2 = 0 
    sqrt0 = (q*px/m - dmu)**2 - delta**2
    if sqrt0 >=0:
        sqrt1 =np.sqrt(sqrt0)
        sqrt2 = 2*m*(mu_q + sqrt1) - px**2
        sqrt3 = 2*m*(mu_q - sqrt1) - px**2
        if sqrt2 > 0:
            k1 = np.sqrt(sqrt2)/hbar
        if sqrt3 > 0:
            k2 = np.sqrt(sqrt3)/hbar

    if k1 > k2:
        k1,k2=k2,k1
    if k1 > 0 and k2 > 0: 
        res1 = quad(func=f, a=a, b=k1 - 1e-10, limit=limit, args=args)[0]
        res2 = quad(func=f, a=k2 + 1e-10, b=b, limit=limit, args=args)[0]
        res3 = quad(func=f, a=k1, b=k2, limit=limit, args=args)[0]
        return res1 + res2 + res3
    if k1 > 0:
        res1 = quad(func=f, a=a, b=k1, limit=limit, args=args)[0]
        res2 = quad(func=f, a=k1, b=b, limit=limit, args=args)[0]
        return res1 + res2
    if k2 > 0:
        res1 = quad(func=f, a=a, b=k2, limit=limit, args=args)[0]
        res2 = quad(func=f, a=k2, b=b, limit=limit, args=args)[0]
        return res1 + res2
    return quad(func=f, a=a, b=b,  limit=limit, args=args)[0] 

def dquad_q(func, mu_a, mu_b, delta, q, hbar=1,
           m_a = 1, m_b=1, k_0=0, k_inf=np.inf, limit=50):
    mu = (mu_a + mu_b)/2
    dmu = (mu_a - mu_b)/2
    args=(mu, dmu, m_a, m_b, delta, hbar, q)
    res = ufloat(*quad(_infunc_q, k_0, k_inf, 
                       (func, k_0, k_inf, args, limit), limit=limit))
    return res * 4