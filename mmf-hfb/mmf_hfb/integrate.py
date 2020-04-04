"""Numerical integration routines."""
from functools import partial

import warnings

import numpy as np

import scipy.integrate
import scipy as sp

from uncertainties import ufloat

__all__ = ['quad', 'dquad', 'quad_k', 'quad_l']
# from scipy.integrate import IntegrationWarning
# warnings.simplefilter('error', IntegrationWarning)

_QUAD_ARGS = dict(
    epsabs=1.49e-08,
    epsrel=1.49e-08)


def quad(func, a, b, points=None, **kw):
    """Simple wrapper over scipy.integrate.quad which allows for
    points with infinite ranges.
    """
    if points is not None and np.any(np.isinf([a, b])):
        sign = 1
        if b < a:
            sign = -1
            a, b = b, a
        points = sorted([a, b] + [p for p in points if a < p and p < b])
        res = [sp.integrate.quad(func=func, a=_a, b=_b, **kw)
               for _a, _b in zip(points[:-1], points[1:])]
        return sign * sum(ufloat(*_r) for _r in res)
    else:
        return ufloat(*sp.integrate.quad(func=func, a=a, b=b,
                                         points=points, **kw))


def dquad(func, x0, x1, y0_x, y1_x, points_x=None, points_y_x=None,
          **kw):
    """Compute a double integral.

    Note: The order of arguments is not the same as dblquad.  They are
    func(x, y) here.

    Arguments
    ---------
    func : callable
       A Python function or method of at least two variables: f(x, y)
    x0, x1 : float
       The limits of integration in x: x0 < x1
    x0_x, y1_x : callable
       The lower (upper) boundary curve in y which is a function taking a
       single floating point argument (x) and returning a floating point
       result or a float indicating a constant boundary curve.
    points_x : []
       List of special points in x (outer integral)
    points_y_x : callable or list
       points_y_x(x) should return a list of special points in y for
       the inner integral.
    """
    def inner_integrand(x):
        points = None
        if points_y_x is not None:
            points = points_y_x(x)

        return quad(partial(func, x), a=y0_x(x), b=y1_x(x),
                    points=points, **kw).n

    return quad(inner_integrand, x0, x1, points=points_x, **kw)


def quad_k(f, kF=None, k_0=0, k_inf=np.inf, dim=1, limit=1000, **kw):
    """Integrate from k_0 to k_inf in dim-dimensions including factors
    of 1/(2*pi)**dim and spherical area."""
    args = dict(_QUAD_ARGS, **kw)

    factor = 1.0
    if dim == 1:
        factor = 1./np.pi
        integrand = f
    elif dim == 2:
        factor = 1./2/np.pi
        
        def integrand(k):
            return f(k)*k
    elif dim == 3:
        factor = 1./2/np.pi**2
        
        def integrand(k):
            return f(k)*k**2
    else:
        raise NotImplementedError(f"Only dim=1,2,3 supported (got {dim}).")

    if kF is None:
        res = ufloat(*sp.integrate.quad(integrand, k_0, k_inf, **args))
    else:
        # One might think that `points=[kF]` could be used here, but
        # this does not work with infinite limits.
        res = (ufloat(*sp.integrate.quad(integrand, k_0, kF, **args))
               + ufloat(*sp.integrate.quad(
                   integrand, kF, k_inf, limit=limit, **args)))

    if abs(res.s) > 1e-6 and abs(res.s/res.n) > 1e-6:
        warnings.warn(f"Integral did not converge: {res}")
        
    return res * factor


def quad_l(f, Nxyz, Lxyz, N_twist=1, **kw):
    """Integrate f(k) using a lattice including factors of 1/(2*pi)**dim.

    Arguments
    ---------
    Nxyz : (int,)
    Lxyz : (float,)
       These tuples specify the size of the lattice and box.  Their
       length specifies the dimension.
    N_twist : int, np.inf
       How many twists to sample in each direction.
       (This is done by just multiplying N and L by this factor.)  If
       N_twist==np.inf, then to the integral (with cutoff).

    BUG: Currently, integration is done over a spherical shell with
    radius k_max.  This only matches the lattice calculation in 1D.
    """
    dim = len(Nxyz)
    Nxyz, Lxyz = np.array(Nxyz), np.array(Lxyz)
    dxyz = Lxyz/Nxyz

    if np.isinf(N_twist):
        k_max = np.pi/np.max(dxyz)

        # This is the idea, but we really need to compute the measure
        # properly so we integrate over the box rather than a
        # sphere.  This is a little tricky.
        return quad_k(f, k_inf=k_max, dim=dim, **kw)

    # Lattice sums.
    Nxyz *= N_twist
    Lxyz *= N_twist
    dxyz = Lxyz/Nxyz
    dkxyz = 2*np.pi/Lxyz
    ks = np.meshgrid(
        *[2*np.pi * np.fft.fftshift(np.fft.fftfreq(_N, _dx))
          for (_N, _dx) in zip(Nxyz, dxyz)],
        indexing='ij', sparse=True)
    k2 = sum(_k**2 for _k in ks)
    k = np.sqrt(k2)
    return ufloat(f(k).sum() * np.prod(dkxyz), 0) / (2*np.pi)**dim
