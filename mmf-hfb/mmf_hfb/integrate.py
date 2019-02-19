"""Numerical integration routines."""
from functools import partial
import numpy as np
import scipy.integrate
import scipy as sp
from uncertainties import ufloat


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
        return ufloat(*sp.integrate.quad(func=func, a=a, b=b, points=points, **kw))


def dquad(func, x0, x1, y0_x, y1_x, points_x=None, points_y_x=None, **kw):
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
