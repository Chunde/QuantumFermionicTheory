"""Numerical integration routines."""
import math
import numpy as np
import scipy.integrate
import scipy as sp
from uncertainties import ufloat
import warnings


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
        return ufloat(*sp.integrate.quad(func=func, a=a, b=b, points=points))
