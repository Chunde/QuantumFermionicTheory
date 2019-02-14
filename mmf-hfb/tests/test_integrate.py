"""Test integrate module."""
from functools import partial
from itertools import chain
import sympy
import numpy as np
import pytest

from mmf_hfb import integrate


def cases():
    def f(x, x0=0, np=np):
        return np.exp(-(x-x0)**2)

    x = sympy.var('x', real=True)
    f_x = f(x, x0=0, np=sympy)
    exact = float(sympy.integrate(f_x, (x, -np.inf, np.inf)).evalf(17))
    exact1 = float(sympy.integrate(f_x, (x, -1, 1)).evalf(17))

    yield (f, -np.inf, np.inf, None, exact)
    yield (f, np.inf, -np.inf, None, -exact)
    yield (f, -1, 1, None, exact1)
    
    for x0 in [0, 1e5]:
        f_ = partial(f, x0=x0)
        yield (f_, -np.inf, np.inf, [x0], exact)
        yield (f_, np.inf, -np.inf, [x0], -exact)
        yield (f_, x0-1, x0+1, [x0], exact1)
        yield (f_, x0-1, x0+1, [x0+0.5, x0-0.5], exact1)


@pytest.fixture(params=list(cases()))
def f_a_b_points_exact(request):
    yield request.param


class TestQuad(object):
    def test(self, f_a_b_points_exact):
        f, a, b, points, exact = f_a_b_points_exact
        res = integrate.quad(f, a, b, points=points)
        assert np.allclose(res.n, exact)
        assert abs(res.n - exact) < 10*res.s
