"""Test integrate module."""
from functools import partial
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
        yield (f_, x0-1, x0+1, [x0+0.5, x0, x0, x0-0.5], exact1)


@pytest.fixture(params=list(cases()))
def f_a_b_points_exact(request):
    yield request.param


class TestQuad(object):
    def test(self, f_a_b_points_exact):
        f, a, b, points, exact = f_a_b_points_exact
        res = integrate.quad(f, a, b, points=points)
        assert np.allclose(res.n, exact)
        assert abs(res.n - exact) < 10*res.s


def cases2():
    def f(x, y, x0=0, y0=0, np=np):
        return (1 + (x-x0) - 2*(y-y0))*np.exp(-(x-x0)**2 - (y-y0)**2)

    x, y = sympy.var(['x', 'y'], real=True)
    f_xy = f(x, y, x0=0, y0=0, np=sympy)
    _exact = f_xy.integrate((x, -np.inf, np.inf),
                            (y, -np.inf, np.inf))
    exact = float(_exact.evalf(17))

    _exact = f_xy.integrate((x, -1, 1),
                            (y, -1, 1))
    exact1 = float(_exact.evalf(17))

    yield (f, -np.inf, np.inf, -np.inf, np.inf, None, None, exact)
    yield (f, np.inf, -np.inf, np.inf, -np.inf, None, None, exact)
    yield (f, -1, 1, -1, 1, None, None, exact1)
    
    for x0 in [0, 1e5]:
        for y0 in [0, 1e5]:
            f_ = partial(f, x0=x0, y0=y0)
            yield (f_, -np.inf, np.inf, -np.inf, np.inf, [x0], [y0], exact)
            yield (f_, np.inf, -np.inf, np.inf, -np.inf, [x0], [y0], exact)
            yield (f_, x0-1, x0+1, y0-1, y0+1, [x0], [y0], exact1)


@pytest.fixture(params=list(cases2()))
def f2_a_b_points_exact(request):
    yield request.param

    
class TestdQuad(object):
    def test(self, f2_a_b_points_exact):
        f, x0, x1, y0, y1, points_x, points_y, exact = f2_a_b_points_exact

        def y0_x(x):
            return y0
        
        def y1_x(x):
            return y1

        if points_y is None:
            points_y_x = None
        else:
            def points_y_x(x):
                return points_y
    
        res = integrate.dquad(f, x0, x1, y0_x, y1_x, points_x, points_y_x)
        assert np.allclose(res.n, exact)
        assert abs(res.n - exact) < 20*res.s

        # Check that switching limits switches sign
        res = integrate.dquad(f, x1, x0, y0_x, y1_x, points_x, points_y_x)
        assert np.allclose(res.n, -exact)
        assert abs(res.n + exact) < 20*res.s

        res = integrate.dquad(f, x0, x1, y1_x, y0_x, points_x, points_y_x)
        assert np.allclose(res.n, -exact)
        assert abs(res.n + exact) < 20*res.s
