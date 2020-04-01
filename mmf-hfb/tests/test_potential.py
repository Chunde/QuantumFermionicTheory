"""Test the potential code."""
import pytest
import numpy as np
import numpy
from mmf_hfb.potentials import HarmonicOscillator

@pytest.fixture(params=[1, 2, 3])
def w(request):
    return request.param

@pytest.fixture(params=[0, 1, 2, 3])
def n(request):
    return request.param


def test_unitary(w, n):
    """check if the integral is one"""
    dx = 0.1
    x = np.linspace(-50, 50, 101)*dx
    h = HarmonicOscillator(w=w)
    y = h.get_wf(x, n=n)
    assert np.allclose(y.conj().dot(y)*dx, 1)
