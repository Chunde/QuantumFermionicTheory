from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import math
import numpy as np

from mmfutils.math.differentiate import differentiate

from pytimeode import evolvers
from pytimeode.utils.testing import TestState as _TestState
from bec import State, u


class Tests(object):
    @classmethod
    def setup_class(cls):
        cls.state1 = State(Nxyz=(2**5,), Lxyz=(200*u.micron,))
        cls.state1.cooling_phase = 1j
        cls.state1.t = -1000.0*u.ms

        cls.state2 = State(Nxyz=(2**4,)*2, Lxyz=(200*u.micron,)*2)
        cls.state2.cooling_phase = 1j
        cls.state2.t = -1000.0*u.ms

        e = evolvers.EvolverABM(cls.state1, dt=0.0001*u.ms)
        e.evolve(10)
        cls.state1 = e.get_y()
        assert not np.allclose(0, cls.state1.get_energy())

        e = evolvers.EvolverABM(cls.state2, dt=0.0001*u.ms)
        e.evolve(10)
        cls.state2 = e.get_y()
        assert not np.allclose(0, cls.state2.get_energy())

        cls.states = [cls.state1, cls.state2]

    def test_state(self):
        """Test for consistency between SplitOperator and ABM interfaces"""
        for s in self.states:
            t = _TestState(s)
            assert all(t.check_split_operator(normalize=True))

    def test_H(self):
        """Test Hamiltonian generation."""
        for s in self.states:
            H = s.get_H()
            dy = s.empty()
            s.compute_dy(dy=dy, subtract_mu=False)
            dy /= s._phase
            assert np.allclose(dy[...].ravel(),
                               H.dot(s[...].ravel()))

    def test_energy(self):
        """Check that H(y) is the derivative of the energy."""
        np.random.seed(10)
        for s in self.states:
            dx = (np.random.random(s.data.shape)
                  + np.random.random(s.data.shape)*1j - 0.5 - 0.5j)

            def f(h):
                s_ = s.copy()
                s_[...] += h*dx
                return s_.get_energy()

            res = differentiate(f, h0=0.001)
            assert not np.allclose(0, res)

            dy = s.empty()
            s.compute_dy(dy, subtract_mu=False)
            dy /= s._phase

            assert np.allclose(res, 2*s.braket(dx, dy[...]).real)
