"""Single component BEC

Here we demonstrate the dynamics of a single component BEC such as 87Rb.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import math
import numpy as np

from mmfutils.performance.fft import fftn, ifftn

from pytimeode import interfaces, mixins, evolvers

import Units as U;
u = units = U.Units()


def step(t, t1, alpha=3.0):
    r"""Smooth step function that goes from 0 at time ``t=0`` to 1 at time
    ``t=t1``.  This step function is $C_\infty$:
    """
    if t < 0.0:
        return 0.0
    elif t < t1:
        return (1 + math.tanh(alpha*math.tan(math.pi*(2*t/t1-1)/2)))/2
    else:
        return 1.0


class State(mixins.ArrayStateMixin):
    interfaces.implements([interfaces.IStateForABMEvolvers,
                           interfaces.IStateForSplitEvolvers,
                           interfaces.IStateWithNormalize])

    def __init__(self,
                 Nxyz = (2**5, 2**5, 2**5),
                 Lxyz = (30*u.micron, 50*u.micron, 50*u.micron),
                 ws = np.array([np.sqrt(8)*126.0, 126.0, 126.0])*u.Hz,
                 barrierVelocity = np.array([0.1,0.1,0.1]),
                 barrierWidth = np.array([0.1, 0.1, 0.1]),
                 g = 4 * np.pi * u.hbar**2 * u.a/u.m,
                 N = 2e4,
                 cooling_phase = 1.0j,
                 ):
        self.Nxyz = Nxyz
        self.Lxyz = Lxyz
        self.ws = ws
        self.N = N
        self.g = g
        self.barrierFlag = False
        self.barrierWidth = barrierWidth
        self.barrierOffset = 0
        self.barrierVelocity = barrierVelocity
        self.barrierIensity = 1.0
        self.cooling_phase = cooling_phase
        self.metric = np.prod(np.divide(self.Lxyz, self.Nxyz))

        # Compute mu using TF approximation
        w3 = (np.prod(self.ws)**(1./len(self.ws)))**3
        self.mu = ((15.0 * self.g * self.N * w3 / (16 * np.pi))**2
                   * u.m**3 / 2)**(1./5.)

        self.xyz = np.meshgrid(
            *[np.arange(_N)*_L/_N - _L/2.0 for _N, _L in zip(self.Nxyz, self.Lxyz)],
            sparse=True, indexing='ij')

        self.kxyz = np.meshgrid(
            *[2*np.pi * np.fft.fftfreq(_N, _L/_N)
              for _N, _L in zip(self.Nxyz, self.Lxyz)],
            sparse=True, indexing='ij')

        self.data = np.empty(self.Nxyz, dtype=complex)
        V_ext = self.get_Vext()
        n = np.maximum(0, (self.mu - V_ext))/self.g
        self.data[...] = np.sqrt(n)
        #what the K means here???
        self.K = sum((u.hbar*_k)**2/2.0/u.m for _k in self.kxyz)

        self.pre_evolve_hook()

    def pre_evolve_hook(self):
        self._phase = 1./ 1j / u.hbar / self.cooling_phase
        self._N = self.get_N()

    @property
    def dim(self):
        return len(self.Nxyz)

    @property
    def shape(self):
        return tuple(self.Nxyz)

    def get_Vext(self):
        """Return the external potentials `(V_F, V_B)`."""
        return 0.5*u.m * sum((_w*_x)**2
                             for _w, _x in zip(self.ws, self.xyz))
    def get_Barrier(self,t_):
        """Return the time dependece barriry"""
        return sum(self.barrierIensity * np.exp(-((-self.barrierOffset + _x - _v * t_)/_w)**2)
                             for _v, _w, _x in zip(self.barrierVelocity,self.barrierWidth, self.xyz))
    def get_density(self):
        return abs(self[...])**2

    def get_N(self):
        n = self.get_density()
        return self.integrate(n)

    def integrate(self, a):
        return self.metric * np.sum(a)

    def braket(self, a, b):
        return self.metric * a[...].ravel().conj().dot(b[...].ravel())

    def get_V(self):
        """Return the complete potential `V` - internal and external."""
        n = self.get_density()
        V_ext = self.get_Vext()
        V_int = self.g * n
        if self.barrierFlag == True:
            return V_ext + V_int + self.get_Barrier(self.t)
        return V_ext + V_int

    ######################################################################
    # Required by interface IStateForABMEvolvers
    def compute_dy_dt(self, dy, subtract_mu=True):
        y = self[...]
        Ky = ifftn(self.K * fftn(y))
        Vy = self.get_V()*y
        Hy = Ky + Vy
        if subtract_mu:
            mu = self.braket(y, Hy)/self.braket(y, y)
            assert np.allclose(0, mu.imag)
            Hy[...] -= mu*y
            self._mu = mu

        dy[...] = Hy*self._phase
        return dy

    ######################################################################
    # Required by interface IStateForSplitEvolvers
    linear = False

    def apply_exp_K(self, dt):
        r"""Apply $e^{-i K dt}$ in place"""
        y = self[...]
        self[...] = ifftn(np.exp(self.K*self._phase*dt)*fftn(y))

    def apply_exp_V(self, dt, state):
        r"""Apply $e^{-i V dt}$ in place using `state` for any
        nonlinear dependence in V. (Linear problems should ignore
        `state`.)"""
        self *= np.exp(self.get_V() * self._phase * dt)

    ######################################################################
    # Required by interface IStateWithNormalize
    def normalize(self):
        """Normalize the state"""
        self *= np.sqrt(self._N/self.get_N())
        assert np.allclose(self._N, self.get_N())

    # End of interface definitions
    ######################################################################

    # The following functions are helpful for analysis but not needed for
    # the evolvers.
    def get_H(self):
        N = np.prod(self.Nxyz)
        Q = fftn(np.eye(N).reshape(self.shape*2),
                 axes=range(self.dim)).reshape((N,N))
        Qinv = Q.T.conj()/N
        K = Qinv.dot(self.K.ravel()[:, None]*Q)
        V = np.diag(self.get_V().ravel() - self._mu)
        return K + V

    def get_Hy(self, subtract_mu=False):
        dy = self.empty()
        self.compute_dy_dt(dy=dy, subtract_mu=subtract_mu)
        Hy = dy/self._phase
        return Hy

    def get_energy_density(self):
        y = self[...]
        n = self.get_density()
        K = y.conj()*ifftn(self.K * fftn(y))
        Vint = self.g*n**2/2.0
        Vext = self.get_Vext()*n
        return (K + Vint + Vext)

    def get_energy(self):
        E = self.integrate(self.get_energy_density())
        assert np.allclose(0, E.imag)
        return E.real

    def plot(self, log=False):  # pragma: nocover
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec
        from mmfutils.plot import imcontourf

        n = self.get_density()
        if log:
            n = np.log10(n)

        if self.dim == 1:
            x = self.xyz[0]/u.micron
            plt.plot(x, n)

        elif self.dim == 3:
            x, y, z = [_x/u.micron for _x in self.xyz]
            nxy = n.sum(axis=2)
            nxz = n.sum(axis=1)
            nyz = n.sum(axis=0)

            gs = GridSpec(1, 3)
            ax = plt.subplot(gs[0])
            imcontourf(x, y, nxy)
            ax.set_aspect(1)
            ax = plt.subplot(gs[1])
            imcontourf(x, z, nxz)
            ax.set_aspect(1)
            ax = plt.subplot(gs[2])
            imcontourf(y, z, nyz)
            ax.set_aspect(1)

        E = self.get_energy()
        N = self.get_N()
        plt.suptitle("N={:.4f}, E={:.4f}".format(N, E))
