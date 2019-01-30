"""BCS Equations in 1D

This module provides a class BCS for solving the BCS equations in 1D for a
two-species Fermi gas with short-range delta-function interactions of strength
v0.
"""
from __future__ import division

import numpy as np
import scipy.integrate

from mmfutils.math.integrate import mquad


class BCS(object):
    hbar = 1.0
    m = 1.0
    w = 1.0                     # Trapping potential

    def __init__(self, N=32, L=10.0, dx=None, T=0):
        """Specify any two of `N`, `L`, or `dx`.

        Arguments
        ---------
        N : int
           Number of lattice points.
        L : float
           Length of the periodic box.
           Can also be understood as the largetest wavelenght of possible waves host in the box.
           Then the minimum wave-vector k0 = 2PI/lambda = 2 * np.pi / L, and all possible ks should 
           be integer times of k0.
        dx : float
           Lattice spacing.
        T : float
           Temperature.
        """
        if dx is None:
            dx = L / N
        elif L is None:
            L = N * dx
        elif N is None:
            N = int(np.ceil(L / dx))

        self.x = np.arange(N) * dx - L / 2
        self.k = 2*np.pi * np.fft.fftfreq(N, dx)
        self.dx = dx

        self.N = N
        self.L = L
        self.T = T

        # External potential
        self.v_ext = self.get_v_ext()

    def get_Ks(self, twist=0):
        """Return the kinetic energy matrix."""
        k_bloch = twist/self.L
        k = self.k + k_bloch
        N = self.N

        # Unitary matrix implementing the FFT including the phase
        # twist
        U = np.exp(-1j*k[:, None]*self.x[None, :])/np.sqrt(N)
        assert np.allclose(U.conj().T.dot(U), np.eye(N))

        # Kinetic energy matrix
        K = np.dot(U.conj().T, (self.hbar * k)[:, None]**2/2/self.m * U)
        return (K, K)

    def get_v_ext(self):
        """Return the external potential."""
        v = self.m * (self.w * self.x)**2 / 2.0
        v = 0 * self.x
        return (v, v)

    def f(self, E):
        """Return the Fermi-Dirac distribution at E."""
        if self.T > 0:
            return 1./(1+np.exp(E/self.T))
        else:
            return (1 - np.sign(E))/2

    def get_H(self, mus_eff, delta, twist=0):
        """Return the single-particle Hamiltonian with pairing.

        Arguments
        ---------
        mus : array
           Effective chemical potentials including the Hartree term but not the
           external trapping potential.
        delta : array
           Pairing field (gap).
        twist : float
           Bloch phase.
        """
        zero = np.zeros_like(self.x)
        delta = delta + zero
        v_a, v_b = self.v_ext
        mu_a, mu_b = mus_eff
        K_a, K_b = self.get_Ks(twist=twist)
        H = np.bmat([[K_a + np.diag(v_a - mu_a), -np.diag(delta)],
                     [-np.diag(delta.conj()), -(K_b + np.diag(v_b - mu_b))]])
        return np.asarray(H)

    def get_R(self, mus, delta, N_twist=1, twists=None):
        """Return the density matrix R."""
        N = self.N
        Rs = []
        if twists is None:
            twists = np.arange(0, N_twist)*2*np.pi/N_twist

        for twist in twists:
            H = self.get_H(mus_eff=mus, delta=delta, twist=twist)
            d, UV = np.linalg.eigh(H)
            R = UV.dot(self.f(d)[:, None]*UV.conj().T)
            # R_ = np.eye(2*N) - UV.dot(self.f(-d)[:, None]*UV.conj().T)
            # assert np.allclose(R, R_)
            Rs.append(R)
        R = sum(Rs)/len(Rs)
        return R

    def get_R_twist_average(self, mus, delta, abs_tol=1e-12):
        """Return the density matrix R."""
        N = self.N
        R0 = 1.0
        def f(twist):
            H = self.get_H(mus_eff=mus, delta=delta, twist=twist)
            d, UV = np.linalg.eigh(H)
            R = UV.dot(self.f(d)[:, None]*UV.conj().T)
            return R/R0

        R0 = f(0)

        R = R0 * mquad(f, -np.pi, np.pi, abs_tol=abs_tol)/2/np.pi
        return R

    def get_LDA(mu_eff, delta):
        """Return the LDA solution"""
