"""BCS Equations in 2D

This module provides a class BCS2D for solving the BCS (BdG) equations in 2D for a
two-species Fermi gas with short-range interaction.
"""
from __future__ import division

from collections import namedtuple
import itertools

import numpy as np
import scipy.integrate

from mmfutils.math.integrate import mquad


class BCS_1D(object):
    """Simple implementation of the BCS equations in a periodic box.

    We use all states in the box, regularizing the theory with a fixed
    coupling constant g_c which will depend on the box parameters.
    """
    hbar = 1.0
    m = 1.0

    def __init__(self, Nx=32, Lx=10.0, dx=None, T=0):
        """Specify any two of `Nx`, `Lx`, or `dx`.

        Arguments
        ---------
        Nx : int
           Number of lattice points.
        Lx : float
           Length of the periodic box.
           Can also be understood as the largetest wavelenght of
           possible waves host in the box. Then the minimum
           wave-vector k0 = 2PI/lambda = 2 * np.pi / L, and all
           possible ks should be integer times of k0.
        dx : float
           Lattice spacing.
        T : float
           Temperature.
        """
        if dx is None:
            dx = Lx/Nx
        elif Lx is None:
            Lx = Nx * dx
        elif Nx is None:
            Nx = np.ceil(Lx / dx).astype(int)

        self.x = np.arange(Nx) * dx - Lx / 2
        self.k = 2*np.pi * np.fft.fftfreq(Nx, dx)
        self.dx = dx
        self.Lx = Lx
        self.Nx = Nx
        self.T = T

        # External potential
        self.v_ext = self.get_v_ext()
        
    def get_Ks(self, twist=0):
        """Return the kinetic energy matrix."""
        k_bloch = twist/self.Lx
        k = self.k + k_bloch

        # Here we use a simple trick of applying the FFT to an
        # identify matrix.  This ensures that we appropriately
        # calculate the matrix structure without having to worry about
        # indices and phase factors.  The following transformation
        # should be correct however:
        #
        # U = np.exp(-1j*k[:, None]*self.x[None, :])/self.Nx
        #
        mat_shape = (self.Nx,)*2
        K = np.eye(self.Nx)
        K = self.ifft((self.hbar*k[:, None])**2/2/self.m * self.fft(K))
        return (K, K)

    def fft(self, y):
            return np.fft.fft(y, axis=0)
            
    def ifft(self, y):
            return np.fft.ifft(y, axis=0)
            
    def get_v_ext(self):
        """Return the external potential."""
        return (0, 0)

    def f(self, E):
        """Return the Fermi-Dirac distribution at E."""
        if self.T > 0:
            f = 1./(1+np.exp(E/self.T))
        else:
            f = (1 - np.sign(E))/2
        return f

    def get_H(self, mus, delta, twist=0):
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
        Delta = np.diag((delta + zero).ravel())
        K_a, K_b = self.get_Ks(twist=twist)
        v_a, v_b = self.v_ext
        mu_a, mu_b = mus
        mu_a += zero
        mu_b += zero
        Mu_a, Mu_b = np.diag((mu_a - v_a).ravel()), np.diag((mu_b - v_b).ravel())
        H = np.bmat([[K_a - Mu_a, Delta],
                     [Delta.conj(), -(K_b - Mu_b)]])
        return np.asarray(H)

    def get_Rs(self, mus, delta,N_twist = 1):
        """Return the density matrix R."""

        Rps = []
        Rms = []
        twists = np.arange(0, N_twist)*2*np.pi/N_twist

        for twist in twists:
            H = self.get_H(mus=mus, delta=delta, twist=twist)
            d, UV = np.linalg.eigh(H)
            Rp = UV.dot(self.f(d)[:, None]*UV.conj().T)
            Rm = UV.dot(self.f(-d)[:, None]*UV.conj().T)
            Rps.append(Rp)
            Rms.append(Rm)
        Rp = sum(Rps)/len(Rps)
        Rm = sum(Rms)/len(Rms)
        return Rp/self.dx, Rm/self.dx



        H = self.get_H(mus=mus, delta=delta)
        d, UV = np.linalg.eigh(H)
        dV = self.dx

        # Factor of dV here to convert to physical densities...
        Rp = UV.dot(self.f(d)[:, None]*UV.conj().T) / dV
        Rm = UV.dot(self.f(-d)[:, None]*UV.conj().T) / dV
        return Rp, Rm

    def get_densities(self, mus, delta,N_twist=1):
        Rp, Rm = self.get_Rs(mus=mus, delta=delta,N_twist=N_twist)
        _N = Rp.shape[0] // 2
        r_a = Rp[:_N, :_N]
        r_b = Rm[_N:, _N:].conj()
        nu_ = (Rp[:_N, _N:] - Rm[_N:, :_N].T.conj())/2.0
        n_a = np.diag(r_a).real
        n_b = np.diag(r_b).real
        nu = np.diag(nu_)
        return namedtuple('Densities', ['n_a', 'n_b', 'nu'])(n_a, n_b, nu)


class BCS_2D(object):
    """Simple implementation of the BCS equations in a periodic square
    box.

    We use all states in the box, regularizing the theory with a fixed
    coupling constant g_c which will depend on the box parameters.
    """
    hbar = 1.0
    m = 1.0

    def __init__(self, Nxy=(32, 32), Lxy=(10.0, 10.0), dx=None, T=0):
        """Specify any two of `Nxy`, `Lxy`, or `dx`.

        Arguments
        ---------
        Nxy : (int, int)
           Number of lattice points.
        Lxy : (float, float)
           Length of the periodic box.
           Can also be understood as the largetest wavelenght of
           possible waves host in the box. Then the minimum
           wave-vector k0 = 2PI/lambda = 2 * np.pi / L, and all
           possible ks should be integer times of k0.
        dx : float
           Lattice spacing.
        T : float
           Temperature.
        """
        dy = dx
        if dx is None:
            dx, dy = np.divide(Lxy, Nxy)
        elif Lxy is None:
            Lxy = np.prod(Nxy, dx)
        elif Nxy is None:
            Nxy = np.ceil(Lxy / dx).astype(int)

        Nx, Ny = Nxy
        Lx, Ly = Lxy
        self.xy = ((np.arange(Nx) * dx - Lx / 2)[:, None],# half of the length
                   (np.arange(Ny) * dy - Ly / 2)[None, :])
        
        self.kxy = (2*np.pi * np.fft.fftfreq(Nx, dx)[:, None],
                    2*np.pi * np.fft.fftfreq(Ny, dy)[None, :])
        self.dx = dx
        
        self.Nxy = tuple(Nxy)
        self.Lxy = tuple(Lxy)
        self.T = T

        self._Ks = self.get_Ks()

        # External potential
        self.v_ext = self.get_v_ext()
        
    def get_Ks(self):
        """Return the kinetic energy matrix."""

        # Here we use a simple trick of applying the FFT to an
        # identify matrix.  This ensures that we appropriately
        # calculate the matrix structure without having to worry about
        # indices and phase factors.
        mat_shape = (np.prod(self.Nxy),)*2
        tensor_shape = self.Nxy + self.Nxy
        K = np.eye(mat_shape[0]).reshape(tensor_shape)
        K = (self.hbar**2/2/self.m
             * self.ifftn(sum(_k**2 for _k in self.kxy)[:, :,  None, None]
                          * self.fftn(K))).reshape(mat_shape)
        return (K, K)

    def fftn(self, y):
            return np.fft.fftn(y, axes=(0, 1))
            
    def ifftn(self, y):
            return np.fft.ifftn(y, axes=(0, 1))
            
    def get_v_ext(self):
        """Return the external potential."""
        return (0, 0)

    def f(self, E):
        """Return the Fermi-Dirac distribution at E."""
        if self.T > 0:
            f = 1./(1+np.exp(E/self.T))
        else:
            f = (1 - np.sign(E))/2
        return f

    def get_H(self, mus, delta):
        """Return the single-particle Hamiltonian with pairing.

        Arguments
        ---------
        mus : array
           Effective chemical potentials including the Hartree term but not the
           external trapping potential.
        delta : array
           Pairing field (gap).
        phi_bloch : float
           Bloch phase.
        """
        zero = np.zeros_like(sum(self.xy))
        Delta = np.diag((delta + zero).ravel())
        K_a, K_b = self._Ks
        v_a, v_b = self.v_ext
        mu_a, mu_b = mus
        mu_a += zero
        mu_b += zero
        Mu_a, Mu_b = np.diag((mu_a - v_a).ravel()), np.diag((mu_b - v_b).ravel())
        H = np.bmat([[K_a - Mu_a, -Delta],
                     [-Delta.conj(), -(K_b - Mu_b)]])
        return np.asarray(H)

    def get_Rs(self, mus, delta):
        """Return the density matrix R."""
        H = self.get_H(mus=mus, delta=delta)
        d, UV = np.linalg.eigh(H)
        Rp = UV.dot(self.f(d)[:, None]*UV.conj().T)
        Rm = UV.dot(self.f(-d)[:, None]*UV.conj().T)
        return Rp, Rm

    def get_densities(self, mus, delta):
        Rp, Rm = self.get_Rs(mus=mus, delta=delta)
        N = Rp.shape[0] // 2
        r_a = Rp[:N, :N]
        r_b = Rm[N:, N:]
        kappa = (Rp[:N, N:] - Rm[N:, :N].T.conj())/2.0
        n_a = np.diag(r_a).reshape(self.Nxy).real
        n_b = np.diag(r_b).reshape(self.Nxy).real
        nu = np.diag(kappa).reshape(self.Nxy)
        return namedtuple('Densities', ['n_a', 'n_b', 'nu'])(n_a, n_b, nu)
        
    def get_LDA(self, mus, delta):
        """Return the LDA solution."""




class BCS(object):
    hbar = 1.0
    m = 1.0
    w = 1.0                     # Trapping potential

    def __init__(self, Nxy=(32, 32), Lxy=(10.0, 10.0), dx=None, T=0):
        """Specify any two of `Nxy`, `Lxy`, or `dx`.

        Arguments
        ---------
        Nxy : (int, int)
           Number of lattice points.
        Lxy : (float, float)
           Length of the periodic box.
           Can also be understood as the largetest wavelenght of
           possible waves host in the box. Then the minimum
           wave-vector k0 = 2PI/lambda = 2 * np.pi / L, and all
           possible ks should be integer times of k0.
        dx : float
           Lattice spacing.
        T : float
           Temperature.
        """
        dy = dx
        if dx is None:
            dx, dy = np.divide(Lxy, Nxy)
        elif Lxy is None:
            Lxy = np.prod(Nxy, dx)
        elif Nxy is None:
            Nxy = np.ceil(Lxy / dx).astype(int)

        Nx, Ny = Nxy
        Lx, Ly = Lxy
        self.xy = ((np.arange(Nx) * dx - Lx / 2)[:, None],# half of the length
                   (np.arange(Ny) * dy - Ly / 2)[None, :])
        
        self.kxy = (2*np.pi * np.fft.fftfreq(Nx, dx)[:, None],
                    2*np.pi * np.fft.fftfreq(Ny, dy)[None, :])
        self.dx = dx
        
        self.Nxy = tuple(Nxy)
        self.Lxy = tuple(Lxy)
        self.T = T

        # External potential
        self.v_ext = self.get_v_ext()
        
    def get_Ks(self, twist=(0, 0)):
        """Return the kinetic energy matrix."""
        k_bloch = np.divide(twist, self.Lxy)
        kxy = [_k + _kb for _k, _kb in zip(self.kxy, k_bloch)]

        mat_shape = (np.prod(self.Nxy),)*2
        tensor_shape = self.Nxy + self.Nxy
        K = np.eye(mat_shape[0]).reshape(tensor_shape)
        N = self.hbar**2/2/self.m   
        K = N * self.ifftn(
            sum(_k**2 for _k in self.kxy)
            [:, :,  None, None]*self.fftn(K)
        ).reshape(mat_shape)
        return (K, K)

    def fftn(self, y):
            return np.fft.fftn(y, axes=(0,1))
            
    def ifftn(self, y):
            return np.fft.ifftn(y, axes=(0,1))
            
    def get_v_ext(self):
        """Return the external potential."""
        return (0, 0)

    def f(self, E, E_c=None):
        """Return the Fermi-Dirac distribution at E."""
        if E_c is None:
            E_c = self.E_c
        if self.T > 0:
            f = 1./(1+np.exp(E/self.T))
        else:
            f = (1 - np.sign(E))/2
        return np.where(abs(E)<E_c, f, 0)

    def get_H(self, mus, delta, twist=(0,0)):
        """Return the single-particle Hamiltonian with pairing.

        Arguments
        ---------
        mus : array
           Effective chemical potentials including the Hartree term but not the
           external trapping potential.
        delta : array
           Pairing field (gap).
        phi_bloch : float
           Bloch phase.
        """
        zero = np.zeros_like(sum(self.xy))
        Delta = np.diag((delta + zero).ravel())
        v_a, v_b = self.v_ext
        mu_a, mu_b = mus
        mu_a += zero
        mu_b += zero
        K_a, K_b = self.get_Ks(twist=twist)
        Mu_a, Mu_b = np.diag((mu_a - v_a).ravel()), np.diag((mu_b - v_b).ravel())
        H = np.bmat([[K_a - Mu_a, -Delta],
                     [-Delta.conj(), -(K_b - Mu_b)]]) # H is 512 * 512?
        return np.asarray(H)

    def get_R(self, mus, delta, N_twist=1, twists=None):
        """Return the density matrix R."""
        N = self.Nxy
        Rs = []
        if twists is None:
            twists = itertools.product(
                (np.arange(0, N_twist)*2*np.pi/N_twist, )*2)

        for twist in twists:
            H = self.get_H(mus=mus, delta=delta, twist=twist)
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
            H = self.get_H(mus=mus, delta=delta, twist=twist)
            d, UV = np.linalg.eigh(H)
            R = UV.dot(self.f(d)[:, None]*UV.conj().T)
            return R/R0

        R0 = f(0)

        R = R0 * mquad(f, -np.pi, np.pi, abs_tol=abs_tol)/2/np.pi
        return R

    def get_LDA(mu_eff, delta):
        """Return the LDA solution"""
