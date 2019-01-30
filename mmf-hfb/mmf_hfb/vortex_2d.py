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
    dim = 1
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
        
    def get_Ks(self, twists=(0,)):
        """Return the kinetic energy matrix."""
        k_bloch = twists[0]/self.Lx
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

    def get_H(self, mus, delta, twists=(0,)):
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
        K_a, K_b = self.get_Ks(twists=twists)
        v_a, v_b = self.v_ext
        mu_a, mu_b = mus
        mu_a += zero
        mu_b += zero
        Mu_a, Mu_b = np.diag((mu_a - v_a).ravel()), np.diag((mu_b - v_b).ravel())
        H = np.bmat([[K_a - Mu_a, Delta],
                     [Delta.conj(), -(K_b - Mu_b)]])
        return np.asarray(H)

    def get_Rs(self, mus, delta, N_twist=1, abs_tol=1e-12):
        """Return the density matrices (R, Rm).

        Arguments
        ---------
        N_twist : int, np.inf
           Number of twists to average over.  Integrate if
           N_twist==np.inf.
        abs_tol : float
           Absolute tolerance if performing twist averaging.

        Returns
        -------                
        R : array
           This is density matrix computed as R = f(M).  This can be
           used to extract n_a from the upper-left block.
        Rm : array
           This is (eye - R) = f(-M).  This can be used to extract n_b
           from the lower-right block.
        """
        # Here we use the notation Rp = R = f(M) and Rm = f(-M) = 1-R
        def get_Rs(twists):
            """Return (R, Rm) with the specified twist."""
            H = self.get_H(mus=mus, delta=delta, twists=twists)
            d, UV = np.linalg.eigh(H)
            Rp = UV.dot(self.f(d)[:, None]*UV.conj().T)
            Rm = UV.dot(self.f(-d)[:, None]*UV.conj().T)
            return np.array([Rp, Rm])
            
        if np.isinf(N_twist):
            Rp_Rm = mquad(get_Rs, -np.pi/2, np.pi/2, abs_tol=abs_tol)
        else:
            twists = np.arange(0, N_twist)*2*np.pi/N_twist
            Rp_Rm = 0
            for twist in twists:
                Rp_Rm += get_Rs(twists=(twist,))
            Rp_Rm /= len(twists)

        # Factor of dV here to convert to physical densities.
        dV = self.dx
        return Rp_Rm / dV

    def get_densities(self, mus, delta, N_twist=1):
        R, Rm = self.get_Rs(mus=mus, delta=delta, N_twist=N_twist)
        _N = R.shape[0] // 2
        r_a = R[:_N, :_N]
        r_b = Rm[_N:, _N:].conj()
        nu_ = (R[:_N, _N:] - Rm[_N:, :_N].T.conj())/2.0
        n_a = np.diag(r_a).real
        n_b = np.diag(r_b).real
        nu = np.diag(nu_)
        return namedtuple('Densities', ['n_a', 'n_b', 'nu'])(n_a, n_b, nu)


class BCS_2D(BCS_1D):
    """Simple implementation of the BCS equations in a periodic square
    box.

    We use all states in the box, regularizing the theory with a fixed
    coupling constant g_c which will depend on the box parameters.
    """
    dim = 2
    
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
        if dx is not None:
            if Lxy is None:
                Lxy = np.prod(Nxy, dx)
            elif Nxy is None:
                Nxy = np.ceil(Lxy / dx).astype(int)
        dxy = np.divide(Lxy, Nxy)

        Nx, Ny = Nxy
        Lx, Ly = Lxy
        self.xy = ((np.arange(Nx) * dxy[0] - Lx / 2)[:, None],# half of the length
                   (np.arange(Ny) * dxy[1] - Ly / 2)[None, :])
        
        self.kxy = (2*np.pi * np.fft.fftfreq(Nx, dxy[0])[:, None],
                    2*np.pi * np.fft.fftfreq(Ny, dxy[1])[None, :])
        self.dxy = dxy
        
        self.Nxy = tuple(Nxy)
        self.Lxy = tuple(Lxy)
        self.T = T

        # External potential
        self.v_ext = self.get_v_ext()
        
    def get_Ks(self, twists=(0, 0)):
        """Return the kinetic energy matrix."""
        k_blochs = np.divide(twists, self.Lxy)
        ks = [_k + _kb for _k, _kb in zip(self.kxy, k_blochs)]

        
        # Here we use a simple trick of applying the FFT to an
        # identify matrix.  This ensures that we appropriately
        # calculate the matrix structure without having to worry about
        # indices and phase factors.
        mat_shape = (np.prod(self.Nxy),)*2
        tensor_shape = self.Nxy + self.Nxy
        K = np.eye(mat_shape[0]).reshape(tensor_shape)
        K = (self.hbar**2/2/self.m
             * self.ifftn(sum(_k**2 for _k in ks)[:, :,  None, None]
                          * self.fftn(K))).reshape(mat_shape)
        return (K, K)

    def fftn(self, y):
            return np.fft.fftn(y, axes=(0, 1))
            
    def ifftn(self, y):
            return np.fft.ifftn(y, axes=(0, 1))
            
    def get_H(self, mus, delta, twists=(0,0)):
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
        K_a, K_b = self.get_Ks(twists=twists)
        v_a, v_b = self.v_ext
        mu_a, mu_b = mus
        mu_a += zero
        mu_b += zero
        Mu_a, Mu_b = np.diag((mu_a - v_a).ravel()), np.diag((mu_b - v_b).ravel())
        H = np.bmat([[K_a - Mu_a, -Delta],
                     [-Delta.conj(), -(K_b - Mu_b)]])
        return np.asarray(H)

    def get_Rs(self, mus, delta, N_twist=1, abs_tol=1e-12):
        """Return the density matrices (R, Rm).

        Arguments
        ---------
        N_twist : int, np.inf
           Number of twists to average over.  Integrate if
           N_twist==np.inf.
        abs_tol : float
           Absolute tolerance if performing twist averaging.

        Returns
        -------                
        R : array
           This is density matrix computed as R = f(M).  This can be
           used to extract n_a from the upper-left block.
        Rm : array
           This is (eye - R) = f(-M).  This can be used to extract n_b
           from the lower-right block.
        """
        # Here we use the notation Rp = R = f(M) and Rm = f(-M) = 1-R
        def get_Rs(twists):
            """Return (R, Rm) with the specified twist."""
            H = self.get_H(mus=mus, delta=delta, twists=twists)
            d, UV = np.linalg.eigh(H)
            Rp = UV.dot(self.f(d)[:, None]*UV.conj().T)
            Rm = UV.dot(self.f(-d)[:, None]*UV.conj().T)
            return np.array([Rp, Rm])
            
        if np.isinf(N_twist):
            Rp_Rm = mquad(get_Rs, -np.pi/2, np.pi/2, abs_tol=abs_tol)
        else:
            twistss = itertools.product(
                *(np.arange(0, N_twist)*2*np.pi/N_twist,)*self.dim)
            
            Rp_Rm = 0
            _N = 0
            for twists in twistss:
                Rp_Rm += get_Rs(twists)
                _N += 1
            Rp_Rm /= _N

        # Factor of dV here to convert to physical densities.
        dV = np.prod(self.dxy)
        return Rp_Rm / dV

    def get_densities(self, mus, delta, N_twist=1):
        R, Rm = self.get_Rs(mus=mus, delta=delta, N_twist=N_twist)
        _N = R.shape[0] // 2
        r_a = R[:_N, :_N]
        r_b = Rm[_N:, _N:].conj()
        nu_ = (R[:_N, _N:] - Rm[_N:, :_N].T.conj())/2.0
        n_a = np.diag(r_a).reshape(self.Nxy).real
        n_b = np.diag(r_b).reshape(self.Nxy).real
        nu = np.diag(nu_).reshape(self.Nxy)
        return namedtuple('Densities', ['n_a', 'n_b', 'nu'])(n_a, n_b, nu)


class BCS_3D(BCS_1D):
    """Simple implementation of the BCS equations in a periodic square
    box.

    We use all states in the box, regularizing the theory with a fixed
    coupling constant g_c which will depend on the box parameters.
    """
    def __init__(self, Nxyz=(32, 32, 32), Lxyz=(10.0, 10.0, 10.0), dx=None, T=0):
        """Specify any two of `Nxyz`, `Lxyz`, or `dx`.

        Arguments
        ---------
        Nxyz : (int, int, int)
           Number of lattice points.
        Lxyz : (float, float, float)
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
        if dx is not None:
            if Lxyz is None:
                Lxyz = np.prod(Nxyz, dx)
            elif Nxyz is None:
                Nxyz = np.ceil(Lxyz / dx).astype(int)
        dxyz = np.divide(Lxyz, Nxyz)

        Nx, Ny, Nz = Nxyz
        Lx, Ly, Nz = Lxyz
        self.xyz = np.meshgrid(*[np.arange(_N) * _d - _L / 2
                                 for _N, _L, _d in zip(Nxyz, Lxyz, dxyz)],
                               indexing='ij', sparse=True)        
        self.kxyz = np.meshgrid(*[2*np.pi * np.fft.fftfreq(_N, _d)
                                  for _N, _d in zip(Nxyz, dxyz)],
                                indexing='ij', sparse=True)
        self.dxyz = dxyz
        
        self.Nxyz = tuple(Nxyz)
        self.Lxyz = tuple(Lxyz)
        self.T = T
        self.E_c = 100
        # External potential
        self.v_ext = self.get_v_ext()

    @property
    def dim(self):
        return len(self.Nxyz)
    
    def get_Ks(self, twists=(0, 0, 0)):
        """Return the kinetic energy matrix."""
        k_blochs = np.divide(twists, self.Lxyz)
        ks = [_k + _kb for _k, _kb in zip(self.kxyz, k_blochs)]

        
        # Here we use a simple trick of applying the FFT to an
        # identify matrix.  This ensures that we appropriately
        # calculate the matrix structure without having to worry about
        # indices and phase factors.
        mat_shape = (np.prod(self.Nxyz),)*2
        tensor_shape = self.Nxyz + self.Nxyz
        K = np.eye(mat_shape[0]).reshape(tensor_shape)
        K = (self.hbar**2/2/self.m
             * self.ifftn(sum(_k**2 for _k in ks)[:, :, :, None, None, None]
                          * self.fftn(K))).reshape(mat_shape)
        return (K, K)

    def fftn(self, y):
            return np.fft.fftn(y, axes=(0, 1, 2))
            
    def ifftn(self, y):
            return np.fft.ifftn(y, axes=(0, 1, 2))
            
    def get_H(self, mus, delta, twists=(0,0,0)):
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
        zero = np.zeros_like(sum(self.xyz))
        Delta = np.diag((delta + zero).ravel())
        K_a, K_b = self.get_Ks(twists=twists)
        v_a, v_b = self.v_ext
        mu_a, mu_b = mus
        mu_a += zero
        mu_b += zero
        Mu_a, Mu_b = np.diag((mu_a - v_a).ravel()), np.diag((mu_b - v_b).ravel())
        H = np.bmat([[K_a - Mu_a, -Delta],
                     [-Delta.conj(), -(K_b - Mu_b)]])
        return np.asarray(H)

    def get_Rs(self, mus, delta, N_twist=1, abs_tol=1e-12):
        """Return the density matrices (R, Rm).

        Arguments
        ---------
        N_twist : int, np.inf
           Number of twists to average over.  Integrate if
           N_twist==np.inf.
        abs_tol : float
           Absolute tolerance if performing twist averaging.

        Returns
        -------                
        R : array
           This is density matrix computed as R = f(M).  This can be
           used to extract n_a from the upper-left block.
        Rm : array
           This is (eye - R) = f(-M).  This can be used to extract n_b
           from the lower-right block.
        """
        # Here we use the notation Rp = R = f(M) and Rm = f(-M) = 1-R
        def get_Rs(twists):
            """Return (R, Rm) with the specified twist."""
            H = self.get_H(mus=mus, delta=delta, twists=twists)
            d, UV = np.linalg.eigh(H)
            Rp = UV.dot(self.f(d)[:, None]*UV.conj().T)
            Rm = UV.dot(self.f(-d)[:, None]*UV.conj().T)
            return np.array([Rp, Rm])
            
        if np.isinf(N_twist):
            Rp_Rm = mquad(get_Rs, -np.pi/2, np.pi/2, abs_tol=abs_tol)
        else:
            twistss = itertools.product(
                *(np.arange(0, N_twist)*2*np.pi/N_twist,)*self.dim)
            
            Rp_Rm = 0
            _N = 0
            for twists in twistss:
                Rp_Rm += get_Rs(twists)
                _N += 1
            Rp_Rm /= _N

        # Factor of dV here to convert to physical densities.
        dV = np.prod(self.dxyz)
        return Rp_Rm / dV

    def get_densities(self, mus, delta, N_twist=1):
        R, Rm = self.get_Rs(mus=mus, delta=delta, N_twist=N_twist)
        _N = R.shape[0] // 2
        r_a = R[:_N, :_N]
        r_b = Rm[_N:, _N:].conj()
        nu_ = (R[:_N, _N:] - Rm[_N:, :_N].T.conj())/2.0
        n_a = np.diag(r_a).reshape(self.Nxyz).real
        n_b = np.diag(r_b).reshape(self.Nxyz).real
        nu = np.diag(nu_).reshape(self.Nxyz)
        return namedtuple('Densities', ['n_a', 'n_b', 'nu'])(n_a, n_b, nu)
