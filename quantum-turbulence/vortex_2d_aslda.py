"""BCS Equations in 2D

This module provides a class BCS2D for solving the BCS (BdG) equations in 2D for a
two-species Fermi gas with short-range interaction.
"""
from __future__ import division

import itertools

import numpy as np
import scipy.integrate

from mmfutils.math.integrate import mquad


class ASLDA(object):
    hbar = 1.0
    m = 1.0
    w = 1.0                     # Trapping potential

    def __init__(self, Nxy=(32, 32), Lxy=(10.0, 10.0), dx=None, T=0,E_c=np.inf):
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
        self.xy = ((np.arange(Nx) * dx - Lx / 2)[:, None], (np.arange(Ny) * dy - Ly / 2)[None, :])
        
        self.kxy = (2*np.pi * np.fft.fftfreq(Nx, dx)[:, None],2*np.pi * np.fft.fftfreq(Ny, dy)[None, :])
        self.dx = dx
        self.Nxy = tuple(Nxy)
        self.Lxy = tuple(Lxy)
        self.T = T
        self.E_c = E_c
        # External potential
        self.v_ext = self.get_v_ext()

    def get_nabla(self, twist=(0,0)):
        k_bloch = np.divide(twist, self.Lxy)
        kxy = [_k + _kb for _k, _kb in zip(self.kxy, k_bloch)]
        mat_shape = (np.prod(self.Nxy),)*2
        tensor_shape = self.Nxy + self.Nxy
        nabla = np.eye(mat_shape[0]).reshape(tensor_shape)
        N= self.hbar**2/2/self.m # the constant here should get some more check!!!
        nabla = N * self.ifft2(-1j*sum(_k  for _k in self.kxy)[:, :,  None, None]*self.fft2(nabla)).reshape((np.prod(self.Nxy),)*2).reshape(mat_shape)
        return nabla

    def get_Ks_Vs(self, ns=(0,0), twist=(0, 0)): 
        """Return the kinetic energy matrix."""
        k_bloch = np.divide(twist, self.Lxy)
        kxy = [_k + _kb for _k, _kb in zip(self.kxy, k_bloch)]

        alpha_a, alpha_b, alpha_p = self.get_alphas(ns)

        if type(alpha_a) != type(np.array):
            alpha_a = np.eye(self.Nxy[0]) * alpha_a # this will be problematic if Nx != Ny
            alpha_b = np.eye(self.Nxy[0]) * alpha_b
        nabla = self.get_nabla(twist)
        mat_shape = (np.prod(self.Nxy),)*2
        tensor_shape = self.Nxy + self.Nxy
        K = np.eye(mat_shape[0]).reshape(tensor_shape)
        K = self.ifft2(sum(_k**2 for _k in self.kxy)[:, :,  None, None]*self.fft2(K)).reshape((np.prod(self.Nxy),)*2).reshape(mat_shape) * self.hbar**2/2/self.m

        K_a = np.diag(nabla.dot(alpha_a.ravel())).dot(nabla) + np.diag(alpha_a.ravel()).dot(K)
        K_b = np.diag(nabla.dot(alpha_b.ravel())).dot(nabla) + np.diag(alpha_b.ravel()).dot(K)
        return (K_a, K_b)

    def fft2(self, y):
            return np.fft.fftn(y, axes=(0,1))
            
    def ifft2(self, y):
            return np.fft.ifftn(y, axes=(0,1))
            
    def get_v_ext(self):
        """Return the external potential."""
        return (0, 0)

    def get_ns(self, H):
        """Return the n_a, n_b"""
        E, psi = np.linalg.eigh(H)
        n = -3
        us, vs = psi[:, ].reshape(2, Nx, Ny)

        n_a, n_b = np.sum(u*u.conj() for u in us), np.sum(v*v.conj() for v in vs)
        return (n_a, n_b)

    def get_p(self, ns):
        # ns start with initialized value (0,0)
        if ns == (0,0):
            return 0
        n_a, n_b = ns
        n_p,n_m = n_a + n_b,n_a - n_b
        p = n_p / n_m # may be wrong
        return p

    def get_alphas(self,ns):
        p = self.get_p(ns)
        p2 = p**2
        p4 = p2**2
        # too make code faster, equation (98) is divived into even part and odd part
        alpha_even,alpha_odd = 1.0094 + 0.532 * p2 * (1 - p2 + p4 / 3.0), 0.156 * p * (1 - 2.0 * p2 / 3.0 + p4 / 5.0)
        alpha_a,alpha_b =  alpha_odd - alpha_even, -alpha_odd - alpha_even
        alpha_p = alpha_even # it's defined as (alpha_a + alpha_b) /2 , which is just alpha_even
        return (alpha_a,alpha_b,alpha_p)

    def get_functionals(self,ns):
        na, nb = ns
        alpha_a, alpha_b, alpha_p = get_alphas(ns)
        G = 0.357 + 0.642 * p2
        gamma = -11.11

        C = alpha_p * (n_a + n_b)**(1.0/3) / gamma
        D = (6 * np.pi**2) ** (5.0/3) / 20 / np.pi**2 *(G + alpha_a * ((1.0 + p)/2.0) ** (5.0/3) - alpha_b * ((1.0-p)/2.0)**(5.0/3))
        return (G, gamma, C, D)


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
        K_a, K_b = self.get_Ks_Vs(twist=twist)
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
