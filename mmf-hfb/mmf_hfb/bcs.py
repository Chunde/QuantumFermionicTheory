"""BCS Equations in 1D, 2D, and 3D.

This module provides a class BCS for solving the BCS (BdG functional)
a two-species Fermi gas with short-range interactions.
"""
from __future__ import division

from collections import namedtuple
import itertools

import numpy as np

from mmfutils.math.integrate import mquad


class BCS(object):
    """Simple implementation of the BCS equations in a periodic box.

    We use all states in the box, regularizing the theory with a fixed
    coupling constant g_c which will depend on the box parameters.
    """
    hbar = 1.0
    m = 1.0

    def __init__(self, Nxyz=None, Lxyz=None, dx=None, T=0, E_c=100):
        """Specify any two of `Nxyz`, `Lxyz`, or `dx`.

        Arguments
        ---------
        Nxyz : (int, int, ...)
           Number of lattice points.  The length specifies the dimension.
        Lxyz : (float, float, ...)
           Length of the periodic box.
           Can also be understood as the largest wavelength of
           possible waves host in the box. Then the minimum
           wave-vector k0 = 2*pi/lambda = 2*pi/L, and all
           possible ks should be integer times of k0.
        dx : float
           Lattice spacing.
        T : float
           Temperature.
        """
        if dx is not None:
            if Lxyz is None:
                Lxyz = np.multiply(Nxyz, dx)
            elif Nxyz is None:
                Nxyz = np.ceil(np.divide(Lxyz, dx)).astype(int)
        dxyz = np.divide(Lxyz, Nxyz)

        self.xyz = np.meshgrid(*[np.arange(_N) * _d - _L / 2
                                 for _N, _L, _d in zip(Nxyz, Lxyz, dxyz)],
                               indexing='ij', sparse=True)
        self.kxyz = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(_N, _d)
                                  for _N, _d in zip(Nxyz, dxyz)],
                                indexing='ij', sparse=True)

        self.dxyz = dxyz
        self.Nxyz = Nxyz
        self.Lxyz = Lxyz

        self.E_c = E_c
        self.T = T

        # External potential
        self.v_ext = self.get_v_ext()

    @property
    def dim(self):
        return len(self.Nxyz)

    @property
    def dV(self):
        return np.prod(self.dxyz)

    def _get_twistss(self, N_twist):
        """return twistss for given twisting Number"""
        twistss = itertools.product(
                *(np.arange(0, N_twist) * 2 * np.pi / N_twist,) * self.dim)
        return list(twistss)

    def _get_K(self, twists=0):
        """Return the kinetic energy matrix."""
        ks_bloch = np.divide(twists, self.Lxyz)
        ks = [_k + _kb for _k, _kb in zip(self.kxyz, ks_bloch)]

        # Here we use a simple trick of applying the FFT to an
        # identify matrix.  This ensures that we appropriately
        # calculate the matrix structure without having to worry about
        # indices and phase factors.  The following transformation
        # should be correct however:
        #
        # U = np.exp(-1j*k[:, None]*self.x[None, :])/self.Nx
        #
        mat_shape = (np.prod(self.Nxyz),) * 2
        tensor_shape = tuple(self.Nxyz) * 2

        K = np.eye(mat_shape[0]).reshape(tensor_shape)
        bcast = (slice(None),) * self.dim + (None,) * self.dim
        K = (self.hbar**2 / 2 / self.m
             * self.ifft(sum(_k**2 for _k in ks)[bcast] *
                       self.fft(K))).reshape(mat_shape)
        return K

    def get_Ks(self, twists):
        K = self._get_K(twists)
        return (K, K)

    def fft(self, y, axes=None):
        if axes is None:
            axes = range(self.dim)
        return np.fft.fftn(y, axes=axes)

    def ifft(self, y, axes=None):
        if axes is None:
            axes = range(self.dim)
        return np.fft.ifftn(y, axes=axes)

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
        return f

    def get_H(self, mus_eff, delta, twists=0):
        """Return the single-particle Hamiltonian with pairing.

        Arguments
        ---------
        mus_eff : array
           Effective chemical potentials including the Hartree term but not the
           external trapping potential.
        delta : array
           Pairing field (gap).
        twist : float
           Bloch phase.
        """
        zero = np.zeros_like(sum(self.xyz))
        Delta = np.diag((delta + zero).ravel())
        K_a, K_b = self.get_Ks(twists=twists)
        v_a, v_b = self.v_ext
        mu_a, mu_b = mus_eff
        mu_a += zero
        mu_b += zero
        Mu_a, Mu_b = np.diag((mu_a - v_a).ravel()), np.diag((mu_b - v_b).ravel())
        H = np.block([[K_a - Mu_a, Delta],
                      [Delta.conj(), -(K_b - Mu_b)]])
        return H

    def get_R(self, mus_eff, delta, N_twist=1, twists=None):
        """Return the density matrix R."""
        Rs = []
        if twists is None:
            twists = np.arange(0, N_twist)*2*np.pi/N_twist

        for twist in twists:
            H = self.get_H(mus_eff=mus_eff, delta=delta, twists=twist)
            d, UV = np.linalg.eigh(H)
            R = UV.dot(self.f(d)[:, None]*UV.conj().T)
            Rs.append(R)
        R = sum(Rs)/len(Rs)
        return R

    def get_Rs(self, mus_eff, delta, N_twist=1, abs_tol=1e-12):
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
        Note on return value scaling problem changed:
                # Factor of dV here to convert to physical densities.
                # this may cause problem when compute densities:
                # as    na = np.diag(R)[:N]
                # while nb = (1 - np.diag(R)[N:]* dV)/dV
                # dV = np.prod(self.dxyz)
                #return Rp_Rm / dV

        """
        # Here we use the notation Rp = R = f(M) and Rm = f(-M) = 1-R
        def get_Rs(twists):
            """Return (R, Rm) with the specified twist."""
            H = self.get_H(mus_eff=mus_eff, delta=delta, twists=twists)
            d, UV = np.linalg.eigh(H)
            Rp = UV.dot(self.f(d)[:, None] * UV.conj().T)
            Rm = UV.dot(self.f(-d)[:, None] * UV.conj().T)
            return np.array([Rp, Rm])

        if np.isinf(N_twist):
            if self.dim == 1:
                Rp_Rm = mquad(get_Rs, -np.pi, np.pi, abs_tol=abs_tol) / 2 / np.pi
            else:
                raise NotImplementedError("N_twist=inf only works for dim=1")
        else:
            twistss = self._get_twistss(N_twist)
            Rp_Rm = 0
            N_ = 0
            for twists in twistss:
                Rp_Rm += get_Rs(twists=twists)
                N_ += 1
            Rp_Rm /= N_
        return Rp_Rm

    def get_densities_R(self, mus_eff, delta, N_twist=1):
        """Get the densities from R."""
        R, Rm = self.get_Rs(mus_eff=mus_eff, delta=delta, N_twist=N_twist)/self.dV
        _N = R.shape[0] // 2
        r_a = R[:_N, :_N]
        r_b = Rm[_N:, _N:].conj()
        nu_ = (R[:_N, _N:] - Rm[_N:, :_N].T.conj()) / 2.0
        n_a = np.diag(r_a).reshape(self.Nxyz).real
        n_b = np.diag(r_b).reshape(self.Nxyz).real
        nu = np.diag(nu_).reshape(self.Nxyz)
        return namedtuple('Densities', ['n_a', 'n_b', 'nu'])(n_a, n_b, nu)

    def get_1d_currents(self, mus_eff, delta, N_twist=1):
        """return current for 1d only"""
        twistss = itertools.product(
            *(np.arange(0, N_twist) * 2 * np.pi / N_twist,) * self.dim)
        j_a = 0
        j_b = 0
        # np.fft.ifft(1j*k*np.fft.fft(f))

        def df(k, f):
            return np.fft.ifft(1j * k * np.fft.fft(f))

        for twists in twistss:
            ks_bloch = np.divide(twists, self.Lxyz)
            k = [_k + _kb for _k, _kb in zip(self.kxyz, ks_bloch)][0]

            H = self.get_H(mus_eff=mus_eff, delta=delta, twists=twists)
            N = self.Nxyz[0]
            d, psi = np.linalg.eigh(H)
            us, vs = psi.reshape(2, N, N * 2)
            us, vs = us.T, vs.T
            j_a_ = -0.5j * sum((us[i].conj() * df(k, us[i]) - us[i] *
                                df(k, us[i]).conj()) * self.f(d[i]) for i in range(len(us)))
            j_b_ = -0.5j * sum((vs[i] * df(k, vs[i]).conj() - vs[i].conj() *
                                df(k, vs[i])) * self.f(-d[i]) for i in range(len(vs)))
            j_a = j_a + j_a_
            j_b = j_b + j_b_
        return (j_a / N_twist / np.prod(self.dxyz), j_b / N_twist / np.prod(self.dxyz))

    def _get_densities_H(self, H, twists):
        """return densities for a given H"""
        d, UV = np.linalg.eigh(H)

        U_V_shape = (2,) + tuple(self.Nxyz) + UV.shape[1:]
        U, V = U_V = UV.reshape(U_V_shape)  # U = us.T, V=vs.T

        # Compute derivatives for currents etc.
        ks_bloch = np.divide(twists, self.Lxyz)
        ks = [_k + _kb for _k, _kb in zip(self.kxyz, ks_bloch)]

        axes = range(1, self.dim + 1)
        U_V_t = self.fft(U_V, axes=axes)

        # Here we compute the derivatives and pack them so that
        # the first component is the derivative in x, y, z, etc.
        dU_Vs = np.array([self.ifft(1j * _k[None, ..., None] * U_V_t, axes=axes) for _k in ks])
        dUs, dVs = dU_Vs[:, 0, ...], dU_Vs[:, 1, ...]
        f_p = self.f(d)
        f_m = self.f(-d)
        n_a = np.dot(U * U.conj(), f_p).real
        n_b = np.dot(V * V.conj(), f_m).real
        nu = np.dot(U * V.conj(), f_p - f_m) / 2
        tau_a = np.dot(sum(dU.conj() * dU for dU in dUs), f_p).real
        tau_b = np.dot(sum(dV.conj() * dV for dV in dVs), f_m).real

        j_a = [0.5 * np.dot((U.conj() * dU - U * dU.conj()), f_p).imag
               for dU in dUs]
        j_b = [0.5 * np.dot((V * dV.conj() - V.conj() * dV), f_m).imag
               for dV in dVs]
        return np.array([n_a, n_b, tau_a, tau_b, nu.real, nu.imag, *j_a, *j_b])

    def _get_densities(self, mus_eff, delta, twists):
        """Return (R, Rm) with the specified twist."""
        H = self.get_H(mus_eff=mus_eff, delta=delta, twists=twists)
        return self._get_densities_H(H, twists=twists)

    def get_densities(self, mus_eff, delta, N_twist=1, abs_tol=1e-12):
        """Return the densities.

        Arguments
        ---------
        N_twist : int, np.inf
           Number of twists to average over.  Integrate if
           N_twist==np.inf.
        abs_tol : float
           Absolute tolerance if performing twist averaging.
        """
        # Here we use the notation Rp = R = f(M) and Rm = f(-M) = 1-R
        def get_dens(twists):
            """Return (R, Rm) with the specified twist."""
            return self._get_densities(mus_eff=mus_eff, delta=delta, twists=twists)

        if np.isinf(N_twist):
            if self.dim == 1:
                dens = mquad(get_dens, -np.pi, np.pi, abs_tol=abs_tol) / 2 / np.pi
            else:
                raise NotImplementedError("N_twist=inf only works for dim=1")
        else:
            twistss = itertools.product(
                *(np.arange(0, N_twist) * 2 * np.pi / N_twist,) * self.dim)
            dens = 0
            N_ = 0
            for twists in twistss:
                dens += get_dens(twists=twists)
                N_ += 1
            dens /= N_

        # Factor of dV here to convert to physical densities.
        dV = np.prod(self.dxyz)

        n_a, n_b, tau_a, tau_b, nu_real, nu_imag = (dens[0:6] / dV)
        j_a, j_b = (dens[6:] / dV).reshape((2, len(self.Nxyz)) + tuple(self.Nxyz))
        Densities = namedtuple('Densities',
                               ['n_a', 'n_b', 'tau_a', 'tau_b', 'nu', 'j_a', 'j_b'])
        return Densities(n_a=n_a, n_b=n_b,
                         tau_a=tau_a, tau_b=tau_b,
                         nu=nu_real + 1j * nu_imag,
                         j_a=j_a, j_b=j_b)
