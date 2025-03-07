"""BCS Equations in 1D, 2D, and 3D.

This module provides a class BCS for solving the BCS (BdG functional)
a two-species Fermi gas with short-range interactions.
"""
from collections import namedtuple
import itertools
import numpy
import numpy as np
from mmfutils.math.integrate import mquad
from mmf_hfb.parallel_helper import PoolHelper
from mmf_hfb.utils import block


def mqaud_worker_thread(obj_args):
    obj, vs, twists, k, args = obj_args
    k_p = obj.hbar**2/2/obj.m*k**2
    H = obj.get_H(vs=vs, k_p=k_p, twists=twists, **args)
    den = obj._get_densities_H(H, twists=twists)
    return den


def twising_worker_thread(obj_args):
    """"""
    obj, k_c, vs, twists, args = obj_args
    abs_tol=1e-6

    def f(k=0):
        k_p = obj.hbar**2/2/obj.m*k**2
        H = obj.get_H(vs=vs, k_p=k_p, twists=twists, **args)
        den = obj._get_densities_H(H, twists=twists)
        return den
    dens = mquad(f, -k_c, k_c, abs_tol=abs_tol)/2/np.pi
    return dens


class BCS(object):
    """Simple implementation of the BCS equations in a periodic box.

    We use all states in the box, regularizing the theory with a fixed
    coupling constant g_c which will depend on the box parameters.
    """
    hbar = 1.0
    m = 1.0

    def __init__(self, Nxyz=None, Lxyz=None, dx=None, T=0, E_c=None, Ec_Emax=None):
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
                Lxyz = numpy.multiply(Nxyz, dx)
            elif Nxyz is None:
                Nxyz = numpy.ceil(numpy.divide(Lxyz, dx)).astype(int)
        dxyz = numpy.divide(Lxyz, Nxyz)

        self.xyz = np.meshgrid(*[np.arange(_N) * _d - _L / 2
                                 for _N, _L, _d in zip(Nxyz, Lxyz, dxyz)],
                               indexing='ij')
        self.kxyz = np.meshgrid(*[2*np.pi*np.fft.fftfreq(_N, _d)
                                  for _N, _d in zip(Nxyz, dxyz)],
                                indexing='ij')

        self.dxyz = dxyz
        self.Nxyz = Nxyz
        self.Lxyz = Lxyz
        self.T = T
        #  should E_max has a factor of len(kxyz)?
        self.E_max = np.max([(self.hbar*_k)**2/2/self.m for _k in self.kxyz])
        if E_c is None and Ec_Emax is not None:
            E_c = Ec_Emax*self.E_max
        # the issue when E_c=None may cause problem.
        # But need to be address very carefully.
        if E_c is not None and E_c > self.E_max:
            raise ValueError("E_c must be no larger than E_max.")
        self.E_c = E_c

    @property
    def dim(self):
        return len(self.Nxyz)

    @property
    def dV(self):
        return numpy.prod(self.dxyz)

    @property
    def shape(self):
        return (2,) + (self.Nxyz)

    def erase_max_ks(self):
        """set the max abs(ks) to zero as they may cause problems"""
        self.max_ks = []
        for i in range(self.dim):
            self.max_ks.append(self.kxyz[i][self.Nxyz[i]//2])
            if self.Nxyz[i] % 2 == 0:
                self.kxyz[i][self.Nxyz[i]//2]=0

    def dotc(self, a, b):
        """Return dot(a.conj(), b) allowing for dim > 1."""
        return np.dot(a.conj().ravel(), b.ravel())

    def Normalize(self, psi, dx=None):
        """Normalize a wave function"""
        if dx is None:
            dx = self.dV
        psi_new = psi/(self.dotc(psi, psi)*dx)**0.5
        return psi_new.reshape(self.Nxyz)

    def restore_max_ks(self):
        """restore the original max ks"""
        for i in range(self.dim):
            if self.Nxyz[i] % 2 == 0 and self.kxyz[i][self.Nxyz[i]//2] == 0:
                self.kxyz[i][self.Nxyz[i]//2] = self.max_ks[i]

    def _get_twistss(self, N_twist):
        """return twistss for given twisting Number"""
        twistss = itertools.product(*(np.arange(0, N_twist)*2*np.pi/N_twist,)*self.dim)
        return list(twistss)

    def _get_modified_taus(self, taus, js):
        """
            return the modified taus with
            currents in it, not implement
        """
        return taus

    def _unpack_densities(self, dens, N_twist=1, struct=False):
        """
            unpack the densities into proper items
        ---------
        struct: bool
            if True, return a struct-like package
            if False, return  bare arrays
        N_twist: int
            the number of actual twisting used to
            average out the result
        """
        dens = dens/N_twist/self.dV
        if struct:
            n_a, n_b, tau_a, tau_b, nu_real, nu_imag = (dens[0:6])
            j_a, j_b = (dens[6:]).reshape((2, len(self.Nxyz)) + tuple(self.Nxyz))
            Densities = namedtuple(
                'Densities', ['n_a', 'n_b', 'tau_a', 'tau_b', 'nu', 'j_a', 'j_b'])
            return Densities(
                n_a=n_a, n_b=n_b,
                tau_a=tau_a, tau_b=tau_b,
                nu=nu_real + 1j*nu_imag,
                j_a=j_a, j_b=j_b)

        n_a, n_b, tau_a, tau_b, nu_real, nu_imag = (dens[0:6])
        nu = (nu_real+ 1j*nu_imag)
        js = (dens[6:]).reshape((2, len(self.Nxyz)) + tuple(self.Nxyz))
        tau_a, tau_b = self._get_modified_taus(taus=(tau_a, tau_b), js=js)
        return (np.array([n_a, n_b]), np.array([tau_a, tau_b]), js, nu)

    def _get_K(self, k_p=0, twists=0, **kw):
        """Return the kinetic energy matrix."""
        ks_bloch = numpy.divide(twists, self.Lxyz)
        ks = [_k + _kb for _k, _kb in zip(self.kxyz, ks_bloch)]

        # Here we use a simple trick of applying the FFT to an
        # identify matrix.  This ensures that we appropriately
        # calculate the matrix structure without having to worry about
        # indices and phase factors.  The following transformation
        # should be correct however:
        #
        # U = np.exp(-1j*k[:, None]*self.x[None, :])/self.Nx
        #
        mat_shape = (numpy.prod(self.Nxyz),) * 2
        tensor_shape = tuple(self.Nxyz) * 2

        K = np.eye(mat_shape[0]).reshape(tensor_shape)
        bcast = (slice(None),)*self.dim + (None,)*self.dim
        K = (
            self.hbar**2/2/self.m*self.ifft(
                sum(_k**2 for _k in ks)[bcast]*self.fft(K))).reshape(mat_shape)
        if not np.allclose(k_p, 0, rtol=1e-16):
            k_p = np.diag(np.ones_like(sum(self.xyz).ravel())*k_p)
            K = K + k_p
        return K

    def _get_Del(self, twists=0, **kw):
        """
            Return the first order derivative matrix
        """
        ks_bloch = numpy.divide(twists, self.Lxyz)
        ks = [_k + _kb for _k, _kb in zip(self.kxyz, ks_bloch)]
        mat_shape = (numpy.prod(self.Nxyz),)*2
        tensor_shape = tuple(self.Nxyz)*2

        K = np.eye(mat_shape[0]).reshape(tensor_shape)
        bcast = (slice(None),) * self.dim + (None,)*self.dim
        K = (
            self.hbar**2/2.0/self.m*self.ifft(
                1j*sum(_k for _k in ks)[bcast]*self.fft(K))).reshape(mat_shape)
        return K

    def _Del(self, alpha, twists=0):
        """
        Apply the Del, or Nabla operation on a function alpha
        -------
        Note:
            Here we compute the first derivatives and pack them so that
            the first component is the derivative in x, y, z, etc.
        Structure of alpha
        --------
            The alpha has shape of [spin=2, d1, d2,...dn, N]
            where first dim is the spin degree of freedom
            d1, d2..dn are sizes of a wavefunction in n dimemstion
            N is the number of wavefunctions
        """
        ks_bloch = numpy.divide(twists, self.Lxyz)
        ks = [_k + _kb for _k, _kb in zip(self.kxyz, ks_bloch)]
        axes = range(1, self.dim + 1)
        aplha_t = self.fft(alpha, axes=axes)
        return np.stack(
            [self.ifft(1j*_k[None, ..., None]*aplha_t, axes=axes) for _k in ks])

    def get_Ks(self, twists, **args):
        K = self._get_K(twists=twists, **args)
        return (K, K)

    def fft(self, y, axes=None):
        if axes is None:
            axes = range(self.dim)
        return np.fft.fftn(y, axes=axes)

    def ifft(self, y, axes=None):
        if axes is None:
            axes = range(self.dim)
        return np.fft.ifftn(y, axes=axes)

    def axes(self, bdg=True):
        if bdg:
            return range(1, self.dim + 1)
        return range(self.dim)

    def get_Vext(self, **kw):
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
        # these line implement step cutoff
        # this is very important for the case
        # when we integrate over another direction
        if E_c is None:
            return f
        mask = 0.5*(numpy.sign(abs(E_c)-abs(E)) + 1)
        return f*mask

    def _get_H(self, mu_eff, twists=0, V=0, **kw):
        K = self._get_K(twists=twists, **kw)
        mu_eff = np.zeros_like(sum(self.xyz)) + mu_eff
        return K - np.diag((mu_eff - V).ravel())

    def get_H(self, mus_eff, delta, twists=0, Vs=None, **kw):
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
        ---------
        NOTE: 
            For ASLDA, the Vs term shouod not be None
            and the mus_eff should be the bare mus
            because Vs here is treated as potential term
            that will be added to the bare mus to get
            the effective mus.
        """
        zero = np.zeros_like(sum(self.xyz))
        Delta = np.diag((delta + zero).ravel())
        K_a, K_b = self.get_Ks(twists=twists, **kw)
        if Vs is None:
            v_a, v_b = self.get_Vext(**kw)
        else:
            v_a, v_b = Vs
        mu_a, mu_b = mus_eff
        mu_a += zero
        mu_b += zero
        Mu_a, Mu_b = np.diag((mu_a - v_a).ravel()), np.diag((mu_b - v_b).ravel())
        H = block([[K_a - Mu_a, Delta],
                   [Delta.conj(), -(K_b - Mu_b)]])  # [check] delta^\dagger?
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
            Rp = UV.dot(self.f(d)[:, None]*UV.conj().T)
            Rm = UV.dot(self.f(-d)[:, None]*UV.conj().T)
            return np.stack([Rp, Rm])

        if np.isinf(N_twist):
            if self.dim == 1:
                Rp_Rm = mquad(get_Rs, -np.pi, np.pi, abs_tol=abs_tol)/2/np.pi
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

    def get_U_V(self, H, UV=None, transpose=False):
        """return U and V"""
        if UV is None:
            _, UV = np.linalg.eigh(H)
        U_V_shape = (2,) + tuple(self.Nxyz) + UV.shape[1:]
        U, V = UV.reshape(U_V_shape)
        if transpose:
            return (U.T, V.T)
        return (U, V)

    def _get_densities_H(self, H, dUV=None, twists=0):
        """return densities for a given H"""
        d, UV = np.linalg.eigh(H) if dUV is None else dUV
        U, V = U_V = self.get_U_V(H=H, UV=UV)
        dU_Vs = self._Del(U_V, twists=twists)
        dUs, dVs = dU_Vs[:, 0, ...], dU_Vs[:, 1, ...]
        f_p, f_m = self.f(d), self.f(-d)
        n_a = np.dot(U*U.conj(), f_p).real
        n_b = np.dot(V*V.conj(), f_m).real
        nu = np.dot(U*V.conj(), f_p - f_m)/2
        tau_a = np.dot(sum(dU.conj()*dU for dU in dUs), f_p).real
        tau_b = np.dot(sum(dV.conj()*dV for dV in dVs), f_m).real
        j_a = [0.5*np.dot((U.conj()*dU - U*dU.conj()), f_p).imag
               for dU in dUs]
        j_b = [0.5*np.dot((V*dV.conj() - V.conj()*dV), f_m).imag
               for dV in dVs]
        return np.stack([n_a, n_b, tau_a, tau_b, nu.real, nu.imag, *j_a, *j_b])

    def _get_densities(self, mus_eff, delta, twists, **args):
        """Return (R, Rm) with the specified twist."""
        H = self.get_H(mus_eff=mus_eff, delta=delta, twists=twists, **args)
        return self._get_densities_H(H, twists=twists)
    
    def get_angular_momentum_spectra(
            self, mus_eff, delta, N_twist=1,
            abs_tol=1e-12, **args):
        """
        return the angular momentum for 2D, 3D.
        Only consider the cases without twisting
        -------------
        NOTE: developing version, not tested yet
        """
        assert self.dim == 2 or self.dim == 3
        H = self.get_H(mus_eff=mus_eff, delta=delta, **args)
        d, uv = np.linalg.eigh(H)
        U, V = U_V = self.get_U_V(H=H, UV=uv)
        dU_Vs = self._Del(U_V)
        dUs, dVs = dU_Vs[:, 0, ...], dU_Vs[:, 1, ...]
        # f_p, f_m = self.f(d), self.f(-d)
        
        def get_L(f, df):
            """
            f is the wave function
            df is the first order derivative of f
            L_x = -i\hbar (y d/dz - z d/dy)
            L_y = -i\hbar (z d/dx - x d/dz)
            L_z = -i\hbar (x d/dy - y d/dx)
            """
            x, y = self.xyz[:2]
            Lz = f*x[:, :, None]*df[1] - f*y[:, :, None]*df[0]
            if self.dim == 2:
                Lx=Ly=Lz*0
            else:
                z = self.xyz[2]
                Lx = f*y[:, :, None]*df[2] - f*z[:, :, None]*df[1]
                Ly = f*z[:, :, None]*df[0] - f*x[:, :, None]*df[2]
            return -1j*self.hbar*np.array([Lx, Ly, Lz])
        L_u = get_L(U, dUs)
        L_v = get_L(V, dVs)
        Results = namedtuple("Results", ['es', 'L_u', 'L_v'])
        return Results(es=d, L_u=L_u, L_v=L_v)

    def get_densities(
            self, mus_eff, delta, N_twist=1,
            abs_tol=1e-12, unpack=True, struct=True, **args):
        """Return the densities.

        Arguments
        ---------
        N_twist : int, np.inf
           Number of twists to average over.  Integrate if
           N_twist==np.inf.
        abs_tol : float
           Absolute tolerance if performing twist averaging.
        unpack: bool
            unpack the result(array) to components(ns, js, taus, nu)
            if False, a stacked 1d array will be returned, this is
            useful for integration and iteration routines
        struct: bool
            specify if the way of unpack, if struct is True,
            the result would be organized a named structure,
            otherwise a list of component array will be returned
        """
        # Here we use the notation Rp = R = f(M) and Rm = f(-M) = 1-R
        def get_dens(twists):
            """Return (R, Rm) with the specified twist."""
            return self._get_densities(
                mus_eff=mus_eff, delta=delta, twists=twists, **args)

        if np.isinf(N_twist):
            if self.dim == 1:
                dens = mquad(get_dens, -np.pi, np.pi, abs_tol=abs_tol) / 2 / np.pi
            else:
                raise NotImplementedError("N_twist=inf only works for dim=1")
        else:
            twistss = self._get_twistss(N_twist=N_twist)
            N_=0
            dens = 0
            for twists in twistss:
                dens += get_dens(twists=twists)
                N_=N_+1
            dens = dens/N_
        if unpack:
            return self._unpack_densities(dens, struct=struct)
        return dens

    def get_entropy(self, mus_eff, delta, N_twist=1):
        """
        Return the entropy
        """
        raise NotImplementedError("Not implemented yet")

    def mquad_(self, obj_args, k_a, k_b, N_factor=10):
        """
            a mquad implementation, should be changed
            later for more general purpose
            ------------
            N_factor: integer
            a factor to increase density of Ks, larger value
            will make the spacing smaller and increase accuracy
            Note: need to test more carefully
        """
        obj, k_c, vs, twists, args = obj_args
        ks = 2*np.pi*np.fft.fftfreq(self.Nxyz[0]*N_factor, self.dxyz[0])
        obj_twists_kp = [(obj, vs, twists, k, args) for k in ks]
        res = PoolHelper.run(mqaud_worker_thread, paras=obj_twists_kp)
        dens = sum(res)/(obj.dxyz[0]*obj.Nxyz[0]*N_factor)
        return dens

    def get_dens_integral(
            self, mus_eff, delta, k_c=None,
            N_twist=1, unpack=True,
            struct=False, abs_tol=1e-6, **args):
        """
            integrate over another dimension by assuming it's homogeneous
            Note: the results are for dim + 1 system.
        """
        if k_c is None:
            k_c = np.sqrt(2*self.m*self.E_c)/self.hbar
        twistss = self._get_twistss(N_twist)
        args = dict(args, mus_eff=mus_eff, delta=delta)
        vs = self.get_Vext(**args)
        paras = [(self, k_c, vs, twists, args) for twists in twistss]
        res = PoolHelper.run(twising_worker_thread, paras=paras)
        dens = sum(res)/len(res)
        if unpack:
            return self._unpack_densities(dens, struct=struct)
        return dens

    def get_ns_e_p(self, mus, delta, **args):
        """
            compute then energy density
            Note: the return value also include the pressure and densities
        """
        ns, taus, _, kappa = self.get_densities(
            mus_eff=mus, delta=delta, struct=False, **args)
        g_eff = -delta/kappa
        energy_density = (
            taus[0] + taus[1])*self.hbar**2/2/self.m - g_eff*kappa.T.conj()*kappa
        pressure = ns[0]*mus[0] + ns[1]*mus[1] - energy_density
        return (ns, energy_density, pressure)


if __name__ == "__main__":
    b = BCS(Lxyz=(1, 1), Nxyz=(4, 4))
    b.get_angular_momentum(mus_eff=(1, 1), delta=5)