"""BCS Equations in 1D, 2D, and 3D.

This module provides a class BCS for solving the BCS (BdG functional)
a two-species Fermi gas with short-range interactions.
"""
from collections import namedtuple
import itertools
import numpy
from mmf_hfb.xp import xp
from mmfutils.math.integrate import mquad
from mmf_hfb.ParallelHelper import PoolHelper

class BCS(object):
    """Simple implementation of the BCS equations in a periodic box.

    We use all states in the box, regularizing the theory with a fixed
    coupling constant g_c which will depend on the box parameters.
    """
    hbar = 1.0
    m = 1.0

    def __init__(self, Nxyz=None, Lxyz=None, dx=None, T=0, E_c=None):
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

        self.xyz = xp.meshgrid(*[xp.arange(_N) * _d - _L / 2
                                 for _N, _L, _d in zip(Nxyz, Lxyz, dxyz)],
                               indexing='ij')
        self.kxyz = xp.meshgrid(*[2 * xp.pi * xp.fft.fftfreq(_N, _d)
                                  for _N, _d in zip(Nxyz, dxyz)],
                                indexing='ij')

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
        return numpy.prod(self.dxyz)

    def _get_twistss(self, N_twist):
        """return twistss for given twisting Number"""
        twistss = itertools.product(
                *(xp.arange(0, N_twist) * 2 * xp.pi / N_twist,) * self.dim)
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
            the number of actual twisings used to
            average out the result
        """
        dens = dens/N_twist/self.dV
        if struct:
            n_a, n_b, tau_a, tau_b, nu_real, nu_imag = (dens[0:6])
            j_a, j_b = (dens[6:] ).reshape((2, len(self.Nxyz)) + tuple(self.Nxyz))
            Densities = namedtuple('Densities',
                                ['n_a', 'n_b', 'tau_a', 'tau_b', 'nu', 'j_a', 'j_b'])
            return Densities(n_a=n_a, n_b=n_b,
                            tau_a=tau_a, tau_b=tau_b,
                            nu=nu_real + 1j * nu_imag,
                            j_a=j_a, j_b=j_b)

        n_a, n_b, tau_a, tau_b, nu_real, nu_imag = (dens[0:6])
        kappa_ = (nu_real+ 1j*nu_imag)
        js = (dens[6:]).reshape((2, len(self.Nxyz)) + tuple(self.Nxyz))
        tau_a, tau_b = self._get_modified_taus(taus=(tau_a, tau_b), js=js)
        return (xp.array([n_a, n_b]), xp.array([tau_a, tau_b]), js, kappa_)

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
        # U = xp.exp(-1j*k[:, None]*self.x[None, :])/self.Nx
        #
        mat_shape = (numpy.prod(self.Nxyz),) * 2
        tensor_shape = tuple(self.Nxyz) * 2

        K = xp.eye(mat_shape[0]).reshape(tensor_shape)
        bcast = (slice(None),) * self.dim + (None,) * self.dim
        K = (self.hbar**2 / 2 / self.m
             * self.ifft(sum(_k**2 for _k in ks)[bcast] *
                       self.fft(K))).reshape(mat_shape)
        if not xp.allclose(k_p, 0, rtol=1e-16):
            k_p = xp.diag(xp.ones_like(sum(self.xyz).ravel()) * k_p) 
            K = K + k_p
        return K

    def _get_Del(self, twists=0, **kw):
        """
            Return the first order derivative matrix
        """
        ks_bloch = numpy.divide(twists, self.Lxyz)
        ks = [_k + _kb for _k, _kb in zip(self.kxyz, ks_bloch)]
        mat_shape = (numpy.prod(self.Nxyz),) * 2
        tensor_shape = tuple(self.Nxyz) * 2

        K = xp.eye(mat_shape[0]).reshape(tensor_shape)
        bcast = (slice(None),) * self.dim + (None,) * self.dim
        K = (self.hbar**2 / 2 / self.m
             * self.ifft(1j*sum(_k for _k in ks)[bcast] *
                       self.fft(K))).reshape(mat_shape)
        return K

    def _Del(self, aplha, twists=0):
        """
        Apply the Del, or Nabla operation on a function alpha
        -------
        Note:
            Here we compute the first derivatives and pack them so that
            the first component is the derivative in x, y, z, etc.
        """
        ks_bloch = numpy.divide(twists, self.Lxyz)
        ks = [_k + _kb for _k, _kb in zip(self.kxyz, ks_bloch)]
        axes = range(1, self.dim + 1)
        aplha_t = self.fft(aplha, axes=axes)
        return xp.stack([self.ifft(1j*_k[None, ..., None]*aplha_t, axes=axes) for _k in ks])

    def get_Ks(self, twists, **args):
        K = self._get_K(twists=twists, **args)
        return (K, K)

    def fft(self, y, axes=None):
        if axes is None:
            axes = range(self.dim)
        return xp.fft.fftn(y, axes=axes)

    def ifft(self, y, axes=None):
        if axes is None:
            axes = range(self.dim)
        return xp.fft.ifftn(y, axes=axes)

    def get_v_ext(self, **kw):
        """Return the external potential."""
        return (0, 0)

    def f(self, E, E_c=None):
        """Return the Fermi-Dirac distribution at E."""
        if E_c is None:
            E_c = self.E_c
        
        if self.T > 0:
            f = 1./(1+xp.exp(E/self.T))
        else:
            f = (1 - xp.sign(E))/2
        # these line implement step cutoff
        # this is very important for the case
        # when we integrate over another direction
        if E_c is None:
            return f
        mask = 0.5 * (numpy.sign(abs(E_c)-abs(E)) + 1)
        return f * mask
        return f

    def block(a11, a12, a21, a22):
        RowBlock1=xp.concatenate((a11,  a12), axis=1)
        RowBlock2=xp.concatenate((a21, a22), axis=1)
        Block=xp.concatenate((RowBlock1, RowBlock2), axis=0)
        return Block

    def get_H(self, mus_eff, delta, twists=0, vs=None, **kw):
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
        zero = xp.zeros_like(sum(self.xyz))
        Delta = xp.diag((delta + zero).ravel())
        K_a, K_b = self.get_Ks(twists=twists, **kw)
        if vs is None:
            v_a, v_b = self.get_v_ext(**kw)
        else:
            v_a, v_b = vs
        mu_a, mu_b = mus_eff
        mu_a += zero
        mu_b += zero
        Mu_a, Mu_b = xp.diag((mu_a - v_a).ravel()), xp.diag((mu_b - v_b).ravel())
        #H = xp.block([[K_a - Mu_a, Delta],
        #              [Delta.conj(), -(K_b - Mu_b)]])
        H = BCS.block(K_a-Mu_a, Delta, Delta.conj(), -(K_b - Mu_b))
        return H

    def get_R(self, mus_eff, delta, N_twist=1, twists=None):
        """Return the density matrix R."""
        Rs = []
        if twists is None:
            twists = xp.arange(0, N_twist)*2*xp.pi/N_twist

        for twist in twists:
            H = self.get_H(mus_eff=mus_eff, delta=delta, twists=twist)
            d, UV = xp.linalg.eigh(H)
            R = UV.dot(self.f(d)[:, None]*UV.conj().T)
            Rs.append(R)
        R = sum(Rs)/len(Rs)
        return R

    def get_Rs(self, mus_eff, delta, N_twist=1, abs_tol=1e-12):
        """Return the density matrices (R, Rm).

        Arguments
        ---------
        N_twist : int, xp.inf
           Number of twists to average over.  Integrate if
           N_twist==xp.inf.
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
                # as    na = xp.diag(R)[:N]
                # while nb = (1 - xp.diag(R)[N:]* dV)/dV
                # dV = xp.prod(self.dxyz)
                #return Rp_Rm / dV

        """
        # Here we use the notation Rp = R = f(M) and Rm = f(-M) = 1-R
        def get_Rs(twists):
            """Return (R, Rm) with the specified twist."""
            H = self.get_H(mus_eff=mus_eff, delta=delta, twists=twists)
            d, UV = xp.linalg.eigh(H)
            Rp = UV.dot(self.f(d)[:, None] * UV.conj().T)
            Rm = UV.dot(self.f(-d)[:, None] * UV.conj().T)
            return xp.stack([Rp, Rm])

        if xp.isinf(N_twist):
            if self.dim == 1:
                Rp_Rm = mquad(get_Rs, -xp.pi, xp.pi, abs_tol=abs_tol) / 2 / xp.pi
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
        n_a = xp.diag(r_a).reshape(self.Nxyz).real
        n_b = xp.diag(r_b).reshape(self.Nxyz).real
        nu = xp.diag(nu_).reshape(self.Nxyz)
        return namedtuple('Densities', ['n_a', 'n_b', 'nu'])(n_a, n_b, nu)

    def get_1d_currents(self, mus_eff, delta, N_twist=1):
        """return current for 1d only"""
        twistss = itertools.product(
            *(xp.arange(0, N_twist) * 2 * xp.pi / N_twist,) * self.dim)
        j_a = 0
        j_b = 0
        # xp.fft.ifft(1j*k*xp.fft.fft(f))

        def df(k, f):
            return xp.fft.ifft(1j * k * xp.fft.fft(f))

        for twists in twistss:
            ks_bloch = xp.divide(twists, self.Lxyz)
            k = [_k + _kb for _k, _kb in zip(self.kxyz, ks_bloch)][0]

            H = self.get_H(mus_eff=mus_eff, delta=delta, twists=twists)
            N = self.Nxyz[0]
            d, psi = xp.linalg.eigh(H)
            us, vs = psi.reshape(2, N, N * 2)
            us, vs = us.T, vs.T
            j_a_ = -0.5j * sum((us[i].conj() * df(k, us[i]) - us[i] *
                                df(k, us[i]).conj()) * self.f(d[i]) for i in range(len(us)))
            j_b_ = -0.5j * sum((vs[i] * df(k, vs[i]).conj() - vs[i].conj() *
                                df(k, vs[i])) * self.f(-d[i]) for i in range(len(vs)))
            j_a = j_a + j_a_
            j_b = j_b + j_b_
        return (j_a / N_twist / xp.prod(self.dxyz), j_b / N_twist / xp.prod(self.dxyz))
    
    def _get_densities_H(self, H, twists):
        """return densities for a given H"""
        d, UV = xp.linalg.eigh(H)
        U_V_shape = (2,) + tuple(self.Nxyz) + UV.shape[1:]
        U, V = U_V = UV.reshape(U_V_shape)  # U = us.T, V=vs.T
        dU_Vs = self._Del(U_V, twists=twists)
        dUs, dVs = dU_Vs[:, 0, ...], dU_Vs[:, 1, ...]
        #dUs, dVs = self._Del(U_V, twists=twists)
        #dUs_, dVs_ = dU_Vs[:, 0, ...], dU_Vs[:, 1, ...]
        #assert xp.allclose(dUs, dUs_)
        #assert xp.allclose(dVs, dVs_)
        f_p = self.f(d)
        f_m = self.f(-d)
        n_a = xp.dot(U*U.conj(), f_p).real
        n_b = xp.dot(V*V.conj(), f_m).real
        nu = xp.dot(U*V.conj(), f_p - f_m)/2
        tau_a = xp.dot(sum(dU.conj()*dU for dU in dUs), f_p).real
        tau_b = xp.dot(sum(dV.conj()*dV for dV in dVs), f_m).real
        j_a = [0.5*xp.dot((U.conj()*dU - U*dU.conj()), f_p).imag
               for dU in dUs]
        j_b = [0.5*xp.dot((V*dV.conj() - V.conj()*dV), f_m).imag
               for dV in dVs]
        return xp.stack([n_a, n_b, tau_a, tau_b, nu.real, nu.imag, *j_a, *j_b])

    def _get_densities(self, mus_eff, delta, twists, **args):
        """Return (R, Rm) with the specified twist."""
        H = self.get_H(mus_eff=mus_eff, delta=delta, twists=twists, **args)
        return self._get_densities_H(H, twists=twists)

    def get_densities(self, mus_eff, delta, N_twist=1, abs_tol=1e-12, unpack=True, struct=True, **args):
        """Return the densities.

        Arguments
        ---------
        N_twist : int, xp.inf
           Number of twists to average over.  Integrate if
           N_twist==xp.inf.
        abs_tol : float
           Absolute tolerance if performing twist averaging.
        """
        # Here we use the notation Rp = R = f(M) and Rm = f(-M) = 1-R
        def get_dens(twists):
            """Return (R, Rm) with the specified twist."""
            return self._get_densities(mus_eff=mus_eff, delta=delta, twists=twists, **args)

        if xp.isinf(N_twist):
            if self.dim == 1:
                dens = mquad(get_dens, -xp.pi, xp.pi, abs_tol=abs_tol) / 2 / xp.pi
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
    
    def mqaud_worker_thread(obj_args):
            obj, vs, twists, k, args = obj_args
            k_p = obj.hbar**2/2/obj.m*k**2
            H = obj.get_H(vs=vs, k_p=k_p, twists=twists, **args)
            den = obj._get_densities_H(H, twists=twists)
            return den

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
        ks = 2 * xp.pi * xp.fft.fftfreq(self.Nxyz[0] * N_factor, self.dxyz[0])
        ks_ = ks[0:-1]
        obj_twists_kp = [(obj, vs, twists, k, args) for k in ks]
        res = PoolHelper.run(BCS.mqaud_worker_thread, paras=obj_twists_kp)
        dens = sum(res)/(obj.dxyz[0] * obj.Nxyz[0] * N_factor)
        return dens 

    def twising_worker_thread(obj_args):
        """"""
        obj, k_c, vs, twists, args = obj_args
        abs_tol=1e-6

        def f(k=0):
            k_p = obj.hbar**2/2/obj.m*k**2
            H = obj.get_H(vs=vs, k_p=k_p, twists=twists, **args)
            den = obj._get_densities_H(H, twists=twists)
            return den
        dens = mquad(f, -k_c, k_c, abs_tol=abs_tol)/2/xp.pi  # factor? It turns out the factor should be 2pi

        #dens_ = obj.mquad_(obj_args=obj_args, k_a=-k_c, k_b=k_c, N_factor=100)
        #assert xp.allclose(dens, dens_, rtol=1e-3)
        return dens


    def get_dens_integral(self, mus_eff, delta, k_c=None, N_twist=1, unpack=True, struct=False, abs_tol=1e-6, **args):
        """
            integrate over another dimension by assuming it's homogeneous
            Note: the results are for dim + 1 system.
        """
        if k_c is None:
            k_c = xp.sqrt(2*self.m*self.E_c)/self.hbar
        twistss = self._get_twistss(N_twist)
        args = dict(args, mus_eff=mus_eff, delta=delta)
        vs = self.get_v_ext(**args)
        paras = [(self, k_c, vs, twists, args) for twists in twistss]
        res = PoolHelper.run(BCS.twising_worker_thread, paras=paras)
        dens = sum(res)/len(res)
        if unpack:
            return self._unpack_densities(dens, struct=struct)
        return dens


    def get_ns_e_p(self, mus_eff, delta, **args):
        """
            compute then energy density
            Note:
                the return value also include the pressure and densities
        """
        ns, taus, _, kappa = self.get_densities(mus_eff=mus_eff, delta=delta, struct=False, **args)
        g_eff = -delta/kappa
        energy_density = (taus[0] + taus[1])*self.hbar**2/2/self.m - g_eff * kappa.T.conj()*kappa
        pressure = ns[0] * mus_eff[0] + ns[1]*mus_eff[1] - energy_density
        return (ns, energy_density, pressure)