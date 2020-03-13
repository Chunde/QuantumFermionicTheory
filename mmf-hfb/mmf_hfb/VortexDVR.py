"""????"""

from collections import namedtuple
import warnings

import numpy as np

from mmfutils.math.integrate import mquad
from mmfutils.math.special import mstep

from .DVRBasis import CylindricalBasis
from .utils import block
from . import homogeneous
from .hfb import BCS


Densities = namedtuple('Densities', ['n_a', 'n_b', 'tau_a', 'tau_b', 'nu', 'j_a', 'j_b'])


def mqaud_worker_thread(obj_args):
    obj, vs, twists, k, args = obj_args
    k_p = obj.hbar**2/2/obj.m*k**2
    H = obj.get_H(vs=vs, k_p=k_p, twists=twists, **args)
    den = obj._get_densities_H(H, twists=twists)
    return den


def get_transform_matrix(dvr_s, dvr_t):
    rs_s = dvr_s.rs
    rs_t = dvr_t.rs
    ws = dvr_t.get_F_rs()
    return np.array(
        [[dvr_s.get_F(
            n=i, rs=rs_t[j])/ws[j] for i in range(
                len(rs_s))] for j in range(len(rs_t))])


class dvr_basis_set(object):
    # define basis set interface
    pass


class dvr_odd_even_set(dvr_basis_set):
    """
    A DVR basis set class, this class only
    use $\nu=0$ and $\nu=1$ bessel DVR basis.
    The $\nu=0$ basis can be used as replacement
    for all even angular momentum, while the
    $\nu=1$ for all odd.
    """
    def __init__(self, **args):
        self.N_basis = 2
        self.bases = [CylindricalBasis(lz=lz, **args) for lz in range(self.N_basis)]
        self.Us =[None, get_transform_matrix(self.bases[1], self.bases[0])]
        self.N_roots = [basis.N_root for basis in self.bases]

    def basis_match_rule(self, lz):
        """
            Assign different bases to different angular momentum lz
            it assign 0 to even lz and 1 to odd lz
        Note:
            inherit a child class to override this function
        """
        assert len(self.bases) > 1
        return lz % 2

    @property
    def zero(self):
        return self.bases[0].zero

    def get_rs(self):
        return self.bases[0].rs

    def get_basis(self, lz):
        return self.bases[lz % 2]

    def get_psi(self, lz, u):
        U_matrix = self.Us[lz % 2]
        if U_matrix is not None:
            u = U_matrix.dot(u.T).T
        b = self.bases[0]
        return b._get_psi(u=u)


class dvr_full_set(dvr_basis_set):
    def __init__(self, l_max, **args):
        self.N_basis = l_max
        self.bases = [CylindricalBasis(lz=lz, **args) for lz in range(self.N_basis)]
        self.Us =[None]
        self.Us.extend(
            [get_transform_matrix(
                self.bases[i], self.bases[0]) for i in range(1, l_max)])

    def basis_match_rule(self, lz):
        """
            Assign different bases to different angular momentum lz
            it assign 0 to even lz and 1 to odd lz
        Note:
            inherit a child class to override this function
        """
        assert lz >= 0
        if lz < len(self.bases):
            return lz
        return lz % 2

    @property
    def zero(self):
        return self.bases[0].zero

    def get_rs(self):
        return self.bases[0].rs

    def get_basis(self, lz):
        return self.bases[self.basis_match_rule(lz=lz)]

    def get_psi(self, lz, u):
        U_matrix = self.Us[self.basis_match_rule(lz=lz)]
        if U_matrix is not None:
            u = U_matrix.dot(u)
        b = self.bases[0]
        return b._get_psi(u=u)


class bdg_dvr(object):
    """
    A 2D and 3D vortex class without external potential
    """
    def __init__(
            self, bases_N=2, mu=1, dmu=0, delta=1, wz=0,
            E_c=None, T=0, l_max=100, g=None, bases=None,
            verbosity=0,
            **args):
        """
        Construct and cache some information of bases
        """
        if bases is None:
            bases = dvr_odd_even_set(l_max=100, **args)
        self.bases = bases  # basis set used
        self.l_max = max(l_max, 1)  # the angular momentum cut_off
        self.E_c = E_c
        self.mus = (mu + dmu, mu - dmu)
        assert T==0
        self.T = T  # only support T=0
        self.wz = wz  # pairing field winding number
        self.g = self.get_g(mu=mu, delta=np.mean(delta)) if g is None else g
        self.rs = self.bases.get_rs()
        self.verbosity = verbosity

    def _log(self, msg, level=1):
        """Log a message."""
        if level <= self.verbosity:
            print(msg)
    
    def f(self, E, T=0):
        if T is None:
            T = self.T
        if self.T > 0:
            f = 1./(1+np.exp(E/self.T))
        else:
            f = (1 - np.sign(E))/2
        return f

    def get_Vext(self, rs):
        """return external potential"""
        return (0, 0)

    def get_H(self, mus, delta, lz=0, lz_offset=0, kz=0):
        """Return the full Hamiltonian (with pairing field).

        Arguments
        ---------
        l_z : int
           Angular momentum quantum number.  The centrifugal piece
           for this is included in the kinetic term, while the remaining
        """
        basis = self.bases.get_basis(lz=lz + lz_offset)
        # kinetic energy component in perpendicular direction
        K_per = np.diag(basis.zero + kz**2/2)
        T = basis.K + K_per
        Delta = np.diag(basis.zero + delta)
        mu_a, mu_b = mus
        V_ext = self.get_Vext(rs=basis.rs)
        V_corr_a = basis.get_V_correction(lz=lz + self.wz)
        V_corr_b = basis.get_V_correction(lz=lz)
        V_eff_a = V_ext[0] + V_corr_a
        V_eff_b = V_ext[1] + V_corr_b
        H_a = T + np.diag(V_eff_a - mu_a)
        H_b = T + np.diag(V_eff_b - mu_b)
        H = block([[H_a, Delta],
                   [Delta.conj(), -H_b]])
        return H

    def get_g(self, mu=1.0, delta=0.2):
        """
        the interaction strength
        """
        # [Check] will be dim = 3 when integrate over z
        k_c = (self.E_c*2)**0.5
        h = homogeneous.Homogeneous(dim=2, k_c=k_c)
        res = h.get_densities(mus_eff=(mu, mu), delta=delta)
        g = 0 if res.nu == 0 else delta/res.nu
        return g

    def get_psi(self, lz, u):
        """
        apply weight on the u(v) to get the actual radial wave-function
        """
        return self.bases.get_psi(lz=lz, u=u)

    def _get_den(self, mus, delta, lz, lz_offset=0, kz=0):
        """
        return the densities for a given H
        """
        H = self.get_H(mus=mus, delta=delta, kz=kz, lz=lz, lz_offset=lz_offset)
        es, phis = np.linalg.eigh(H)
        phis = phis.T
        offset = phis.shape[0] // 2
        
        # N_states = np.sum(abs(es) <= self.E_c)
        # if N_states > 0:
        #     self._log(f"{N_states} states included", 1)

        dens = (self.bases.zero,)*5
        start_index = -1
        end_index = -1
        flags = abs(es) <= self.E_c
        for i, flag in enumerate(flags):
            if start_index == -1 and flag:
                start_index = i
                continue
            if start_index != -1 and not flag:
                end_index = i
                break
        if start_index == -1 or end_index == -1:
            return dens
        es = es[start_index:end_index]
        phis = phis[start_index:end_index]
        us, vs = phis[:, 0:offset], phis[:, offset:]
        us = self.get_psi(lz=lz + lz_offset, u=us)
        vs = self.get_psi(lz=lz + lz_offset, u=vs)
        f_p, f_m = self.f(es), self.f(-es)
        n_a = sum(us*us.conj()*f_p[:, None]).real
        n_b = sum(vs*vs.conj()*f_m[:, None]).real
        nu = sum(us*vs.conj()*(f_p - f_m)[:, None])/2
        j_a = -n_a*self.wz/self.rs/2  # WRONG!
        j_b = -n_b*self.wz/self.rs/2  # WRONG!
        return np.array([n_a, j_a, n_b, j_b, nu])
        # old implementation easier to read
        # speed is the same,.
        # for i in range(len(es)):
        #     E, uv = es[i], phis[i]
        #     if abs(E) > self.E_c:
        #         continue
        #     u, v = uv[: offset], uv[offset:]
        #     u = self.get_psi(lz=lz, u=u)
        #     v = self.get_psi(lz=lz, u=v)

        #     f_p, f_m = self.f(E=E), self.f(E=-E)
        #     n_a = u*u.conj()*f_p
        #     n_b = v*v.conj()*f_m
        #     j_a = -n_a*self.wz/self.rs/2  # WRONG!
        #     j_b = -n_b*self.wz/self.rs/2  # WRONG!
        #     nu = u*v.conj()*(f_p - f_m)/2
        #     dens = dens + np.array([n_a, j_a, n_b, j_b, nu])
        # return dens

    def get_densities(
            self, mus, delta, kz=0, wz=None, struct=True,
            lz_offset=0, basis_interpolation=True):
        """
        return the particle number density and anomalous density
        Note: Here the anomalous density is represented as kappa
        instead of lz so it's not that confusing when lz has
        been used as angular momentum quantum number.
        Parameters:
        --------------
        wz: Integer
            winding number of a vortex
        struct: bool
            return result type: True will return a structed type
            else, return tuple type
        lz_offset: Integer
            Shift the basis angular momentum by the offset, this is
            equivalent to shift the basis
        basis_interpolation: bool
            indicate if to interpolate results from different bases
            when the winding number is odd.
        ----------------
        Potential issue: it seems the delta pass into this function
            is in a specific basis(kz=0 basis), then when do iteration
            should it be converted to the proper new basis? Yes, so this
            need to be fixed
        """
        if wz is None:
            wz = self.wz
        else:
            self.wz = wz

        dens = (self.bases.zero,)*5
        lzs = [0]
        for lz in range(1, self.l_max):
            lzs.append(-lz)
            lzs.append(lz)
        # the average over \nu is not good, a better solution
        # may be save the u and v from different basis, and then
        # compute all densities when all are accurate, as the
        # \nu=u*v, u and v need different bases to have good
        # accuracy. But the eigen energy from different bases
        # may are different.
        if self.wz % 2 == 0:
            # for even winding, both results for both spins
            # are accurate
            def get_den(lz):
                return self._get_den(
                    mus=mus, delta=delta, kz=kz, lz=lz, lz_offset=lz_offset)
        else:
            if basis_interpolation:
                warnings.warn(
                    f"The winding number is odd, and bases interpolation is used,"
                    +"this may lead to in accurate result")

                def get_den(lz):
                    den1 = self._get_den(
                        mus=mus, delta=delta, kz=kz, lz=lz, lz_offset=lz_offset)
                    den2 = self._get_den(
                        mus=mus, delta=delta, kz=kz, lz=lz, lz_offset=lz_offset + 1)
                    if lz_offset % 2 != 0:
                        return np.array(
                            [den1[0], den1[1], den2[2], den2[3], (den1[4]+den2[4])/2])
                    else:
                        return np.array(
                            [den2[0], den2[1], den1[2], den1[3], (den1[4]+den2[4])/2])
            else:
                def get_den(lz):
                    return self._get_den(
                        mus=mus, delta=delta, kz=kz, lz=lz, lz_offset=lz_offset)

        for lz in lzs:  # range(-self.l_max, self.l_max):  # sum over angular momentum
            # Fun fact, calling the following line will be much slower than calling its
            # next line, that means the function defined above runs faster, may be due
            # to the stack operation? as only lz is past to it in the get_den(...) call.
            # get_den(...) takes only half of the time compared to self._get_den(...)
            # -------------------------------------------------------------------------
            # den = self._get_den(mus=mus, delta=delta, kz=kz, lz=lz, lz_offset=lz_offset)
            den = get_den(lz)
            if np.alltrue(den==0):
                break
            dens = dens + den
       
        if struct:
            n_a, j_a, n_b, j_b, kappa = dens
            return Densities(
                n_a=n_a, n_b=n_b,
                tau_a=None, tau_b=None,
                nu=kappa,
                j_a=j_a, j_b=j_b)
        return dens


class VortexMixin(object):
    barrier_width = 0.2
    barrier_height = 100.0
    R = 5

    def get_Vext(self, rs):
        R0 = self.barrier_width*self.R
        V = self.barrier_height*mstep(rs-self.R+R0, R0)
        return (V, V)


class PeriodicDVR(VortexMixin, BCS):
    """Vortex in periodic basis."""
    def __init__(self, delta, mus_eff, **args):
        BCS.__init__(self, **args)
        h = homogeneous.Homogeneous(Nxyz=self.Nxyz, Lxyz=self.Lxyz)
        res = h.get_densities(mus_eff=mus_eff, delta=delta)
        self.g = delta/res.nu

    def get_Vext(self, **kw):
        # self.R = min(self.Lxyz)/2
        rs = np.sqrt(sum([_x**2 for _x in self.xyz[:2]]))
        return VortexMixin.get_Vext(self, rs=rs)


class CylindricalDVR(VortexMixin, bdg_dvr):
    """Vortex in cylindrical basis."""


class CylindricalDVR3D(CylindricalDVR):
    """Vortex in cylindrical basis."""

    def get_densities(self, mus, delta, wz=None, abs_tol=1e-6):
        k_max = (2.0*self.E_c)**0.5

        def f(kz):
            return CylindricalDVR.get_densities(
                self, mus=mus, delta=delta, wz=wz, kz=kz, struct=False)
        dens = 2*mquad(f, 0, k_max, abs_tol=abs_tol)/2/np.pi
        return dens


class bdg_dvr_ho(bdg_dvr):
    """a 2D DVR with harmonic potential class"""

    def get_Vext(self, rs):
        return rs**2/2
