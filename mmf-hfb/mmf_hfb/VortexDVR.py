from mmf_hfb.DVRBasis import CylindricalBasis
from mmfutils.math.integrate import mquad
from mmf_hfb.utils import block
from mmf_hfb import homogeneous
import numpy as np
import sys
from mmfutils.math.special import mstep
from collections import namedtuple
from mmf_hfb.bcs import BCS


Densities = namedtuple('Densities', ['n_a', 'n_b', 'tau_a', 'tau_b', 'nu', 'j_a', 'j_b'])


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
            u = U_matrix.dot(u)
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
        assert T==0
        self.T = T  # only support T=0
        self.wz = wz  # pairing field winding number
        self.g = self.get_g(mu=mu, delta=np.mean(delta)) if g is None else g
        self.mus = (mu + dmu, mu - dmu)
        self.E_c = E_c
        self.rs = self.bases.get_rs()
        self.verbosity = verbosity

    def _log(self, msg, level=1):
        """Log a message."""
        if level <= self.verbosity:
            print(msg)

    def f(self, E, T=0):
        if T is None:
            T = self.T
        if T == 0:
            if E < 0:
                return 1
            return 0
        else:
            return 1./(1+np.exp(E/T))

    def get_Vext(self, rs):
        """return external potential"""
        return (0, 0)

    def get_H(self, mus, delta, lz=0, kz=0):
        """Return the full Hamiltonian (with pairing field).

        Arguments
        ---------
        l_z : int
           Angular momentum quantum number.  The centrifugal piece
           for this is included in the kinetic term, while the remaining
        """
        basis = self.bases.get_basis(lz=lz)
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
        h = homogeneous.Homogeneous(dim=2)
        res = h.get_densities(mus_eff=(mu, mu), delta=delta)
        g = 0 if res.nu == 0 else delta/res.nu
        return g

    def get_psi(self, lz, u):
        """
        apply weight on the u(v) to get the actual radial wave-function
        """
        return self.bases.get_psi(lz=lz, u=u)

    def _get_den(self, H, lz):
        """
        return the densities for a given H
        """
        es, phis = np.linalg.eigh(H)
        phis = phis.T
        offset = phis.shape[0] // 2
        dens = 0
        # N_states = np.sum(abs(es) <= self.E_c)
        # if N_states > 0:
        #     self._log(f"{N_states} states included", 1)
        for i in range(len(es)):
            E, uv = es[i], phis[i]
            if abs(E) > self.E_c:
                continue

            u, v = uv[: offset], uv[offset:]
            u = self.get_psi(lz=lz, u=u)
            v = self.get_psi(lz=lz, u=v)

            f_p, f_m = self.f(E=E), self.f(E=-E)
            n_a = u*u.conj()*f_p
            n_b = v*v.conj()*f_m
            j_a = -n_a*self.wz/self.rs/2  # WRONG!
            j_b = -n_b*self.wz/self.rs/2  # WRONG!
            nu = u*v.conj()*(f_p - f_m)/2
            dens = dens + np.array([n_a, j_a, n_b, j_b, nu])
        return dens

    def get_densities(self, mus, delta, kz=0, wz=None, struct=True):
        """
        return the particle number density and anomalous density
        Note: Here the anomalous density is represented as kappa
        instead of lz so it's not that confusing when lz has
        been used as angular momentum quantum number.
        """
        if wz is None:
            wz = self.wz
        else:
            self.wz = wz

        lzs = [0]
        dens = 0
        for lz in range(1, self.l_max):
            lzs.append(-lz)
            lzs.append(lz)
        for lz in lzs:  # range(-self.l_max, self.l_max):  # sum over angular momentum
            H = self.get_H(mus=mus, delta=delta, kz=kz, lz=lz)
            den = self._get_den(H, lz=lz)
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
        self.g = delta/res.nu.n

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
