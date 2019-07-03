from collections import namedtuple
import warnings
import numpy as np
import scipy.integrate
import scipy as sp
from mmf_hfb import tf_completion as tf

from uncertainties import ufloat

_QUAD_ARGS = dict(
    epsabs=1.49e-08,
    epsrel=1.49e-08)


def quad_k(f, kF=None, k_0=0, k_inf=np.inf, dim=1, limit=1000, **kw):
    """Integrate from k_0 to k_inf in dim-dimensions including factors
    of 1/(2*pi)**dim and spherical area."""
    args = dict(_QUAD_ARGS, **kw)

    factor = 1.0
    if dim == 1:
        factor = 1./np.pi
        integrand = f
    elif dim == 2:
        factor = 1./2/np.pi
        
        def integrand(k):
            return f(k) * k
    elif dim == 3:
        factor = 1./2/np.pi**2
        
        def integrand(k):
            return f(k) * k**2
    else:
        raise NotImplementedError(f"Only dim=1,2,3 supported (got {dim}).")

    if kF is None:
        res = ufloat(*sp.integrate.quad(integrand, k_0, k_inf, **args))
    else:
        # One might think that `points=[kF]` could be used here, but
        # this does not work with infinite limits.
        res = (ufloat(*sp.integrate.quad(integrand, k_0, kF, **args))
               + ufloat(*sp.integrate.quad(
                   integrand, kF, k_inf, limit=limit, **args)))

    if abs(res.s) > 1e-6 and abs(res.s/res.n) > 1e-6:
        warnings.warn(f"Integral did not converge: {res}")
        
    return res * factor


def quad_l(f, Nxyz, Lxyz, N_twist=1, **kw):
    """Integrate f(k) using a lattice including factors of 1/(2*pi)**dim.

    Arguments
    ---------
    Nxyz : (int,)
    Lxyz : (float,)
       These tuples specify the size of the lattice and box.  Their
       length specifies the dimension.
    N_twist : int, np.inf
       How many twists to sample in each direction.
       (This is done by just multiplying N and L by this factor.)  If
       N_twist==np.inf, then to the integral (with cutoff).

    BUG: Currently, integration is done over a spherical shell with
    radius k_max.  This only matches the lattice calculation in 1D.
    """
    dim = len(Nxyz)
    Nxyz, Lxyz = np.array(Nxyz), np.array(Lxyz)
    dxyz = Lxyz/Nxyz
    
    if np.isinf(N_twist):
        k_max = np.pi/np.max(dxyz)
        
        ## This is the idea, but we really need to compute the measure
        ## properly so we integrate over the box rather than a
        ## sphere.  This is a little tricky.
        return quad_k(f, k_inf=k_max, dim=dim, **kw)

    # Lattice sums.
    Nxyz *= N_twist
    Lxyz *= N_twist
    dxyz = Lxyz/Nxyz
    dkxyz = 2*np.pi/Lxyz
    ks = np.meshgrid(
        *[2*np.pi * np.fft.fftshift(np.fft.fftfreq(_N, _dx))
          for (_N, _dx) in zip(Nxyz, dxyz)],
        indexing='ij', sparse=True)
    k2 = sum(_k**2 for _k in ks)
    k = np.sqrt(k2)
    return ufloat(f(k).sum() * np.prod(dkxyz), 0) / (2*np.pi)**dim


class Homogeneous(object):
    """Solutions to the homogeneous BCS equations at finite T.

    Allows for modified dispersion as well as asymmetric populations.
    """
    T = 0.0
    m = 1
    hbar = 1
    
    def __init__(self, Nxyz=None, Lxyz=None, dx=None, dim=None, k_c=None, E_c=None, **kw):
        if Nxyz is None and Lxyz is None and dx is None:
            self._dim = dim
        elif dx is not None:
            if Lxyz is None:
                Lxyz = np.multiply(Nxyz, dx)
            elif Nxyz is None:
                Nxyz = np.ceil(np.divide(Lxyz, dx)).astype(int)

            self.dxyz = np.divide(Lxyz, Nxyz)
            self._dim = len(Nxyz)
            
        self.Nxyz = Nxyz
        self.Lxyz = Lxyz
        self.k_c = k_c
        self.E_c = k_c**2*self.hbar**2/2.0/self.m
        self.__dict__.update(kw)

    @property
    def dim(self):
        return self._dim

    def f(self, E):
        """Return the Fermi distribution function."""
        if self.T == 0:
            return (1+np.sign(-E))/2.0
        else:
            return 1./(1+np.exp(E/self.T))

    def get_es(self, k, mus_eff):
        e = (self.hbar*k)**2/2.0/self.m
        return (e - mus_eff[0], e - mus_eff[1])

    Results = namedtuple('Results', ['e_p', 'E', 'w_p', 'w_m'])

    def get_res(self, k, mus_eff, delta):
        e = (self.hbar*k)**2/2.0/self.m
        e_a, e_b = e - mus_eff[0], e - mus_eff[1]
        e_p, e_m = (e_a + e_b)/2, (e_a - e_b)/2
        E = np.sqrt(e_p**2 + abs(delta)**2)
        w_p, w_m = e_m + E, e_m - E

        # This is a bit of a hack to get all of the desired Results
        # from the field names.  They are extracted from the local
        # dictionary.  In the final version of the code, these should
        # just be explicitly set.
        args = dict(locals())
        res = self.Results(*[args[_n] for _n in self.Results._fields])
        return res

    def _get_densities_tf(
            self, mus_eff, delta, q=0, dq=0, **args):
        """
        extended the homogeneous code to support FF state
        calculation, seems to be much slower.
        """
        mu_a, mu_b = mus_eff
        args.update(
            mu_a=mu_a, mu_b=mu_b, m_a=self.m, m_b=self.m, delta=delta,
            dim=self.dim, hbar=self.hbar, T=self.T, k_c=self.k_c)

        n_m = tf.integrate_q(tf.n_m_integrand, **args)
        n_p = tf.integrate_q(tf.n_p_integrand, **args)
        n_a = (n_p + n_m)/2.0
        n_b = (n_p - n_m)/2.0
        tau_m = tf.integrate_q(tf.tau_m_integrand, **args)
        tau_p = tf.integrate_q(tf.tau_p_integrand, **args)
        tau_a = (tau_p + tau_m)/2.0
        tau_b = (tau_p - tau_m)/2.0
        nu_delta = tf.integrate_q(tf.nu_delta_integrand, **args)
        nu = nu_delta*delta
        return namedtuple('Densities', ['n_a', 'n_b', 'tau_a', 'tau_b', 'nu'])(
            n_a, n_b, tau_a, tau_b, nu)

    def get_densities(self, mus_eff, delta, N_twist=1, **args):
        """
        Return the densities (ns, taus, nu).
        --------------
        Note: if dq is in args, that means we try to calculate FF state
        """
        if 'dq' in args:
            return self._get_densities_tf(mus_eff=mus_eff, delta=delta, **args)
        kF = np.sqrt(2*max(0, max(mus_eff)))
        
        if self.Nxyz is None:
            def quad(f):
                return quad_k(f, dim=self.dim, kF=kF, k_inf=self.k_c)
        else:
            def quad(f):
                return quad_l(f, Nxyz=self.Nxyz, Lxyz=self.Lxyz,
                              N_twist=N_twist)

        def np_integrand(k):
            """Density"""
            res = self.get_res(k=k, mus_eff=mus_eff, delta=delta)
            n_p = 1 - res.e_p/res.E*(self.f(res.w_m) - self.f(res.w_p))
            return n_p

        def nm_integrand(k):
            """Density"""
            res = self.get_res(k=k, mus_eff=mus_eff, delta=delta)
            n_m = self.f(res.w_p) - self.f(-res.w_m)
            return n_m

        def tau_p_integrand(k):
            k2 = k**2
            res = self.get_res(k=k, mus_eff=mus_eff, delta=delta)
            f_nu = self.f(res.w_m) - self.f(res.w_p)
            f_p = 1 - res.e_p/res.E*f_nu
            f_m = self.f(res.w_p) - self.f(-res.w_m)
            f_a = (f_p + f_m)/2
            f_b = (f_p - f_m)/2
            return k2*(f_a + f_b)

        def tau_m_integrand(k):
            k2 = k**2
            res = self.get_res(k=k, mus_eff=mus_eff, delta=delta)
            f_nu = self.f(res.w_m) - self.f(res.w_p)
            f_p = 1 - res.e_p/res.E*f_nu
            f_m = self.f(res.w_p) - self.f(-res.w_m)
            f_a = (f_p + f_m)/2
            f_b = (f_p - f_m)/2
            return k2*(f_a - f_b)

        def nu_delta_integrand(k):
            res = self.get_res(k=k, mus_eff=mus_eff, delta=delta)
            f_nu = self.f(res.w_m) - self.f(res.w_p)
            return -0.5/res.E*f_nu

        n_m = quad(nm_integrand)
        n_p = quad(np_integrand)
        n_a = (n_p + n_m)/2.0
        n_b = (n_p - n_m)/2.0
        tau_m = quad(tau_m_integrand)
        tau_p = quad(tau_p_integrand)
        tau_a = (tau_p + tau_m)/2.0
        tau_b = (tau_p - tau_m)/2.0
        if self.k_c is None:  # if not cutoff
            if self.Nxyz is None and self.dim != 1:
                # This is divergent:
                nu = np.inf
            else:
                nu_delta = quad(nu_delta_integrand)
                nu = nu_delta * delta
        else:
            nu_delta = quad(nu_delta_integrand)
            nu = nu_delta*delta

        return namedtuple('Densities', ['n_a', 'n_b', 'tau_a','tau_b', 'nu'])(
            n_a, n_b, tau_a, tau_b, nu)

    def get_entropy(self, mus_eff, delta, N_twist=1):
        """Return the entropy"""
        kF = np.sqrt(2*max(0, max(mus_eff)))
        
        if self.Nxyz is None:
            def quad(f):
                return quad_k(f, dim=self.dim, kF=kF, k_inf=self.k_c)
        else:
            def quad(f):
                return quad_l(f, Nxyz=self.Nxyz, Lxyz=self.Lxyz,
                              N_twist=N_twist)

        def entropy_integrand(k):
            res = self.get_res(k=k, mus_eff=mus_eff, delta=delta)
            R_p = self.f(res.w_p)
            R_m = self.f(res.w_m)
            R = 0
            if R_p > 0 and R_p < 1:
                R = R + R_p*np.log(R_p) + (1-R_p)*np.log(1-R_p)
            if R_m > 0 and R_m < 1:
                R = R + R_m*np.log(R_m) + (1-R_m)*np.log(1-R_m)
            return R

        s = quad(entropy_integrand)
        return s
        
    def get_BCS_v_n_e(self, mus_eff, delta, N_twist=1, k_inf=np.inf):
        """Return `(v_0, n, mu, e)` for the 1D BCS solution at T=0."""
        kF = np.sqrt(2*max(0, max(mus_eff)))
        
        if self.Nxyz is None:
            def quad(f):
                return quad_k(f, dim=self.dim, kF=kF, k_inf=k_inf)
        else:
            def quad(f):
                return quad_l(f, Nxyz=self.Nxyz, Lxyz=self.Lxyz,
                              N_twist=N_twist)

        def nu_delta_integrand(k):
            res = self.get_res(k=k, mus_eff=mus_eff, delta=delta)
            f_nu = self.f(res.w_m) - self.f(res.w_p)
            return -0.5/res.E*f_nu
        
        nu_delta = quad(nu_delta_integrand)
        v_0 = -1/nu_delta

        def np_integrand(k):
            """Density"""
            res = self.get_res(k=k, mus_eff=mus_eff, delta=delta)
            f_nu = self.f(res.w_m) - self.f(res.w_p)
            n_p = 1 - res.e_p/res.E*f_nu
            return n_p

        def nm_integrand(k):
            """Density"""
            res = self.get_res(k=k, mus_eff=mus_eff, delta=delta)
            n_m = self.f(res.w_p) - self.f(-res.w_m)
            return n_m

        def kappa_integrand(k):
            res = self.get_res(k=k, mus_eff=mus_eff, delta=delta)
            f_nu = self.f(res.w_m) - self.f(res.w_p)
            f_p = 1 - res.e_p/res.E*f_nu
            tau_p = k**2*f_p
            nu_delta = nu_delta_integrand(k)
            return self.hbar**2 * tau_p/self.m/2 + abs(delta)**2 * nu_delta

        n_m = quad(nm_integrand)
        n_p = quad(np_integrand)
        kappa = quad(kappa_integrand)
        n_a = (n_p + n_m)/2.0
        n_b = (n_p - n_m)/2.0
        ns = np.array([n_a, n_b])
        mus = mus_eff - np.array([n_b, n_a])*v_0
        e = kappa - v_0*n_a*n_b

        return namedtuple('BCS_Results', ['v_0', 'ns', 'mus', 'e'])(
            v_0, ns, mus, e)


class Homogeneous1D(Homogeneous):
    dim = 1


class Homogeneous2D(Homogeneous):
    dim = 2


class Homogeneous3D(Homogeneous):
    dim = 3
