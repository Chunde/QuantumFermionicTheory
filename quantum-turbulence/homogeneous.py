from collections import namedtuple
import warnings
import numpy as np
import scipy.integrate
import scipy as sp

from uncertainties import ufloat

m = 1.0

_QUAD_ARGS = dict(
    epsabs=1.49e-08,
    epsrel=1.49e-08)


def quad(f, kF=None, k_0=0, k_inf=np.inf, limit=1000, **kw):
    """Wrapper for quad that deals with singularities
    at the Fermi surface.
    """
    args = dict(_QUAD_ARGS, **kw)
    if kF is None:
        res = ufloat(*sp.integrate.quad(f, k_0, k_inf, **args))
    else:
        # One might think that `points=[kF]` could be used here, but
        # this does not work with infinite limits.
        res = (ufloat(*sp.integrate.quad(f, k_0, kF, **args))
               +
               ufloat(*sp.integrate.quad(f, kF, k_inf, limit=limit, **args)))

    if abs(res.s) > 1e-6 and abs(res.s/res.n) > 1e-6:
        warnings.warn("Integral did not converge: res, err = %g, %g"
                      % (res, err))
    return 2*res   # Accounts for integral from -inf to inf


def quadl(f, N, L, twist=1):
    """Integrate f(k) using a lattice.

    Arguments
    ---------
    N : int
    L : float
    twist : int, np.inf
       How many twists to sample.  (This is done by just multiplying N
       and L by this factor.)  If twist==np.inf, then to the integral
       (with cutoff).
    """
    dx = L/N
    k_max = np.pi/dx
    
    if np.isinf(twist):
        return quad(f, k_inf=k_max)

    N *= twist
    L *= twist
    k = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(N, dx))
    return np.trapz(f(k), k)


def dquad(f, kF=None, k_0=0, k_inf=np.inf, limit=1000, int_name="Gap"):
    """Wrapper for quad that deals with singularities
    at the Fermi surface.
    """
    print(f"Computing {int_name}...")
    if kF is None:
        res, err = sp.integrate.dblquad(f, k_0, k_inf,
                                        lambda x: k_0, lambda x: k_inf)
    else:
        # One might think that `points=[kF]` could be used here, but
        # this does not work with infinite limits.
        res0, err0 = sp.integrate.dblquad(f, k_0, kF,
                                          lambda x: k_0, lambda x: k_inf)
        #res1, err1 = sp.integrate.dblquad(f, kF, np.inf,
        #                                  lambda x: kF, lambda x:
        #                                  k_inf)
        # Chunde made a very stupid mistake, in this line, I integrate
        # in y direction from KF to infinity, it should be from zero
        # to infinity 
        res1, err1 = sp.integrate.dblquad(f, kF, np.inf,
                                          lambda x: k_0, lambda x: k_inf)
        res = res0 + res1
        err = max(err0, err1)

    if abs(err) > 1e-6 and abs(err/res) > 1e-6:
        warnings.warn(
            f"{int_name} integral did not converge: res, err = {res}, {err}")
    return 2*res   # Accounts for integral from -inf to inf for 3D
                   # case, should be a factor of 4 instead of 2? 


def get_BCS_v_n_e(mu_eff, delta):
    m = hbar = 1.0
    kF = np.sqrt(2*m*max(0, mu_eff))/hbar

    def gap_integrand(k):
        e_p = (hbar*k)**2/2.0/m - mu_eff
        return 1./np.sqrt(e_p**2 + abs(delta)**2)

    v_0 = 4*np.pi / quad(gap_integrand, kF).n

    def n_integrand(k):
        """Density"""
        e_p = (hbar*k)**2/2.0/m - mu_eff
        denom = np.sqrt(e_p**2 + abs(delta)**2)
        return (denom - e_p)/denom

    n = quad(n_integrand, kF).n / 2/np.pi

    def e_integrand(k):
        """Energy"""
        e_p = (hbar*k)**2/2.0/m - mu_eff
        denom = np.sqrt(e_p**2 + abs(delta)**2)
        return (hbar*k)**2/2.0/m * (denom - e_p)/denom
        """Where this fomula comes from?"""
    e = quad(e_integrand, kF).n / 2/np.pi - v_0*n**2/4.0 - abs(delta)**2/v_0
    mu = mu_eff - n*v_0/2

    return namedtuple('BCS_Results', ['v_0', 'n', 'mu', 'e'])(v_0, n, mu, e)


def BCS(mu_eff, delta=1.0):
    m = hbar = 1.0
    """Return `(E_N_E_2, lam)` for comparing with the exact Gaudin
    solution.

    Arguments
    ---------
    delta : float
       Pairing gap.  This is the gap in the energy spectrum.
    mu_eff : float
       Effective chemical potential including both the bare chemical
       potential and the self-energy correction arising from the
       Hartree term.

    Returns
    -------
    E_N_E_2 : float
       Energy per particle divided by the two-body binding energy
       abs(energy per particle) for 2 particles.
    lam : float
       Dimensionless interaction strength.
    """
    v_0, n, mu, e = get_BCS_v_n_e(mu_eff=mu_eff, delta=delta)
    lam = m*v_0/n/hbar**2

    # Energy per-particle
    E_N = e/n

    # Energy per-particle for 2 particles
    E_2 = -m*v_0**2/4.0 / 2.0
    E_N_E_2 = E_N/abs(E_2)
    return E_N_E_2, lam


class Homogeneous1D(object):
    """Solutions to the homogeneous BCS equations in 1D at finite T.

    Allows for modified dispersion as well as asymmetric populations.
    """
    T = 0.0

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def f(self, E):
        """Return the Fermi distribution function."""
        if self.T == 0:
            return (1+np.sign(-E))/2.0
        else:
            # May overflow when T is too small
            return 1./(1+np.exp(E/self.T))

    def get_es(self, k, mus_eff):
        return (k**2/2.0/m - mus_eff[0],
                k**2/2.0/m - mus_eff[1])

    Results = namedtuple('Results', ['e_p', 'E', 'w_p', 'w_m'])

    def get_res(self, k, mus_eff, delta):
        e_a, e_b = self.get_es(k, mus_eff=mus_eff)
        e_p, e_m = (e_a + e_b)/2, (e_a - e_b)/2
        E = np.sqrt(e_p**2 + abs(delta)**2)
        w_p, w_m = e_m + E, e_m - E
        args = dict(locals())
        rets = self.Results(*[args[_n] for _n in self.Results._fields])
        return rets

    def get_BCS_v_n_e(self, mus_eff, delta, N=None, L=None, dx=None, twist=1):
        """Return `(v_0, n, mu, e)` for the 1D BCS solution at T=0."""

        if N is None and L is None and dx is None:
            quad_ = quad
        else:
            # Use finite lattice for integration
            if dx is None:
                dx = L/N
            elif L is None:
                L = N * dx
            elif N is None:
                N = np.ceil(L / dx).astype(int)

            def quad_(f, kF):
                return quadl(f, N=N, L=L, twist=twist)

        kF = np.sqrt(2*max(0, max(mus_eff)))

        def gap_integrand(k):
            res = self.get_res(k=k, mus_eff=mus_eff, delta=delta)
            return (self.f(res.w_m) - self.f(res.w_p))/res.E

        v_0 = 4*np.pi / quad_(gap_integrand, kF)

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

        n_m = quad_(nm_integrand, kF) / 2/np.pi
        n_p = quad_(np_integrand, kF) / 2/np.pi
        n_a = (n_p + n_m)/2.0
        n_b = (n_p - n_m)/2.0
        ns = np.array([n_a, n_b])
        mus = mus_eff - np.array([n_b, n_a])*v_0

        return namedtuple('BCS_Results', ['v_0', 'ns', 'mus'])(v_0, ns, mus)


class Homogeneous3D(object):
    """3D homogenous system with regularization"""
    T = 0.0
    m = hbar = 1
    k_cutoff = np.inf
    q = 0.0
    
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def f(self, E):
        if self.T == 0:
            return (1+np.sign(-E))/2.0
        else:
            return 1./(1+np.exp(E/self.T))

    def get_es(self, kz, kp, mus_eff, q):
        return (((kz+q)**2 + kp**2)/2.0/m - mus_eff[0],
                ((kz+q)**2 + kp**2)/2.0/m - mus_eff[1])

    Results = namedtuple('Results', ['e_p', 'E', 'w_p', 'w_m'])

    def get_res(self, kz, kp, mus_eff, delta, q):
        e_a, e_b = self.get_es(kz, kp, mus_eff=mus_eff,q=q)
        e_p, e_m = (e_a + e_b)/2, (e_a - e_b)/2
        E = np.sqrt(e_p**2 + abs(delta)**2)
        w_p, w_m = e_m + E, e_m - E
        args = dict(locals())
        rets = self.Results(*[args[_n] for _n in self.Results._fields])
        return rets

    def get_inverse_scattering_length(self, mus_eff, delta, k_c):
        kF = np.sqrt(2*max(0, max(mus_eff)))
        
        def gap_integrand(kz_, kp_): 
            # this integration will diverge?
            res = self.get_res(kz=kz_,kp=kp_, mus_eff=mus_eff, delta=delta,
                               q=self.q)
            return kp_* (1 - self.f(res.w_p) - self.f(-res.w_m))/res.E

        self._gap_integrand = gap_integrand

        if k_c < kF:
            res, err = sp.integrate.dblquad(
                gap_integrand, 0, k_c,
                lambda x: 0, lambda x: np.sqrt(k_c**2-x**2))
        else:
            res0, err0 = sp.integrate.dblquad(
                gap_integrand, 0, kF,
                lambda x: 0, lambda x: np.sqrt(k_c**2-x**2))
            res1, err1 = sp.integrate.dblquad(
                gap_integrand, kF, k_c,
                lambda x: 0, lambda x: np.sqrt(k_c**2-x**2))
            res, err = res0 + res1, err0 + err1
        
        if abs(err) > 1e-6 and abs(err/res) > 1e-6:
                warnings.warn("scattering integral did not converge:" +
                              f"res, err = {res}, {err})")

        # Factor of 1/4/pi**2 but we include the factor of 2 from -k_z to k_z
        res /= 2*np.pi**2   # the result should be the 1/g, where g=-v_0

        # the shift of mu to impprove  convergence, due to 
        k0 = (np.mean(mus_eff) * 2 * self.m)**0.5 / self.hbar

        # Lambda_c = k_c/2/np.pi**2 (1 - k0/2/k_c * np.log((k_c+k0) / (k_c-k0)))
        # res = g_inv / 2.0
        # C = Lambda_c - g_inv
        # a_inv = 4*np.pi * C
        shift_correction = (
            -k_c/2/np.pi**2 * k0/2/k_c
            * np.log((k_c+k0) / (k_c-k0))
            *4*np.pi)
        return (-np.pi * 2.0 * res + 2 * k_c / np.pi) + shift_correction

    def get_BCS_v_n_e_in_cylindrical(self, mus_eff, delta,
                                     k_c=10000.0,
                                     unitary=False):
        kF = np.sqrt(2*max(0, max(mus_eff)))

        ainv_s = self.get_inverse_scattering_length(
            mus_eff=mus_eff, delta=delta, k_c=k_c)

        def np_integrand(kz_, kp_):
            res = self.get_res(kz=kz_,kp=kp_, mus_eff=mus_eff, delta=delta,
                               q=self.q)
            n_p = 1 + res.e_p/res.E*(
                self.f(res.w_p) + self.f(-res.w_m) - 1)  # --> Dr. Forbes's equation
            return n_p * kp_

        
        def nm_integrand(kz_, kp_):
            res = self.get_res(kz=kz_, kp=kp_, mus_eff=mus_eff,
                               delta=delta, q=self.q)
            n_m = self.f(res.w_p) - self.f(-res.w_m) #--> Dr. Forbes's equation
            return n_m * kp_

        n_m = dquad(f=nm_integrand, kF=kF, #k_inf=k_c,
                    int_name="Density Difference")/4/np.pi**2
        n_p = dquad(f=np_integrand, kF=kF, #k_inf=k_c,
                    int_name="Total Density")/4/np.pi**2
        n_a = (n_p + n_m)/2.0
        n_b = (n_p - n_m)/2.0
        ns = np.array([n_a, n_b])
        mus = mus_eff
        return ainv_s, ns, mus

    def get_BCS_v_n_e_in_spherical(self, mus_eff, delta, k_c=10000.0,
                                   unitary=False):
        """Return `(v_0, n, mu, e)` for the 3D BCS solution at T=0 or T > 0."""
        assert self.q == 0
        kF = np.sqrt(2*max(0, max(mus_eff)))

        if not unitary:
            ainv_s = self.get_inverse_scattering_length(
                mus_eff=mus_eff, delta=delta, k_c=k_c)
            v_0 = np.pi * 4.0 / (ainv_s  - 2.0 * k_c/np.pi)
        else:
            v_0 = 2 * np.pi **2 / k_c
            
        def np3(kr):
             res = self.get_res(kz=kr,kp=0, mus_eff=mus_eff, delta=delta)
             n_p = 1 + res.e_p/res.E*(self.f(res.w_p) + self.f(-res.w_m) - 1)
             return n_p
         
        def np_integrand(kr):
            return np3(kr) * kr**2

        self._np3 = np3
        
        def nm_integrand(kr):
            res = self.get_res(kz=kr,kp=0, mus_eff=mus_eff, delta=delta)
            n_m = self.f(res.w_p) - self.f(-res.w_m)
            return n_m * kr**2

        n_m = sp.integrate.quad(nm_integrand, 0, k_c)[0]/2/np.pi**2
        #quad(f=nm_integrand, kF=kF)/2/np.pi**2#check the factor, should change
        n_p = sp.integrate.quad(np_integrand, 0, k_c)[0]/2/np.pi**2
        #quad(f=np_integrand, kF=kF)/2/np.pi**2#check the factor, should change
        n_a = (n_p + n_m)/2.0
        n_b = (n_p - n_m)/2.0
        ns = np.array([n_a, n_b])
        mus = mus_eff

        return v_0, ns, mus

