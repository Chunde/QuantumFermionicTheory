from collections import namedtuple
import warnings
import numpy as np
import scipy.integrate
import scipy as sp

m = 1.0

def quad(f, kF=None, k_0=0, k_inf=np.inf, limit=1000):
    """Wrapper for quad that deals with singularities
    at the Fermi surface.
    """
    if kF is None:
        res, err = sp.integrate.quad(f, k_0, k_inf)
    else:
        # One might think that `points=[kF]` could be used here, but
        # this does not work with infinite limits.
        res0, err0 = sp.integrate.quad(f, k_0, kF)
        res1, err1 = sp.integrate.quad(f, kF, k_inf, limit=limit)
        res = res0 + res1
        err = max(err0, err1)

    if abs(err) > 1e-6 and abs(err/res) > 1e-6:
        warnings.warn(
            "Gap integral did not converge: res, err = %g, %g" % (res, err))
    return 2*res   # Accounts for integral from -inf to inf

def dquad(f, kF=None, k_0=0, k_inf=np.inf, limit=1000, int_name="Gap"):
    """Wrapper for quad that deals with singularities
    at the Fermi surface.
    """
    if kF is None:
        res, err = sp.integrate.dblquad(f, k_0, k_inf, lambda x: k_0, lambda x: k_inf)
    else:
        # One might think that `points=[kF]` could be used here, but
        # this does not work with infinite limits.
        res0, err0 = sp.integrate.dblquad(f, k_0, kF, lambda x: k_0, lambda x: k_inf)
        res1, err1 = sp.integrate.dblquad(f, kF, np.inf, lambda x: kF, lambda x: k_inf)
        res = res0 + res1
        err = max(err0, err1)

    if abs(err) > 1e-6 and abs(err/res) > 1e-6:
        warnings.warn(
            int_name + " integral did not converge: res, err = %g, %g" % (res, err))
    return 2*res   # Accounts for integral from -inf to inf for 3D case, shoud be a facotr of 4 instead of 2?

def get_BCS_v_n_e(delta, mu_eff):
    """Return `(v_0, n, mu, e)` for the 1D BCS solution at T=0.

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
    v_0 : float
       Strength of the interaction.
    n : float
       Total density.
    mu : float
       Bare chemical potential (without the self-energy)
    e : float
       Energy density.
    """
    m = hbar = 1.0
    kF = np.sqrt(2*m*max(0, mu_eff))/hbar

    def gap_integrand(k):
        e_p = (hbar*k)**2/2.0/m - mu_eff
        return 1./np.sqrt(e_p**2 + abs(delta)**2)

    v_0 = 4*np.pi / quad(gap_integrand, kF)

    def n_integrand(k):
        """Density"""
        e_p = (hbar*k)**2/2.0/m - mu_eff
        denom = np.sqrt(e_p**2 + abs(delta)**2)
        return (denom - e_p)/denom

    n = quad(n_integrand, kF) / 2/np.pi

    def e_integrand(k):
        """Energy"""
        e_p = (hbar*k)**2/2.0/m - mu_eff
        denom = np.sqrt(e_p**2 + abs(delta)**2)
        return (hbar*k)**2/2.0/m * (denom - e_p)/denom
        """Where this fomula comes from?"""
    e = quad(e_integrand, kF) / 2/np.pi - v_0*n**2/4.0 - abs(delta)**2/v_0
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
    v_0, n, mu, e = get_BCS_v_n_e(delta=delta, mu_eff=mu_eff)
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

    def get_BCS_v_n_e(self, delta, mus_eff):
        """Return `(v_0, n, mu, e)` for the 1D BCS solution at T=0."""
        kF = np.sqrt(2*max(0, max(mus_eff)))

        def gap_integrand(k):
            res = self.get_res(k=k, mus_eff=mus_eff, delta=delta)
            return (self.f(res.w_m) - self.f(res.w_p))/res.E

        v_0 = 4*np.pi / quad(gap_integrand, kF,)

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

        n_m = quad(nm_integrand, kF) / 2/np.pi
        n_p = quad(np_integrand, kF) / 2/np.pi
        n_a = (n_p + n_m)/2.0
        n_b = (n_p - n_m)/2.0
        ns = np.array([n_a, n_b])
        mus = mus_eff - np.array([n_b, n_a])*v_0

        return namedtuple('BCS_Results', ['v_0', 'ns', 'mus'])(v_0, ns, mus)

"""3D homogenous system with regularization"""
class Homogeneous3D(object):
    T = 0.0
    k_cutoff = np.inf
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def f(self, E):
        if self.T == 0:
            return (1+np.sign(-E))/2.0
        else:
            return 1./(1+np.exp(E/self.T))

    def get_es(self, kz,kp, mus_eff):
        return ((kz**2+kp**2)/2.0/m - mus_eff[0],
               (kz**2+kp**2)/2.0/m - mus_eff[1])

    Results = namedtuple('Results', ['e_p', 'E', 'w_p', 'w_m'])

    def get_res(self, kz,kp, mus_eff, delta):
        e_a, e_b = self.get_es(kz,kp, mus_eff=mus_eff)
        e_p, e_m = (e_a + e_b)/2, (e_a - e_b)/2
        E = np.sqrt(e_p**2 + abs(delta)**2)
        w_p, w_m = e_m + E, e_m - E
        args = dict(locals())
        rets = self.Results(*[args[_n] for _n in self.Results._fields])
        return rets

    #def get_gap_unitary(self, delta,mus_eff, k_c):


    def get_inv_scattering_length(self, delta, mus_eff, k_c):
        kF = np.sqrt(2*max(0, max(mus_eff)))
        def gap_integrand(kz_, kp_): 
            # this integration will diverge?
            res = self.get_res(kz=kz_,kp=kp_, mus_eff=mus_eff, delta=delta)
            return kp_* (1 - self.f(res.w_p) - self.f(-res.w_m))/res.E

        self._gap_integrand = gap_integrand

        if k_c < kF:
            res, err = sp.integrate.dblquad(gap_integrand, 0, k_c, lambda x: 0, lambda x: np.sqrt(k_c**2-x**2))
        else:
            res0, err0 = sp.integrate.dblquad(gap_integrand, 0, kF, lambda x: 0, lambda x: np.sqrt(k_c**2-x**2))
            res1, err1 = sp.integrate.dblquad(gap_integrand, kF, k_c, lambda x: 0, lambda x: np.sqrt(k_c**2-x**2))
            res, err = res0 + res1, err0 + err1
        
        if abs(err) > 1e-6 and abs(err/res) > 1e-6:
                warnings.warn("scalttering integral did not converge: res, err = %g, %g" % (res, err))

        # Factor of 1/4/pi**2 but we include the factor of 2 from -k_z to k_z
        res /= 2*np.pi**2 # the result should be the 1/g, where g=-v_0
        # print(-np.pi * 2.0 * res, 2 * k_c / np.pi)
        return (-np.pi * 2.0 * res + 2 * k_c / np.pi)

    def get_BCS_v_n_e(self, delta, mus_eff, k_c=10000.0, unitary = False):
        """Return `(v_0, n, mu, e)` for the 3D BCS solution at T=0 or T > 0."""
        kF = np.sqrt(2*max(0, max(mus_eff)))

        #def gap_integrand(kz_,kp_): #this integration will divergnce?
        #    res = self.get_res(kz=kz_,kp=kp_, mus_eff=mus_eff, delta=delta)
        #    return kp_* (1 - self.f(res.w_p) - self.f(-res.w_m))/res.E
        #gap_int = dquad(f=gap_integrand, kF = kF, int_name="Gap")# bad, the result is finite, something goes wrong
        #v_0 = 4*np.pi / gap_int #without regularization, v_0 should be zero?
        if not unitary:
            ainv_s = self.get_inv_scattering_length(delta=delta, mus_eff=mus_eff, k_c=k_c)
            v_0 = np.pi * 4.0 / (ainv_s  - 2.0 * k_c/np.pi)
        else:
            v_0 = 2 * np.pi **2 / k_c

        def np_integrand(kz_, kp_):
            """Density"""
            res = self.get_res(kz=kz_,kp=kp_, mus_eff=mus_eff, delta=delta)
            n_p = 1 + res.e_p/res.E*(self.f(res.w_p) + self.f(-res.w_m) - 1)
            return n_p * kp_

        def nm_integrand(kz_, kp_):
            """Density"""
            res = self.get_res(kz=kz_,kp=kp_, mus_eff=mus_eff, delta=delta)
            n_m = self.f(res.w_p) - self.f(-res.w_m)
            return n_m * kp_

        n_m = dquad(f=nm_integrand, kF=kF, k_inf=k_c, int_name="Density Difference")/4/np.pi**2#check the factor, should change
        n_p = dquad(f=np_integrand, kF=kF, k_inf=k_c, int_name="Total Density")/4/np.pi**2#check the factor, should change
        n_a = (n_p + n_m)/2.0
        n_b = (n_p - n_m)/2.0
        ns = np.array([n_a, n_b])
        mus = mus_eff - np.array([n_b, n_a])*v_0

        return v_0, ns, mus

