from collections import namedtuple
import warnings
import numpy as np
import scipy.integrate
import scipy as sp

from uncertainties import ufloat


hbar = m = 1.0


def quad(f, kF=None, k_0=0, k_inf=np.inf, limit=1000):
    """Wrapper for quad that deals with singularities at the Fermi surface.
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
        warnings.warn(f"Gap integral did not converge: res,err={res},{err}")

    return ufloat(res, err)


class Homogeneous(object):
    """Homogeneous infinite-matter two-component systems with
    regularization.

    Notes
    =====
    We use the following arguments and variables.

    mus : Effective chemical potentials
    mu0s : Bare chemical potentials (when the theory has renormalization)
    delta : Gap
    """
    m = 1.0
    hbar = 1.0
    T = 0.0
    k_cutoff = np.inf
    dim = 2
    
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def f_thermal(self, E):
        """Fermi distribution function."""
        if self.T == 0:
            return (1+np.sign(-E))/2.0
        else:
            return 1./(1+np.exp(E/self.T))

    def f(self, E):
        """Weighting function including thermal and cutoff effects."""
        # This version is just the thermal factor.  Overload to
        # include different regularizations.
        return self.f_thermal(E)
    
    Results = namedtuple('Results', ['e_p', 'e0_p', 'E', 'w_p', 'w_m'])

    def _integrate(self, integrand, kF, **kw):
        if self.dim == 2:
            f = lambda k: integrand(k)*k/2/np.pi
        elif self.dim == 3:
            f = lambda k: integrand(k)*k**2/2/np.pi**2
        return quad(f, kF=kF, **kw)
        
    def _get_res(self, k, mus, delta):
        """This function computes all local results for use by the
        various integration routines.
        """
        e_a, e_b = (self.hbar*k)**2/2.0/self.m - mus[0], k**2/2.0/m - mus[1]
        e0_p = (self.hbar*k)**2/2.0/m
        e_p, e_m = (e_a + e_b)/2, (e_a - e_b)/2
        E = np.sqrt(e_p**2 + abs(delta)**2)
        w_p, w_m = e_m + E, e_m - E
        args = dict(locals())
        res = self.Results(*[args[_n] for _n in self.Results._fields])
        return res

    def get_C0(self, mus, delta):
        """Return the quantity C_tilde with the mu=0 regulator."""
        kF = np.sqrt(2*self.m*max(0, max(mus)))/self.hbar
        def _integrand(k):
            res = self._get_res(k=k, mus=mus, delta=delta)
            # Note: we use this form corresponding to (80) in
            # [Bulgac:2011] as it allows us to use f(E) for the
            # regularization.
            nu_ = delta/res.E * (self.f(res.w_m) - self.f(res.w_p))/2.0
            # nu_ = delta/res.E * (1 - self.f(-res.w_m) - self.f(res.w_p))/2.0
            return 0.5/res.e0_p - nu_/delta
        return self._integrate(_integrand, kF=kF)

    def get_nu(self, mus, delta):
        kF = np.sqrt(2*self.m*max(0, max(mus)))/self.hbar
        def _integrand(k):
            res = self._get_res(k=k, mus=mus, delta=delta)
            # Note: we use this form corresponding to (80) in
            # [Bulgac:2011] as it allows us to use f(E) for the
            # regularization.
            nu_ = delta/res.E * (self.f(res.w_m) - self.f(res.w_p))/2.0
            # nu_ = delta/res.E * (1 - self.f(-res.w_m) - self.f(res.w_p))/2.0
            return 0.5/res.e0_p - nu_/delta

    def get_kappa(self, mus, delta):
        """Return the convergent kinetic energy 
        $\kappa = \hbar^2\tau/2m - \Delta\nu$.
        """
        kF = np.sqrt(2*self.m*max(0, max(mus)))/self.hbar
        def _integrand(k):
            res = self._get_res(k=k, mus=mus, delta=delta)
            n_p = 1 - res.e_p/res.E*(self.f(res.w_m) - self.f(res.w_p))
            nu = delta/res.E * (self.f(res.w_m) - self.f(res.w_p))/2.0
            kappa_ = (self.hbar*k)**2*n_p/2/self.m - np.conj(delta)*nu
            return kappa_
        return self._integrate(_integrand, kF=kF)

    def get_n_p(self, mus, delta):
        """Return the densities (np, nm)."""
        kF = np.sqrt(2*self.m*max(0, max(mus)))/self.hbar
        def _integrand(k): 
            res = self._get_res(k=k, mus=mus, delta=delta)
            n_p = 1 - res.e_p/res.E*(self.f(res.w_m) - self.f(res.w_p))
            return n_p
        return self._integrate(_integrand, kF=kF)
        
    def get_n_m(self, mus, delta):
        """Return the densities (np, nm)."""
        kF = np.sqrt(2*self.m*max(0, max(mus)))/self.hbar
        def _integrand(k): 
            res = self._get_res(k=k, mus=mus, delta=delta)
            n_m = self.f(res.w_p) - self.f(-res.w_m)
            return n_m
        return self._integrate(_integrand, kF=kF)

    def get_E(self, mus, delta):
        """Return the energy density."""
        kappa = self.get_kappa(msus=mus, delta=delta)
        ### Missing Hartree term here
        return kappa - 0
    
