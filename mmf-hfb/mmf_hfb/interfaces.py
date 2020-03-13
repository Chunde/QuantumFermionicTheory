"""Interfaces for the HFB code.
"""
from zope.interface import Attribute, Interface, implementer


class IBasis(Interface):
    """Interface for a DVR basis."""


class IHFB(Interface):
    """Interface for HFB-like codes."""
    basis = Attribute("IBasis instance.")
    functional = Attribute("IFunctional")

    def get_densities(mus_eff, delta, **args):
        """
        return the densities(may include ns, taus, js, nu)
        ----------------
        mus_eff: the effective chemical potential(mu_a, mu_b)
        delta: the gap or coupling strength
        args: other parameters
        """

    def get_ns_e_p(mus, delta, **args):
        """
        return the particle densities(n_a, n_b)
        energy density and pressure
        """


class IFunctional(Interface):
    """
    Interface for functional
    Note: Interface is unnecessary for Python
    because python function call is made by looking up
    a table with based on base classes order
    """
    def get_alpha(p, d=0):
        """
        Parameters
        ----------
        p: polarization in range of (0, 1)
        d: order of derivative(-1, 0, 1)
        """

    def get_alphas(ns, d=0):
        """
        Parameters
        ----------
        ns: densities (na, nb)
        d: order of derivative
        Note: call the get_alpha
        """

    def get_p(ns, d=0):
        """
        Parameters
        ----------
        ns: densities (na, nb)
        d: order of derivative
        """

    def get_C(ns, d=0):
        """
        Parameters
        ----------
        ns: densities (na, nb)
        d: order of derivative
        """

    def get_D(ns, d=0):
        """
        Parameters
        ----------
        ns: densities (na, nb)
        d: order of derivative
        """

    def get_beta(ns, d=0):
        """
        Parameters
        ----------
        ns: densities (na, nb)
        d: order of derivative
        """

    def get_Vs(**kw):
        """
        return the Vs(effective potential)
        """
