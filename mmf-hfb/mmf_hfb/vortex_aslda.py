"""ASLDA class
This module provides a ASLDA method for solving the polarized
two-species Fermi gas with short-range interaction.
"""
from mmf_hfb.Functionals import Functional
from mmfutils.math.integrate import mquad
from mmf_hfb.bcs import BCS
import numpy as np


class ASLDA(Functional, BCS):
    
    def __init__(self, Nxyz, Lxyz, dx=None, T=0, E_c=100):
        BCS.__init__(self, Nxyz=Nxyz, Lxyz=Lxyz, dx=dx, T=T, E_c=E_c)
        Functional.hbar, Functional.m = BCS.hbar, BCS.m
        self.E_c = E_c
        # Second order derivative operator without twisting
        self._D2 = BCS._get_K(self)
  

    def _g_eff(self, ns, Vs, mus_eff, alpha_p, dim=1):
        """
            get the effective g
        """
        V_a, V_b = Vs
        mu_p = (sum(mus_eff) - V_a + V_b) / 2
        k0 = (2*self.m/self.hbar**2*mu_p/alpha_p)**0.5
        k_c = (2*self.m/self.hbar**2 * (self.E_c + mu_p)/alpha_p)**0.5
        C = alpha_p * (sum(ns)**(1.0/3))/self.gamma
        Lambda = self._get_Lambda(k0=k0, k_c=k_c, dim=dim)
        g = alpha_p/(C - Lambda)
        return g

    def _get_modified_K(self, D2, alpha,  twists=0, **args):
        """"
            return a modified kinetic density matrix
        """
        A = np.diag(alpha)
        K = (D2.dot(A) - np.diag(self._D2.dot(alpha)) + A.dot(D2)) / 2
        return K

    def get_Ks(self, twists=0, ns=None, k_p=0, **args):
        """
            return the modified kinetic density  matrix
        Arguments
        ---------
        k_p: kinetic energy offset added to the diagonal elements
        """
        K = BCS._get_K(self, twists)

        k_p = np.diag(np.ones_like(sum(self.xyz)) * k_p)
        K = K + k_p
        alpha_a, alpha_b, alpha_p = self._get_alphas(ns)
        if alpha_a is None or alpha_b is None:
            return (K, K)
        # K( A U') = [(A u')'= (A u)'' - A'' u + A u'']/2
        K_a = self._get_modified_K(K, alpha_a)
        K_b = self._get_modified_K(K, alpha_b)
        assert np.allclose(K_b, K_b.conj().T)
        return (K_a, K_b)

    def get_v_ext(self, delta=0, ns=None, taus=None, kappa=0, **args):
        """
            get the modified V functional terms
        """
        if ns is None or taus is None:
            return BCS.get_v_ext(self)
        U_a, U_b = self.v_ext # external trap
        tau_a, tau_b = taus
        tau_p, tau_m = tau_a + tau_b, tau_a - tau_b
        alpha_a, alpha_b, alpha_p = self._get_alphas(ns)
        p = self._get_p(ns)
        dp_n_a, dp_n_b = self._dp_dn(ns)
        dalpha_p = self._dalpha_p_dp(p)
        dalpha_m = self._dalpha_m_dp(p)
        dalpha_p_dn_a, dalpha_p_dn_b = dalpha_p*dp_n_a, dalpha_p*dp_n_b
        dalpha_m_dn_a, dalpha_m_dn_b = dalpha_m*dp_n_a, dalpha_m*dp_n_b
        dC_dn_a, dC_dn_b = self._dC_dn(ns)
        dD_dn_a, dD_dn_b = self._dD_dn(ns=ns)
        C0 = self.hbar**2 /self.m
        C1 = C0 / 2
        C2 = tau_p * C1 - delta.conj().T * kappa / alpha_p
        V_a = dalpha_m_dn_a*tau_m*C1 + dalpha_p_dn_a*C2 + dC_dn_a + C0*dD_dn_a + U_a
        V_b = dalpha_m_dn_b*tau_m*C1 + dalpha_p_dn_b*C2 + dC_dn_b + C0*dD_dn_b + U_b
        return (V_a, V_b)

    def _get_modified_taus(self, taus, js):
        """
            return the modified taus with 
            currents in it, not implement
        """
        return taus

    def _unpack_densities(self, dens, N_twist=1):
        """
            unpack the densities into proper items
        """
        n_a, n_b, tau_a, tau_b, nu_real, nu_imag = (dens[0:6] /N_twist/ self.dV)
        kappa_ = (nu_real+ 1j*nu_imag)
        js = (dens[6:] /N_twist / self.dV).reshape((2, len(self.Nxyz)) + tuple(self.Nxyz))
        tau_a, tau_b = self._get_modified_taus(taus=(tau_a, tau_b), js=js)
        return ((n_a, n_b), (tau_a, tau_b), js, kappa_)

    def get_dens_integral(self, mus_eff, delta,
                                        ns=None, taus=None, kappa=None,
                                        k_c=None, N_twist=8, abs_tol=1e-6):
        """
            integrate over other dimensions by assuming it's homogeneous
            on those dimensions
            Note: These code does not work
        """
        if k_c is None:
            k_c = np.sqrt(2*self.m*self.E_c)/self.hbar

        twistss = self._get_twistss(N_twist)
        vs = self.get_v_ext(delta=delta, ns=ns, taus=taus, kappa=kappa)
        dens = 0
        args = dict(mus_eff=mus_eff,  delta=delta,ns=ns, taus=taus, kappa=kappa)
        for twists in twistss:
            def f(k=0):
                k_p = self.hbar**2/2/self.m*k**2
                H = self.get_H(vs=vs, k_p=k_p, twists=twists, **args)
                den = self._get_densities_H(H, twists=twists)
                return den
            dens = dens + mquad(f, -k_c, k_c, abs_tol=abs_tol)/2/k_c

        return self._unpack_densities(dens, N_twist=N_twist)

    def get_dens_twisting(self, mus_eff, delta,
                                    ns=None, taus=None, kappa=None,
                                    N_twist=8, abs_tol=1e-12):
        """
            average with twisting
        """
        twistss = self._get_twistss(N_twist)
        dens = 0
        args = dict(mus_eff=mus_eff, delta=delta,ns=ns, taus=taus, kappa=kappa)
        for twists in twistss:
            H = self.get_H(twists=twists, **args)
            den = self._get_densities_H(H, twists=twists)
            dens = dens + den
        return self._unpack_densities(dens, N_twist=N_twist)
