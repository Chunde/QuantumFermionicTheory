"""ASLDA class
This module provides a ASLDA method for solving the polarized
two-species Fermi gas with short-range interaction.
"""
from mmf_hfb.Functionals import Functional
from mmfutils.math.integrate import mquad
from mmf_hfb.bcs import BCS
import numpy as np


class ASLDA(Functional, BCS):
    hbar = 1.0
    m = 1.0
    
    def __init__(self, Nxyz, Lxyz, dx=None, T=0, E_c=100):
        BCS.__init__(self, Nxyz=Nxyz, Lxyz=Lxyz, dx=dx, T=T, E_c=E_c)
        Functionals.hbar, Functionals.m = BCS.hbar, BCS.m
        self.E_c = E_c
        # Second order derivative operator without twisting
        self._D2 = BCS._get_K(self)
  
    def _get_modified_K(self, D2, alpha):
        """"
            return a modified kinetic density matrix
        """
        A = np.diag(alpha)
        K = (D2.dot(A) - np.diag(self._D2.dot(alpha)) + A.dot(D2)) / 2
        return K

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

    def _get_H(self, mus_eff, delta, ns=None,
                taus=None, kappa=0, ky=0, kz=0, twists=0):
        """
            Return the single-particle Hamiltonian with pairing. 
        """
        zero = np.zeros_like(sum(self.xyz))
        Delta = np.diag((delta + zero).ravel())
        mu_a, mu_b = mus_eff
        mu_a += zero
        mu_b += zero
        (K_a, K_b), (V_a, V_b) = self._get_Ks_Vs(
            delta=delta, mus_eff=mus_eff, kappa=kappa,
            taus=taus, ns=ns, ky=ky, kz=kz, twists=twists)
        Mu_a, Mu_b = np.diag((mu_a - V_a).ravel()), np.diag((mu_b - V_b).ravel())
        H = np.block([[K_a - Mu_a, Delta],
                     [Delta.conj(), -(K_b - Mu_b)]])
        assert np.allclose(H.real, H.conj().T.real)
        return H

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

    def _get_Ks_Vs(self, delta, mus_eff=(0, 0), ns=None,
                  taus=None, kappa=0, ky=0, kz=0, twists=None):
        """
            Return the kinetic energy and modified potential matrices.
        """
        k_p = self.hbar**2/2/self.m *(kz**2 + ky**2)
        K_a, K_b = self.get_modified_Ks(ns=ns, k_p=k_p, twists=twists)
        V_a, V_b = self.get_modified_Vs(delta=delta, ns=ns, taus=taus, kappa=kappa)
        # self.g_eff = self._g_eff(ns=ns, Vs=(V_a, V_b), mus_eff=mus_eff, alpha_p=alpha_p)
        return ((K_a , K_b), (V_a, V_b))
             
    def _get_ns_taus_js_kappa(self, H, twists):
        """
            Return densities
        """
        dens = self._get_densities_H(H, twists=twists)
        return self._unpack_densities(dens)

    def get_modified_Ks(self, ns=None, k_p=0, twists=0):
        """
            return the modified kinetic density  matrix
        Arguments
        ---------
        k_p: kinetic energy offset added to the diagonal elements
        """
        K = BCS._get_K(self, twists)  # the K already has factor of hbar^2/2m
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

    def get_modified_Vs(self, delta, ns=None, taus=None, kappa=0):
        """
            get the modified V functional terms
        """
        if ns is None or taus is None:
            return self.v_ext
        U_a, U_b = self.v_ext
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
        zero = np.zeros_like(sum(self.xyz))
        Delta = np.diag((delta + zero).ravel()) 
        mu_a, mu_b = mus_eff
        V_a, V_b = self.get_modified_Vs(delta=delta, ns=ns, taus=taus, kappa=kappa)
        mu_a, mu_b= zero + mu_a - V_a, zero + mu_b - V_b
        mu_a, mu_b = np.diag(mu_a.ravel()), np.diag(mu_b.ravel())
        dens = 0
        for twists in twistss:
            def f(k=0):
                k_p = self.hbar**2/2/self.m*k**2
                K_a, K_b = self.get_modified_Ks(ns=ns, k_p=k_p, twists=twists)
                K_a, K_b = K_a + k_p, K_b + k_p
                H = np.block([[K_a - mu_a, Delta],
                    [Delta.conj(), -(K_b - mu_b)]])
                assert np.allclose(H.real, H.conj().T.real)
                den = self._get_densities_H(H, twists=twists)
                return den
            dens = dens + mquad(f, -k_c, k_c, abs_tol=abs_tol)/2/k_c

        return self._unpack_densities(dens, N_twist=N_twist)

    def get_dens_twisting(self, mus_eff, delta,
                                    ns=None, taus=None, kappa=None,
                                    N_twist=8, abs_tol=1e-12):
        """
            average over multiple twists
        """
        twistss = self._get_twistss(N_twist)
        dens = 0
        for twists in twistss:
            H = self._get_H(
                mus_eff=mus_eff, delta=delta, ns=ns,
                taus=taus, kappa=kappa, twists=twists)
            den = self._get_densities_H(H, twists=twists)
            dens = dens + den
        return self._unpack_densities(dens, N_twist=N_twist)
