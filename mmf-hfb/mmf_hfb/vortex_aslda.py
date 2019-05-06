"""BCS Equations in 2D

This module provides a class BCS2D for solving the BCS (BdG) equations in 2D for a
two-species Fermi gas with short-range interaction.
"""
from mmf_hfb.Functionals import Functionals
from mmf_hfb.bcs import BCS
import numpy as np
from mmfutils.math.integrate import mquad


class ASLDA(Functionals, BCS):
    hbar = 1.0
    m = 1.0
    
    def __init__(self, Nxyz, Lxyz, dx=None, T=0, E_c=100):
        BCS.__init__(self, Nxyz=Nxyz, Lxyz=Lxyz, dx=dx, T=T, E_c=E_c)
        self.E_c = E_c
        self.g_eff = 1.0

    def get_alphas(self, ns=None):
        p = self._get_p(ns)
        p2 = p**2
        p4 = p2**2
        # too make code faster, equation (98) 
        # is divided into even part and odd part
        alpha_even, alpha_odd = 1.0094 + 0.532*p2*(1 - p2 + p4/3.0), 0.156*p* (1 - 2.0*p2/3.0 + p4/5.0)
        alpha_a, alpha_b = alpha_odd + alpha_even, -alpha_odd + alpha_even  #fixed an error here
        alpha_p = alpha_even # it's defined as (alpha_a + alpha_b) /2 , which is just alpha_even
        return (alpha_a, alpha_b, alpha_p)

    def get_Del(self, twist=0):
        """This should only be applied to wave functions"""
        """return the second order derivative operator matrix"""
        """Be careful:
        D = np.fft.ifft(-1j*k*np.fft.fft(np.eye(N), axis=1), axis=1)
        1) D.dot(f)
        2) np.fft.ifft(1j*k*np.fft.fft(f))
        1) and 2) will yield different results if f is complex
           In general, the fft method is better in accuracy
        """

        k_bloch = twist/self.Lx
        k = self.kx + k_bloch
        N = self.Nx

        # Unitary matrix implementing the FFT including the phase
        # twist
        U = np.exp(-1j*k[:, None]*self.x[None, :])/np.sqrt(N)
        assert np.allclose(U.conj().T.dot(U), np.eye(N))
        
        Del = np.dot(U.conj().T, (1j * k)[:, None]/2/self.m * U)
        return Del


    def get_Ks(self, k_p=0, twists=0):
        """return the original kinetic density matrix for homogeneous system"""
        K = self._get_K(twists)
        K = K + k_p
        return (K, K)

    def get_modified_K(self, D2, alpha):
        """"return a modified kinetic density  matrix"""
        "[Numerical Test Status:Pass]"
        A = np.diag(alpha)
        K = (D2.dot(A) - np.diag(self._D2.dot(alpha)) + A.dot(D2)) / 2
        return K

    def get_modified_Ks(self, alpha_a, alpha_b, twist=0):
        """return the modified kinetic density  matrix"""
        "[Numerical Test Status:Pass]"
        D2 = self.get_Laplacian(twist=twist)
        # K( A U') = [(A u')'= (A u)'' - A'' u + A u'']/2
        K_a = self.get_modified_K(D2, alpha_a)
        K_b = self.get_modified_K(D2, alpha_b)
        #assert np.allclose(K_b, K_b.conj().T)
        return (self.hbar**2/2/self.m*K_a, self.hbar**2/2/self.m*K_b)

    def get_modified_Vs(self, delta, ns=None, taus=None, kappa=0):
        """get the modified V functional terms"""
        if ns is None or taus is None:
            return self.v_ext
        U_a, U_b = self.v_ext
        tau_a, tau_b = taus
        tau_p, tau_m = tau_a + tau_b, tau_a - tau_b
        alpha_a, alpha_b, alpha_p = self.get_alphas(ns)
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

    def get_effective_g(self, ns, Vs, mus, alpha_p, dim=1):
        """get the effective g"""
        V_a, V_b = Vs
        mu_p = (sum(mus) - V_a + V_b) / 2
        k0 = (2*self.m/self.hbar**2*mu_p/alpha_p)**0.5
        kc = (2*self.m/self.hbar**2 * (self.E_c + mu_p)/alpha_p)**0.5
        C = alpha_p * (sum(ns)**(1.0/3))/self.gamma
        Lambda = self._get_Lambda(k0=k0, kc=kc, dim=dim)
        g = alpha_p/(C - Lambda)
        return g

    def get_Ks_Vs(self, delta, mus=(0, 0), ns=None, taus=None, kappa=0, ky=0, kz=0, twists=None):
        """Return the kinetic energy and modified potential matrices."""
        alphas = self.get_alphas(ns)
        k_p = self.hbar**2/2/self.m *(kz**2 + ky**2)
        k_p = np.diag(np.ones_like(sum(self.xyz))) * k_p
        if alphas is None or ns is None or taus is None:
            return (self.get_Ks(k_p=k_p, twists=twists), self.get_modified_Vs(delta=delta, ns=ns, taus=taus, kappa=kappa))
        return (self.get_Ks(k_p=k_p, twists=twists), self.get_modified_Vs(delta=delta, ns=ns, taus=taus, kappa=kappa, alphas=alphas))
        alpha_a, alpha_b, alpha_p = alphas
       
        K_a, K_b = self.get_modified_Ks(alpha_a=alpha_a, alpha_b=alpha_b, twist=twist)
        V_a, V_b = self.get_modified_Vs(delta=delta, ns=ns, taus=taus, kappa=kappa, alphas=alphas)
        self.g_eff = self.get_effective_g(ns=ns, Vs=(V_a, V_b), mus=mus, alpha_p=alpha_p)
        return ((K_a + k_p, K_b + k_p), (V_a, V_b))
            
 
    def get_H_ext(self, mus, delta, ns=None, taus=None, kappa=0, ky=0, kz=0, twists=0):
        """Return the single-particle Hamiltonian with pairing. """
        zero = np.zeros_like(sum(self.xyz))
        Delta = np.diag((delta + zero).ravel())
        mu_a, mu_b = mus
        mu_a += zero
        mu_b += zero
        (K_a, K_b), (V_a, V_b) = self.get_Ks_Vs(delta=delta, mus=mus, kappa=kappa, taus=taus, ns=ns, ky=ky, kz=kz, twists=twists)
        Mu_a, Mu_b = np.diag((mu_a - V_a).ravel()), np.diag((mu_b - V_b).ravel())
        H = np.block([[K_a - Mu_a, Delta],  # may need remove the minus sign for Delta, need to check
                     [Delta.conj(), -(K_b - Mu_b)]])
        assert np.allclose(H.real, H.conj().T.real)
        return H

    def _get_modified_taus(self, taus, js):
        """return the modified taus with currents in it, not implement"""
        return taus

    def get_ns_taus_kappa(self, H):
        """Return the n_a, n_b"""
        N = self.Nx
        Es, psi = np.linalg.eigh(H) 
        us, vs = psi.reshape(2, N, N*2)
        us, vs = us.T, vs.T
        j_a, j_b =None, None
        n_a = sum(np.abs(us[i])**2*self.f(Es[i]) for i in range(len(us)))/self.dx
        n_b = sum(np.abs(vs[i])**2*self.f(-Es[i]) for i in range(len(vs)))/self.dx
        assert not np.allclose(n_a, 0) and not np.allclose(n_b, 0)
        nabla = self.get_Del()  # should be careful, can be wrong!
        tau_a = sum(np.abs(nabla.dot(us[i]))**2*self.f(Es[i]) for i in range(len(us)))/self.dx
        tau_b = sum(np.abs(nabla.dot(vs[i]))**2*self.f(-Es[i]) for i in range(len(vs)))/self.dx
        kappa = 0.5 * sum(us[i]*vs[i].conj()*(self.f(Es[i]) - self.f(-Es[i])) for i in range(len(us)))/self.dx
        j_a = -0.5*sum((us[i].conj()*nabla.dot(us[i]) - us[i]*nabla.dot(us[i].conj()))*self.f(Es[i]) for i in range(len(us)))
        j_b = -0.5*sum((vs[i]*nabla.dot(vs[i]).conj() - vs[i].conj()*nabla.dot(vs[i]))*self.f(-Es[i]) for i in range(len(vs)))
        return ((n_a, n_b), self._get_modified_taus(taus=(tau_a, tau_b), js=(j_a, j_b)), kappa)

    def get_ns_taus_kappa_average_3d(self, mus, delta, ns=None, taus=None, kappa=None, kc=None, N_twist=8, abs_tol=1e-6):
        if kc is None:
            kc = 100*np.sqrt(2*self.m*self.E_c)/self.hbar
        twists = np.arange(0, N_twist)*2*np.pi/N_twist
        n_a, n_b, tau_a_, tau_b_, kappa_ = 0, 0, 0, 0, 0

        zero = np.zeros_like(sum((self.Nx, self.Nx)))
        Delta = np.diag((delta.reshape(self.Nx) + zero)) 
        mu_a, mu_b = mus
        mu_a, mu_b= zero + mu_a, zero + mu_b
        V_a, V_b = self.get_modified_Vs(delta=delta, ns=ns, taus=taus, kappa=kappa)
        Mu_a, Mu_b = np.diag((mu_a - V_a).ravel()), np.diag((mu_b - V_b).ravel())
        alpha_a, alpha_b, alpha_p = self.get_alphas(ns)
        for twist in twists:
            def g(kz=0):
                def f(ky=0):
                    k_p = self.hbar**2/2/self.m*ky**2
                    K_a, K_b = self.get_modified_Ks(alpha_a=alpha_a, alpha_b=alpha_b, twist=twist)
                    H = np.bmat([[K_a - Mu_a, Delta], [Delta.conj(), -(K_b - Mu_b)]])
                    assert np.allclose(H.real, H.conj().T.real)
                    _ns, _taus, _kappa = self.get_ns_taus_kappa(np.asarray(H))
                    return np.concatenate((_ns[0].ravel(), _ns[1].ravel(), _taus[0].ravel(), _taus[1].ravel(), _kappa.ravel()))
                rets = mquad(f, -kc, kc, abs_tol=abs_tol)/2/kc
                return rets
            rets = mquad(g, -kc, kc, abs_tol=abs_tol)/2/kc
            rets = rets.reshape(5, self.Nx)
            n_a = n_a + rets[0]
            n_b = n_b + rets[1]
            tau_a_ = tau_a_ + rets[2]
            tau_b_ = tau_b_ + rets[3]
            kappa_ = kappa_ + rets[4]
        return ((n_a/N_twist, n_b/N_twist), (tau_a_/N_twist, tau_b_/N_twist), kappa_/N_twist)

    def get_ns_taus_kappa_average_2d(self, mus, delta, ns=None, taus=None, kappa=None, kc=None, N_twist=8, abs_tol=1e-6):
        if kc is None:
            kc = 100*np.sqrt(2*self.m*self.E_c)/self.hbar
        twists = np.arange(0, N_twist)*2*np.pi/N_twist
        n_a, n_b, tau_a_, tau_b_, kappa_ =0, 0, 0, 0, 0

        zero = np.zeros_like(sum((self.Nx, self.Nx)))
        Delta = np.diag((delta.reshape(self.Nx) + zero)) 
        mu_a, mu_b = mus
        mu_a, mu_b= zero + mu_a, zero + mu_b
        V_a, V_b = self.get_modified_Vs(delta=delta, ns=ns, taus=taus, kappa=kappa)
        Mu_a, Mu_b = np.diag((mu_a - V_a).ravel()), np.diag((mu_b - V_b).ravel())
        alpha_a, alpha_b, alpha_p = self.get_alphas(ns)
        for twist in twists:
            def f(ky=0):
                k_p = self.hbar**2/2/self.m*ky**2
                K_a, K_b = self.get_modified_Ks(alpha_a=alpha_a, alpha_b=alpha_b, twists=twists)
                H = np.bmat([[K_a - Mu_a, Delta], # may need remove the minus sign for Delta, need to check
                     [Delta.conj(), -(K_b - Mu_b)]])
                assert np.allclose(H.real, H.conj().T.real)
                _ns, _taus, _kappa = self.get_ns_taus_kappa(np.asarray(H))
                return np.concatenate((_ns[0].ravel(), _ns[1].ravel(), _taus[0].ravel(), _taus[1].ravel(), _kappa.ravel()))
            rets = mquad(f, -kc, kc, abs_tol=abs_tol)/2/kc
            rets = rets.reshape(5, self.Nx)
            n_a = n_a + rets[0]
            n_b = n_b + rets[1]
            tau_a_ = tau_a_ + rets[2]
            tau_b_ = tau_b_ + rets[3]
            kappa_ = kappa_ + rets[4]
        return ((n_a/N_twist, n_b/N_twist), (tau_a_/N_twist, tau_b_/N_twist), kappa_/N_twist)

    def get_ns_taus_kappa_average_1d(self, mus, delta, ns=None, taus=None, kappa=None, N_twist=8, abs_tol=1e-12):
        kc = np.sqrt(2 * self.m * self.E_c)/self.hbar
        twists = np.arange(0, N_twist)*2*np.pi/N_twist
        n_a, n_b, tau_a_, tau_b_, kappa_ = 0, 0, 0, 0, 0
        for twist in twists:
            H = self.get_H(mus=mus, delta=delta, ns=ns, taus=taus, kappa=kappa, twist=twist)
            _ns, _taus, _kappa = self.get_ns_taus_kappa(H)
            n_a = n_a + _ns[0]
            n_b = n_b + _ns[1]
            tau_a_ = tau_a_ + _taus[0]
            tau_b_ = tau_b_ + _taus[1]
            kappa_ = kappa_ + _kappa
        return ((n_a/N_twist, n_b/N_twist), (tau_a_/N_twist, tau_b_/N_twist), kappa_/N_twist)

    def get_R(self, mus, delta, N_twist=1, ky=0, kz=0, twists=None):
        """Return the density matrix R."""
        Rs = []
        if twists is None:
            twists = np.arange(0, N_twist)*2*np.pi/N_twist

        for twist in twists:
            H = self.get_H_ext(mus=mus, delta=delta, ky=ky, kz=kz, twists=twist)
            d, UV = np.linalg.eigh(H)
            R = UV.dot(self.f(d)[:, None]*UV.conj().T)
            Rs.append(R)
        R = sum(Rs)/len(Rs)
        return R

    def get_R_twist_average(self, mus, delta, ky=0, kz=0, abs_tol=1e-12):
        """Return the density matrix R."""
        R0 = 1.0

        def f(twist):
            H = self.get_H(mus=mus, delta=delta, ky=ky, kz=kz, twist=twist)
            d, UV = np.linalg.eigh(H)
            R = UV.dot(self.f(d)[:, None]*UV.conj().T)
            return R/R0
        R0 = f(0)
        R = R0 * mquad(f, -np.pi, np.pi, abs_tol=abs_tol)/2/np.pi
        return R

    def get_R_per_average(self, mus, delta, abs_tol=1e-12):
        kc = np.sqrt(2 * self.m * self.E_c)/self.hbar
        
        def f(kz=0):
            def g(ky=0):
                R = self.get_R(mus=mus, delta=delta, N_twist=4, ky=ky, kz=kz)
                return R
            R = mquad(g, -kc, kc, abs_tol=abs_tol)/2/kc
            return R

        R = mquad(f, -kc, kc, abs_tol=abs_tol)/2/kc
        return R