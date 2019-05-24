"""ASLDA class
This module provides a ASLDA method for solving the polarized
two-species Fermi gas with short-range interaction.
"""
from mmf_hfb.Functionals import FunctionalASLDA, FunctionalBdG, FunctionalSLDA
from mmfutils.math.integrate import mquad
from mmf_hfb.bcs import BCS
from mmf_hfb.xp import xp
import numpy
import scipy.optimize


class LDA(FunctionalBdG, BCS):
    
    def __init__(self, Nxyz, Lxyz, dx=None, T=0, E_c=None):
        BCS.__init__(self, Nxyz=Nxyz, Lxyz=Lxyz, dx=dx, T=T, E_c=E_c)
        FunctionalBdG.__init__(self)
        self.E_c = E_c
        self._D2 = BCS._get_K(self)
        self._D1 = BCS._get_Del(self)

    def _get_modified_K(self, D2, alpha, twists=0, Laplacian_only=True, **args):
        """"
            return a modified kinetic density matrix
            -------------
            Laplacian_only: bool
            if True, use the relation:
                (ab')'=[(ab)'' -a''b + ab'']/2
            if False:
                (ab')=a'b'+ab''
        """
        A = xp.diag(alpha.ravel())  #[Check] ????
        if Laplacian_only:
            K = (D2.dot(A) - xp.diag(self._D2.dot(alpha.ravel())) + A.dot(D2)) / 2
        else:
            D1 =self._get_Del(twists=twists)
            dalpha = self._D1.dot(alpha.ravel())
            K = xp.diag(dalpha.ravel()).dot(D1) + A.dot(D2)
        return K

    def _get_modified_taus(self, taus, js):
        """
            return the modified taus with
            currents in it, not implement
        """
        return taus

    def get_Ks(self, twists=0, ns=None, k_p=0,  **args):
        """
            return the modified kinetic density  matrix
        Arguments
        ---------
        k_p: kinetic energy offset added to the diagonal elements
        """
        K = BCS._get_K(self, k_p=k_p, twists=twists, **args)
        #k_p = xp.diag(xp.ones_like(sum(self.xyz).ravel()) * k_p)   #[Check] the shape of the k_p matrix
        #K = K + k_p
        if ns is None:
            return (K, K)
        alpha_a, alpha_b, alpha_p = self._get_alphas(ns)
        
        if alpha_a is None or alpha_b is None:
            return (K, K)
        # K( A U') = [(A u')'= (A u)'' - A'' u + A u'']/2
        K_a = self._get_modified_K(K, alpha_a, **args)
        K_b = self._get_modified_K(K, alpha_b, **args)
        if xp == numpy:
            assert xp.allclose(K_b, K_b.conj().T)

        return (K_a, K_b)

    def get_v_ext(self, delta=0, ns=None, taus=None, kappa=0, **args):
        """
            return the modified V functional terms
        """
        if ns is None or taus is None:
            return BCS.get_v_ext(self)
        U_a, U_b = self.v_ext  # external trap
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
        C0_ = self.hbar**2/self.m
        C1_ = C0_/2
        C2_ = tau_p*C1_ - xp.conj(delta).T*kappa/alpha_p
        V_a = dalpha_m_dn_a*tau_m*C1_ + dalpha_p_dn_a*C2_ + dC_dn_a + C0_*dD_dn_a + U_a
        V_b = dalpha_m_dn_b*tau_m*C1_ + dalpha_p_dn_b*C2_ + dC_dn_b + C0_*dD_dn_b + U_b
        return (V_a, V_b)


    def get_ns_e_p(self, mus_eff, delta, N_twist=32, **args):
        """
            compute then energy density for BdG, equation(78) in page 39
            Note:
                the return value also include the pressure and densities
        """
        
        ns, taus, js, kappa = self.get_densities(mus_eff=mus_eff, delta=delta, N_twist=N_twist, struct=False)
        energy_density = self._energy_density(delta=delta, ns=ns, taus=taus, kappa=kappa)
        
        pressure = ns[0] * mus_eff[0] + ns[1]*mus_eff[1] - energy_density
        return (ns, energy_density, pressure)

class SLDA(LDA, FunctionalSLDA):

    def __init__(self, Nxyz, Lxyz, dx=None, T=0, E_c=None):
        LDA.__init__(self, Nxyz=Nxyz, Lxyz=Lxyz, dx=dx, T=T, E_c=E_c)
        FunctionalSLDA.__init__(self)

    def get_ns_e_p(self, mus_eff, delta, ns=None, taus=None, kappa=None,
            N_twist=32, max_iter=None, use_Broyden=False, **args):
        """
            compute then energy density for SLDA, equation(78) in page 39
            Note:
                the return value also include the pressure and densities
        """
        
        ns, taus, js, kappa = self.get_densities(mus_eff=mus_eff, delta=delta, N_twist=N_twist, struct=False)
        energy_density = self._energy_density(delta=delta, ns=ns, taus=taus, kappa=kappa)
        pressure = ns[0] * mus_eff[0] + ns[1]*mus_eff[1] - energy_density
        return (ns, energy_density, pressure)


class ASLDA(LDA, FunctionalASLDA):

    def __init__(self, Nxyz, Lxyz, dx=None, T=0, E_c=None):
        LDA.__init__(self, Nxyz=Nxyz, Lxyz=Lxyz, dx=dx, T=T, E_c=E_c)
        FunctionalASLDA.__init__(self)

    def get_ns_e_p(self, mus_eff, delta, ns=None, taus=None, kappa=None,
                    N_twist=32, max_iter=None, use_Broyden=False, **args):
        """
            compute then energy density for ASLDA, equation(78) in page 39
            Note:
                the return value also include the pressure and densities
        """
        assert self.dim == 2
        args = dict(args, mus_eff=mus_eff, delta=delta, struct=False, N_twist=N_twist)
        # Broyden1 method
        if use_Broyden:
            print("Use Broyden method")
            args.update(unpack=False)
            x0 = self.get_densities(**args)  # initial guess

            def f(x):
                if x is not None:
                    ns, taus, js, kappa = self._unpack_densities(x)
                    args.update(ns=ns, taus=taus, kappa=kappa)
                x_ = self.get_dens_integral(**args)
                ret = (x-x_)**2
                print(ret.max())
                return ret

            x = scipy.optimize.broyden1(f, x0, maxiter=max_iter, f_tol=1e-4)
            ns, taus, js, kappa = self._unpack_densities(x)
        # simple iteration method
        if not use_Broyden:
            print("Use simple iteration")
            ns_ = taus_ = js_ =kappa_ = None
            iter = 0
            lr = .1
            Vs = self.get_v_ext(**args)
            args.update(unpack=True, Vs=Vs, E_c=self.E_c, dim=self.dim)
            while(True):
                args.update(ns=ns, taus=taus, kappa=kappa)
                ns, taus, js, kappa = self.get_dens_integral(**args)
                print(f"{iter}:\tns={ns[0][0].max(), ns[1][0].max()}\ttaus={taus[0][0].max(),taus[1][0].max()}\tkappa={kappa[0].max().real}")
                if ns_ is not None:
                    if xp.allclose(ns_[0], ns[0]):
                        break
                    if lr < 1:
                        mr = 1 - lr
                        n, t, j, k = (ns*lr + ns_*mr), (taus*lr + taus_*mr), (js*lr + js_*mr), (kappa*lr + kappa_*mr)
                        ns_, taus_, js_, kappa_= ns, taus, js, kappa
                        ns, taus, js, kappa = n, t, j, k
                else:
                    ns_, taus_, js_, kappa_= ns, taus, js, kappa
                iter = iter + 1
                if max_iter is not None and iter > max_iter:
                    break
        energy_density = self._energy_density(ns=ns, taus=taus, kappa=kappa, **args)
        pressure = ns[0] * mus_eff[0] + ns[1]*mus_eff[1] - energy_density
        return (ns, energy_density, pressure)

