"""ASLDA class
This module provides a ASLDA method for solving the polarized
two-species Fermi gas with short-range interaction.
"""
from mmf_hfb.Functionals import FunctionalASLDA as Functional, FunctionalType
from mmfutils.math.integrate import mquad
from mmf_hfb.bcs import BCS
from mmf_hfb.xp import xp, allclose
import numpy

class ASLDA(Functional, BCS):
    
    def __init__(self, Nxyz, Lxyz, dx=None, T=0, E_c=100):
        BCS.__init__(self, Nxyz=Nxyz, Lxyz=Lxyz, dx=dx, T=T, E_c=E_c)
        Functional.__init__(self)
        self.E_c = E_c
        self._D2 = BCS._get_K(self)
        self._D1 = BCS._get_Del(self)
    @property
    def dim(self):
        """Need to override this property in BCS"""
        return len(self.Nxyz)

    def _get_Lambda(self, k0, k_c, dim=1):
        """return the renormalization condition parameter Lambda"""
        if dim ==3:
            Lambda = self.m/self.hbar**2/2/xp.pi**2*(1.0 - k0/k_c/2*xp.log((k_c+k0)/(k_c-k0)))
        elif dim == 2:
            Lambda = self.m /self.hbar**2/4/xp.pi*xp.log((k_c/k0)**2 - 1)
        elif dim == 1:
            Lambda = self.m/self.hbar**2/2/xp.pi*xp.log((k_c-k0)/(k_c+k0))/k0
        return Lambda

    
    def _get_modified_K(self, D2, alpha, twists=0,  Laplacian_only=True, **args):
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

    def get_Ks(self, twists=0, ns=None, k_p=0,  **args):
        """
            return the modified kinetic density  matrix
        Arguments
        ---------
        k_p: kinetic energy offset added to the diagonal elements
        """
        K = BCS._get_K(self, twists)
        if ns is None:
            return (K, K)
        k_p = xp.diag(xp.ones_like(sum(self.xyz).ravel()) * k_p)   #[Check] the shape of the k_p matrix
        K = K + k_p
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

    def _get_modified_taus(self, taus, js):
        """
            return the modified taus with
            currents in it, not implement
        """
        return taus

    def get_dens_integral(self, mus_eff, delta,
                            ns=None, taus=None, kappa=None,
                            k_c=None, N_twist=32, abs_tol=1e-6):
        """
            integrate over other dimensions by assuming it's homogeneous
            on those dimensions
            Note: These code does not work
        """
        if k_c is None:
            k_c = xp.sqrt(2*self.m*self.E_c)/self.hbar

        twistss = self._get_twistss(N_twist)
        args = dict(mus_eff=mus_eff, delta=delta, ns=ns, taus=taus, kappa=kappa)
        vs = self.get_v_ext(**args)
        dens = 0
        N_=0
        for twists in twistss:
            def f(k=0):
                k_p = self.hbar**2/2/self.m*k**2
                H = self.get_H(vs=vs, k_p=k_p, twists=twists, **args)
                den = self._get_densities_H(H, twists=twists)
                return den
            dens = dens + mquad(f, -k_c, k_c, abs_tol=abs_tol)/2/k_c
            N_=N_ + 1
        dens = dens/N_

        return self._unpack_densities(dens, struct=False)

    def get_ns_e_p(self, mus_eff, delta,
            ns=None, taus=None, kappa=None, N_twist=32, **args):
        """
            compute then energy density for ASLDA, equation(78) in page 39
            Note:
                the return value also include the pressure and densities
        """
        
        if self.FunctionalType == FunctionalType.BDG:
            ns, taus, js, kappa = self.get_densities(mus_eff=mus_eff, delta=delta, N_twist=N_twist,  struct=False)
            g_eff = self._g_eff(mus_eff=mus_eff, delta=delta, kappa=kappa)
            energy_density = (taus[0] + taus[1])*self.hbar**2/2/self.m
        elif self.FunctionalType == FunctionalType.SLDA:
            raise Exception("Not implemented")
            assert xp.allclose(alpha_a, alpha_b)
            energy_density = alpha_a*(taus[0] + taus[1])/2.0  + self._Beta(ns)*(3*xp.pi**2.0)**(2.0/3)*(ns[0] + ns[1])**(5.0/3)*3.0/10
            energy_density = energy_density *self.hbar**2/self.m
        elif self.FunctionalType == FunctionalType.ASLDA:
            args = dict(args, mus_eff=mus_eff, delta=delta, struct=False,N_twist=N_twist)
            ns_ = None
            while(True):
                args.update(ns=ns, taus=taus, kappa=kappa)
                ns, taus, js, kappa = self.get_densities( **args)
                print(f"ns={ns[0][0].max(), ns[1][0].max()}\ttaus={taus[0][0].max(),taus[1][0].max()}\tkappa={kappa[0].max().real}")
                
                if ns_ is not None and xp.allclose(ns_[0], ns[0]):
                    break
                ns_ = ns
            D = self._D(ns)
            Vs = self.get_v_ext(**args)
            alpha_a, alpha_b, alpha_p = self._get_alphas(ns)
            g_eff = self._g_eff(alpha_p=alpha_p, Vs=Vs, dim=self.dim, **args)
            energy_density = (alpha_a*taus[0]/2.0 + alpha_b*taus[1]/2.0 + D)*self.hbar**2/self.m
        else:
            raise ValueError('Unsupported functional type')

        energy_density = energy_density - g_eff * kappa.T.conj()*kappa
        pressure = ns[0] * mus_eff[0] + ns[1]*mus_eff[1] - energy_density
        return (ns, energy_density, pressure)
    