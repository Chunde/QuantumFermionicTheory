"""ASLDA class
This module provides a ASLDA method for solving the polarized
two-species Fermi gas with short-range interaction.
"""
from mmf_hfb.Functionals import FunctionalASLDA, FunctionalBdG, FunctionalSLDA
from mmf_hfb.bcs import BCS
import numpy as np
import numpy
import scipy.optimize


class BDG(FunctionalBdG, BCS):
    
    def __init__(self, Nxyz, Lxyz, dx=None, T=0, E_c=None, C=None, fix_C=False):
        BCS.__init__(self, Nxyz=Nxyz, Lxyz=Lxyz, dx=dx, T=T, E_c=E_c)
        FunctionalBdG.__init__(self)
        self.E_c = E_c
        self.C = C
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
        A = np.diag(alpha.ravel())  # [Check] ????
        if Laplacian_only:
            K = (D2.dot(A) - np.diag(self._D2.dot(alpha.ravel())) + A.dot(D2)) / 2
        else:
            D1 =self._get_Del(twists=twists)
            dalpha = self._D1.dot(alpha.ravel())
            K = np.diag(dalpha.ravel()).dot(D1) + A.dot(D2)
        return K

    def _get_modified_taus(self, taus, js):
        """
            return the modified taus with
            currents in it, not implement
        """
        return taus

    def get_Ks(self, twists=0, ns=None, k_p=0, **args):
        """
            return the modified kinetic density  matrix
        Arguments
        ---------
        k_p: kinetic energy offset added to the diagonal elements
        """
        K = BCS._get_K(self, k_p=k_p, twists=twists, **args)
        #k_p = np.diag(np.ones_like(sum(self.xyz).ravel()) * k_p)   #[Check] the shape of the k_p matrix
        #K = K + k_p
        if ns is None:
            return (K, K)
        alpha_a, alpha_b = self._get_alphas(ns)
        
        if alpha_a is None or alpha_b is None:
            return (K, K)
        # K( A U') = [(A u')'= (A u)'' - A'' u + A u'']/2
        K_a = self._get_modified_K(K, alpha_a, **args)
        K_b = self._get_modified_K(K, alpha_b, **args)
        if np == numpy:
            assert np.allclose(K_b, K_b.conj().T)
        return (K_a, K_b)

    def get_v_ext(self, **args):
        """
            return the external potential
        """
        return np.array([0, 0])

    def solve(self, mus, delta, use_Broyden=True):
        """use the Broyden solver may be much faster"""
        mu, dmu = mus
        mu_a, mu_b = mu + dmu, mu - dmu
        mu_a_eff, mu_b_eff = np.array([mu_a, mu_b]) + self.get_Vs(delta=delta)
        mu_a_eff, mu_b_eff = mu_a_eff*np.ones_like(sum(self.xyz)), mu_b_eff*np.ones_like(sum(self.xyz))
        delta = delta*np.ones_like(sum(self.xyz))
        if use_Broyden:

            def fun(x):
                mu_a_eff, mu_b_eff, delta = x
                res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta)
                ns, taus, nu = (res.n_a.n, res.n_b.n), (res.tau_a.n, res.tau_b.n), res.nu.n
                mu_a_eff_, mu_b_eff_ = (
                    np.array([mu_a, mu_b])
                    + self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu))
                g_eff = self._g_eff(
                    mus_eff=(mu_a_eff_, mu_b_eff_), ns=ns,
                    dim=self.dim, E_c=self.E_c)
                delta_ = g_eff*nu
                print(
                    f"mu_a_eff={mu_a_eff},\tmu_b_eff={mu_b_eff},\tdelta={delta}"
                    +f"\tC={self.C},\tg={g_eff},\tn={ns[0]},\ttau={taus[0]},\tnu={nu}")
                x_ = np.array([mu_a_eff_, mu_b_eff_, delta_])
                return x - x_

            x0 = np.array([mu_a_eff, mu_b_eff, delta])  # initial guess
            x = scipy.optimize.broyden1(fun, x0, maxiter=100, f_tol=1e-4)
            mu_a_eff, mu_b_eff, delta = x
            res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta)
            ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
            g_eff = self._g_eff(
                mus_eff=(mu_a_eff, mu_b_eff), ns=ns,
                dim=self.dim, k_c=self.k_c, E_c=self.k_c**2/2/self.m)
        else:
            while(True):
                res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta)
                ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
                V_a, V_b = self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu)
                mu_a_eff_, mu_b_eff_ = mu_a + V_a, mu_b + V_b
                g_eff = self._g_eff(
                    mus_eff=(
                        mu_a_eff_, mu_b_eff_), ns=ns,
                        dim=self.dim, E_c=self.E_c)
                delta_ = g_eff*nu
                if np.allclose(np.stack([mu_a_eff_, mu_b_eff_, delta_],axis=1), np.stack([mu_a_eff, mu_b_eff, delta],axis=1), rtol=1e-8):
                    break
                delta, mu_a_eff, mu_b_eff = delta_, mu_a_eff_, mu_b_eff_
                print(f"mu_a_eff={mu_a_eff},\tmu_b_eff={mu_b_eff},\tdelta={delta}"
                      +f"\tC={self.C},\tg={g_eff},\tn={ns[0]},\ttau={taus[0]},\tnu={nu}")
        return (ns, taus, nu, g_eff, delta, mu_a_eff, mu_b_eff)

    # def get_ns_e_p(self, mus_eff, delta, N_twist=32, **args):
    #     """
    #         compute then energy density for BdG, equation(78) in page 39
    #         Note:
    #             the return value also include the pressure and densities
    #     """
    #     ns, taus, js, kappa = self.get_densities(mus_eff=mus_eff, delta=delta, N_twist=N_twist, struct=False)
    #     energy_density = self._energy_density(delta=delta, ns=ns, taus=taus, kappa=kappa)  # [bad]
    #     pressure = ns[0] * mus_eff[0] + ns[1]*mus_eff[1] - energy_density
    #     return (ns, energy_density, pressure)
    def get_ns_e_p(self, mus, delta, update_C, use_Broyden=False, **args):
        """
            compute then energy density for BdG, equation(77) in page 39
            Note:
                the return value also include the pressure and densities
            -------------
            mus = (mu, dmu)
        """
        if delta is None:
            delta = self.delta
        mu, dmu = mus
        mu_a, mu_b = mu + dmu, mu - dmu
        ns, taus, nu, g_eff, delta, mu_a_eff, mu_b_eff = self.solve(
            mus=mus, delta=delta, use_Broyden=use_Broyden)
        alpha_a, alpha_b = self.get_alphas(ns=ns)
        D = self.get_D(ns=ns)
        energy_density = taus[0]/2.0 + taus[1]/2.0 + g_eff*abs(nu)**2
        if self.T !=0:
            energy_density = (
                energy_density
                + self.T*self.get_entropy(mus_eff=(mu_a_eff, mu_b_eff), delta=delta).n)
        energy_density = energy_density - D
        pressure = ns[0]*mu_a + ns[1]*mu_b - energy_density
        if update_C:
            self.C = self.get_C(ns)
        return (ns, energy_density, pressure)


class SLDA(BDG, FunctionalSLDA):

    def get_alphas(self, ns, d=0):
        if d==0:
            return (1.0, 1.0)
        elif d==1:
            return (0, 0, 0, 0)


class ASLDA(SLDA, FunctionalASLDA):
    pass
