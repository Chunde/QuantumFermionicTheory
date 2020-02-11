"""ASLDA class
This module provides a ASLDA method for solving the polarized
two-species Fermi gas with short-range interaction.
"""
from mmf_hfb.bcs import BCS
import numpy as np


class KernelBCS(BCS):
    
    def __init__(
        self, Nxyz, Lxyz, dx=None, T=0, 
            E_c=None, C=None, fix_C=False, **args):
        BCS.__init__(self, Nxyz=Nxyz, Lxyz=Lxyz, dx=dx, T=T, E_c=E_c)
        self.E_c = E_c
        if E_c is None:  # the max k_c need to be checked again
            self.k_c = np.max(self.kxyz)
        else:
            self.k_c = None
        self.C = C
        self._D2 = BCS._get_K(self)
        self._D1 = BCS._get_Del(self)
        self.ones = np.ones_like(sum(self.xyz))
        if bool(args):
            print(f"Unused args: {args}")
        
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
        # k_p = np.diag(np.ones_like(sum(self.xyz).ravel()) * k_p)
        # #[Check] the shape of the k_p matrix
        # K = K + k_p
        if ns is None:
            return (K, K)
        alpha_a, alpha_b = self.get_alphas(ns)
        
        if alpha_a is None or alpha_b is None:
            return (K, K)
        # K( A U') = [(A u')'= (A u)'' - A'' u + A u'']/2
        if not hasattr(alpha_a, '__iter__'):
            alpha_a = alpha_a*self.ones
            alpha_b = alpha_b*self.ones
        K_a = self._get_modified_K(K, alpha_a, **args)
        K_b = self._get_modified_K(K, alpha_b, **args)
        assert np.allclose(K_b, K_b.conj().T)
        return (K_a, K_b)

    def get_Vext(self, **args):
        """
            return the external potential
        """
        return np.array([0*np.ones_like(sum(self.xyz)), 0*np.ones_like(sum(self.xyz))])

    def output_res(self, mu_a_eff, mu_b_eff, delta, g_eff, ns, taus, nu):
        if self.C is None or not hasattr(self.C, "__iter__"):
            C = self.C
        else:
            C = self.C.flat[0]
        print(
            f"mu_a_eff={mu_a_eff.flat[0].real},\tmu_b_eff={mu_b_eff.flat[0].real},"
            +f"\tdelta={delta.flat[0].real}\tC={C},\tg={g_eff.flat[0].real},"
            +f"\tn={ns[0].flat[0].real},\ttau={taus[0].flat[0].real},"
            +f"\tnu={nu.flat[0].real}")
