"""ASLDA class
This module provides a ASLDA method for solving the polarized
two-species Fermi gas with short-range interaction.
"""
from mmf_hfb.homogeneous import Homogeneous
import numpy as np
import scipy.optimize


class homogeneous_kernel(Homogeneous):
    
    def __init__(
        self, mu_eff, dmu_eff, delta=1, m=1, T=0,
            hbar=1, k_c=None, C=None, dim=3, fix_C=False, **args):
        kcs=[1000, 1000, 50]
        if k_c is None:
            k_c = kcs[dim - 1]
        self.m = m
        self._tf_args = dict(m_a=m, m_b=m, dim=dim, hbar=hbar, T=T, k_c=k_c)
        Homogeneous.__init__(self, **self._tf_args)
        self.C = C
        self.mus_eff = (mu_eff, dmu_eff)
        self.delta = delta
        self.k_c = k_c
        self.xyz = np.array([[1]])  # used for dimensionality check

    def get_Vext(self, **args):
        """
            return the external potential
        """
        return np.array([0, 0])

    def output_res(self, mu_a_eff, mu_b_eff, delta, g_eff, ns, taus, nu):
        print(
            f"mu_a_eff={mu_a_eff},\tmu_b_eff={mu_b_eff},\tdelta={delta}"
            +f"\tC={self.C},\tg={g_eff},\tn={ns[0]},\ttau={taus[0]},\tnu={nu}")
