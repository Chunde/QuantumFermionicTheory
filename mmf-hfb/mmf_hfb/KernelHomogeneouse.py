"""ASLDA class
This module provides a ASLDA method for solving the polarized
two-species Fermi gas with short-range interaction.
"""
from mmf_hfb.homogeneous import Homogeneous
import numpy as np


class KernelHomogeneous(Homogeneous):
    
    def __init__(
        self, mu_eff, dmu_eff, delta=1, m=1, T=0,
            hbar=1, k_c=None, C=None, dim=3, fix_C=False):
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
     
    def get_v_ext(self):
        """
            return the external potential
        """
        return np.array([0, 0])
