# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import *

# # Harmonic Potential

# +
import math
import numpy as np
import scipy as sp


class HarmonicOscillator(object):
    """
    1D quantum harmonic Oscillator class
    to give exact wave function
    """
    w = m = hbar= 1

    def __init__(self, w=1, m=1, dim=1):
        """support 1d, will be generalized later"""
        assert dim == 1
        self.w = w
        self.m = m
    
    def get_wf(self, x, n=0):
        """return the wavefunction"""
        C1 = 1/np.sqrt(2**math.factorial(n))*(self.m*self.w/np.pi/self.hbar)**0.25
        C2 = np.exp(-1*self.m*self.w*x**2/2/self.hbar)
        Hn = sp.special.eval_hermite(n, np.sqrt(self.m*self.w/self.hbar)*x)
        return C1*C2*Hn
    
    def get_E(self, n):
        """return eigen value"""
        return self.hbar*self.w*(n+0.5)


# -

x = np.linspace(-5, 5, 200)
qho = HarmonicOscillator()
for n in range(5):
    wf = qho.get_wf(x, n=n)
    plt.plot(x, wf, label=f'n={n}')
plt.axhline(0, linestyle='dashed')
plt.legend()

# # Pöschl–Teller potential
#



# # Morse potential


