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

# A [Pöschl–Teller](https://en.wikipedia.org/wiki/P%C3%B6schl%E2%80%93Teller_potential) potential is defined:
# $$V(x)=-{\frac {\lambda (\lambda +1)}{2}}\mathrm {sech} ^{2}(x)$$
# and the solutions of the time-independent Schrödinger equation
#
# $$-{\frac {1}{2}}\psi ''(x)+V(x)\psi (x)=E\psi (x)$$
# with this potential can be found by virtue of the substitution $u=\mathrm {tanh(x)}$, which yields
#
# $$\left[(1-u^{2})\psi '(u)\right]'+\lambda (\lambda +1)\psi (u)+{\frac {2E}{1-u^{2}}}\psi (u)=0$$
# Thus the solutions $\psi (u)$ are just the Legendre functions $P_{\lambda }^{\mu }(\tanh(x))$ with $E={\frac {-\mu ^{2}}{2}}$, and $\lambda =1,2,3\cdots$, $\mu =1,2,\cdots ,\lambda -1,\lambda$. Moreover, eigenvalues and scattering data can be explicitly computed. In the special case of integer $\lambda$ , the potential is reflectionless and such potentials also arise as the N-soliton solutions of the Korteweg-de Vries equation.
#
# The more general form of the potential is given by:
#
# $$V(x)=-{\frac {\lambda (\lambda +1)}{2}}\mathrm {sech} ^{2}(x)-{\frac {\nu (\nu +1)}{2}}\mathrm {csch} ^{2}(x)$$



# # Morse potential

# * From [Wikipedia](https://en.wikipedia.org/wiki/Morse_potential), the Morse potential energy function is of the form
#
# $$ V'(r)=D_{e}(1-e^{-a(r-r_{e})})^{2}$$
# Here $ r$ is the distance between the atoms, $ r_{e}$ is the equilibrium bond distance, $ D_{e}$ is the well depth (defined relative to the dissociated atoms), and $a$ controls the 'width' of the potential (the smaller $a$ is, the larger the well). The dissociation energy of the bond can be calculated by subtracting the zero point energy $E_{0}$ from the depth of the well. The force constant (stiffness) of the bond can be found by Taylor expansion of $ V'(r)$ around $ r=r_{e}$ to the second derivative of the potential energy function, from which it can be shown that the parameter, $ a$, is
#
# $$a={\sqrt {k_{e}/2D_{e}}},$$
# where $ k_{e}$ is the force constant at the minimum of the well.
#
# Since the zero of potential energy is arbitrary, the equation for the Morse potential can be rewritten any number of ways by adding or subtracting a constant value. When it is used to model the atom-surface interaction, the energy zero can be redefined so that the Morse potential becomes
#
# $$V(r)=V'(r)-D_{e}=D_{e}(1-e^{-a(r-r_{e})})^{2}-D_{e}$$
# which is usually written as
#
# $$V(r)=D_{e}(e^{-2a(r-r_{e})}-2e^{-a(r-r_{e})})$$
# where $r$ is now the coordinate perpendicular to the surface. This form approaches zero at infinite $r$ and equals $-D_{e}$ at its minimum, i.e. $r=r_{e}$. It clearly shows that the Morse potential is the combination of a short-range repulsion term (the former) and a long-range attractive term (the latter), analogous to the Lennard-Jones potential.


