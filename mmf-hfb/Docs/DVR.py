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
import numpy as np
import matplotlib.pyplot as plt
import mmf_hfb.DVRBasis as HarmonicDVR; reload(HarmonicDVR)
from mmf_hfb.DVRBasis import HarmonicDVR
from scipy.integrate import quad
from mmfutils.math import bessel
from mmf_hfb.bcs import BCS


# # Definition of DVR

# Let $\phi_1(x), \phi_2(x)\dots \phi_n(x)$ be normalized and orthogonal basis in the Hilbert space $H$, $\{x_\alpha\}=(x_1, x_2, \dots, x_m)$ be a set of grid point in the configuration space of the system on which the coordinate system is based. Define the projector operator as:
#
# $$
# P=\sum_n{\ket{\phi_n}\bra{\phi_n}}\qquad \text{It may be easy to prove}:  P^2=P=P^{\dagger}
# $$
# Then let:
# $$
# \ket{\Delta_\alpha}=P\ket{x_\alpha}=\sum_n{\ket{\phi_n}\braket{\phi_n| x_\alpha}}=\sum_n{\phi_n^*(x_\alpha)}\ket{\phi_n}
# $$
# If these $(\ket{\Delta_1}, \ket{\Delta_2},\dots, \ket{\Delta_m})$ is complete in the subspace $S=PH$, and orthogonal, ie:
#
# $$
# \braket{\Delta_\alpha|\Delta_\beta}=N_{\alpha}\delta_{\alpha\beta}\qquad N_\alpha > 0 \tag{1}
# $$
#
# Then we say $(\ket{\Delta_1}, \ket{\Delta_2},\dots, \ket{\Delta_m})$ is the DVR set of the space $S$, we may also call $\ket{\Delta_{\alpha}}$ a DVR state, and each of such state is associated with a grid point, i.e: $x_{\alpha}$ as it's defined upon.

# ## Example

# Say we have three function in the basis: $\phi_1(x), \phi_2(x), \phi_3(x)$ associated with a set of abscissa{$x_n$}={$x_1, x_2, x_3, x_4$}, they are orthogonal,ie:
# $$
# \braket{\phi_i|\phi_j}=\int_a^b{\phi^*_i(x)\phi_j(x)}dx=\delta_{ij} \qquad
# $$
# Then:
# $$
# \ket{\Delta_1}=\sum_{n=1}^3{\phi_n^*(x_1)}\ket{\phi_n}=\phi_1^*(x_1)\ket{\phi_1}+\phi_2^*(x_1)\ket{\phi_2}+\phi_3^*(x_1)\ket{\phi_3}\\
# \ket{\Delta_2}=\sum_{n=1}^3{\phi_n^*(x_2)}\ket{\phi_n}=\phi_1^*(x_2)\ket{\phi_1}+\phi_2^*(x_2)\ket{\phi_2}+\phi_3^*(x_2)\ket{\phi_3}\\
# \ket{\Delta_3}=\sum_{n=1}^3{\phi_n^*(x_3)}\ket{\phi_n}=\phi_1^*(x_3)\ket{\phi_1}+\phi_2^*(x_3)\ket{\phi_2}+\phi_3^*(x_3)\ket{\phi_3}\\
# \ket{\Delta_4}=\sum_{n=1}^3{\phi_n^*(x_4)}\ket{\phi_n}=\phi_1^*(x_4)\ket{\phi_1}+\phi_2^*(x_4)\ket{\phi_2}+\phi_3^*(x_4)\ket{\phi_3}\\
# $$
#
# With the condition (1):

# $$
# \braket{\Delta_i|\Delta_j}=\phi^*_1(x_i)\phi_1(x_j)+\phi^*_2(x_i)\phi_2(x_j)+\phi^*_3(x_i)\phi_3(x_j)=N_i\delta_{ij}
# $$
#
# where
# $$
# N_i=\sum_{n=1}^3{\phi_n^*(x_i)\phi_n(x_i)}
# $$

# Let:
# $$
# \mat{G}=\begin{pmatrix}
# \phi_1(x_1) & \phi_1(x_2) & \phi_1(x_3) &\phi_1(x_4)\\
# \phi_2(x_1) & \phi_2(x_2) & \phi_2(x_3) &\phi_2(x_4)\\
# \phi_3(x_1) & \phi_3(x_2) & \phi_3(x_3) &\phi_3(x_4)\\
# \end{pmatrix}
# $$
# Hence, we arrive:
# $$
# G^{\dagger}G=\mat{G}=\begin{pmatrix}
# N_1 & 0 & 0 & 0\\
# 0 & N_2 & 0 & 0\\
# 0 & 0 & N_3 & 0\\
# 0 & 0 & 0 & N_4
# \end{pmatrix}
# $$
#

# Now let consider the properity of $\braket{x|\Delta_i}$
# $$
# \Delta_i(x)=\braket{x|\Delta_i}=\psi^*_1(x_i)\psi_1(x)+\psi^*_2(x_i)\psi_2(x)+\psi^*_3(x_i)\psi_3(x)
# $$
#
# if evaluate the $\Delta_i(x)$ at those grid points, it can be found:
#
# $$
# \Delta_i(x_j)=N_i\delta_{ij}
# $$
#
# This is an interesting property, the DVR state $\ket{\Delta_i}$ is localized at it's own grid point $x_i$, which means, it's only non-zero at it's own grid point. In other words, the DVR states satisfy simultaneously two properties: $Orthogonality$ and $Interpolation.$

# ## Normalized DVR
# if define:
# $$
# \ket{F_{\alpha}}=\frac{1}{\sqrt{N_{\alpha}}}\ket{\Delta_{\alpha}}\qquad\\
# $$
#
# Then
#
# $$
# \braket{F_i|F_j}=\delta_{ij} \qquad (Normalized)
# $$

# ## Expansion of States
# For a general state $\ket{\psi}$ in the sub space $\mat{H}$, then it can be expanded exactly in the DVR basis:
#
# $$
# \ket{\psi}=\sum_{n=1}^m\ket{F_n}\braket{F_n|\psi}
# $$
#
# As:
# $$
# \ket{F_i}=\frac{1}{\sqrt{N_i}}\ket{\Delta_i}=\frac{1}{\sqrt{N_i}}P\ket{x_i}
# $$

# So:
# $$
# \braket{F_i|\psi}=\frac{1}{\sqrt{N_i}}\bra{x_i}P^{\dagger}\ket{\psi}=\frac{1}{\sqrt{N_i}}\bra{x_i}P\ket{\psi}
# $$

# Because we assume $\ket{\psi}$ is in the subspace spaned by the basis, then $P\ket{\psi}=\psi$ as it's being projected to the same space.
# So the result is:
# $$
# \braket{F_i|\psi}=\frac{1}{\sqrt{N_i}}\psi(x_i)\\
# \ket{\psi}=\sum_{n=1}^m\frac{1}{\sqrt{N_i}}\psi(x_i)\ket{F_n}
# $$

# This result shows that the expansion coefficient of a state is simply connect to its value at grid points.

# ## Scalar Product
# To compute integral $\braket{\phi|\psi}$, we insert the unitary relation into the integral to get:
# $$
# \braket{\phi|\psi}=\sum_i\braket{\phi|F_i}\braket{F_i|\psi}=\sum_i \frac{1}{N_i}\phi^*(x_i)\psi(x_i)
# $$

# # 2D Harmonic System
#
# $\begin{aligned} \psi_{00} &=\left(\frac{m \omega}{\pi \hbar}\right)^{1 / 2} e^{-m \omega \rho^{2} / 2 \hbar} \\ \psi_{10} &=\sqrt{\frac{2 m \omega}{\hbar}}\left(\frac{m \omega}{\pi \hbar}\right)^{1 / 2} e^{-m \omega \rho^{2} / 2 \hbar} \rho \cos \phi \\ \psi_{01} &=\sqrt{\frac{2 m \omega}{\hbar}}\left(\frac{m \omega}{\pi \hbar}\right)^{1 / 2} e^{-m \omega \rho^{2} / 2 \hbar} \rho \sin \phi \end{aligned}$

# +
def Normalize(psi):
    """Normalize a wave function"""
    return psi/(psi.conj().dot(psi))**0.5

def HO_psi(n, m, rs):
    """
    n = E -1
        e.g if E=1, to select the corresponding
        wavefunction, use n=E-1=0, and m = 0
    m is used to pick the degerated wavefunciton
    m <=n
    """
    assert n < 4 and n >=0
    assert m <=n
    P=1
    if n ==1:
        P = rs
    elif n == 2:
        P=rs**2
        if m == 1:
            P=P-1
    elif n == 3:
        P = rs**3
        if m == 1 or m==2:
            P=P - rs/2
    return P*np.exp(-rs**2/2)


# -

# ## Construct Wavefunction from a basis

# +
plt.figure(figsize=(16, 16))
h = HarmonicDVR(nu=0, dim=2, w=1)
H = h.get_H()
Es, us = np.linalg.eigh(H)
Fs = h.get_F_rs()
print(Es[:10])
rs = np.linspace(0.01, 8, 500)
wf =sum([u*h.get_F(nu=0, n=i, rs=rs) for (i, u) in enumerate(us.T[1])])
wf_ = us.T[1]*h.ws
plt.subplot(211)
scale_factor = HO_psi(n=2, m=1, rs=rs[0])*rs[0]**0.5/wf[0]
plt.plot(rs, HO_psi(n=2, m=1, rs=rs), '+', label='Analytical')
plt.plot(h.rs, wf_*scale_factor,'o', label='Reconstructed(Fs)')
plt.plot(rs, (wf*scale_factor/rs**0.5), '-',label='Reconstructed')

plt.axhline(0, c='r', linestyle='dashed')
plt.legend()

plt.subplot(212)
plt.plot(h.rs, Fs, 'o')
for i in range(10):
    l, = plt.plot(rs, h.get_F(n=i, rs=rs), label=r'$\nu$'+f'{i}')
    plt.axvline(h.rs[i], linestyle='dashed', c=l.get_c())
plt.xlim(0, 3)
plt.legend()
# -

plt.figure(figsize=(16, 8))
h = HarmonicDVR(nu=1, dim=2, w=1)
H = h.get_H()
Es, us = np.linalg.eigh(H)
print(Es[:10])
rs = np.linspace(0.01, 5, 200)
wf =sum([u*h.get_F(n=i, rs=rs) for (i, u) in enumerate(us.T[0])])
plt.plot(rs, Normalize(wf/rs**0.5), label='Reconstructed')
plt.plot(rs, Normalize(HO_psi(n=1, m=1, rs=rs)), '+', label='Analytical')
plt.axhline(0, linestyle='dashed')
plt.legend()

# # DVR Vortices

from mmf_hfb.VortexDVR import VortexDVR

# +
mu = 10
dmu = 0
mus = (mu + dmu, mu - dmu)
delta=5
dvr = VortexDVR(mu=mu, delta=delta)
delta = delta + dvr.bases[0].zero

while(True):
    n_a, n_b, kappa = dvr.get_densities(mus=(mu,mu), delta=delta)
    delta_ = -dvr.g*kappa
    plt.plot(dvr.bases[0].rs, delta_)
    plt.plot(dvr.bases[0].rs, delta,'+')
    plt.title(f"Error={(delta-delta_).max()}")
    plt.ylabel(r"$\Delta$")
    plt.show()
    clear_output(wait=True)
    if np.allclose(delta, delta_):
        break      
    delta=delta_
# -


