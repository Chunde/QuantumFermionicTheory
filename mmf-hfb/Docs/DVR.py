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

# # Schrodinger Equaiton in Spherical Coordinates

# In spherical coordinates, the Laplancian can be put:
# $$
# \begin{aligned} \nabla^{2} f=\frac{1}{r^{2}} \frac{\partial}{\partial r} &\left(r^{2} \frac{\partial f}{\partial r}\right)+\frac{1}{r^{2} \sin \theta} \frac{\partial}{\partial \theta}\left(\sin \theta \frac{\partial f}{\partial \theta}\right) +\frac{1}{r^{2} \sin ^{2} \theta}\left(\frac{\partial^{2} f}{\partial \phi^{2}}\right) \end{aligned}
# $$
#
# After some manipulation:
#
# $$
# {\frac{d^{2} \Phi}{d \phi^{2}}=-m^{2} \Phi} \\ 
# {\sin \theta \frac{d}{d \theta}\left(\sin \theta \frac{d \Theta}{d \theta}\right)+l(l+1) \sin ^{2} \theta \Theta=m^{2} \Theta} \\
# {\frac{d}{d r}\left(r^{2} \frac{d R}{d r}\right)-\frac{2 m r^{2}}{\hbar^{2}}[V(r)-E] R=l(l+1) R}
# $$
#
# where $m^2$ and $l(l+1)$ are constants

# The equation for radial wavefunction can be simplified by define $u(r)=rR(r)$:
#
# $$
# -\frac{\hbar^{2}}{2 m} \frac{d^{2} u}{d r^{2}}+\left[V+\frac{\hbar^{2}}{2 m} \frac{l(l+1)}{r^{2}}\right] u=E u
# $$

# ## Dimensionality $d\ne 3$
# $$
#   \nabla^2\psi(r, \Omega) 
#   = \frac{Y_l^m(\uvect{x})}{r^{(d-1)/2}}\left[
#     \diff[2]{}{r} - \frac{\nu_{d,l}^2 - 1/4}{r^2}
#   \right]u(r), \qquad
#   \nu_{d,l} = l + \frac{d}{2} - 1.
# $$
# where
#
# $$
#   \psi(\vect{x}) = \frac{1}{r^{(d-1)/2}}u(r)Y_l^m(\uvect{x}),
# $$

# # Example: 2D Harmonic

from scipy.integrate import quad
from mmfutils.math import bessel

xs = np.linspace(0, 10*np.pi, 500)
for nu in range(4):
    js = bessel.J(nu, 0)(xs)
    plt.plot(xs,js)
plt.axhline(0, linestyle='dashed')


# # The first three state wavefunctions
#
# $\begin{aligned} \psi_{00} &=\left(\frac{m \omega}{\pi \hbar}\right)^{1 / 2} e^{-m \omega \rho^{2} / 2 \hbar} \\ \psi_{10} &=\sqrt{\frac{2 m \omega}{\hbar}}\left(\frac{m \omega}{\pi \hbar}\right)^{1 / 2} e^{-m \omega \rho^{2} / 2 \hbar} \rho \cos \phi \\ \psi_{01} &=\sqrt{\frac{2 m \omega}{\hbar}}\left(\frac{m \omega}{\pi \hbar}\right)^{1 / 2} e^{-m \omega \rho^{2} / 2 \hbar} \rho \sin \phi \end{aligned}$

# +
def Normalize(psi):
    return psi
    return psi/psi.dot(psi.conj())**0.5

def psi00(rs):
    psi = np.sqrt(1.0/np.pi)*np.exp(-rs**2/2)
    return psi

def psi10(rs):
    psi = np.sqrt(2.0)*psi00(rs)*rs
    return psi


# -

rs = np.linspace(0, 2, 100)
psi = psi10(rs)
plt.plot(rs, psi)

import mmf_hfb.HarmonicDVR as HarmonicDVR; reload(HarmonicDVR)
from mmf_hfb.HarmonicDVR import HarmonicDVR

h = HarmonicDVR(nu=2, dim=2)
rs = np.linspace(0, 5, 100)
for r in h.rs:
    plt.axvline(r, linestyle='dashed')
plt.axhline(0, linestyle='dotted')
Fs = h.get_F(nu=0, n=3, rs=rs)
plt.plot(rs, Fs)
Fs = h.get_F_rs()
plt.plot(h.rs, Fs, '+')


def spectrum(nu=0):
    h = HarmonicDVR(nu=nu, dim=2, w=1)
    H = h.get_H()
    Fs = h.get_F_rs()    
    Es, us = np.linalg.eigh(H)
    for i in range(2):
        psi = h.get_psi(us[:,i])*Fs
        plt.plot(h.rs,psi, label=f'{i}')
    plt.plot(psi00(h.rs), '--', label='00')
    plt.plot(psi10(h.rs), '+', label='01')
    print(Es)
    plt.axhline(0, linestyle='dashed')
    plt.legend()


spectrum(nu=0)

# # Harmonic in Plane Basis
# * Make sure the plane wave basis yields the desired result

from mmf_hfb.bcs import BCS

Nx = 128
L = 23
dim = 1
dx = L/Nx
bcs = BCS(Nxyz=(Nx,)*dim, Lxyz=(L,)*dim)
x = bcs.xyz

V=sum(np.array(x)**2/2.0).ravel()
K = bcs._get_K()
H = K + np.diag(V)

Es, phis = np.linalg.eigh(H)

Es[:40]

for i in range(4):
    plt.plot(x[0], np.array(phis).T[i])
plt.axhline(0, linestyle='dashed')


# # Harmonic in Sinc Basis

# +
class DVR1D(object):
    r"""Sinc function basis for non-periodic functions over an interval
    `x0 +- L/2` with `N` points."""
    def __init__(self, N, L, x0=0.0):
        L = float(L)
        self.N = N
        self.L = L
        self.x0 = x0
        self.a = L/N
        self.n = np.arange(N)
        self.x = self.x0 + self.n*self.a - self.L/2.0 + self.a/2.0
        self.k_max = np.pi/self.a

    def H(self, V):
        """Return the Hamiltonian with the give potential."""
        _m = self.n[:, None]
        _n = self.n[None, :]
        K = 2.0*(-1.0)**(_m-_n)/(_m-_n)**2/self.a**2
        K[self.n, self.n] = np.pi**2/3/self.a**2
        K *= 0.5   # p^2/2/m
        V = np.diag(V(self.x))
        return K + V

    def F(self, x=None):
        """Return the DVR basis vectors"""
        if x is None:
            x_m = self.x[:, None]
        else:
            x_m = np.asarray(x)[:, None]
        x_n = self.x[None, :]
        return np.sinc((x_m-x_n)/self.a)/np.sqrt(self.a)


class DVRPeriodic(DVR1D):
    r"""Sinc function basis for periodic functions over an interval
    `x0 +- L/2` with `N` points."""
    def __init__(self, *v, **kw):
        # Small shift here for consistent abscissa
        DVR1D.__init__(self, *v, **kw)
        self.x -= self.a/2.0
        
    def H(self, V):
        """Return the Hamiltonian with the give potential."""
        _m = self.n[:, None]
        _n = self.n[None, :]
        _arg = np.pi*(_m-_n)/self.N
        if (0 == self.N % 2):
            K = 2.0*(-1.0)**(_m-_n)/np.sin(_arg)**2
            K[self.n, self.n] = (self.N**2 + 2.0)/3.0
        else:
            K = 2.0*(-1)**(_m-_n)*np.cos(_arg)/np.sin(_arg)**2
            K[self.n, self.n] = (self.N**2 - 1.0)/3.0
        K *= 0.5*(np.pi/self.L)**2   # p^2/2/m
        V = np.diag(V(self.x))
        return K + V

    def F(self, x=None):
        """Return the DVR basis vectors"""
        if x is None:
            x_m = self.x[:, None]
        else:
            x_m = np.asarray(x)[:, None]
        x_n = self.x[None, :]
        F = np.sinc((x_m-x_n)/self.a)/np.sinc((x_m-x_n)/self.L)/np.sqrt(self.a)
        if (0 == self.N % 2):
            F *= np.exp(-1j*np.pi*(x_m-x_n)/self.L)
        return F


# +
Ns = [30, 40, 50]

def V(x):
   r"""HO potential"""
   return w**2*x**2/2.0

fig = plt.figure(figsize=(10,5))
ax = []
for _n, DVR in enumerate([DVR1D, DVRPeriodic]):
    ax.append(plt.subplot(1,2,_n+1))
    a = 1.0
    k_c = np.pi/a
    for _N in Ns:
        _L = np.pi*_N/k_c
        w = 2.0*np.pi/_L
        dvr = DVR(N=_N, L=_L)
        E = np.linalg.eigvalsh(dvr.H(V=V))
        n = np.arange(_N)
        _En = (n + 0.5)*w
        plt.semilogy(n, abs(E-_En) + 1e-16, 'b-+')

    _L = 30.0
    w = 2.0*np.pi/_L
    for _N in [60, 90]:
        a = _L / _N
        k_c = np.pi/a
        dvr = DVR(N=_N, L=_L)

        E = np.linalg.eigvalsh(dvr.H(V=V))
        n = np.arange(_N)
        _En = (n + 0.5)*w
        plt.semilogy(n, abs(E-_En) + 1e-16, 'r:.')

    plt.axis([0, 50, 1e-16, 100])
    plt.xlabel(r'$n$')
    if _n == 0:
        plt.ylabel(r'$\Delta E$')
    plt.title(DVR.__name__)
# -

# # Discrete Variable Representation (DVR) Method
# Create a program code which applies the discrete variable representation (DVR) method to calculate the eigenvectors and eigenstates for a one-dimensional quantum system in a Morse oscillator potential.
#
# ## DVR in a Nutshell
# The DVR (Discrete Variable Representation) method also known as pseudo-spectral method is one of the most widely-used numerical schemes for wave packet propagation. This general and grid-based method can be summarized in three essential steps:

# ## The DVR Algorithm
# Consider a one-dimensional quantum system with coordinate x ∈ (a, b). The kinetic energy operator is deﬁned by:
# $$
# \hat{T}=-\frac{\hbar^{2}}{2 m} \frac{\mathrm{d}^{2}}{\mathrm{d} \mathrm{x}^{2}}
# $$
#
# and the potential energy is given by the Morse potential
#
# $$
# V(x)=D_{0}\left(1-\exp \left(-\alpha\left(x-x_{0}\right)\right)\right)^{2}
# $$
# The grid xi is equally spaced
# $$
# x_{i}=a+(b-a) i / N, \quad i=1, \ldots, N
# $$
# ### Basis Functions
# $$
# \operatorname{sinc}^{2}(\mathrm{x})=\frac{\sin ^{2}(\mathrm{mx})}{(\mathrm{mx})^{2}}, \quad \mathrm{m} \in \mathbb{N}
# $$

# First, construct the overlap matrix $S^{FBR}$ in ﬁnite basis representation
#
# $$
# {S}^{\mathrm{FBR}}=\left(\begin{array}{ccc}{s_{11}} & {\cdots} & {s_{N 1}} \\ {\vdots} & {\ddots} & {\vdots} \\ {s_{1 N}} & {\cdots} & {s_{N N}}\end{array}\right)
# $$
#
# With
#
# $$
# s_{m n}=\int_{a}^{b} \frac{\sin ^{2}(m x) \sin ^{2}(n x)}{(m n)^{2} x^{4}} \mathrm{d} \mathrm{x}
# $$

from  scipy.integrate import quad

a=-5
b=5
N=11
D0=1
alpha=1
x0=0
x = np.linspace(a,b,N)
V = D0*(1-np.exp(-alpha*(x-x0)))**2
ms = np.linspace(1, N, N)
ns = ms
mm, nn = np.meshgrid(ms, ns)

plt.plot(x, V)


def fillM(fun):
    M = np.zeros_like(mm)
    for i in reversed(range(N)):
        for j in reversed(range(i)):
            M[i][j] = fun(i+1, j+1)
            M[j][i]= M[i][j]
    for i in range(1, N):
        M[i][i]=fun(i+1, i +1)
    return M


def smn(m, n):
    def integrand(x):
        return (np.sinc(m*x)*np.sinc(n*x)/m/n)**2
    return quad(integrand, a, b)[0]


S_FBR = fillM(smn)


# Compute the quadrature matrix $X^{FBR}$
#
# $$
# {X}^{\mathrm{FBR}}=\left(\begin{array}{ccc}{x_{11}} & {\ldots} & {x_{N 1}} \\ {\vdots} & {\ddots} & {\vdots} \\ {x_{1 N}} & {\ldots} & {x_{N N}}\end{array}\right)
# $$
#
# With
#
# $$
# x_{m n}=\int_{a}^{b} \frac{\sin ^{2}(m x) \sin ^{2}(n x)}{(m n)^{2}} \mathrm{d} \mathrm{x}
# $$

def xmn(m, n):
    def integrand(x):
        return (np.sin(m*x)*np.sin(n*x)/m/n)**2
    return quad(integrand, a, b)[0]


X_FBR=fillM(xmn)




