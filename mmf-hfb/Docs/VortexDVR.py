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
import sys
import matplotlib.pyplot as plt
from scipy.integrate import quad
from mmfutils.math import bessel
from mmf_hfb.bcs import BCS
from mmf_hfb import homogeneous
from mmfutils.plot import imcontourf
from collections import namedtuple
from mmfutils.math.special import mstep
from mmf_hfb.DVRBasis import CylindricalBasis
from mmf_hfb.VortexDVR import bdg_dvr, bdg_dvr_ho
from mmf_hfb.utils import block

# # BCS

Nx = 64
L = 16
dim = 2
dx = L/Nx
b0 = BCS(Nxyz=(Nx,)*dim, Lxyz=(L,)*dim)
x = b0.xyz
V=sum(np.array(x)**2/2.0).ravel()
K = b0._get_K()
H = K + np.diag(V)
Es0, psis0 = np.linalg.eigh(H)
psis0 = psis0.T


# # DVR

# ## Analytical Radial Wavefunction
#
# $$
# \phi(r)=C R(r)=CP(r)e^{-r^2/2}
# $$
# where $P(r)$ is a polynomial function of $r$, and $C$ is a normalization factor.
#
# For $E=1 \qquad \text{Ground State}$
#
# $P(r)=1$
# Then 
# $$
# \int_0^{\infty} R(r)^2 dr = \int_0^{\infty} e^{-r^2} dr = \frac{\sqrt{\pi}}{2}, \qquad C=\sqrt{\frac{\sqrt{\pi}}{2}}
# $$
#
# For $E=2$, there are two degenerate states, with $L=1$, and $L=-1$
#
# $P(r)=r$
# $$
# \int_0^{\infty} R(r)^2 dr = \int_0^{\infty}r^2 e^{-r^2} dr = \frac{\sqrt{\pi}}{4}, \qquad C=\sqrt{\frac{\sqrt{\pi}}{4}}
# $$
#
# For $E=3$
# There are two different $P(r)$, corresponding to $L=0$ and $L=\pm2$
# $$
# \int_0^{\infty} R(r)^2 dr=\frac{3\sqrt{\pi}}{8}, \qquad P(r)=r^2 \qquad \text{and} \qquad P(r)=r^2 -1, \qquad C=\sqrt{\frac{3\sqrt{\pi}}{8}}
# $$
#
# For $E=4$
# There are two different $P(r)$
# $$
# \int_0^{\infty} R(r)^2 dr=\frac{15\sqrt{\pi}}{16}, \qquad P(r)=r^3, \qquad C=\sqrt{\frac{15\sqrt{\pi}}{16}}\\
# \int_0^{\infty} R(r)^2 dr=\frac{5\sqrt{\pi}}{8}, \qquad P(r)=r^3 -r/2, \qquad C=\sqrt{\frac{5\sqrt{\pi}}{8}}
# $$
#

# ### Another nomalization scheme 
# * Above scheme does not take the angular component into consideration, the real wavefunction in 2D polar coordinate can be put as:
# $$
# \braket{r,\theta|\psi}=\psi(r, \theta)=\psi(r)e^{in\theta}
# $$
# The physical way to normalize a single particle wavefunction is:
# $$
# \braket{\psi|\psi}=1=\int dr d\theta{\braket{\psi|r, \theta}\braket{r, \theta|\psi}}=\int_{r=0}^{r=\infty}\int_{\theta=0}^{\theta=2\pi}\phi(r)^*\phi(r)r dr d\theta= 2\pi\int_{r=0}^{r=\infty}\phi(r)^*\phi(r)r dr d\theta
# $$

#
# For $E=1 \qquad \text{Ground State}$
#
# $P(r)=1$
# Then 
# $$
# 2\pi\int_0^{\infty} R(r)^2 r dr = 2\pi\int_0^{\infty} r e^{-r^2} dr =\pi, \qquad C=\sqrt{\pi}
# $$
#
# For $E=2$, there are two degenerate states, with $L=1$, and $L=-1$
#
# $P(r)=r$
# $$ 
# 2\pi\int_0^{\infty} r R(r)^2 dr = 2\pi\int_0^{\infty} r^3e^{-r^2} dr =\pi, \qquad C=\sqrt{\pi}
# $$
#
# For $E=3$
# There are two different $P(r)$, corresponding to $L=0$ and $L=\pm2$
# $$
# 2\pi\int_0^{\infty} r R(r)^2 dr=2\pi, \qquad P(r)=r^2 ,\qquad C=\sqrt{2\pi}\\
# 2\pi\int_0^{\infty} r R(r)^2 dr=\pi,\qquad \text{and} \qquad P(r)=r^2 -1,\qquad C=\sqrt{\pi}
# $$
#
# For $E=4$
# There are two different $P(r)$
# $$
# 2\pi\int_0^{\infty} r R(r)^2 dr=6\pi, \qquad P(r)=r^3, \qquad C=\sqrt{6\pi}\\
# 2\pi\int_0^{\infty} r R(r)^2 dr=\frac{17\pi}{4}, \qquad P(r)=r^3 -r/2, \qquad C=\sqrt{\frac{17\pi}{4}}
# $$

# +
def Normalize(psi):
    """Normalize a wave function"""
    return psi/(psi.conj().dot(psi))**0.5


def nan0(data):
    """convert nan to zero"""
    return np.nan_to_num(data, 0)


def get_2d_ho_wf_p(n, m, rs):
    """
    return 2d radial wave function for a 
    harmonic oscillator.
    ------------------------------------
    n = E -1
        e.g if E=1, to select the corresponding
        wavefunction, use n=E-1=0, and m = 0
    m is used to pick the degerated wavefunciton
    m <=n
    """
    assert n < 4 and n >=0
    assert m <=n
    P, pi = 1, np.pi
    C= (pi)**0.5
    if n ==1:  # E=2
        P = rs
    elif n == 2: # E=3
        P=rs**2
        C=(2*pi)**0.5
        if m == 1:
            P=P-1
            C = pi**0.5
    elif n == 3: #  E=4
        P = rs**3
        C= (6*pi)**0.5
        if m == 1 or m==2:
            P=P - rs/2
            C= (17*pi/4)**0.5
    return P*np.exp(-rs**2/2)/C

def get_2d_ho_wf(n, m, rs, p=False):
    """
    return 2d radial wave function for a 
    harmonic oscillator.
    ------------------------------------
    NOTE: if p is true, will use the physical
        normalization scheme
        
    n = E -1
        e.g if E=1, to select the corresponding
        wavefunction, use n=E-1=0, and m = 0
    m is used to pick the degerated wavefunciton
    m <=n
    """
    if p:
        return get_2d_ho_wf(n=n, m=m, rs=rs)
    assert n < 4 and n >=0
    assert m <=n
    P, pi = 1, np.pi
    C= (pi**0.5/2)**0.5
    if n ==1:  # E=2
        P = rs
        C=(pi**0.5/4)**0.5
    elif n == 2: # E=3
        P=rs**2
        if m == 1:
            P=P-1
        C = (3*pi**0.5/8)**0.5
    elif n == 3: #  E=4
        P = rs**3
        C= (15*pi**0.5/16)**0.5
        if m == 1 or m==2:
            P= P - rs/2
            C= (5*pi**0.5/8)**0.5
    return P*np.exp(-rs**2/2)/C

def show_2d_harmonic_oscillator_den(m=0, n=0, L=5, N=100):
    """Show 2D harmonic oscillator density"""
    ho = HarmonicOscillator2D()
    rs = np.linspace(-L, L, N)
    zs = ho.get_wf(rs, n=n, m=m)
    imcontourf(rs, rs, zs.conj()*zs)


# -

# ## Anaylical wavefunctions normalization

rs = np.linspace(0.0000, 5, 200)
for n in range(4):
    for m in range(n + 1):
        def f(r):
            return get_2d_ho_wf(n, m, r)**2
        ret =quad(f, 0, 10)
        assert np.allclose(ret[0], 1)

rs = np.linspace(0.0000, 5, 200)
plt.figure(figsize(16, 6))
for n in range(4):
    for m in range(n + 1):
        def f(r):
            return 2*np.pi*r*get_2d_ho_wf_p(n, m, r)**2
        ret =quad(f, 0, 10)
        assert np.allclose(ret[0], 1)
        plt.plot(rs, get_2d_ho_wf_p(n=n, m=m, rs=rs), label=f"n={n},m={m}")
plt.axhline(0, linestyle='dashed', c='red')
plt.xlabel("r")
plt.ylabel(r"$phi(r)$")
plt.legend()

n=0
rs = np.linspace(0.0000, 5, 100)
wf_an = get_2d_ho_wf(n=2, m=1, rs=rs)**2
sum(wf_an)*np.diff(rs).mean()


# ## Cylindrical DVR Class

# ## Harmonic Oscillator Class

class HarmonicDVR(CylindricalBasis):
    m=hbar=w=1

    def __init__(self, w=1, nu=0, dim=2, **args):
        CylindricalBasis.__init__(self, nu=nu, dim=dim, **args)
        self.w = w

    def get_V(self):
        """return the external potential"""
        r2 = (self.rs)**2
        return self.w**2*r2/2

    def get_H(self, nu=None):
        if nu is None:
            nu = self.nu
        K = self.K
        V = self.get_V()
        V_corr = self.get_V_correction(nu=nu)  # correction centrifugal piece due to different angular quantum number
        H = K + np.diag(V + V_corr)
        return H


# ### Make sure DVR basis funtioncs are normalized

h = HarmonicDVR(nu=0, dim=2, w=1)
H = h.get_H()
for i in range(h.N_root):
    def f(r):
        return h.get_F(n=i, rs=r)**2
    ret =quad(f, 0, np.inf)
    assert np.allclose(ret[0], 1)
clear_output()

# ### Make sure DVR basis functions are Orthogonal

h = HarmonicDVR(nu=0, dim=2, w=1)
H = h.get_H()
N = min(4, h.N_root)
for i in range(N):
    for j in range(i + 1, N):
        def f(r):
            return h.get_F(n=j, rs=r)*h.get_F(n=i, rs=r)
        ret =quad(f, 0, h.R_max*10)
        assert np.allclose(ret[0], 0, atol=1e-6)
clear_output()

# ### Reproduce the graphs in the Paper
# * Figure 2 in paper [Bessel discrete variable representation bases](https://aip.scitation.org/doi/10.1063/1.1481388)

h.K_max =1
paras = [(0,0),(3,0), (10,0), (3, 10)]
for i in range(len(paras)):
    rs = np.linspace(0 if i < 3 else 20, 30 if i < 3 else 50, 1000)
    plt.subplot(2,2,i+1)
    plt.plot(rs, h.get_F(nu=paras[i][0], n=paras[i][1], rs=rs))
    plt.axhline(0, linestyle='dashed', c='red')

# ### Check Errors
# * check how energy spectrum errors scales as number of abscissa and level of energy

plt.figure(figsize=(16,6))
linestyles = ['--', '+']
parities = ['odd', 'even']
for c, N in enumerate([10, 20, 30, 40]):
    dvr_o = HarmonicDVR(nu=0, w=1, N_root=N)
    dvr_e = HarmonicDVR(nu=1, w=1, N_root=N)
    c = None
    for (i, dvr) in enumerate([dvr_o, dvr_e]):
        H = dvr.get_H()
        Es, us = np.linalg.eigh(H)
        ns = np.array(list(range(len(Es))))
        Es0 = 2*ns + 2**i  # analytica energy spectrum
        errs = (Es - Es0)/Es0
        if c is not None:
            plt.semilogy(ns, errs, linestyles[i], c=c, label=f"{parities[i]}, N={N}")
        else:
            l, = plt.semilogy(ns, errs, linestyles[i], label=f"{parities[i]}, N={N}")
            c = l.get_c()      
plt.xlabel(r"$n$")
plt.ylabel(r"$(E-E_0)/E_0$")
plt.legend()

# ### Construct Wave Function from DVR Basis
# * Note: To get the radial wavefunction, we should divide the functionconstructed from the DVR basis by a factor of $\sqrt{r}$, the $\phi(r)$ is not the radial wavefunction:
# $$
#  \ket{\phi}=\sum_i{ u_i\ket{F_i}} \qquad \text{Normalized}
# $$
# by doing this, the resulted radio wavefunction $\psi(r)$:
# $$
# \psi(r)=\frac{\phi(r)}{\sqrt{r}} \qquad \text{Not normalized}
# $$
# will be not properly normalized, so we should renomalize it if necessary

# In other world, to properly normalize single particle state, ie:
# $$
# \braket{\Psi|\Psi}=1
# $$
# where $\Psi(r,\theta)=\psi(r)e^{in\theta}$
# $$
# \braket{\Psi|\Psi}=2\pi\int {r\psi^*(r)\psi(r) dr}=2\pi\int{\phi^*(r)\phi(r) dr}=2\pi\braket{\phi|\phi}=2\pi\sum_i{u^2_i}=1
# $$
# <font color='red'>Which means the results from diagonizing the Hamiltonian should have a weight factor of $\frac{1}{\sqrt{2\pi}}$</font>

plt.figure(figsize=(16, 8))
h = HarmonicDVR(nu=0, dim=2, w=1, R_max=None, N_root=32)
H = h.get_H()
Es, us = np.linalg.eigh(H)
Fs = h.get_F_rs()
print(Es[:10])
rs = np.linspace(0.000001, 5, 250)
dr = rs[1]-rs[0]
for n in [0, 1]:  # E=1, E=3
    u = us.T[n]
    phi_dvr_full =sum([u*h.get_F(nu=0, n=i, rs=rs) for (i, u) in enumerate(us.T[n])])
    assert np.allclose(sum(abs(phi_dvr_full)**2)*dr, 1, atol=1e-4)  # phi is normalized
    psi_dvr_full = phi_dvr_full/rs**0.5 # psi=phi/sqrt(r) is not normalized
    psi_dvr_abscissa = us.T[n]*h.ws
    psi_analytical = get_2d_ho_wf(n=2*n, m=2*n-1, rs=rs)
    factor = get_2d_ho_wf(n=2*n, m=2*n-1, rs=h.rs[0])/psi_dvr_abscissa[0]
    plt.plot(rs, psi_analytical, '+', label='Analytical')
    plt.plot(h.rs, factor*psi_dvr_abscissa,'o', label='Reconstructed(Fs)')
    plt.plot(rs, factor*(psi_dvr_full), '-',label='Reconstructed')
plt.xlabel("r")
plt.ylabel("F(r)")
plt.axhline(0, c='black', linestyle='dashed')
plt.legend()


# # Current and Kenitic Terms
# The Current terms are defined as:
# $$
# \begin{aligned}
# &\mathbf{j}_{a}(\mathbf{r})=\frac{i}{2} \sum_{n}\left[u_{n}^{*}(\mathbf{r}) \nabla u_{n}(\mathbf{r})-u_{n}(\mathbf{r}) \nabla u_{n}^{*}(\mathbf{r})\right] f_{\beta}\left(E_{n}\right)\\
# &\mathbf{j}_{b}(\mathbf{r})=\frac{i}{2} \sum_{n}\left[v_{n}^{*}(\mathbf{r}) \nabla v_{n}(\mathbf{r})-v_{n}(\mathbf{r}) \nabla v_{n}^{*}(\mathbf{r})\right] f_{\beta}\left(-E_{n}\right)
# \end{aligned}
# $$
#
# And the kenitic terms are defined as:
# $$
# \tau_{a}(\mathbf{r})=\sum_{n}\left|\nabla u_{n}(\mathbf{r})\right|^{2} f_{\beta}\left(E_{n}\right), \quad \tau_{b}(\mathbf{r})=\sum_{n}\left|\nabla v_{n}(\mathbf{r})\right|^{2} f_{\beta}\left(-E_{n}\right)
# $$

# For these terms, the first order derivative of the wavefunction should be computed. In 2D spherical DVR basis, the wavefunction is of this form:
# $$
# \Psi(r,\theta)=\psi(r)e^{in\theta}
# $$
# In the polar coordinate system, the gradient of a function is:
# $$
# \nabla f = \frac{\partial f}{\partial r} \hat{r} +\frac{1}{r}\frac{\partial f}{\partial \theta} \hat{\theta}
# $$

# ## Current in a Vortex 
#
# Apply the $\nabla$ operator on the wavefunction for a vortex
# $$
# \nabla\Psi(r, \theta)=\frac{\partial \psi(r)}{\partial r}\hat{r}e^{in\theta} + \frac{in}{r}\psi(r)e^{in\theta}
# $$
# and 
# \begin{align}
# \vec{J}
# &=\frac{i}{2}\left[\Psi^*(r,\theta)\left(\frac{\partial \psi(r)}{\partial r}\hat{r}e^{in\theta}
# + \frac{in}{r}\psi(r)\hat{\theta}e^{in\theta}\right)
# -\Psi(r,\theta)\left(\frac{\partial \psi^*(r)}{\partial r}\hat{r}e^{-in\theta}
# - \frac{in}{r}\psi^*(r)
# \hat{\theta}e^{-in\theta}\right)\right]\\
# &=\frac{i}{2}\left[\psi^*(r)\left(\frac{\partial \psi(r)}{\partial r}\hat{r} 
# + \frac{in}{r}\psi(r)\hat{\theta}\right)
# -\psi(r)\left(\frac{\partial \psi^*(r)}{\partial r}\hat{r} 
# - \frac{in}{r}\psi^*(r)\hat{\theta}\right)\right]\\
# &=\frac{i}{2}\left[\psi^*(r)\frac{\partial \psi(r)}{\partial r}\hat{r} 
# + \frac{in}{r}\psi^*(r)\psi(r)\hat{\theta}
# -\psi(r)\frac{\partial \psi^*(r)}{\partial r}\hat{r} 
# + \frac{in}{r}\psi(r)\psi^*(r)\hat{\theta}\right]\\
# &=\frac{i}{2}\left[\psi^*(r)\frac{\partial \psi(r)}{\partial r}
# -\psi(r)\frac{\partial \psi^*(r)}{\partial r} 
# \right]\hat{r}-\frac{n}{r}\psi^*(r)\psi(r)\hat{\theta}\\
# &=-\frac{n}{r}\psi^*(r)\psi(r)\hat{\theta} \qquad \text{no radial current}
# \end{align}

# ## Kinetic Terms
# To compute the kinetic terms, the first order derivative on abssicas should be computed:

# # Bases Transform Matrix

# Since differnet bases have different abcissas, the final result should be presented expanded in single basis or we can't compare anything. 
# Expand a function in two different bases $\ket{F}, \ket{F'}$. where $\braket{F_{n}|F_{n'}}=\delta(n, n')$ and $\braket{F'_{n}|F'_{n'}}=\delta(n, n')$
# $$
# \ket{\psi}=\sum_i C_i\ket{F_i}=\sum_i C'_i\ket{F'_i}
# $$
# To get $C_i$, multiply $\bra{F_j}$ from the left to get:
#
# $$
# \bra{F_j}\sum_i C_i\ket{F_i}=\bra{F_j}\sum_i C'_i\ket{F'_i}
# $$
#
# Use the orthognal relations:
#
# $$
# C_j=\bra{F_j}\sum_i C_i\ket{F_i}=\bra{F_j}\sum_i C'_i\ket{F'_i}=\sum_i \braket{F_j|F'_i}C'_i
# $$
#
# $$
# \begin{pmatrix}
# C_1\\
# C_2\\
# \vdots\\
# C_n
# \end{pmatrix}=
# \begin{pmatrix}
# \braket{F_1|F'_1}&\braket{F_1|F'_2}&\dots\braket{F_1|F'_m}\\
# \braket{F_2|F'_1}&\braket{F_2|F'_2}&\dots\braket{F_2|F'_m}\\
# \vdots&\vdots&\vdots\\
# \braket{F_n|F'_1}&\braket{F_n|F'_2}&\dots\braket{F_n|F'_m}\\
# \end{pmatrix}\begin{pmatrix}
# C'_1\\
# C'_2\\
# \vdots\\
# C'_m
# \end{pmatrix}
# $$

# ## Use the interploation properties of a DVR basis
# * In DVR, we may only want to evaluate the general function at abssisas, where the basis functions are local and interpolated, let the abassias for these two DVR bases be $r_i$ where $i=1,2,\dots  n$ and $r'_j$ where $j=1,2,\dots m$.
#
# First, evalute the function at the abssicas of the first basis
# $$
# \psi(r_i)=\sum_i C_i F_i(r_i)
# $$
# From the interprolation propoties of the DVR basis, we know that:
# $F_i(r_k)=F_i(r_i)\delta_{ik}$, so the weight factor $C_i$ can be easily evaluated:
# $$
# \psi(r_i)=C_i F_i(r_i) \qquad C_i=\frac{\psi(r_i)}{F_i(r_i)}
# $$
# Once we know all the weight factors $C_i$ in the basis $\ket{F_i}$, we can evalute the value of the $\psi(r)$ at arbitray point $r$. That means we can get the values: 
# $$
# \psi_j=\psi(r'_j)
# $$
# where the $r'_j$ for $j=1,2,\dots m$ are abssicas for the second DVR basis. Since the function can also be expand in terms of the second DVR basis functions $\ket{F'_j}$ with weight $C'_j$ , i.e:
# $$\psi=\sum_j{C'_j F'_j(r)}$$
#
# Resue the properties of interpolation at those abssicas of the DVR basis, we can get $C'_j$ as:
# $$
# C'_j =\frac{ \psi(r'_j)}{F'_j(r'_j)}= \frac{ \sum_i{C_i F_i(r'_j)}}{F'_j(r'_j)}
# $$

# $$
# C'_j =\frac{1}{F'_j(r'_j)}\sum_i{C_i F_i(r'_j)}=\frac{1}{F'_j(r'_j)}\left[C_0 F_0(r'_j)+C_1 F_1(r'_j)+\dots C_n F_n(r'_j)\right]
# $$

def get_transform_matrix(dvr_s, dvr_t):
    rs_s = dvr_s.rs
    rs_t = dvr_t.rs
    ws = dvr_t.get_F_rs()
    return np.array([[dvr_s.get_F(n=i, rs=rs_t[j])/ws[j] for i in range(len(rs_s))] for j in range(len(rs_t))])


U10 = get_transform_matrix(dvr1, dvr0)


def transform(dvr_s, dvr_t, us_s):
    def f(r):
        fs = [us_s[n]*dvr_s.get_F(n=n, rs=r) for n in range(len(us_s))]
        return sum(fs)
    psi = [f(r) for r in dvr_t.rs]
    Fs = dvr_t.get_F_rs()
    us_t = np.array(psi)/np.array(Fs)
    return us_t


us0 = U10.dot(us1) # transform(dvr_s=dvr1, dvr_t=dvr0, us_s=us1)
psi1= dvr1._get_psi(us1)
psi0 = dvr0._get_psi(us0)
plt.plot(dvr1.rs, psi1, '-', label="DVR1")
plt.plot(dvr0.rs, psi0, '--', label="DVR0")
plt.legend()

# # Compare Radial Wavefunctions & Densities
# * When compute DVR radial densites, since there are more than one DVR bases, all resulted wavefunctions should be transformed to the same basis(say basis when $\nu=0$), or simply adding up the densities from different bases may not yield right results as the grid point set are general not the same.

# ## 2D BCS Lattice

Nx = 32
L = 10
dim = 2
dx = L/Nx
b1 = BCS(Nxyz=(Nx,)*dim, Lxyz=(L,)*dim)
x = b1.xyz
V=sum(np.array(x)**2/2.0).ravel()
K = b1._get_K()
H = K + np.diag(V)
Es, psis = np.linalg.eigh(H)
psis=psis.T
Es[:10]


# ## DVR Densities Check
# For example, if $E=3$, there are three states with the same energy(degeneracy)
# * Triple degeneracy, sum up all three state densities
# * To see how close they are, increase the DVR absissa number to 64

# +
def get_dvr(nu=0, N_root=None):
    dvr = HarmonicDVR(nu=nu%2, w=1, R_max=None, N_root=N_root)
    H = dvr.get_H(nu=nu)
    Es, us = np.linalg.eigh(H)
    res = namedtuple(
                'res', ['dvr', 'Es', 'us'])
    return res(dvr=dvr, Es=Es, us=us)

ds = [get_dvr(nu=nu, N_root=64) for nu in range(10)]

def get_den_dvr(n, m):
    d, Es, us = ds[n]
    psi_dvr = d._get_psi(us.T[m])
    return abs(psi_dvr)**2
    
def plot_den_dvr(n, m, d0=1):
    den_dvr = get_den_dvr(n, m)
    plt.plot(d.rs, d0*den_dvr, '--', label="DVR")


# -

def compare_bcs_dvr_dens(E=3):
    d, Es, us = ds[E]
    start_index = sum(list(range(E)))
    end_index = start_index + E
    b = b0
    x, y = b.xyz
    rs = np.sqrt(sum(_x**2 for _x in b.xyz)).ravel()
    # BCS densities
    psis_bcs = np.array([b.Normalize((psis0[i]).reshape(b.Nxyz)) for i in range(start_index, end_index)])
    den_bcs=0
    for i in range(len(psis_bcs)):
        den_bcs = abs(psis_bcs[i])**2
    den_bcs = sum(abs(psis_bcs)**2)
    plt.figure(figsize=(14,5))
    plt.subplot(121)
    imcontourf(x, y, den_bcs)
    plt.colorbar()

    # DVR densities
    plt.subplot(122)
    parity = E%2
    den_dvr = 0
    if parity == 1:
        psi_index = E//2
        den_dvr += get_den_dvr(0, psi_index) # abs(psi_dvr)**2
        
    for i in range(1 + parity, E + 1, 2):
        psi_index = E//2 + parity - 1 - i//2
        den_dvr += 2*get_den_dvr(i, psi_index)  # 2*abs(psi_dvr)**2

    plt.plot(rs, den_bcs.ravel(), '+', label="Grid")
    plt.plot(d.rs, den_dvr, '-', label="DVR")
    plt.legend()


for E in range(7, 10):
    compare_bcs_dvr_dens(E=E)
    plt.show()


# # 2D BdG
# In BCS, to compute the total densities $n_a, n_b$, we sum up over all possible states. In principle, for the DVR case, same number of states(include the double-degenerate states) should be used to caculate the densities. However, <font color='red'>it turns out larger angular momentum $L$ contribute much less to the densities(I do not know excatly why), so only the first tens of them are significant. </font>

class BCS_ho(BCS):
    """2D harmonic"""
    def get_v_ext(self, **kw):
        """Return the external potential."""
        V=sum(np.array(self.xyz)**2/2.0)
        return (V, V)


mu, dmu, delta = 5, 3.5, 2

# ## BCS in a Box

b2 = BCS_ho(Nxyz=(32,)*2, Lxyz=(10,)*2)
res = b2.get_densities(mus_eff=(mu + dmu, mu - dmu), delta=delta)
n_a, n_b = res.n_a, res.n_b
x, y = b2.xyz
rs = np.sqrt(sum(_x**2 for _x in b2.xyz)).ravel()
plt.figure(figsize=(18, 4))
plt.subplot(131)
imcontourf(x, y, n_a)
plt.colorbar()
plt.subplot(132)
imcontourf(x, y, n_b)
plt.colorbar()
plt.subplot(133)
plt.plot(rs, n_a.ravel(), '+', label=r"$n_a$")
plt.plot(rs, n_b.ravel(), 'o', label=r"$n_b$")
plt.legend()

# ## Compare to DVR case

dvr = bdg_dvr_ho(mu=mu, dmu=dmu, E_c=None, N_root=64, delta=delta)
dvr.l_max=20  # 20 is good enough
delta = delta + dvr.bases[0].zero
res = dvr.get_densities(mus=(mu + dmu, mu - dmu), delta=delta)
na, nb, kappa, j_a, j_b = res.n_a, res.n_b, res.nu, res.j_a, res.j_b
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(dvr.bases[0].rs, (na), label=r'$n_a$(DVR)')
plt.plot(rs, n_a.ravel(), '+', label=r'$n_a$(Grid)')
plt.legend()
plt.subplot(122)
plt.plot(dvr.bases[0].rs, (nb), label=r'$n_b$(DVR)')
plt.plot(rs, n_b.ravel(), '+', label=r'$n_b$(Grid)')
plt.legend()
clear_output();plt.show();


# # Vortices

# +
def get_transform_matrix(dvr_s, dvr_t):
        rs_s = dvr_s.rs
        rs_t = dvr_t.rs
        ws = dvr_t.get_F_rs()
        return np.array([[dvr_s.get_F(n=i, rs=rs_t[j])/ws[j] for i in range(len(rs_s))] for j in range(len(rs_t))])

Densities = namedtuple(
                'Densities', ['n_a', 'n_b', 'tau_a', 'tau_b', 'nu', 'j_a', 'j_b'])

class bdg_dvr(object):
    """
    A 2D and 3D vortex class without external potential
    """
    def __init__(
            self, bases_N=2, mu=1, dmu=0, delta=1, lz=0,
                E_c=None, T=0, l_max=100, g=None, **args):
        """
        Construct and cache some information of bases

        """
        self.bases = [CylindricalBasis(nu=nu, **args) for nu in range(bases_N)]
        self.l_max = max(l_max, 1)  # the angular momentum cut_off
        assert T==0
        self.T=T
        self.lz=lz
        self.g = self.get_g(mu=mu, delta=np.mean(delta)) if g is None else g
        self.mus = (mu + dmu, mu - dmu)
        self.E_c = sys.maxsize if E_c is None else E_c
        self.U10 = get_transform_matrix(self.bases[1], self.bases[0])
        self.rs = self.bases[0].rs
        
    def f(self, E, T=0):
        if T is None:
            T = self.T
        if T == 0:
            if E < 0:
                return 1
            return 0
        else:
            return 1./(1+np.exp(E/T))

    def basis_match_rule(self, nu):
        """
            Assign different bases to different angular momentum \nu
            it assign 0 to even \nu and 1 to odd \nu
        Note:
            inherit a child class to override this function
        """
        assert len(self.bases) > 1  # make sure the number of bases is at least two
        return nu % 2

    def get_Vext(self, rs):
        """return external potential"""
        return 0

    def get_H(self, mus, delta, lz=0, nu=0):
        """
        return the full Hamiltonian(with pairing field)
        """
        basis = self.bases[self.basis_match_rule(nu)]
        T = basis.K
        Delta = np.diag(basis.zero + delta)
        mu_a, mu_b = mus
        V_ext = self.get_Vext(rs=basis.rs)
        V_corr = basis.get_V_correction(nu=nu)
        V_eff = V_ext + V_corr
        lz2 = lz**2/basis.rs**2/2
        lz2 = lz*(lz + 1)/basis.rs**2/2
        H_a = T + np.diag(V_eff - mu_a + lz2)
        H_b = T + np.diag(V_eff - mu_b + lz2)
        H = block(H_a, Delta, Delta.conj(), -H_b)
        return H

    def get_g(self, mu=1.0, delta=0.2):
        """
        the interaction strength
        """
        # [Check] will be dim = 3 when integrate over z
        h = homogeneous.Homogeneous(dim=2)
        res = h.get_densities(mus_eff=(mu, mu), delta=delta)
        g = 0 if res.nu == 0 else delta/res.nu
        return g

    def get_psi(self, nu, u):
        """
        apply weight on the u(v) to get the actual radial wave-function
        """
        if nu%2 == 1:
            u = self.U10.dot(u)
        b = self.bases[0]
        return b._get_psi(u=u)
    
    def transform(self, nu_s, nu_t, us_s):
        if nu_s == nu_t:
            return us_s
        dvr_s = self.bases[self.basis_match_rule(nu_s)]
        dvr_t = self.bases[self.basis_match_rule(nu_t)]
        def f(r):
            fs = [us_s[n]*dvr_s.get_F(n=n, rs=r) for n in range(len(us_s))]
            return sum(fs)
        psi = [f(r) for r in dvr_t.rs]
        Fs = dvr_t.get_F_rs()
        us_t = np.array(psi)/np.array(Fs)
        return us_t

    def _get_den(self, H, nu):
        """
        return the densities for a given H
        """
        es, phis = np.linalg.eigh(H)
        phis = phis.T
        offset = phis.shape[0] // 2
        den = 0
        for i in range(len(es)):
            E, uv = es[i], phis[i]
            if abs(E) > self.E_c:
                continue
            
            u, v = uv[: offset], uv[offset:]
            u = self.get_psi(nu=nu, u=u)
            v = self.get_psi(nu=nu, u=v)
            
            f_p, f_m = self.f(E=E), self.f(E=-E)
            n_a = u*u.conj()*f_p
            n_b = v*v.conj()*f_m
            j_a = -n_a*self.lz/self.rs
            j_b = -n_b*self.lz/self.rs
            kappa = u*v.conj()*(f_p - f_m)/2
            den = den + np.array([n_a, n_b, kappa, j_a, j_b])
        return den

    def get_densities(self, mus, delta, lz=None):
        """
        return the particle number density and anomalous density
        Note: Here the anomalous density is represented as kappa
        instead of \nu so it's not that confusing when \nu has
        been used as angular momentum quantum number.
        """
        if lz is None:
            lz = self.lz
        else:
            self.lz = lz
        
        dens = self._get_den(self.get_H(mus=mus, delta=delta, nu=0, lz=lz), nu=0)
        for nu in range(1, self.l_max):  # sum over angular momentum
            H = self.get_H(mus=mus, delta=delta, nu=nu, lz=lz)
            dens = dens + 2*self._get_den(H, nu=nu)  # double-degenerate
        n_a, n_b, kappa, j_a, j_b = dens
        return Densities(
            n_a=n_a, n_b=n_b,
            tau_a=None, tau_b=None,
            nu=kappa,
            j_a=j_a, j_b=j_b)
        #return (n_a, n_b, kappa, j_a, j_b)

class BCS_vortex(BCS):
    """BCS Vortex"""
    barrier_width = 0.2
    barrier_height = 100.0
    
    def __init__(self, delta, mus_eff, **args):
        BCS.__init__(self, **args)
        h = homogeneous.Homogeneous(Nxyz=self.Nxyz, Lxyz=self.Lxyz) 
        res = h.get_densities(mus_eff=mus_eff, delta=delta)
        self.g = delta/res.nu.n
        
    def get_v_ext(self, **kw):
        self.R = min(self.Lxyz)/2
        r = np.sqrt(sum([_x**2 for _x in self.xyz[:2]]))
        R0 = self.barrier_width * self.R
        V = self.barrier_height * mstep(r-self.R+R0, R0)
        return (V, V)
    
class dvr_vortex(bdg_dvr):
    """BCS Vortex"""
    barrier_width = 0.2
    barrier_height = 100.0
    
    def get_Vext(self, rs):
        self.R = 5
        R0 = self.barrier_width * self.R
        V = self.barrier_height * mstep(rs-self.R+R0, R0)
        return V


# +
loop = 1
mu = 5
dmu = 3
delta_bcs=delta_dvr=delta = 2
# BCS
b3 = BCS_vortex(Nxyz=(32,)*2, Lxyz=(10,)*2, mus_eff=(mu+dmu, mu-dmu), delta=delta)
E_c = np.max(b3.kxyz)**2*b3.dim/2
x, y = b3.xyz
rs = np.sqrt(sum(_x**2 for _x in b3.xyz)).ravel()
# DVR
dvr = dvr_vortex(mu=mu, dmu=dmu, delta=delta, g=b3.g, E_c=0.65*E_c, N_root=32, R_max=5, l_max=50)
delta_bcs = delta*(x+1j*y)
delta_dvr = delta*dvr.rs
dvr.lz = 0 if np.size(delta_bcs)==1 else 0.5

with NoInterrupt() as interrupted:
    for _ in range(loop):
        # BCS plot
        res_bcs = b3.get_densities(mus_eff=(mu + dmu, mu - dmu), delta=delta_bcs)
        na_bcs, nb_bcs, nu_bcs, ja_bcs, jb_bcs = res_bcs.n_a, res_bcs.n_b, res_bcs.nu, res_bcs.j_a, res_bcs.j_b
        
        res_dvr = dvr.get_densities(mus=(mu + dmu,mu - dmu), delta=delta_dvr)
        na_dvr, nb_dvr, nu_dvr, ja_dvr, jb_dvr =res_dvr.n_a, res_dvr.n_b, res_dvr.nu, res_dvr.j_a, res_dvr.j_b
        delta_dvr = -dvr.g*nu_dvr
        
        plt.figure(figsize=(18, 15))
        plt.subplot(331)
        imcontourf(x, y, na_bcs)
        plt.colorbar()
        plt.title(r"$n_a$")
        plt.subplot(332)
        
        plt.colorbar()
        ds.append(delta)
        delta_bcs = -b3.g*nu_bcs       
        if np.size(delta_bcs) == np.prod(b3.Nxyz):
            imcontourf(x, y, abs(delta_bcs))
        plt.title(r"$\Delta$")    
        # n_a      
        plt.subplot(334)
        plt.plot(dvr.rs, na_dvr, label=r'$n_a$(DVR)')
        plt.plot(rs, na_bcs.ravel(), '+', label=r'$n_a$(Grid)')
        plt.legend()
        # n_b
        plt.subplot(335)
        plt.plot(dvr.rs, nb_dvr, label=r'$n_b$(DVR)')
        plt.plot(rs, nb_bcs.ravel(), '+', label=r'$n_b$(Grid)')
        plt.legend()
        # nu
        plt.subplot(336)
        plt.plot(dvr.rs, abs(nu_dvr), label=r'$\nu$(DVR)')
        plt.plot(rs, abs(nu_bcs).ravel(), '+', label=r'$\nu$(Grid)')
        plt.legend()
        # Delta
        plt.subplot(333)
        plt.plot(dvr.rs, abs(delta_dvr), label=r'$\Delta$(DVR)')
        plt.plot(rs, abs(delta_bcs).ravel(), '+', label=r'$\Delta$(Grid)')
        plt.legend()
        # current
        plt.subplot(337)
        plt.plot(dvr.rs, -ja_dvr, label=r'$j_a$(DVR)')
        plt.plot(rs, np.sqrt(sum(ja_bcs**2)).ravel(), '+', label=r'$j_a$(Grid)')
        plt.legend()
        plt.subplot(338)
        plt.plot(dvr.rs, -jb_dvr, label=r'$j_b$(DVR)')
        plt.plot(rs, np.sqrt(sum(jb_bcs**2)).ravel(), '+', label=r'$j_b$(Grid)')
        plt.legend()
        clear_output(wait=True)
        plt.show()
# -



# ## The Additional Term in DVR
# * Numerically, if the term is $n(n-1)$, the densities $n_a, n_b$ match the BCS results perfectly
# * if $n(n+1)$, not nicely
# * if $n^2$ as given by the derivation, not that good.
# * Need to double check the derivation
# * the $\nu$ term still not fit nicely

# ## Check how $n_a, n_b, \nu$ change with $L$

def get_den(nu=0):
    mus = (mu + dmu, mu - dmu)
    return dvr._get_den(dvr.get_H(mus=mus, delta=delta, nu=nu), nu=nu)


import time
for l in range(32):
    a, b, v = get_den(l)
    plt.plot(dvr.bases[0].rs, a, '--', label=r'$n_a$')
    plt.plot(dvr.bases[0].rs, b, '+', label=r'$n_b$')
    plt.plot(dvr.bases[0].rs, v, 'o', label=r'$\nu$')
    plt.legend()
    plt.title(r"$\nu$"+f"={l}")
    clear_output(wait=True)
    plt.show()
    time.sleep(0.5)

# # Problems:
# There are some unsolved problems in the DVR case. need to fully understand where the issues  come from. HEADACKE!
# ## Errors
# * The calculation of $\nu$ does not agree with the BCS in a box, something wrong.
#     * <font color='red'>It's found that changing the max $L$ will change the $\nu$, not much on $n_s$</font>
#     * Solution: <font color='green'> Pick an energy cutoff to match the $\nu$ in both cases</font>, the $\nu$ seems to be senstive to the cutoff.

#
# ## To-Do
# * Compute $\tau$ and $j_{\pm a/b}$ terms
# * Integral over the Z dirction
# * Implement ASLDA

# # BdG in Rotating Frame Transform

# In polar coordinates, the Del operator $\nabla^2$ is defined as:
# $$
# \begin{align}
# \nabla^2
# &=\frac{1}{r} \frac{\partial}{\partial r}\left(r \frac{\partial f}{\partial r}\right)+\frac{1}{r^{2}} \frac{\partial^{2} f}{\partial \theta^{2}}\\
# &=\frac{\partial^2 f}{\partial r^2}+\frac{1}{r} \frac{\partial f}{\partial r}+\frac{1}{r^{2}} \frac{\partial^{2} f}{\partial \theta^{2}}
# \end{align}
# $$

# To be general, assume $f=f(r,\theta)$
# $$
# \begin{aligned}
# \nabla^2 \left[ fe^{in\theta}\right]
# &=\frac{\partial^{2} }{\partial r^{2}}\left[f(r,\theta)e^{in\theta}\right]
# +\frac{1}{r} \frac{\partial}{\partial r}\left[f(r,\theta)e^{in\theta}\right]
# +\frac{1}{r^{2}} \frac{\partial^{2} }{\partial \theta^{2}}\left[f(r,\theta)e^{in\theta}\right]\\
# &=\bigg\{\frac{\partial^{2} }{\partial r^{2}}\left[f(r,\theta)\right]
# +\frac{1}{r} \frac{\partial}{\partial r}\left[f(r,\theta)\right]
# +\frac{1}{r^{2}} \left[\left(\frac{\partial^{2} }{\partial \theta^{2}}+i2n\frac{\partial}{\partial \theta} - n^2\right)f(r, \theta)\right]\bigg\}e^{in\theta}\\
# &=\left[\left(\nabla^2 +i2n\frac{\partial}{r^2\partial \theta} - \frac{n^2}{r^2})f(r,\theta)\right)\right]e^{in\theta}
# \end{aligned}
# $$
# if $f(r,\theta)=f(r)$, i.e. $f$ only depends on $r$, the above result can be simplified:
# $$
# \nabla^2 \left[ fe^{in\theta}\right]=\left[\left(\nabla^2  - \frac{n^2}{r^2})f(r,\theta)\right)\right]e^{in\theta}
# $$
#
# To compute the pairing field of a vortex in BdG formulism, let the pairing field to be of this form:
# $$
# \Delta = \Delta_0 g(r) e^{i2n\theta}
# $$
#
# $$
# R\psi(x,y)=e^{i\theta \hat{L}_z/\hbar}\psi(x,y)
# $$
#
# Then, to be explicit:

# $$
# \begin{align}
# \begin{pmatrix}
# -\frac{\nabla^2}{2}-\mu_a & \Delta g(r)e^{i2n\theta}\\
# \Delta^*g(r)^*e^{-i2n\theta} & \frac{\nabla^2}{2} + \mu_b\\
# \end{pmatrix}
# \begin{pmatrix}
# U(r)e^{in\theta}\\
# V^*(r)e^{-in\theta}
# \end{pmatrix}
# &=\begin{pmatrix}
# (-\frac{\nabla^2}{2}-\mu_a)U(r)e^{in\theta}+ \Delta g(r) e^{i2n\theta}V^*(r)e^{-in\theta}\\
# \Delta^* g(r)^* e^{-i2\theta}U(r)e^{in\theta} + (\frac{\nabla^2}{2} + \mu_b)V^*(x)e^{-in\theta}\\
# \end{pmatrix}\\
# &=\begin{pmatrix}\left[-\frac{\nabla^2}{2}-\mu_a \right]U(r)e^{in\theta}+ \Delta g(r)V^*(r)e^{in\theta}\\
# \Delta^* g(r)^* U(r)e^{-in\theta} + \left[\frac{\nabla^2}{2} + \mu_b\right]V^*(r)e^{-in\theta}\\
# \end{pmatrix}\\
# &=\begin{pmatrix}\left[-\frac{\nabla^2}{2}-\mu_a + \frac{n^2}{2r^2} \right]U(r)e^{in\theta}+ \Delta g(r)V^*(r)e^{in\theta}\\
# \Delta^* g(r)^* U(r)e^{-in\theta} + \left[\frac{\nabla^2}{2} + \mu_b - \frac{n^2}{2r^2}\right]V^*(r)e^{-in\theta}\\
# \end{pmatrix}\\
# &=\begin{pmatrix}
# E & 0\\
# 0&-E
# \end{pmatrix}\begin{pmatrix}
# U(x)e^{in\theta}\\
# V(x)^*e^{-in\theta}
# \end{pmatrix}
# \end{align}
# $$
# * By canceling out the phase terms:
# $$
# \begin{pmatrix}\left[-\frac{\nabla^2}{2}-\mu_a + \frac{n^2}{2r^2} \right]& \Delta g(r)\\
# \Delta^* g(r)^* & \left[\frac{\nabla^2}{2} + \mu_b - \frac{n^2}{2r^2}\right]\\
# \end{pmatrix}\begin{pmatrix}
# U(x)\\
# V(x)^*
# \end{pmatrix}=\begin{pmatrix}
# E & 0\\
# 0&-E
# \end{pmatrix}\begin{pmatrix}
# U(x)\\
# V(x)^*
# \end{pmatrix}
# $$
#
#
# * So:
# To introduce vortex pairing field, an additional terms ($\frac{n^2}{2r^2}$) can be added to the diagnoal of the BdG matrix

# ## Careful Check
# The radial Schrodinger equation is given as:
# $$
# \frac{1}{r^{d-1}} \frac{d}{d r}\left(r^{d-1} \frac{d \psi}{d r}\right)-\frac{\lambda(\lambda+d-2)}{r^{2}} \psi+\frac{2 m}{\hbar^{2}}[E-V(r)] \psi=0
# $$
# By defining:
# $\phi(r)=r^{(d-1) / 2} \psi(r)$, with normalization $\int_{0}^{\infty} d r|\phi(r)|^{2}=1$. 
# The $\phi(r)$ satisfies:
# $$
# \frac{d^{2} \phi}{d r^{2}}-\frac{\nu^{2}-1 / 4}{r^{2}} \phi+\frac{2 m}{\hbar^{2}}[E-V(r)] \phi=0
# $$
# where $\nu=\lambda+d / 2-1$

# # First Order Derivative Operator

# In the case of free particles of energy E5\2k2/2m, the radial wave function is:
# $$
# \phi_{k}(r)=\langle r | k \nu\rangle=\sqrt{k r} J_{\nu}(k r)
# $$
#
# where we have the relation:
# $$
# \left\langle k \nu | k^{\prime} \nu\right\rangle=\int_{0}^{\infty} d r\langle k \nu | r\rangle\left\langle r | k^{\prime} \nu\right\rangle=\delta\left(k-k^{\prime}\right)
# $$
#
# Useful integrals
#
# $$
# \begin{array}{l}{\int_{0}^{R} r d r J_{\nu}(k r) J_{\nu}\left(k^{\prime} r\right)} \\ {\quad=\frac{R}{k^{2}-k^{\prime 2}}\left[k^{\prime} J_{\nu}(k R) J_{\nu}^{\prime}\left(k^{\prime} R\right)-k J_{\nu}^{\prime}(k R) J_{\nu}\left(k^{\prime} R\right)\right]}\end{array}
# $$
# when $k\rightarrow k'$
#
# $$
# \begin{array}{l}{\int_{0}^{R} r d r J_{\nu}(k r)^{2}} \\ {\quad=\frac{1}{2 k^{2}}\left[k^{2} R^{2} J_{\nu}^{\prime}(k R)^{2}+\left(k^{2} R^{2}-\nu^{2}\right) J_{\nu}(k R)^{2}\right]}\end{array}
# $$
#
# Given the DVR basis function to be:
# $$
# F_{\nu n}(r)=(-1)^{n+1} \frac{K z_{\nu n} \sqrt{2 r}}{K^{2} r^{2}-z_{\nu n}^{2}} J_{\nu}(K r)
# $$
#
# The matrix element for the kenitic with centrifugal term can be computed as：
# $$
# \begin{array}{l}{\left\langle F_{\nu n}\left|k_{r}^{2}+\frac{\nu^{2}-\frac{1}{4}}{r^{2}}\right| F_{\nu n^{\prime}}\right\rangle} \\ {\qquad=\left\{\begin{array}{ll}{\frac{K^{2}}{3}\left[1+\frac{2\left(\nu^{2}-1\right)}{z_{\nu n}^{2}}\right],} & {n=n^{\prime}} \\ {(-1)^{n-n^{\prime}} 8 K^{2} \frac{z_{\nu n} z_{\nu n^{\prime}}}{\left(z_{\nu n}^{2}-z_{\nu n^{\prime}}^{2}\right)^{2}},} & {n \neq n^{\prime}}\end{array}\right.}\end{array}
# $$
#
# Where $K_r^2=-\frac{d^2}{dr^2}$

# ## Addtional Terms in Rotating Frame
# * To get rid of the phase term in the pairing field, the BdG matrix can be transformed into the rotating frame, where two additional terms will show up in the kenitic matrix, one of them is just constant($q^2$) which is trivial, the other term is the first order derivatie $\nabla=\frac{d}{dr}$
# * Bessel Functions are so messy, hard to compute by hand????
#
# $$
# \begin{array}{l}{\left\langle F_{\nu n}\left|\frac{d}{dr}\right| F_{\nu n^{\prime}}\right\rangle} \\ {\qquad=\left\{\begin{array}{ll}{\frac{K^{2}}{3}\left[1+\frac{2\left(\nu^{2}-1\right)}{z_{\nu n}^{2}}\right],} & {n=n^{\prime}} \\ {(-1)^{n-n^{\prime}} 8 K^{2} \frac{z_{\nu n} z_{\nu n^{\prime}}}{\left(z_{\nu n}^{2}-z_{\nu n^{\prime}}^{2}\right)^{2}},} & {n \neq n^{\prime}}\end{array}\right.}\end{array}
# $$

# Let try to compute $\frac{d F_{vn}}{dr}$
#
# \begin{align}
# \frac{d F_{vn}}{dr}
# &=(-1)^{n+1} K z_{\nu n}\frac{d }{dr}\left[\frac{ \sqrt{2 r}}{K^{2} r^{2}-z_{\nu n}^{2}} J_{\nu}(K r)\right]\\
# &=(-1)^{n+1} K z_{\nu n}\left[\frac{J_{\nu}(Kr)}{\sqrt{2r}(K^2r^2-z_{\nu n}^2)} - \frac{2\sqrt{2r}K^2rJ_{\nu}(Kr) }{(K^2r^2-z_{\nu n}^2)^2}+\frac{K\sqrt{2r}J'_{\nu}(Kr)}{K^2r^2-z_{\nu n}^2}\right]
# \end{align}

# * Check the defination of $F_{\nu n}$, we can try to rewrite the above expression as:
# $$
# \frac{d F_{\nu n}}{dr}= \frac{F_{\nu n}}{2r} - \frac{2K^2r F_{\nu n}}{K^2r^2-z^2_{\nu n}} +\frac{1}{2}\left(F_{(\nu-1) n}-F_{(\nu+1) n}\right)
# $$
#
# In the last term, we use the following relations: 
# $$
# \frac{\partial J_{v}(z)}{\partial z}=\frac{1}{2}\left[J_{v-1}(z)-J_{v+1}(z)\right]
# $$
