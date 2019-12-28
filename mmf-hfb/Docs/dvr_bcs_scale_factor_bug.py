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
from scipy.integrate import quad
from mmfutils.math import bessel
from mmf_hfb.bcs import BCS
from mmf_hfb import homogeneous
from mmfutils.plot import imcontourf
from collections import namedtuple
from mmfutils.math.special import mstep
from mmf_hfb.DVRBasis import CylindricalBasis
from mmf_hfb.VortexDVR import bdg_dvr, bdg_dvr_ho

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
            plt.semilogy(ns, errs, linestyles[i], c=c)
        else:
            l, = plt.semilogy(ns, errs, linestyles[i])
            c = l.get_c()      
plt.xlabel(r"$E_n$")
plt.ylabel(r"$(E-E_0)/E_0$")

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
# <font color='red'>Which means the results from diagonizing the Hamiltonian should have a weight of $\frac{1}{\sqrt{2\pi}}$</font>

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

# ## 3D Case
# In 3D case, the grid point set are the zeros  of spherical Bessel function. For frist four sphereical Bessel functions are:
# $$
# \begin{aligned}
# &j_{0}(x)=\frac{\sin x}{x}\\
# &j_{1}(x)=\frac{\sin x}{x^{2}}-\frac{\cos x}{x}\\
# &j_{2}(x)=\left(\frac{3}{x^{2}}-1\right) \frac{\sin x}{x}-\frac{3 \cos x}{x^{2}}\\
# &j_{3}(x)=\left(\frac{15}{x^{3}}-\frac{6}{x}\right) \frac{\sin x}{x}-\left(\frac{15}{x^{2}}-1\right) \frac{\cos x}{x}
# \end{aligned}
# $$

# The transform matrix from $\nu=1$ to $\nu=0$ is given(Aurel's code):
# \begin{align}
# a_i &= \frac{sin(z_{1i})}{\sqrt{z_{1i}}}\\
# b_j &= -\frac{cos(z_{0j})}{\sqrt{z_{0j}}}\\
# U_{ji}&=\frac{2b_j\sqrt{z_{0j}z_{1i}}}{a_i(z^2_{0j}-z^2_{1i})}
# \end{align}

dvr0 = CylindricalBasis(nu=0, R_max=9, N_root=128)
dvr1 = CylindricalBasis(nu=1, R_max=9, N_root=127)
z0 = dvr0.zs
z1 = dvr1.zs
a = np.sin(z1)/np.sqrt(z1)
b = -np.cos(z0)/np.sqrt(z0)
U10 = 2*b[:,None]*(z0[:, None]*z1[None,:])**0.5/a[None,:]/(z0[:,None]**2-z1[None,:]**2)

# ## Test the Transform Matrix in 2D
# * if the basis set have similar basis function size, the $U$ matrix in 3D seems to work nicely in 2D case

import random
us1 = np.cos(np.linspace(0, 5 + 15*np.random.random(), len(z1))) 
us0 = U10.dot(us1)
psi1= dvr1._get_psi(us1)
psi0 = dvr0._get_psi(us0)
plt.plot(dvr1.rs, psi1, label="DVR1")
plt.plot(dvr0.rs, psi0, '+', label="DVR0")
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

# +
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
# -

dvr = bdg_dvr_ho(mu=mu, dmu=dmu, E_c=None, N_root=64, delta=delta)
dvr.l_max=20  # 20 is good enough
delta = delta + dvr.bases[0].zero
na, nb, kappa = dvr.get_densities(mus=(mu + dmu, mu - dmu), delta=delta)
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

# # BdG in Rotating Frame Transform
#
# $$
# \Delta = \Delta_0 r e^{i2n\theta}=\Delta_0(x + iy)^{2n}/r^{2n-1}=\Delta_0f(r)(x + iy)^{2n}
# $$
#
# $$
# R\psi(x,y)=e^{i\theta \hat{L}_z/\hbar}\psi(x,y)
# $$

# * Mathmatical identity
# \begin{align}
# \nabla^2\left[U(x)e^{iqx}\right]
# &=\nabla\left[\nabla U(x)e^{iqx}+U(x)iqe^{iqx}\right]\\
# &=\nabla^2U(x)e^{iqx}+2iq\nabla U(x)e^{iqx}-q^2U(x)e^{iqx}\\
# &=(\nabla+iq)^2U(x)e^{iqx}
# \end{align}
#
# * Let $2q=q_a + q_b$
# \begin{align}
# \begin{pmatrix}
# -\nabla^2-\mu_a & \Delta e^{2iqx}\\
# \Delta^*  e^{-2iqx} & \nabla^2 + \mu_b\\
# \end{pmatrix}
# \begin{pmatrix}
# U(x)e^{iq_a x}\\
# V^*(x)e^{-iq_bx}
# \end{pmatrix}
# &=\begin{pmatrix}
# (-\nabla^2-\mu_a)U(x)e^{iq_a x}+ \Delta e^{2iqx}V^*(x)e^{-iq_bx}\\
# \Delta^*  e^{-2iqx}U(x)e^{iq_a x} + (\nabla^2 + \mu_b)V^*(x)e^{-iq_bx}\\
# \end{pmatrix}\\
# &=\begin{pmatrix}\left[-(\nabla+iq_a)^2-\mu_a\right]U(x)e^{iq_a x}+ \Delta V^*(x)e^{iq_ax}\\
# \Delta^*  U(x)e^{-iq_b x} + \left[(\nabla-iq_b)^2 + \mu_b)\right]V^*(x)e^{-iq_bx}\\
# \end{pmatrix}\\
# &=\begin{pmatrix}\left[-(\nabla+iq_a)^2-\mu_a\right]U(x)e^{iq_a x}+ \Delta V*(x)e^{iq_ax}\\
# \Delta^* U(x)e^{-iq_b x} + \left[(\nabla-iq_b)^2 + \mu_b)\right]V^*(x)e^{-iq_bx}\\
# \end{pmatrix}
# =\begin{pmatrix}
# E & 0\\
# 0&-E
# \end{pmatrix}\begin{pmatrix}
# U(x)e^{iq_a x}\\
# V(x)^*e^{-iq_bx}
# \end{pmatrix}
# \end{align}
# * By canceling out the phase terms:
#
# \begin{align}
# \begin{pmatrix}\left[(i\nabla-q_a)^2-\mu_a\right] &\Delta\\
# \Delta^* & -\left[(i\nabla + q_b)^2 - \mu_b)\right]\\
# \end{pmatrix}
# \begin{pmatrix}
# U(x)\\
# V^*(x)\\
# \end{pmatrix}=\begin{pmatrix}
# E & 0\\
# 0&-E
# \end{pmatrix}
# \begin{pmatrix}
# U(x)\\
# V^*(x)
# \end{pmatrix}
# \end{align}
#
# * Let $\delta q = q_a - q_b$, then:
# $$
# q_a = q + \delta q\\
# q_b = q - \delta q
# $$
# * So:
#     
# \begin{align}
# \begin{pmatrix}\left[(i\nabla-q - \delta q)^2-\mu_a\right] &\Delta\\
# \Delta^* & -\left[(i\nabla + q - \delta q)^2 - \mu_b)\right]\\
# \end{pmatrix}
# \begin{pmatrix}
# U(x)\\
# V^*(x)\\
# \end{pmatrix}=\begin{pmatrix}
# E & 0\\
# 0&-E
# \end{pmatrix}
# \begin{pmatrix}
# U(x)\\
# V^*(x)
# \end{pmatrix}
# \end{align}

# For simplest case, $q_a=q_b=q$:
# \begin{align}
# \begin{pmatrix}\left[(i\nabla-q )^2-\mu_a\right] &\Delta\\
# \Delta^* & -\left[(i\nabla + q)^2 - \mu_b)\right]\\
# \end{pmatrix}
# \begin{pmatrix}
# U(x)\\
# V^*(x)\\
# \end{pmatrix}=\begin{pmatrix}
# E & 0\\
# 0&-E
# \end{pmatrix}
# \begin{pmatrix}
# U(x)\\
# V^*(x)
# \end{pmatrix}
# \end{align}
#

#
# In a vortex, the pairing field is
# $$
# \Delta = \Delta_0(x + iy)=\Delta_0 r e^{i\theta}
# $$
#

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
