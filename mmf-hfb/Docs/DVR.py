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
from mmf_hfb.DVRBasis import CylindricalBasis
from mmf_hfb import homogeneous
from mmfutils.plot import imcontourf
from collections import namedtuple
from mmf_hfb.Potentials import HarmonicOscillator2D
from mmf_hfb.VortexDVR import vortex_dvr, vortex_dvr_ho


# # 2D Harmonic System
#
# $\begin{aligned} \psi_{00} &=\left(\frac{m \omega}{\pi \hbar}\right)^{1 / 2} e^{-m \omega \rho^{2} / 2 \hbar} \\ \psi_{10} &=\sqrt{\frac{2 m \omega}{\hbar}}\left(\frac{m \omega}{\pi \hbar}\right)^{1 / 2} e^{-m \omega \rho^{2} / 2 \hbar} \rho \cos \phi \\ \psi_{01} &=\sqrt{\frac{2 m \omega}{\hbar}}\left(\frac{m \omega}{\pi \hbar}\right)^{1 / 2} e^{-m \omega \rho^{2} / 2 \hbar} \rho \sin \phi \end{aligned}$

# +
def Normalize(psi):
    """Normalize a wave function"""
    return psi/(psi.conj().dot(psi))**0.5

def HO_psi(n, m, rs):
    """
    return 2d radia wave function for a 
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

def get_2d_den(m=0, n=0, L=5, N=100):
    """Show 2D harmonic osciallater density"""
    ho = HarmonicOscillator2D()
    rs = np.linspace(-L, L, N)
    zs = ho.get_wf(rs, n=n, m=m)
    imcontourf(rs, rs, zs.conj()*zs)


# -

# ## Visualize single 2D wavefunction
# * For BCS uses a plance wave basis, the angular momentum part is automatically incooporated to the wavefunction

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


Es0[:10]

n = 5
print(Es0[n])
x, y = b0.xyz
plt.figure(figsize=(7, 5))
psi = psis0[n].reshape((b0.Nxyz))
n0 = abs(psi)**2
imcontourf(x, y, n0.real)
plt.colorbar()


# ## Harmonic DVR Class

class HarmonicDVR(CylindricalBasis):
    m=hbar=w=1
    eps = 7./3 - 4./3 -1  # machine accuracy

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
        V_corr = self.get_V_correction(nu=nu)  # correction centrifigal piece due to different angular quantum number
        H = K + np.diag(V + V_corr)
        return H


# ## Construct Wavefunction from a basis

# +
plt.figure(figsize=(16, 16))
h = HarmonicDVR(nu=0, dim=2, w=1)
H = h.get_H()
Es, us = np.linalg.eigh(H)
Fs = h.get_F_rs()
print(Es[:10])
rs = np.linspace(0.01, 8, 100)

for n in [0, 1]:
    wf =sum([u*h.get_F(nu=0, n=i, rs=rs) for (i, u) in enumerate(us.T[n])])
    wf_ = us.T[n]*h.ws
    plt.subplot(211)
    scale_factor = HO_psi(n=2*n, m=2*n-1, rs=rs[0])*rs[0]**0.5/wf[0]

    plt.plot(rs, HO_psi(n=2*n, m=2*n-1, rs=rs), '+', label='Analytical')
    plt.plot(h.rs, wf_*scale_factor,'o', label='Reconstructed(Fs)')
    plt.plot(rs, (wf*scale_factor/rs**0.5), '-',label='Reconstructed')

plt.xlabel("r")
plt.ylabel("F(r)")
plt.axhline(0, c='r', linestyle='dashed')
plt.legend()

plt.subplot(212)
rs = np.linspace(0.01, 8, 500)
plt.plot(h.rs, Fs, 'o')
for i in range(10):
    l, = plt.plot(rs, h.get_F(n=i, rs=rs), label=r'$\nu$'+f'{i}')
    plt.axvline(h.rs[i], linestyle='dashed', c=l.get_c())
plt.xlim(0, 3)
plt.legend()
clear_output()
# -

# # Compare to 2D Box


# +
dim = 2
T=0
N_twist = 1

"""Compare the BCS lattice class with the homogeneous results."""
np.random.seed(1)
hbar, m, kF = 1, 1, 10
eF = (hbar*kF)**2/2/m
mu = 0.28223521359748843*eF
delta = 0.411726229961806*eF

N, L, dx = 16, None, 0.1
args = dict(Nxyz=(N,)*dim, dx=dx)
args.update(T=T)

h = homogeneous.Homogeneous(**args)
b = BCS(**args)

res_h = h.get_densities((mu, mu), delta, N_twist=N_twist)
res_b = b.get_densities((mu, mu), delta, N_twist=N_twist)
print(res_h.n_a.n, res_b.n_a.mean())
print(res_h.n_b.n, res_b.n_b.mean())
print(res_h.nu.n, res_b.nu.mean().real)
# -

# ## Lattice Spectrum

Nx = 32
L = 16
dim = 2
dx = L/Nx
bcs = BCS(Nxyz=(Nx,)*dim, Lxyz=(L,)*dim)
x = bcs.xyz
V=sum(np.array(x)**2/2.0).ravel()
K = bcs._get_K()
H = K + np.diag(V)
Es, psis = np.linalg.eigh(H)
psis=psis.T
Es[:10]

# ### Total Density

n = sum([psi.conj()*psi for psi in psis])
x, y = bcs.xyz
imcontourf(x, y, abs(n.reshape(bcs.Nxyz)))
plt.colorbar()


# ## DVR Spectrum
# * <font color='red'> It can be seen, all states in $\nu !=0$ basis are double degeneracy<font/>

# ## Check Radia Wavefunction

def get_dvr(nu=0):
    dvr = HarmonicDVR(nu=nu, w=1, N_root=30)
    H = dvr.get_H(nu=nu)
    Es, us = np.linalg.eigh(H)
    print(Es[:10])
    res = namedtuple(
                'res', ['dvr', 'Es', 'us'])
    return res(dvr=dvr, Es=Es, us=us)
ds = [get_dvr(nu=nu) for nu in range(3)]

# ### $\nu=0$

get_2d_den(5, 5, N=256)

b = bcs
d, Es, us = ds[2]
print(Es[:10])
x, y = b.xyz
rs = np.sqrt(sum(_x**2 for _x in b.xyz)).ravel()
psi = b.Normalize((psis[5]).reshape(b.Nxyz))
plt.figure(figsize=(13,5))
plt.subplot(121)
imcontourf(x, y, abs(psi))
plt.colorbar()
plt.subplot(122)
plt.plot(rs, abs(psi.ravel()), '+', label="Grid")
plt.plot(d.rs, abs(Normalize(d._get_psi(us.T[0]))), 'o', label="DVR")
plt.plot(d.rs, abs(Normalize(HO_psi(n=2, m=0, rs=d.rs))), '-', label='Analytical')
plt.legend()

x, y = bcs.xyz
d, Es, us = ds[0]
rs = np.sqrt(sum(_x**2 for _x in bcs.xyz)).ravel()
psi = bcs.Normalize((sum(psis[3:5])).reshape(bcs.Nxyz))
plt.figure(figsize=(13,5))
plt.subplot(121)
imcontourf(x, y, abs(psi))
plt.colorbar()
plt.subplot(122)
plt.plot(rs, abs(psi.ravel()), '+', label="Grid")
plt.plot(d.rs, abs(Normalize(d._get_psi(us.T[1]))), 'o', label="DVR")
plt.plot(d.rs, abs(Normalize(HO_psi(n=2, m=1, rs=d.rs))), '-', label='Analytical')
plt.legend()

# ### $\nu=2$

b = b0
d, Es, us = ds[2]
x, y = b.xyz
rs = np.sqrt(sum(_x**2 for _x in b.xyz)).ravel()
psi = b.Normalize((psis0[3]).reshape(b.Nxyz))
plt.figure(figsize=(14,5))
plt.subplot(121)
imcontourf(x, y, abs(psi))
plt.colorbar()
plt.subplot(122)
plt.plot(rs, abs(psi.ravel()), '+', label="Grid")
plt.plot(d.rs, abs(Normalize(d._get_psi(us.T[0]))), 'o', label="DVR")
plt.plot(d.rs, abs(Normalize(HO_psi(n=2, m=0, rs=d.rs))), '-', label='Analytical')
plt.legend()

# ### Overall Density

d0, d1, d2, = ds
def den_dvr(di, n):
    d, Es, us = di
    psi = d._get_psi(us.T[n]) 
    return psi.conj()*psi


# * within a factor of normalization, things look right!

d=d0.dvr
plt.plot(d.rs, den_dvr(d0, 0))
psi = HO_psi(n=0, m=0, rs=d.rs)
plt.plot(d.rs, psi.conj()*psi)

# * Summing up the density of the first 6 states
# * It's clear that the DVR case should count the degeneracy properly

b = b0
x, y = b.xyz
n0 = b.Normalize((sum(abs(psis0[0:6])**2)).reshape(b.Nxyz))
plt.figure(figsize=(13,5))
plt.subplot(121)
imcontourf(x, y, n0)
plt.colorbar()
plt.subplot(122)
n1 = den_dvr(d0, 0) + den_dvr(d0, 1) + den_dvr(d1, 0)*2 + den_dvr(d2, 0)*2
plt.plot(d0.dvr.rs, Normalize(n1), label="DVR")
plt.plot(rs, n0.ravel(), '+', label="Grid")
plt.legend()


# ## 2D Harmonic in a lattice

class BCS_ho(BCS):
    """2D harmonic"""
    def get_v_ext(self, **kw):
        """Return the external potential."""
        V=sum(np.array(self.xyz)**2/2.0)
        return (V, V)


# ## 1D BCS double-check

Nx = 128
L = 23
dim = 1
dx = L/Nx
mu = 5
dmu = 0
delta = 1
b1 = BCS_ho(Nxyz=(Nx,)*dim, Lxyz=(L,)*dim)
res = b1.get_densities(mus_eff=(mu + dmu, mu - dmu), delta=delta)
n_a, n_b = res.n_a, res.n_b
b1._d[Nx: Nx+10]

# ## 2D BCS double-check

Nx = 32
L = 10
dim = 2
dx = L/Nx
mu = 5
dmu = 3.5
delta = 2
b2 = BCS_ho(Nxyz=(Nx,)*dim, Lxyz=(L,)*dim)
res = b2.get_densities(mus_eff=(mu + dmu, mu - dmu), delta=delta)
n_a, n_b = res.n_a, res.n_b
n_a = b2.Normalize(n_a)
n_b = b2.Normalize(n_b)

x, y = b2.xyz
plt.figure(figsize=(18, 4))
plt.subplot(131)
imcontourf(x, y, n_a)
plt.colorbar()
plt.subplot(132)
imcontourf(x, y, n_b)
plt.colorbar()
plt.subplot(133)
rs = np.sqrt(sum(_x**2 for _x in b2.xyz)).ravel()
plt.plot(rs, n_a.ravel(), '+')
plt.plot(rs, n_b.ravel(), 'o')


# # DVR Vortex Class
#
# Steps:
#
# * compute $u, v$
# * compute $\phi_a=wu, \phi_b = wv$, which is the radial wave function
# * compute $n_a = \psi_a^*\psi_a, n_b=\psi_b^*\psi_b, \kappa = \psi_a\psi_b$
# * update $\Delta=-g\kappa$, return to first step to iterate

# ## 2D harmonic in DVR basis

# ### Bugs

# * when N_root = 32, the $n_b$ is different from N_root=33, where the former value yields zero $n_b$, and the later yields more consistent result.
# * the $n_a$ $n_b$ are not excaly the same even when $d\mu=0$, some thing get wrong.
# * Seem for current version of code, N_root=48 works "best" due to the way of normalization(which is not right).

delta = 2
dvr = vortex_dvr_ho(mu=mu, dmu=dmu, E_c=None, N_root=48, delta=delta)
delta = delta + dvr.bases[0].zero
dvr.l_max=100
na, nb, kappa = dvr.get_densities(mus=(mu + dmu,mu - dmu), delta=delta)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(dvr.bases[0].rs, Normalize(na))
plt.plot(rs, n_a.ravel(), '+')
plt.subplot(122)
plt.plot(dvr.bases[0].rs, Normalize(nb))
plt.plot(rs, n_b.ravel(), '+')
clear_output()

# #  Test Bed

mu=0
dmu=0
delta=0
dvr = DVR2D(mu=mu, dmu=dmu, delta=delta)
delta = delta + dvr.bases[0].zero
dvr.l_max=1
while(True):
    n_a, n_b, kappa = dvr.get_densities(mus=(mu,mu), delta=delta)
    delta_ = -dvr.g*kappa
    plt.figure(figsize=(16, 5))
    plt.subplot(121)
    plt.plot(dvr.bases[0].rs, delta_)
    plt.plot(dvr.bases[0].rs, delta,'+')
    plt.title(f"Error={(delta-delta_).max()}")
    plt.ylabel(r"$\Delta$")
    plt.subplot(122)
    plt.plot(dvr.bases[0].rs, n_a)
    plt.plot(dvr.bases[0].rs, n_b, '+')
    plt.show()
    clear_output(wait=True)
    if np.allclose(delta, delta_, atol=1e-8):
        break
    delta=delta_
# + {}
# dmu = 0
# mus = (mu + dmu, mu - dmu)
# dvr = VortexDVR(mu=mu, delta=delta)
# delta = delta + dvr.bases[0].zero

# while(True):
#     n_a, n_b, kappa = dvr.get_densities(mus=(mu,mu), delta=delta)
#     delta_ = -dvr.g*kappa
#     plt.figure(figsize=(16, 5))
#     plt.subplot(121)
#     plt.plot(dvr.bases[0].rs, delta_)
#     plt.plot(dvr.bases[0].rs, delta,'+')
#     plt.title(f"Error={(delta-delta_).max()}")
#     plt.ylabel(r"$\Delta$")
#     plt.subplot(122)
#     plt.plot(dvr.bases[0].rs, n_a)
#     plt.plot(dvr.bases[0].rs, n_b, '+')
#     plt.show()
#     clear_output(wait=True)
#     if np.allclose(delta, delta_, atol=1e-8):
#         break
#     delta=delta_

