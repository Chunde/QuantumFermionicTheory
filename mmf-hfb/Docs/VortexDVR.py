# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
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
from mmf_hfb.VortexDVR import bdg_dvr, bdg_dvr_ho,dvr_full_set
from mmf_hfb.utils import block

# # Vortices
# Let us get started with the single partilce Harmitonian
# $$
#   \op{H}\psi_{n,l_z}(r, \theta) = E\psi_{n,l_z}(r, \theta)
# $$
# where $\op{H} = \frac{-\hbar^2\nabla^2}{2m} - \mu$
#
# Let the full wavefunction $\psi_{n, l_z}(r,\theta)=R_{n, l_z}(r)e^{il_z\theta}$, plugin back to the above equation:
# \begin{align}
#   E\psi_{n,l_z}(r, \theta)
#   &=\left(\frac{-\hbar^2}{2m}\nabla^2 - \mu\right)R_{n, l_z}(r)e^{il_z\theta}\\
# \end{align}

# Let $R(r) = r^{-1/2}f(r)$, the the schrodinger equation changes to:
#
# $$
# \left[-\frac{\hbar^2}{2m}\left(\frac{\partial^2 }{\partial r^2}- \frac{1_z^2 - 1/4}{r^2}\right)-\mu\right]f(r)=Ef(r)
# $$
#
# then in the DVR basis,$f(r)$ can be expanded using the bais function $F(r)$, where $u_i$ are the coefficients:
# $$
# f(r) = \sum_i F_i(r)u_i
# $$
# When diagonalizing the BdG-like Hamontonian, we will the $u$s and $v$s, from that we compute the original $f(r)$. After that we computer $R(r)$ by:
# $$
# R(r)=\frac{f(r)}{\sqrt{2\pi r}}
# $$
# where the $\sqrt{2\pi}$ is a nomalization factor to make the full wave-function unitary.
# ## BdG

# $$
#   \begin{pmatrix}
#     -\frac{\hbar^2}{2m}\nabla^2 - \mu_a & \Delta(r)e^{\I w \theta}\\
#     \Delta(r)e^{-i w \theta} & \frac{\hbar^2}{2m}\nabla^2 + \mu_b
#   \end{pmatrix}
#   \begin{pmatrix}
#     \sqrt{r}u_{n, l_z}(r)e^{\I w \theta}\\
#     \sqrt{r}v^*_{n, l_z}(r)
#   \end{pmatrix}
#   e^{\I l_z \theta}
# $$
#
# $$
#   \begin{pmatrix}
#     -\frac{\hbar^2}{2m}\left(\diff[2]{}{r} - \frac{(l_z+w)^2-1/4}{r^2}\right) - \mu_a & \Delta(r)\\
#     \Delta(r) & \frac{\hbar^2}{2m}\left(\diff[2]{}{r} -\frac{l_z^2-1/4}{r^2}\right) + \mu_b
#   \end{pmatrix}
#   \begin{pmatrix}
#     u_{n, l_z}(r)\\
#     v^*_{n, l_z}(r)
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#   E_{n, l_z}&0\\
#   0&-E_{n, l_z}
#   \end{pmatrix}
#   \begin{pmatrix}
#     u_{n, l_z}(r)\\
#     v^*_{n, l_z}(r)
#   \end{pmatrix}
# $$

# +
import mmf_hfb.VortexDVR  as vd; reload(vd)
from mmf_hfb.VortexDVR import bdg_dvr,dvr_vortex,BCS_vortex
mu, dmu = 5, 0
mus=(mu + dmu,mu - dmu)
E_c=20
delta_bcs=delta_dvr=delta=2
# BCS
bcs = BCS_vortex(Nxyz=(32,)*2, Lxyz=(10,)*2, mus_eff=(mu+dmu, mu-dmu), delta=delta)
x, y = bcs.xyz
rs = np.sqrt(sum(_x**2 for _x in bcs.xyz)).ravel()
# DVR
dvr = dvr_vortex(mu=mu, dmu=dmu, delta=delta, g=bcs.g, E_c=E_c, bases=None, N_root=32, R_max=5, l_max=200)
delta_bcs = delta*(x+1j*y) # if n == 1 else delta*(x**2-y**2+2j*x*y)# 
delta_dvr = delta*dvr.rs
dvr.wz = 0 if np.size(delta_bcs)==1 else 1
bcs.E_c=E_c
dvr.E_c=E_c
def update_plot(delta_bcs_, delta_dvr_):
    res_bcs = bcs.get_densities(mus_eff=mus, delta=delta_bcs_)
    na_bcs, nb_bcs, nu_bcs, ja_bcs, jb_bcs = res_bcs.n_a, res_bcs.n_b, res_bcs.nu, res_bcs.j_a, res_bcs.j_b
    res_dvr = dvr.get_densities(mus=mus, delta=delta_dvr_)
    na_dvr, nb_dvr, nu_dvr, ja_dvr, jb_dvr =res_dvr.n_a, res_dvr.n_b, res_dvr.nu, res_dvr.j_a, res_dvr.j_b
    
    delta_dvr_tmp = dvr.g*nu_dvr
    delta_bcs_tmp = bcs.g*nu_bcs   
    err_dvr = np.max(abs((delta_dvr_tmp - delta_dvr_)))
    err_bcs = np.max(abs((delta_bcs_tmp - delta_bcs_)))
    plt.figure(figsize=(18, 15))    
    plt.subplot(331)
    imcontourf(x, y, na_bcs)
    plt.colorbar()
    plt.title(r"$n_a$")
    plt.subplot(332)
    if np.size(delta_bcs_tmp) == np.prod(bcs.Nxyz):
        imcontourf(x, y, abs(delta_bcs_tmp))
        plt.colorbar()
    plt.title(r"$\Delta$")    
    plt.subplot(334)  # n_a 
    plt.plot(dvr.rs, na_dvr, label=r'$n_a$(DVR)')
    plt.plot(rs, na_bcs.ravel(), '+', label=r'$n_a$(Grid)')
    plt.legend()
    plt.subplot(335)  # n_b
    plt.plot(dvr.rs, nb_dvr, label=r'$n_b$(DVR)')
    plt.plot(rs, nb_bcs.ravel(), '+', label=r'$n_b$(Grid)')
    plt.legend()
    plt.subplot(336)  # nu
    plt.plot(dvr.rs, abs(delta_dvr_tmp), label=r'$\Delta$(DVR)')
    plt.plot(rs, abs(delta_bcs_tmp).ravel(), '+', label=r'$\Delta$(Grid)')
    plt.legend()
    plt.subplot(333)
    plt.plot(dvr.rs, nb_dvr + na_dvr, label=r'$n_+$(DVR)')
    plt.plot(rs, (nb_bcs + na_bcs).ravel(), '+', label=r'$n_+$(Grid)')
    plt.legend()
    clear_output(wait=True)
    plt.show()
    return (delta_bcs_tmp, delta_dvr_tmp, err_bcs, err_dvr)

with NoInterrupt() as interrupted:
    for n in range(1):
        delta_bcs_, delta_dvr_, err_bcs, err_dvr = update_plot(delta_bcs, delta_dvr)
        if err_bcs<1e-5 or err_dvr <1e-5:
            break
        err_dvr = np.max(abs(delta_dvr - delta_dvr_))
        err_bcs = np.max(abs(delta_bcs - delta_bcs_))
        # delta_bcs, delta_dvr = delta_bcs_, delta_dvr_
        print(n, err_dvr, err_bcs)
# -

# ## Spectrum

H = bcs.get_H(mus_eff=mus, delta=delta_bcs)
Eb, UV = np.linalg.eigh(H)
(n_a_bcs, n_b_bcs), (tau_a_bcs, tau_b_bcs),(j_a_bcs, j_b_bcs), nu_bcs = bcs._unpack_densities(bcs._get_densities_H(H, dUV=(Eb, UV)))

# +
Ed = []
def _get_den(self, H, lz):
    es, phis = np.linalg.eigh(H)
    phis = phis.T
    offset = phis.shape[0] // 2
    dens_a,dens_b,dens_nu = [], [], []
    for i in range(len(es)):
        E, uv = es[i], phis[i]
        if abs(E) > self.E_c:
            continue
        Ed.append(E)
        u, v = uv[: offset], uv[offset:]
        u = self.get_psi(lz=lz, u=u)
        v = self.get_psi(lz=lz, u=v)
        f_p, f_m = self.f(E=E), self.f(E=-E)
        n_a = u*u.conj()*f_p
        n_b = v*v.conj()*f_m
        j_a = -n_a*self.wz/self.rs/2
        j_b = -n_b*self.wz/self.rs/2
        kappa = u*v.conj()*(f_p - f_m)/2
        dens_a.append(np.array([n_a, j_a]))
        dens_b.append(np.array([n_b, j_b]))
        dens_nu.append(kappa)
    return np.array([sum(dens_a), sum(dens_b), sum(dens_nu)])

dens_a=dens_b=dens_nu=0
for lz in range(-dvr.l_max, dvr.l_max):  # sum over angular momentum
    H = dvr.get_H(mus=mus, delta=delta_dvr, lz=lz)
    den = _get_den(dvr, H, lz=lz)
    den_a, den_b, den_nu = den
    dens_a = dens_a + den_a
    dens_b = dens_b + den_b
    dens_nu = dens_nu + den_nu
n_a_dvr, j_a_dvr = dens_a
n_b_dvr, j_b_dvr = dens_b
kappa_dvr = dens_nu
Ed=np.sort(Ed)
# -

plt.plot(dvr.rs, n_a_dvr, label=r'$n_a$(DVR)')
plt.plot(rs, n_a_bcs.ravel(), '+', label=r'$n_a$(Grid)');plt.legend()

np.sort(abs(Eb))[:32], np.sort(abs(Ed))[:32]

plt.figure(figsize=(16, 6))
plt.plot(np.sort(abs(Eb))[:100],'+', label="BCS")
plt.plot(np.sort(abs(Ed))[:100] + 0.1,'+', label="DVR")
plt.ylabel('E');plt.legend()

# # BCS

Nx = 32
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

Es0[:32]


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
#plt.figure(figsize(16, 6))
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


# ### Spectrum

h = HarmonicDVR(nu=1)
H = h.get_H()
d,u =np.linalg.eigh(H)
d

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

#plt.figure(figsize=(16,6))
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

# +
#plt.figure(figsize=(16,9))
linestyles = ['-', '--']
parities = ['odd', 'even']
E_max = 30
Ess = [[] for _ in range(E_max)]
Nss = [[] for _ in range(E_max)]
Ns = list(range(1, 40))
for N in Ns:
    dvr_o = HarmonicDVR(nu=0, w=1, N_root=N)
    dvr_e = HarmonicDVR(nu=1, w=1, N_root=N)
    for (i, dvr) in enumerate([dvr_o, dvr_e]):
        H = dvr.get_H()
        Es, us = np.linalg.eigh(H)
        ns = np.array(list(range(len(Es))))
        Es0 = 2*ns + 2**i  # analytica energy spectrum
        errs = (Es - Es0)/Es0
        for j, E in enumerate(Es0):
            if E > E_max:
                break
            Ess[E-1].append(errs[j])
            Nss[E-1].append(N)
# plotting
for i, Es in enumerate(Ess):
    if i <= 20:
        continue
    plt.semilogy(Nss[i],Es, linestyles[i%2],label=f'E={i+1}')

plt.legend()
# -

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

#plt.figure(figsize=(16, 8))
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

# # Derivatives in DVR

# To evaluate the first order derivate of a function $f(x)$ in a DVR basis the function should be firstly expressed using the basis functions:
#
# $$
# f(x)=\sum_n{u_i F_i(x)}
# $$
# where $u_i$ is the weigth for the corresponding basis function $F_i(x)$. Assume we just want to evaluate the derivative at the abscisas $x_1, x_2, \dots, x_n$, then, due to the property of interpolation, we can evaluate $u_i$ easily. i.e:
# $$
# u_i = \frac{f(x_i)}{F_i(x_i)}
# $$
#
# To compute the first order derivative:
# $$
# \frac{\partial f}{\partial x}\big|_{x_i} = u_i\frac{\partial F_i}{\partial x}\big|_{x_i}\qquad i=1, 2, \dots n
# $$
# Similarily, the second order derivative can be computed as:
# $$
# \frac{\partial^2 f}{\partial x^2}\big|_{x_i} = u_i\frac{\partial^2 F_i}{\partial x^2}\big|_{x_i}
# $$

import random
dvr = CylindricalBasis(nu=0, R_max=9, N_root=64)
zs, rs = dvr.zs, dvr.rs
Fs = dvr.get_F_rs()
fs = np.sin(rs)
us = fs/Fs
dFs = dvr.get_F_rs(d=1)
dfs = us*dFs
plt.plot(rs, np.cos(rs))
plt.plot(rs, us*dFs, '+')

from mmfutils.math import bessel
nu=0
d=1
Fs = [(-1)**(n+1)*dvr.K_max*np.sqrt(2*rs[n]*zs[n])/(2*zs[n])*bessel.J_sqrt_pole(
            nu=nu, zn=zs[n], d=d)(zs[n]) for n in range(len(rs))]

Z = np.linspace(zs[0]-.1,zs[0]+.1 , 1000)
def f(z, d=1):
    return bessel.J_sqrt_pole(nu=0.5, zn=zs[0], d=d)(z)
#plt.plot(Z, f(Z, d=1))
plt.plot(Z, f(Z, d=1))


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


dvr0 = CylindricalBasis(nu=0, R_max=9, Nroot=40)
dvr1 = CylindricalBasis(nu=1, R_max=9, Nroot=38)
U10 = get_transform_matrix(dvr1, dvr0)
z0 = dvr0.zs
z1 = dvr1.zs


def transform(dvr_s, dvr_t, us_s):
    def f(r):
        fs = [us_s[n]*dvr_s.get_F(n=n, rs=r) for n in range(len(us_s))]
        return sum(fs)
    psi = [f(r) for r in dvr_t.rs]
    Fs = dvr_t.get_F_rs()
    us_t = np.array(psi)/np.array(Fs)
    return us_t


import random
us1 = np.cos(np.linspace(0,5+15*np.random.random(), len(z1)))
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
   # plt.figure(figsize=(14,5))
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


mu, dmu, delta0 = 5, 3.5, 2

# ## BCS in a Box

delta=delta0
b2 = BCS_ho(Nxyz=(32,)*2, Lxyz=(10,)*2)
res = b2.get_densities(mus_eff=(mu + dmu, mu - dmu), delta=delta)
n_a, n_b = res.n_a, res.n_b
x, y = b2.xyz
rs = np.sqrt(sum(_x**2 for _x in b2.xyz)).ravel()
#plt.figure(figsize=(18, 4))
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
delta = delta + dvr.bases.zero
res = dvr.get_densities(mus=(mu + dmu, mu - dmu), delta=delta)
na, nb, kappa, j_a, j_b = res.n_a, res.n_b, res.nu, res.j_a, res.j_b
#plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(dvr.rs, (na), label=r'$n_a$(DVR)')
plt.plot(rs, n_a.ravel(), '+', label=r'$n_a$(Grid)')
plt.legend()
plt.subplot(122)
plt.plot(dvr.rs, (nb), label=r'$n_b$(DVR)')
plt.plot(rs, n_b.ravel(), '+', label=r'$n_b$(Grid)')
plt.legend()
clear_output();plt.show();

# ## Check Energy Spectrum in DVR & Grid

delta=delta0
mus = (mu+dmu, mu-dmu)
H = b2.get_H(mus_eff=mus, delta=delta)
d, UV = np.linalg.eigh(H)
U, V = U_V = b2.get_U_V(H=H, UV=UV)
dU_Vs = b2._Del(U_V)
dUs, dVs = dU_Vs[:, 0, ...], dU_Vs[:, 1, ...]
f_p, f_m = b2.f(d), b2.f(-d)
n_a = np.dot(U*U.conj(), f_p).real
n_b = np.dot(V*V.conj(), f_m).real
nu = np.dot(U*V.conj(), f_p - f_m)/2
tau_a = np.dot(sum(dU.conj()*dU for dU in dUs), f_p).real
tau_b = np.dot(sum(dV.conj()*dV for dV in dVs), f_m).real
j_a = [0.5*np.dot((U.conj()*dU - U*dU.conj()), f_p).imag for dU in dUs]
j_b = [0.5*np.dot((V*dV.conj() - V.conj()*dV), f_m).imag for dV in dVs]

Es = []
dvr.lz=0
dvr.l_max=20
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
        Es.append(E)
        if nu != 0:
            Es.append(E)
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
dvr.E_c = max(d)
H = dvr.get_H(mus=mus, delta=delta, nu=0)
dens = _get_den(self=dvr, H=H, nu=0)
for nu in range(1, dvr.l_max):  # sum over angular momentum
    H = dvr.get_H(mus=mus, delta=delta, nu=nu)
    dens = dens + 2*_get_den(self=dvr,H=H, nu=nu)  # double-degenerate
Es = np.sort(Es)

np.sort(abs(d))[0:40]

np.sort(abs(Es))[:40]



# ## To-Do
# * Compute $\tau$ and $j_{\pm a/b}$ terms: The derivative term from the bessel package seems to have bug.
# * Integral over the Z dirction
# * Implement ASLDA
# * figure out why J_sqrt_pole always gives 0s
# $$
# \Delta(r,\theta) = r^2e^{i2\theta}=r^2\left[1-2sin^2(\theta)+2isin(\theta)cos(\theta)\right]=(x^2-y^2+2ixy)
# $$

# # Vortex in BdG
# Let us get started with the single partilce Harmitonian
# $$
#   \op{H}\psi_{n,l_z}(r, \theta) = E\psi_{n,l_z}(r, \theta)
# $$
# where $\op{H} = \frac{-\hbar^2\nabla^2}{2m} - \mu$
#
# Let the full wavefunction $\psi_{n, l_z}(r,\theta)=R_{n, l_z}(r)e^{il_z\theta}$, plugin back to the above equation:
# \begin{align}
#   E\psi_{n,l_z}(r, \theta)
#   &=\left(\frac{-\hbar^2}{2m}\nabla^2 - \mu\right)R_{n, l_z}(r)e^{il_z\theta}\\
# \end{align}

# In polar coordinates, the Del operator $\nabla^2$ is defined as:
# $$
# \begin{align}
# \nabla^2
# &=\frac{1}{r} \frac{\partial}{\partial r}\left(r \frac{\partial }{\partial r}\right)+\frac{1}{r^{2}} \frac{\partial^{2} }{\partial \theta^{2}}\\
# &=\frac{\partial^2 }{\partial r^2}+\frac{1}{r} \frac{\partial }{\partial r}+\frac{1}{r^{2}} \frac{\partial^{2} }{\partial \theta^{2}}
# \end{align}
# $$
# The Schrodinger equatin can rewritten as:
# \begin{align}
#   E\psi_{n,l_z}(r, \theta)
#   &=\left(\frac{-\hbar^2}{2m}\left[
#     \frac{1}{r}\pdiff{}{r}\left(r \pdiff{}{r}R_{n, l_z}(r)\right)
#     - \frac{l_z^2}{r^2}R_{n, l_z}(r)\right]
#     - \mu R_{n, l_z}(r)\right)e^{il_z\theta},\\
# \end{align}
#
# Let $R(r) = r^{-1/2}f(r)$, the kenitic part can  be expressed:
#
# \begin{align}
# \left[\frac{\partial^2 }{\partial r^2}+\frac{1}{r} \frac{\partial }{\partial r}\right]\left[r^{-1/2}f(r)\right]
# &=\frac{\partial }{\partial r}\left[-\frac{1}{2}r^{-3/2}f(r)+r^{-1/2}\frac{\partial f}{\partial r}\right]+\frac{1}{r}\left[-\frac{1}{2}r^{-3/2}f+r^{-1/2}\frac{\partial f}{\partial r}\right]\\
# &=\left[\frac{3}{4}r^{-5/2}f(r)-\frac{1}{2}r^{-3/2}\frac{\partial f}{\partial r}-\frac{1}{2}r^{-3/2}\frac{\partial f}{\partial r}+ r^{-1/2}\frac{\partial^2 f}{\partial r^2}\right]+\left[-\frac{1}{2}r^{-5/2}f(r)+r^{-3/2}\frac{\partial f}{\partial r}\right]\\
# &=\frac{1}{4}r^{-5/2}f(r)+r^{-1/2}\frac{\partial^2 f}{\partial r^2}\\
# &=r^{-1/2}\left[\frac{\partial^2 }{\partial r^2}+ \frac{1}{4r^2}\right]f(r)
# \end{align}

# Combinaition with th angular part yields:
# \begin{align}
# \left(-\frac{\hbar^2}{2m}\nabla^2 - \mu\right)R_{n, l_z}(r)e^{il_z\theta}
# &=
#   \left(-\frac{\hbar^2}{2m}\left[
#     \frac{1}{r}\pdiff{}{r}\left(r \pdiff{}{r}R_{n, l_z}(r)\right)
#     - \frac{l_z^2}{r^2}R_{n, l_z}(r)\right]
#     - \mu R_{n, l_z}(r)\right)e^{il_z\theta},\\
# &=r^{-1/2}\left[\frac{-\hbar^2}{2m}\left(\frac{\partial^2 }{\partial r^2}+ \frac{1}{4r^2}-\frac{l_z^2}{r^2}\right)-\mu\right]f(r)e^{il_z\theta}\\
# &=r^{-1/2}\left[\frac{-\hbar^2}{2m}\left(\frac{\partial^2 }{\partial r^2}- \frac{1_z^2 - 1/4}{r^2}\right)-\mu\right]f(r)e^{il_z\theta}
# \end{align}
#
# Substituion back to the Harmitonian gives:
# $$
# \left[-\frac{\hbar^2}{2m}\left(\frac{\partial^2 }{\partial r^2}- \frac{1_z^2 - 1/4}{r^2}\right)-\mu\right]f(r)=Ef(r)
# $$
# Define an operator $\op{K}$ as:
#
# $$
# \op{K}=-\frac{\hbar^2}{2m}\left(\frac{\partial^2 }{\partial r^2}- \frac{1_z^2 - 1/4}{r^2}\right)
# $$

# $$
#   \begin{pmatrix}
#     -\frac{\hbar^2}{2m}\nabla^2 - \mu_a & \Delta(r)e^{\I w \theta}\\
#     \Delta(r)e^{-i w \theta} & \frac{\hbar^2}{2m}\nabla^2 + \mu_b
#   \end{pmatrix}
#   \begin{pmatrix}
#     \sqrt{r}u_{n, l_z}(r)e^{\I w \theta}\\
#     \sqrt{r}v^*_{n, l_z}(r)
#   \end{pmatrix}
#   e^{\I l_z \theta}
# $$
#
# $$
#   \begin{pmatrix}
#     -\frac{\hbar^2}{2m}\left(\diff[2]{}{r} - \frac{(l_z+w)^2-1/4}{r^2}\right) - \mu_a & \Delta(r)\\
#     \Delta(r) & \frac{\hbar^2}{2m}\left(\diff[2]{}{r} -\frac{l_z^2-1/4}{r^2}\right) + \mu_b
#   \end{pmatrix}
#   \begin{pmatrix}
#     u_{n, l_z}(r)\\
#     v^*_{n, l_z}(r)
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#   E_{n, l_z}&0\\
#   0&-E_{n, l_z}
#   \end{pmatrix}
#   \begin{pmatrix}
#     u_{n, l_z}(r)\\
#     v^*_{n, l_z}(r)
#   \end{pmatrix}
# $$

# # Find a Vortex

# +
plt.figure(figsize(10,10))
x = np.linspace(-1, 1,100)
y = np.linspace(-1, 1,100)
z0 = 0.5+0.5j
xi =0.1
z=x[:, None]+y[None,:]*1j

def f(r):
    return r/np.sqrt(r**2 + xi**2)

r = z - z0
psi = f(abs(r))*r/abs(r)*np.exp(1j*0.3)
n=abs(psi)**2
imcontourf(x,y, n)

ix, iy = np.unravel_index(np.argmin(n), psi.shape)
z0_ = z.flat[ix]
P = np.polyfit(z[ix-5:ix+5, iy-5:iy+5].ravel(),
              psi[ix-5:ix+5, iy-5:iy+5].ravel(), deg=3)
print(z0_, np.roots(P))
# -


