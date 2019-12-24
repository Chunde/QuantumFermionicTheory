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

# ## Some Helper functions

# +
def Normalize(psi):
    """Normalize a wave function"""
    return psi/(psi.conj().dot(psi))**0.5


def nan0(data):
    """convert nan to zero"""
    return np.nan_to_num(data, 0)


def HO_psi(n, m, rs):
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

def show_2d_harmonic_oscillator_den(m=0, n=0, L=5, N=100):
    """Show 2D harmonic oscillator density"""
    ho = HarmonicOscillator2D()
    rs = np.linspace(-L, L, N)
    zs = ho.get_wf(rs, n=n, m=m)
    imcontourf(rs, rs, zs.conj()*zs)


# -

# ## Cylindrical DVR Class

# +
from mmfutils.math import bessel

class CylindricalBasis(object):
    eps = 7./3 - 4./3 -1  # machine precision
    m = hbar = 1

    def __init__(self, N_root=None, R_max=None, K_max=None, a0=None, nu=0, **args):
        """
        Parameters
        --------------
        N_root: int
            number of roots
        R_max: float
            max radius range
        K_max: float
            momentum cutoff
        a0: float
            wavefunction position scale
        nu: int
            angular momentum quantum number
        dim: int
            dimensionality
        """
        self.N_root = N_root
        self.R_max = R_max
        self.K_max = K_max
        if N_root is None or R_max is None or K_max is None:
            self._init(a0=a0)
            
        self._align_K_max()
        self.dim = 2
        self.nu = nu
        self.zs = self.get_zs(nu=nu)
        self.rs = self.get_rs(zs=self.zs)
        self.K = self.get_K(zs=self.zs, nu=nu)
        self.zero = np.zeros_like(self.zs)
        self.rs_scale = self._rs_scaling_factor(zs=self.zs)
        self.ws = self.get_F_rs()/self.rs_scale  # weight

    def _init(self, a0=None):
        """evaluate R_max and K_max using Gaussian wavefunction"""
        if a0 is None:
            a0 = 1
        if self.R_max is None:
            self.R_max = np.sqrt(-2*a0**2*np.log(self.eps))
        if self.K_max is None:
            self.K_max = np.sqrt(-np.log(self.eps)/a0**2)
        if self.N_root is None:
            self.N_root = int(np.ceil(self.K_max*2*self.R_max/np.pi))

    def _align_K_max(self):
        """
        For large n, the roots of the bessel function are approximately
        z[n] = (n + 0.75)*pi, so R = R_max = z_max/K_max = (N-0.25)*pi/K_max
        """
        self.K_max = (self.N_root - 0.25)*np.pi/self.R_max
    
    def get_zs(self, nu=None):
        """
        return roots for order $\nu$
        """
        if nu is None:
            nu = self.nu
        zs = bessel.j_root(nu=nu, N=self.N_root)
        return zs

    def get_rs(self, zs=None, nu=None):
        """
        return cooridnate in postition space
        """
        if nu is None:
            nu = self.nu
        if zs is None:
            zs = self.get_zs(nu=nu)
        return zs/self.K_max

    def get_F(self, nu=None, n=0, rs=None):
        """return the nth basis function for nu"""
        if nu is None:
            nu = self.nu
        if rs is None:
            rs = self.rs
        zs = self.get_zs(nu=nu)
        F = (-1)**(n+1)*self.K_max*zs[n]*np.sqrt(2*rs)/(
            self.K_max**2*rs**2-zs[n]**2)*bessel.J(nu, 0)(self.K_max*rs)
        F=nan0(F)
        return F

    def get_F_rs(self, zs=None, nu=None):
        """
        return the basis function values at abscissa r_n
        for the nth basis function. Since each basis function
        have non-zero value only at its own r_n, we just need to
        compute that value, all other values are simply zero

        """
        if nu is None:
            nu = self.nu
        if zs is None:
            rs = self.rs
            zs = self.zs
        else:
            rs = zs/self.K_max
        Fs = [(-1)**(n+1)*self.K_max*np.sqrt(2*rs[n]*zs[n])/(2*zs[n])*bessel.J_sqrt_pole(
            nu=nu, zn=zs[n])(zs[n]) for n in range(len(rs))]
        return Fs

    def _rs_scaling_factor(self, zs=None):
        """
        the dimension dependent scaling factor used to
        convert from u(r) to psi(r)=u(r)/rs_, or u(r)=psi(r)*rs_
        """
        if zs is None:
            zs = self.zs
        rs = self.get_rs(zs=zs)
        rs_ = rs**((self.dim - 1)/2.0)
        return rs_
        
    def _get_psi(self, u):
        """apply weight on the u(v) to get the actual radial wave-function"""
        return u*self.ws

    def get_nu(self, nu=None):
        """
         `nu + d/2 - 1` for the centrifugal term
         Note:
            the naming convention use \nu as angular momentum quantum number
            but it's also being used as the order number of bessel function
         """
        if nu is None:
            nu = self.nu
        return nu + self.dim/2.0 - 1

    def get_K(self, zs=None, nu=None):
        """
        return the kinetic matrix for a given nu
        Note: the centrifugal potential is already include
        """
        if nu is None:
            nu = self.nu
        if zs is None:
            zs = self.get_zs(nu=nu)
        zi = np.array(list(range(len(zs)))) + 1
        xx, yy = np.meshgrid(zi, zi, sparse=False, indexing='ij')
        zx, zy = np.meshgrid(zs, zs, sparse=False, indexing='ij')
        nu = self.get_nu(nu)  # see get_nu(...)
        K_diag = (1+2*(nu**2 - 1)/zs**2)/3.0  # diagonal terms
        K_off = 8*(-1)**(abs(xx - yy))*zx*zy/(zx**2 - zy**2 + self.eps)**2
        np.fill_diagonal(K_off, K_diag)
        K = self.K_max**2*K_off/2.0  # factor of 1/2 include
        return K


    def get_V_correction(self, nu):
        """
            if nu is not the same as the basis, a piece of correction
            should be made to the centrifugal potential
        """
        return (nu**2 - self.nu**2)*self.hbar**2/2.0/self.rs**2

# -

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

# -

# # Compare Radial Wavefunctions

# ## 2D BCS Lattice

Nx = 32
L = 16
dim = 2
dx = L/Nx
b = BCS(Nxyz=(Nx,)*dim, Lxyz=(L,)*dim)
x = b.xyz
V=sum(np.array(x)**2/2.0).ravel()
K = b._get_K()
H = K + np.diag(V)
Es, psis = np.linalg.eigh(H)
psis=psis.T
Es[:10]


# ## DVR

def get_dvr(nu=0):
    dvr = HarmonicDVR(nu=nu, w=1, N_root=30)
    H = dvr.get_H(nu=nu)
    Es, us = np.linalg.eigh(H)
    print(Es[:10])
    res = namedtuple(
                'res', ['dvr', 'Es', 'us'])
    return res(dvr=dvr, Es=Es, us=us)
ds = [get_dvr(nu=nu) for nu in range(3)]

# ## Compare Radial Functions

# +
d, Es, us = ds[0]
print(Es[:10])
x, y = b.xyz
rs = np.sqrt(sum(_x**2 for _x in b.xyz)).ravel()
psi_bcs = b.Normalize(psis[0]).reshape(b.Nxyz)

plt.figure(figsize=(18,4))
plt.subplot(131)
imcontourf(x, y, abs(psi_bcs))
plt.colorbar()

plt.subplot(132)
psi_dvr = d._get_psi(us.T[0])
plt.plot(rs, abs(psi_bcs.ravel()), '+', label="Grid")
plt.plot(d.rs, abs(psi_dvr), 'o', label="DVR")
plt.title("Unnormalized Wavefunctions")
plt.legend()

plt.subplot(133)
psi_bcs = b.Normalize(psi = psis[0].reshape(b.Nxyz))
psi_dvr = Normalize(d._get_psi(us.T[0]))
plt.plot(rs, abs(psi_bcs.ravel()), '+', label="Grid")
plt.plot(d.rs, abs(psi_dvr), 'o', label="DVR")
plt.title("Normalized Wavefunction")
plt.legend()
# -
np.diff(d.rs)


# # BdG in Ratating Frame Transform

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

# # Transform Matrix

# Since differnet bases have different abcissas, the final result should be presented expanded in single basis or we can't compare anything. 
# Expand a function in two different bases $\ket{F}, \ket{F'}$. where $\braket{F_{n}|F_{n'}}=\delta(n, n')$ and $\braket{F'_{n}|F'_{n'}}=\delta(n, n')$
# $$
# \ket{\psi}=\sum_i C_i\ket{F_i}=\sum_i C'_i\ket{F'_i}
# $$
# To get $C_i$, multiply $\bra{F_j}$ from the left to get:

# $$
# \bra{F_j}\sum_i C_i\ket{F_i}=\bra{F_j}\sum_i C'_i\ket{F'_i}
# $$
#
# Use the orthognal relations:
#
# $$
# C_j=\bra{F_j}\sum_i C_i\ket{F_i}=\bra{F_j}\sum_i C'_i\ket{F'_i}=\sum_i \braket{F_j|F'_i}C'_i
# $$

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

# ## 3D cases


