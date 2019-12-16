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


# -

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
# -

plt.figure(figsize=(16, 8))
h = HarmonicDVR(nu=1, w=1)
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
#
# Steps:
#
# * compute $u, v$
# * compute $\phi_a=wu, \phi_b = wv$, which is the radial wave function
# * compute $n_a = \psi_a^*\psi_a, n_b=\psi_b^*\psi_b, \kappa = \psi_a\psi_b$
# * update $\Delta=-g\kappa$, return to first step to iterate

# +
from mmf_hfb.DVRBasis import CylindricalBasis
from mmf_hfb.utils import block
from mmf_hfb import homogeneous
import numpy as np


class VortexDVR(object):
    """
    A 2D and 3D vortex class without external potential
    """
    def __init__(self, bases_N=2, mu=1, dmu=0, delta=1, T=0, l_max=100, **args):
        """
        Construct and cache some information of bases

        """
        self.bases = [CylindricalBasis(nu=nu, **args) for nu in range(bases_N)]
        self.l_max = max(l_max, 1)  # the angular momentum cut_off
        assert T==0
        self.T=T
        self.g = self.get_g(mu=mu, delta=delta)
        self.mus = (mu + dmu, mu - dmu)

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

    def get_H(self, mus, delta, nu=0):
        """
        return the full Hamiltonian(with pairing field)
        """
        basis = self.bases[self.basis_match_rule(nu)]
        T = basis.K
        Delta = np.diag(basis.zero + delta)
        mu_a, mu_b = mus
        V_ext = self.get_Vext(rs=basis.rs)
        V_corr = basis.get_V_correction(nu=nu)
        V_mean_field = basis.get_V_mean_field(nu=nu)
        V_eff = V_ext + V_corr + V_mean_field
        H_a = T + np.diag(V_eff - mu_b)
        H_b = T + np.diag(V_eff - mu_a)
        H = block(H_a, Delta, Delta.conj(), -H_b)
        return H

    def get_g(self, mu=1.0, delta=0.2):
        """
        the interaction strength
        """
        h = homogeneous.Homogeneous(dim=3)
        res = h.get_densities(mus_eff=(mu, mu), delta=delta)
        g = 0 if res.nu == 0 else delta/res.nu
        return g

    def _get_psi(self, nu, u):
        """apply weight on the u(v) to get the actual radial wave-function"""
        b = self.bases[self.basis_match_rule(nu)]
        return u*b.ws

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
            u, v = uv[: offset], uv[offset:]
            u = self._get_psi(nu=nu, u=v)
            v = self._get_psi(nu=nu, u=v)
            f_p, f_m = self.f(E=E), self.f(E=-E)
            n_a = u*u.conj()*f_p
            n_b = v*v.conj()*f_m
            kappa = u*v.conj()*(f_p - f_m)/2
            # fe = self.f(E=E)
            # n_a = (1 - fe)*v**2
            # n_b = fe*u**2
            # kappa = (1 - 2*fe)*u*v
            den = den + np.array([n_a, n_b, kappa])
        return den

    def get_densities(self, mus, delta):
        """
        return the particle number density and anomalous density
        Note: Here the anomalous density is represented as kappa
        instead of \nu so it's not that confusing when \nu has
        been used as angular momentum quantum number.
        """
        dens = 0
        for nu in range(self.l_max):  # sum over angular momentum
            H = self.get_H(mus=mus, delta=delta, nu=nu)
            dens = dens + self._get_den(H, nu=nu)
        n_a, n_b, kappa = dens
        return (n_a, n_b, kappa)
# -

# # Compare to 2D Box


from mmfutils.plot import imcontourf

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

# +
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
# -

# ### Lattice Spectrum

Nx = 32
L = 16
dim = 2
dx = L/Nx
bcs = BCS(Nxyz=(Nx,)*dim, Lxyz=(L,)*dim)
x = bcs.xyz
V=sum(np.array(x)**2/2.0).ravel()
K = bcs._get_K()
H = K + np.diag(V)
Es, phis = np.linalg.eigh(H)
Es[:10]

# ### DVR Spectrum

Es = []
h = HarmonicDVR(nu=0, w=1, N_root=10)
H = h.get_H(nu=None)
Es_, us = np.linalg.eigh(H)
Es.extend(Es_)
print(np.sort(Es))

Nx = 128
L = 23
dim = 1
dx = L/Nx
mu = 0
dmu = 0
delta = 0
b2 = HO_bcs(Nxyz=(Nx,)*dim, Lxyz=(L,)*dim)
res = b2.get_densities(mus_eff=(mu + dmu, mu - dmu), delta=delta)
n_a, n_b = res.n_a, res.n_b

b2._d[Nx: Nx+32]


# ## 2D Harmonic in a lattice

class HO_bcs(BCS):
    """2D harmonic"""
    def get_v_ext(self, **kw):
        """Return the external potential."""
        V=sum(np.array(self.xyz)**2/2.0)
        return (V, V)


Nx = 32
L = 10
dim = 2
dx = L/Nx
mu = 5
dmu = 0
delta = 1
b2 = HO_bcs(Nxyz=(Nx,)*dim, Lxyz=(L,)*dim)
res = b2.get_densities(mus_eff=(mu + dmu, mu - dmu), delta=delta)
n_a, n_b = res.n_a, res.n_b

x, y = b2.xyz
plt.figure(figsize=(14, 5))
plt.subplot(121)
imcontourf(x, y, n_a)
plt.colorbar()
plt.subplot(122)
imcontourf(x, y, n_b)
plt.colorbar()


b2._d[1024:1050]

# ## Visualize single 2D wavefunction

Nx = 64
L = 16
dim = 2
dx = L/Nx
bcs = BCS(Nxyz=(Nx,)*dim, Lxyz=(L,)*dim)
x = bcs.xyz
V=sum(np.array(x)**2/2.0).ravel()
K = bcs._get_K()
H = K + np.diag(V)
Es, psis = np.linalg.eigh(H)
psis = psis.T
Es[:10]

x, y = bcs.xyz
plt.figure(figsize=(7, 5))
psi = psis[1].reshape((bcs.Nxyz))
n0 = abs(psi)**2
imcontourf(x, y, n0.real)
plt.colorbar()


# ## 2D harmonic in DVR basis

class DVR2D(VortexDVR):
    """a 2D DVR with harmonic potential class"""
    def get_Vext(self, rs):
        return rs**2/2


# +
mu=10
dmu=0
delta=0
dvr = DVR2D(mu=mu, dmu=dmu, delta=delta)
delta = delta + dvr.bases[0].zero

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
# -


