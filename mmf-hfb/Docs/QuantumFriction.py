# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import *
import matplotlib.pyplot as plt

# # BCS Cooling Class

# +
from mmf_hfb.bcs import BCS
from scipy.integrate import solve_ivp
import numpy as np
import scipy as sp


def Assert(a, b, rtol=1e-10):
    assert np.allclose(a, b, rtol=rtol)

    
def assert_orth(psis):
    y1, y2 = psis
    inner_prod = y1.dot(y2.conj())
    ret = np.allclose(inner_prod, 0, rtol=1e-16)
    if not ret:
        print(inner_prod)
    assert ret

    
def H_exp(H, psi):
    return H.dot(psi).dot(psi.conj()).real

    
class BCSCooling(BCS):
    """
    1d Local Quamtum Friction class
    """

    def __init__(self, N=256, L=32, beta_0=1.0, beta_V=1.0, beta_K=1.0, dt_Emax=1.0):
        """
        Arguments
        ---------
        beta_0 : float
           Portion of the original Hamiltonian H to include.
        beta_V : float
           Portion of the position cooling potential V_c.
        beta_K : float
           Portion of the momentum cooling potential K_c.
        """
        BCS.__init__(self, Nxyz=(N,), Lxyz=(L,))
        self.beta_0 = beta_0
        self.beta_V = beta_V
        self.beta_K = beta_K
        self._K2 = (self.hbar*np.array(self.kxyz[0]))**2/2/self.m
        Emax = self._K2.max()
        self.dt = dt_Emax*self.hbar/Emax      
        self.g = 0
    
    def get_V(self, psi, V):
        return self.g*abs(psi)**2 + V
    
    def apply_H(self, psi, V):
        """compute dy/dt"""
        V = self.get_V(psi, V=V)
        psi_k = np.fft.fft(psi)
        Kpsi = self._K2*psi_k
        Vpsi = V*psi
        return Kpsi + Vpsi

    def get_Vc(self, psi, V):
        N = psi.dot(psi.conj())*self.dV
        Hpsi = self.apply_H(psi, V=V)
        Vc = 2*(psi.conj()*Hpsi).imag*self.dV/N
        return Vc

    def get_Kc(self, psi, V):
        N = psi.dot(psi.conj())*self.dV
        psi_k = np.fft.fft(psi)*self.dV
        Hpsi = self.apply_H(psi, V=V)
        Vpsi_k = np.fft.fft(Hpsi)*self.dV
        Kc = 2*(psi_k.conj()*Vpsi_k).imag/N /self.dV
        return Kc
 
    def apply_expK(self, psi, V, factor=1):
        len0 = psi.dot(psi.conj())
        psi_k = np.fft.fft(psi)
        Kc = self.beta_K*self.get_Kc(psi=psi, V=V)
        psi = np.fft.ifft(np.exp(-1j*self.dt*factor*(self.beta_0*self._K2 + Kc))*psi_k)
        len1 = psi.dot(psi.conj())
        Assert(len0, len1)
        return psi
        
    def apply_expV(self, psi, V, factor=1):
        len0 = psi.dot(psi.conj())
        Vc = self.beta_V*self.get_Vc(psi, V=V)
        psi = np.exp(-1j*self.dt*factor*(self.beta_0*V + Vc))*psi
        len1 = psi.dot(psi.conj())
        Assert(len0, len1)
        return psi
    
    def apply_Hc(self, psi, V):
        """
        Apply the cooling Hamiltonian.
        or, compute dy/dt w.r.t to Hc
        """
        H_psi = self.apply_H(psi, V=V)
        Vc_psi = self.get_Vc(psi, V=V)*psi
        Kc_psi = self.ifft(self.get_Kc(psi, V=V)*self.fft(psi))
        return (self.beta_0*H_psi + self.beta_V*Vc_psi + self.beta_K*Kc_psi)
    
    def step(self, psi, V, n=1):
        """
        Evolve the state psi by applying n steps of the
        Split-Operator method.
        """
        psi = self.apply_expK(psi=psi, V=V, factor=0.5)
        for n in range(n):
            psi = self.apply_expV(psi=psi, V=V)
            psi = self.apply_expK(psi=psi, V=V)
        psi = self.apply_expK(psi=psi, V=V, factor=-0.5)
        return psi
       
    def compute_dy_dt(self, t, psi, subtract_mu=True):
        """Return dy/dt for ODE integration."""
        Hpsi = self.apply_Hc(psi, V=self.V)
        if subtract_mu:
            Hpsi -= psi.dot(Hpsi.conj())/psi.dot(psi.conj())*psi
        return Hpsi/(1j*self.hbar)
    
    def get_U_E(self, H, transpose=False):
        """return Us and Vs and energy"""
        Es, U = sp.linalg.eigh(H)
        if transpose:
            return (U.T, Es)
        return (U, Es)

    def evolve(self, Us, V, n=1):
        """Evolve all stats"""  
        us = []
        for i in range(len(Us)):
            us.append(self.step(psi=Us[i], V=V, n=n))
        return us
    
    def solve(self, psi0, T, V, **kw):
        self.V = V
        res = solve_ivp(fun=self.compute_dy_dt, t_span=(0, T), y0=psi0, **kw)
        if not res.success:
            raise Exception(res.message)
        return(res.t, list(res.y.T))
            
    def get_E_N(self, psi, V):
        """Return the energy and particle number `(E,N)`."""
        K = psi.dot(self.ifft(self._K2*self.fft(psi)))
        n = abs(psi)**2
        E = (K + sum(V*n)).real*self.dV
        N = sum(n)*self.dV
        return E, N
    
    def get_E_Ns(self, psis, V):
        E = 0
        N = 0
        for psi in psis:
            E_, N_ = self.get_E_N(psi=psi, V=V)
            E = E + E_
            N = N + N_
        return E, N


# -

# ## Analytical vs Numerical

Nx = 256
Lx = 4
transpose=True
bcs = BCSCooling(N=Nx, L=Lx, beta_0=1j, beta_K=0, beta_V=0)
np.random.seed(1)
psi_ = np.exp(-bcs.xyz[0]**2/2)/np.pi**0.25

H0 = bcs._get_H(mu_eff=0, V=0)  # free particle
x = bcs.xyz[0]
V = x**2/2
H1 = bcs._get_H(mu_eff=0, V=V)  # harmonic trap
U0, Es0 = bcs.get_U_E(H0, transpose=transpose)
U1, Es1 = bcs.get_U_E(H1, transpose=transpose)

index = 0
psi1, psi2 = U0[index], U0[index + 1]
psi1_, psi2_ = U1[index], U1[index + 1]

u0 = np.exp(-x**2/2)/np.pi**4
u0 = u0/u0.dot(u0.conj())**0.5
u1=(np.sqrt(2)*x*np.exp(-x**2/2))/np.pi**4
u1 = u1/u1.dot(u1.conj())**0.5
ax1, = plt.plot(x, u0, '--')
ax2, = plt.plot(x, u1, '--')
plt.plot(x, U1[index], c=ax1.get_c())
plt.plot(x, U1[index+1], c=ax2.get_c())

ax1, = plt.plot(psi1)
plt.plot(psi2, c=ax1.get_c())
ax2, = plt.plot(psi1_, '--')
plt.plot(psi2_, '--', c=ax2.get_c())
plt.show()

# ## Check Energy

eg = eg_VK = BCSCooling(N=Nx, L=Lx)
eg_K = BCSCooling(N=Nx, L=Lx, beta_0=1, beta_V=0.0, beta_K=1.0)
eg_V = BCSCooling(N=Nx, L=Lx, beta_0=1, beta_V = 1, beta_K=0.0)

index = 0
psi1, psi2 = U0[index], U0[index + 1]
psi1_, psi2_  = U1[index], U1[index + 1]
bcs.get_E_Ns(U0[:2], V=0)[0], bcs.get_E_Ns(U0[:2], V=V)[0], bcs.get_E_Ns(U1[:2], V=0)[0],bcs.get_E_Ns(U1[:2], V=V)[0]

H_exp(H0, psi1), H_exp(H0, psi2), H_exp(H1, psi1_), H_exp(H1, psi2_)

Es0[:2], Es1[:2], bcs.get_E_N(psi2, V=V)[0]

# ## Evolve in Time

# +
plt.figure(figsize=(10,5))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
for Nx in [128, 256, 512]:
    s = BCSCooling(N=Nx, L=Lx, beta_0=-1j, beta_K=0, beta_V=0)
    s.g = -1
    r2 = sum(_x**2 for _x in s.xyz)
    V = s.xyz[0]**2/2
    psi_0 = np.exp(-r2/2.0)*np.exp(1j*s.xyz[0])
    ts, psis = s.solve(psi_0, T=20, rtol=1e-5, atol=1e-6, V=V, method='BDF')
    psi0 = psis[-1]
    E0, N0 = s.get_E_N(psi0)
    Es = [s.get_E_N(_psi)[0] for _psi in psis]
    line, = ax1.semilogy(ts[:-2], (Es[:-2] - E0)/abs(E0), label=f"Nx={Nx}")
    plt.sca(ax2)
    s.plot(psi0, c=line.get_c(), alpha=0.5)

plt.sca(ax1)
plt.legend()
plt.xlabel('t')
plt.ylabel('abs((E-E0)/E0)');
# -

bcs.solve(psi0=u1, T=20, V=V)

from IPython.display import display, clear_output
psi1, psi2 = U0[index], U0[index + 1]
psi1_, psi2_  = U1[index], U1[index + 1]
psis0 = [psi1_, psi2_]
E0, N0 = eg.get_E_Ns(psis0, V=V)
Es = [[], [], []]
psi2_ = [psi1, psi2]
psis = [psi2_, psi2_, psi2_]
egs = [eg_K]
Ndata = 50
Nstep = 100
steps = list(range(Ndata))
step=0
for _n in range(Ndata):
    for n, eg in enumerate(egs):
        step = step + 1
        psis[n] = eg.evolve(psis[n], V=V, n=Nstep)
        E, N = eg.get_E_Ns(psis[n], V=V)
        Es[n].append(abs(E - E0)/E0)
    for n, eg in enumerate(egs):
        ax, = plt.plot(x, psis[n][0])
        plt.plot(x, psis[n][1], c=ax.get_c())
    ax, = plt.plot(x, psis0[0], '--')
    plt.plot(x, psis0[1], c=ax.get_c())
    plt.legend(['V+K', 'K', 'V'])
    plt.title(f"E0={E0},E={E}")
    plt.show()
    clear_output(wait=True)


for n, eg in enumerate(egs):
    plt.plot(steps, Es[n])
plt.xlabel("Step")
plt.ylabel("E-E0/E0")
plt.legend(['V+K', 'K', 'V'])
plt.show()


