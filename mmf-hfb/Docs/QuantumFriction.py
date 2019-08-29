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
import numpy as np

def Assert(a, b, rtol=1e-10):
    assert np.allclose(a, b, rtol=rtol)
    
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
        
        
    def apply_expK(self, psi, V, factor=1):
        len0 = psi.dot(psi.conj())
        psi_k = np.fft.fft(psi)
        Kc = self.get_Kc(psi=psi, V=V)
        psi = np.fft.ifft(
            np.exp(-1j*self.dt*factor*(self.beta_0*self._K2 + Kc))*psi_k)
        len1 = psi.dot(psi.conj())
        Assert(len0, len1)
        return psi
        
    def apply_expV(self, psi, V, factor=1):
        len0 = psi.dot(psi.conj())
        Vc = self.get_Vc(psi)
        psi = np.exp(-1j*self.dt*factor*(self.beta_0*V + Vc)) * psi
        len1 = psi.dot(psi.conj())
        Assert(len0, len1)
        return psi

    def apply_H(self, psi, V):
        psi_k = np.fft.fft(psi)
        Kpsi = self._K2*psi_k
        Vpsi = V*psi
        return Kpsi + Vpsi

    def get_Vc(self, psi):
        psi2=(abs(psi)**2).max()
        if psi2 == 0:
            return psi
        psi_k = np.fft.fft(psi)
        Kpsi = np.fft.ifft(self._K2*psi_k)
        Vc = 2*(psi.conj()*Kpsi).imag/psi2
        return self.beta_V*Vc

    def get_Kc(self, psi, V):
        psi2=(abs(psi)**2).max()
        if psi2 == 0:
            return psi
        psi_k = np.fft.fft(psi)
        Vpsi_k = np.fft.fft(V*psi)
        Kc = 2*(psi_k.conj()*Vpsi_k).imag/psi2
        return self.beta_K*Kc

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

    def get_UV_E(self, H, UV=None, transpose=True):
        """return U and V"""
        if UV is None:
            Es, UV = np.linalg.eigh(H)
        U_V_shape = (2,) + tuple(self.Nxyz) + UV.shape[1:]
        U, V = UV.reshape(U_V_shape)
        if transpose:
            return (U.T, V.T, Es)
        return (U, V, Es)

    def evolve(self, n=1):
        self.V0 = 1
        Vs = self.get_v_ext()
        Us = self.UV0
        for U, V in zip(Us, Vs):
            for i in range(len(U)):
                U[i] = self.step(psi=U[i], V=V, n=n)

   
    def get_E_N(self, psi):
        """Return the energy and particle number `(E,N)`."""
        dx = self.dV
        n = abs(psi)**2
        K = abs(np.fft.ifft(self.hbar*self.kxyz[0]*np.fft.fft(psi)))**2/2/self.m
        E = (K.sum()*dx).real
        N = n.sum()*dx
        return E, N

# -

# ## Test

Nx = 64
Lx = 4
bcs = BCSCooling(N=Nx, L=Lx)

# ### Free Fermionic Gas

H0 = bcs.get_H(mus_eff=(0, 0), delta=0)

# ### Add some noise as pertubation

np.random.seed(1)
E0 = 0.1*(np.pi/bcs.Lxyz[0])**2
V = E0*np.random.random(bcs.Nxyz[0])
H1 = bcs.get_H(mus_eff=(0, 0), delta=0, Vs=(V,V))

U0, V0, Es0 = bcs.get_UV_E(H0)

U1, V1, Es0 = bcs.get_UV_E(H1)

index = 50
psi0 = V0[index]
psi = V1[index]
plt.plot(psi0)
plt.plot(psi)

eg = eg_VK = BCSCooling(N=Nx, L=Lx)
eg_K = BCSCooling(N=Nx, L=Lx, beta_0=1, beta_V=0.0)
eg_V = BCSCooling(N=Nx, L=Lx, beta_0=1, beta_K=0.0)
x = eg.xyz[0]

# +
from IPython.display import display, clear_output

E0, N0 = eg.get_E_N(psi0)
Es = [[], [], []]
psis = [psi, psi, psi]
egs = [eg_K]
Ndata = 100
Nstep = 300
steps = list(range(100))
for _n in range(Ndata):
    for n, eg in enumerate(egs):
        psis[n] = eg.step(psis[n], V=V, n=Nstep)
        E, N = eg.get_E_N(psis[n])
        Es[n].append(E - E0)
    for n, eg in enumerate(egs):
        plt.plot(x, psis[n])
    plt.plot(x, psi0, '--')
    plt.legend(['V+K', 'K', 'V'])
    plt.show()
    clear_output(wait=True)


# -

for n, eg in enumerate(egs):
    plt.plot(steps, Es[n])
plt.xlabel("Step")
plt.ylabel("E-E0")
plt.legend(['V+K', 'K', 'V'])
plt.show()


