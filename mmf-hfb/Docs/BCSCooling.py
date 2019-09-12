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
import numpy as np

# ## Some Helper Functions

# +
from mmf_hfb.BCSCooling import BCSCooling

def H_exp(H, psi):
    return H.dot(psi).dot(psi.conj()).real

def Normalize(psi):
    return psi/psi.dot(psi.conj())**0.5

def Prob(psi):
    return np.abs(psi)**2


# -

# ## Analytical vs Numerical

Nx = 128
L = 23.0
dx = L/Nx
bcs = BCSCooling(N=Nx, L=None, dx=dx, beta_0=1j, beta_K=0, beta_V=0)
np.random.seed(1)
psi_ = np.exp(-bcs.xyz[0]**2/2)/np.pi**0.25

H0 = bcs._get_H(mu_eff=0, V=0)  # free particle
x = bcs.xyz[0]
V = x**2/2
H1 = bcs._get_H(mu_eff=0, V=V)  # harmonic trap
U0, Es0 = bcs.get_U_E(H0, transpose=True)
U1, Es1 = bcs.get_U_E(H1, transpose=True)

# ## Check relation of $V_c(x)$, $K_c(k)$ with $H_c$
# * By defination, $V_c$ should be equal to the diagonal terms of $H_c$ in position space while $K_c$ in momentum space

np.random.seed(2)
psi = [np.random.random(np.prod(bcs.Nxyz)) - 0.5]
Vc = bcs.get_Vc(psi, V=0)
Kc = bcs.get_Kc(psi, V=0)
Hc = bcs.get_Hc(psi, V=0)
Hc_k = np.fft.ifft(np.fft.fft(Hc, axis=0), axis=1)
np.allclose(np.diag(Hc_k).real - Kc, 0), np.allclose(np.diag(Hc) - Vc, 0)

# ## Check Derivatives
# * As derivatitves will be used, we need to make sure the numerical method works by comparing its results to analytical ones

y = np.cos(x)**2
plt.subplot(211)
plt.plot(x, y)
dy = bcs.Del(y, n=1)
plt.plot(x, dy)
plt.plot(x, -np.sin(2*x), '+')
plt.subplot(212)
dy = bcs.Del(y, n=2)
plt.plot(x, dy)
plt.plot(x, -2*np.cos(2*x), '+')


# ## Evolve with Imaginary Time

def ImaginaryCooling():
    plt.figure(figsize(16, 8))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for Nx in [64, 128, 256]:
        s = BCSCooling(N=Nx, dx=dx,  beta_0=-1j, beta_K=1, beta_V=1)
        s.g = 0# -1
        x = s.xyz[0]
        r2 = x**2
        V = x**2/2
        u0 = np.exp(-x**2/2)/np.pi**4
        u0 = u0/u0.dot(u0.conj())**0.5
        u1=(np.sqrt(2)*x*np.exp(-x**2/2))/np.pi**4
        u1 = u1/u1.dot(u1.conj())**0.5

        psi_0 = Normalize(V*0 + 1) # np.exp(-r2/2.0)*np.exp(1j*s.xyz[0])
        ts, psis = s.solve([psi_0], T=10, rtol=1e-5, atol=1e-6, V=V, method='BDF')
        psi0 = psis[0][-1]
        E0, N0 = s.get_E_Ns([psi0], V=V)
        Es = [s.get_E_Ns([_psi], V=V)[0] for _psi in psis[0]]
        line, = ax1.semilogy(ts[0][:-2], (Es[:-2] - E0)/abs(E0), label=f"Nx={Nx}")
        plt.sca(ax2)
        plt.plot(x, psi0)  # ground state
        plt.plot(x, psi_0, '--')  # initial state
        plt.plot(x, u0, '+')  # desired ground state
        E, N = s.get_E_Ns([V], V=V)
        plt.title(f"E={E:.4f}, N={N:.4f}")

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('abs((E-E0)/E0)')
    plt.sca(ax2)
    plt.xlim(-5,5)
    clear_output()
    plt.show()


ImaginaryCooling()


# ## Demostrate the $V_c$ and $K_c$ are Independent of Box Size
# * with fixed $dx$

# $$
# \hat{R}=\sum_n \ket{\psi_n}\bra{\psi_n}\qquad
# \hat{V}_c(x)=\int dx V_c(x) \ket{x}\bra{x} \qquad\\
# N=\braket{\psi|\psi}=\int dx\psi(x)^*\psi(x)\qquad
# V_c(x) =\braket{x|H_c|x}
# $$

def Check_Vc():
    for Nx in [64, 128, 256]:
        offset = np.log(Nx)*0.1
        s = BCSCooling(N=Nx, dx=dx,  beta_0=-1j, beta_K=1, beta_V=1)
        s.g = -1
        x = s.xyz[0]
        V_ext = x**2/2
        psi0 = np.exp(-x**2/2.0)*np.exp(1j*x)
    #     H1 = s._get_H(mu_eff=0, V=V)  # harmonic trap
    #     U1, _ = bcs.get_U_E(H1, transpose=True)
    #     psi0 = U1[0] 
        plt.subplot(121)
        plt.plot(x, Prob(psi0) + offset)
        plt.subplot(122)
        Vc = s.get_Vc(s.apply_H([psi0], V=V_ext), V=V_ext) 
        l, = plt.plot(x, Vc + offset)  # add some offset in y direction to seperate plots
    plt.subplot(121)
    plt.xlim(-10, 10)
    plt.subplot(122)
    plt.xlim(-10,10)
    plt.xlabel("x"); plt.ylabel(f"$V_c$");
    clear_output()


Check_Vc()

# ## Split-operator method

from IPython.display import display, clear_output
from IPython.core.debugger import set_trace


def PlayCooling(psis0, psis, N_data=10, N_step=100, **kw):
    b = BCSCooling(N=Nx, L=None, dx=dx, **kw)
    E0, N0 = bcs.get_E_Ns(psis0, V=V)
    Es, cs, steps = [], [], list(range(N_data))
    for _n in range(N_data):
        psis = b.step(psis, V=V, n=N_step)
        E, N = b.get_E_Ns(psis, V=V)
        Es.append(abs(E - E0)/E0)
        for psi in psis:
            ax, = plt.plot(x, abs(psi)**2)
            cs.append(ax.get_c())
        for i, psi in enumerate(psis0):
            plt.plot(x, abs(psi)**2,'+', c=cs[i])
        #for i, psi in enumerate(psis):
        #    dpsi = bcs.Del(psi, n=1)
        #   plt.plot(x, abs(dpsi)**2,'--', c=cs[i])
        plt.title(f"E0={E0},E={E}, $" + r"\beta_0$" +f"={b.beta_0}, "+r"$\beta_V$"+f"={b.beta_V}, "+r" $\beta_K$" +f"={b.beta_K}")
        plt.show()
        clear_output(wait=True)
    return psis


bcs = BCSCooling(N=Nx, L=None, dx=dx, beta_0=1, beta_V=1, beta_K=0, smooth=True) 
H = bcs._get_H(mu_eff=0, V=V)  # harmonic trap
Us, Es = bcs.get_U_E(H, transpose=True)
plt.plot(x, np.log10(abs(Us[0])))


def Cooling(beta_0=1, N=1, **args):
    H0 = bcs._get_H(mu_eff=0, V=0)  # free particle
    H1 = bcs._get_H(mu_eff=0, V=V)  # harmonic trap
    U0, Es0 = bcs.get_U_E(H0, transpose=True)
    U1, Es1 = bcs.get_U_E(H1, transpose=True)
    psis0 = U1[:N]
    psis = U0[:N]
    psis=PlayCooling(psis0=psis0, psis=psis, **args)


# ### Evolve with Original Harmitonian

Cooling(N_data=10, N_step=100, beta_V=0, beta_K=0,divs=(0, 0))

# ### With $V_c$ Only

Cooling(N_data=10, N_step=1000, beta_V=1, beta_K=0, divs=(0, 0))

# ### With $K_c$ only

Cooling(N_data=10, N_step=1000, beta_V=0, beta_K=2, divs=(0, 0))

# ### With $V_c$ and $K_c$

Cooling(N_data=10, N_step=1000,beta_V=1, beta_K=1, divs=(0, 0))

# ### With Derivatives

Cooling(N_data=30, N_step=1000, beta_V=0.01, beta_K=0, divs=(1, 1))

# ### Test code

# +
from mmf_hfb.BCSCooling import BCSCooling
import matplotlib.pyplot as plt

def get_V(x):
    return x**2/2

def PlayCooling(bcs, psis0, psis, V=None, N_data=10, N_step=100, **kw):
    x = bcs.xyz[0]
    if V is None:
        V = get_V(x)
    E0, _ = bcs.get_E_Ns(psis0, V=V)
    Es, cs= [], []
    for _n in range(N_data):
        psis = bcs.step(psis, V=V, n=N_step)
        # assert np.allclose(psis[0].dot(psis[1].conj()), 0)
        E, N = bcs.get_E_Ns(psis, V=V)
        Es.append(abs(E - E0)/E0)
        for psi in psis:
            ax, = plt.plot(x, abs(psi)**2)
            cs.append(ax.get_c())
        for i, psi in enumerate(psis0):
            plt.plot(x, abs(psi)**2, '+', c=cs[i])
        # for i, psi in enumerate(psis):
        #    dpsi = bcs.Del(psi, n=1)
        #   plt.plot(x, abs(dpsi)**2,'--', c=cs[i])
        plt.title(f"E0={E0},E={E}")
        plt.show()
        clear_output(wait=True)
    return psis


def CoolingEx(bcs, N=1, **args):
    V = get_V(bcs.xyz[0])
    H0 = bcs._get_H(mu_eff=0, V=0)  # free particle
    H1 = bcs._get_H(mu_eff=0, V=V)  # harmonic trap
    U0, _ = bcs.get_U_E(H0, transpose=True)
    U1, _ = bcs.get_U_E(H1, transpose=True)
    psis0 = U1[:N]
    psis = U0[:N]
    psis=PlayCooling(bcs=bcs, psis0=psis0, psis=psis, V=V, **args)


# -

Nx = 128
L = 23.0
dx = L/Nx
bcs = BCSCooling(N=Nx, L=None, dx=dx, beta_0=1, beta_V=0.2, beta_K=0, divs=(1, 1), smooth=True)
bcs.erase_max_ks()
CoolingEx(bcs=bcs, N_data=100, N_step=100)


