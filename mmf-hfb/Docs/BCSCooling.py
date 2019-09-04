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

from mmf_hfb.BCSCooling import BCSCooling
def H_exp(H, psi):
    return H.dot(psi).dot(psi.conj()).real
def Normalize(psi):
    return psi/psi.dot(psi.conj())**0.5


# ## Analytical vs Numerical

Nx = 128
bcs = BCSCooling(N=Nx, L=None, dx=0.1, beta_0=1j, beta_K=0, beta_V=0)
np.random.seed(1)
psi_ = np.exp(-bcs.xyz[0]**2/2)/np.pi**0.25

H0 = bcs._get_H(mu_eff=0, V=0)  # free particle
x = bcs.xyz[0]
V = x**2/2
H1 = bcs._get_H(mu_eff=0, V=V)  # harmonic trap
U0, Es0 = bcs.get_U_E(H0, transpose=True)
U1, Es1 = bcs.get_U_E(H1, transpose=True)

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

index = 0
psi1, psi2 = U0[index], U0[index + 1]
psi1_, psi2_  = U1[index], U1[index + 1]
bcs.get_E_Ns(U0[:2], V=0)[0], bcs.get_E_Ns(U0[:2], V=V)[0], bcs.get_E_Ns(U1[:2], V=0)[0],bcs.get_E_Ns(U1[:2], V=V)[0]

H_exp(H0, psi1), H_exp(H0, psi2), H_exp(H1, psi1_), H_exp(H1, psi2_)

Es0[:2], Es1[:2], bcs.get_E_Ns([psi2], V=V)[0]

# ## Evolve in Imaginary Time

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
for Nx in [128]:
    s = BCSCooling(N=Nx, dx=0.1,  beta_0=-1j, beta_K=1, beta_V=1)
    s.g = 0# -1
    x = s.xyz[0]
    r2 = x**2
    V = x**2/2
    psi_0 = Normalize(V*0 + 1) # np.exp(-r2/2.0)*np.exp(1j*s.xyz[0])
    ts, psis = s.solve([psi_0], T=10, rtol=1e-5, atol=1e-6, V=V, method='BDF')
    psi0 = psis[-1]
    E0, N0 = s.get_E_N(psi0, V=V)
    Es = [s.get_E_N(_psi, V=V)[0] for _psi in psis]
    line, = ax1.semilogy(ts[:-2], (Es[:-2] - E0)/abs(E0), label=f"Nx={Nx}")
    plt.sca(ax2)
    plt.plot(x, psi0)
    plt.plot(x,psi_0, '--')
    plt.plot(x, u0, '+')
    E, N = s.get_E_N(psi0, V=V)
    plt.title(f"E={E:.4f}, N={N:.4f}")
plt.sca(ax1)
plt.legend()
plt.xlabel('t')
plt.ylabel('abs((E-E0)/E0)')
plt.show()

# ## Split-operator method

from IPython.display import display, clear_output
eg = BCSCooling(N=Nx, dx=0.1, beta_0=1, beta_V=0.95, beta_K=0)
psi1, psi2 = U0[index], U0[index + 1]
psi1_, psi2_  = U1[index], U1[index + 1]
psis0 = [psi1_, psi2_]
E0, N0 = eg.get_E_Ns(psis0, V=V)
Es = []
psis = [psi1, psi2]
egs = [eg]
Ndata = 10
Nstep = 25
steps = list(range(Ndata))
step=0
for _n in range(Ndata):
    step = step + 1
    psis = eg.step(psis, V=V, n=Nstep)
    E, N = eg.get_E_Ns(psis, V=V)
    Es.append(abs(E - E0)/E0)
    ax, = plt.plot(x, abs(psis[0])**2)
    plt.plot(x, abs(psis[1])**2, c=ax.get_c())
    ax, = plt.plot(x, abs(psis0[0])**2, '--')
    plt.plot(x, abs(psis0[1])**2,'--', c=ax.get_c())
    plt.legend(['V+K', 'K', 'V'])
    plt.title(f"E0={E0},E={E}")
    plt.show()
    clear_output(wait=True)


plt.plot(steps, Es)
plt.xlabel("Step")
plt.ylabel("E-E0/E0")
plt.legend(['V+K', 'K', 'V'])
plt.show()

from IPython.core.debugger import set_trace

eg = BCSCooling(N=Nx, dx=0.1, beta_0=1, beta_V=1.95, beta_K=0)
N_psi = 2
psis0 = U1[:N_psi]
E0, N0 = eg.get_E_Ns(psis0, V=V)
Es = []
psis = U0[:N_psi]
egs = [eg]
Ndata = 50
Nstep = 500
steps = list(range(Ndata))
plt.figure(figsize(16,8))
cs = []
for _n in range(Ndata):
    psis = eg.step(psis, V=V, n=Nstep)
    print(psis[0].dot(psis[1].conj()))
    assert np.allclose(psis[0].dot(psis[1].conj()), 0)
    E, N = eg.get_E_Ns(psis, V=V)
    Es.append(abs(E - E0)/E0)
    for psi in psis:
        ax, = plt.plot(x, abs(psi)**2)
        cs.append(ax.get_c())
    for i, psi in enumerate(psis0):
        plt.plot(x, abs(psi)**2,'--', c=cs[i])
    plt.legend(['V+K', 'K', 'V'])
    plt.title(f"E0={E0},E={E}")
    plt.show()
    clear_output(wait=True)

print(psis[0].dot(psis[1].conj()))


