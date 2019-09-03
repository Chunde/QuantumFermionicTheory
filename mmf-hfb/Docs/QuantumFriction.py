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

from mmf_hfb.QuantumFriction import BCSCooling
def H_exp(H, psi):
    return H.dot(psi).dot(psi.conj()).real


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
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
for Nx in [128]:
    s = BCSCooling(N=Nx, dx=0.1,  beta_0=-1j, beta_K=1, beta_V=1)
    s.g = -1
    r2 = sum(_x**2 for _x in s.xyz)
    V = s.xyz[0]**2/2
    psi_0 = np.exp(-r2/2.0)*np.exp(1j*s.xyz[0])
    ts, psis = s.solve(psi_0, T=20, rtol=1e-5, atol=1e-6, V=V, method='BDF')
    psi0 = psis[-1]
    E0, N0 = s.get_E_N(psi0, V=V)
    Es = [s.get_E_N(_psi, V=V)[0] for _psi in psis]
    line, = ax1.semilogy(ts[:-2], (Es[:-2] - E0)/abs(E0), label=f"Nx={Nx}")
    plt.sca(ax2)
    s.plot(psi0, V=V, c=line.get_c(), alpha=0.5)
    plt.plot(s.xyz[0],psi_0, '--')

plt.sca(ax1)
plt.legend()
plt.xlabel('t')
plt.ylabel('abs((E-E0)/E0)')
plt.show()
# -

plt.plot(ts, Es)

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


