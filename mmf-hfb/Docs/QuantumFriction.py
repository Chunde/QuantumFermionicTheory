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

from mmf_hfb.QuantumFriction import BCSCooling

N=256
L=30
eg = eg_VK = BCSCooling(N=N, L=L)
eg_K = BCSCooling(N=N, L=L, beta_0=1, beta_V=0.0)
eg_V = BCSCooling(N=N, L=L, beta_0=1, beta_K=0.0)
x = eg.xyz[0]

psi, psi0, V = eg.get_configuration(id=256)
# psi = np.random.random(eg.N) + 1j*np.random.random(eg.N) - 0.5 - 0.5j
psi = 0*eg.xyz[0] + 1 + 1.0*np.exp(-eg.xyz[0]**2/2)
#psi_ground = 0*psi0 + np.sqrt((abs(psi0)**2).mean())
#plt.plot(x, psi)
plt.plot(x, psi0, '--')
plt.show()

# +
from IPython.display import display, clear_output

E0, N0 = eg.get_E_N(psi0, V=V)
Es = [[], [], []]
psis = [psi, psi, psi]
egs = [eg_VK, eg_K, eg_V]
Ndata = 100
Nstep = 300
steps = list(range(100))
for _n in range(Ndata):
    for n, eg in enumerate(egs):
        psis[n] = eg.step(psis[n], V=V, n=Nstep)
        E, N = eg.get_E_N(psis[n], V=V)
        Es[n].append(E - E0)
    for n, eg in enumerate(egs):
        plt.plot(x, psis[n][0])
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


