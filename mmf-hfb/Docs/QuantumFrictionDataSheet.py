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
import matplotlib.pyplot as plt
# %pylab inline --no-import-all
from nbimports import *
import numpy as np
from mmf_hfb.BCSCooling import BCSCooling
from mmf_hfb.Potentials import HarmonicOscillator
from mmf_hfb.SolverABM import ABMEvolverAdapter
from os.path import join
import inspect
import json
import glob
import os
from mmf_hfb.CoolingCaseTests import TestCase, Prob, Normalize, dict_to_complex, deserialize_object, load_json_data


# # Load Data from Files

def random_gaussian_mixing(x):
    n = np.random.randint(1, 10)
    cs = np.random.random(n)
    ns = np.random.randint(1, 10, size=n)    
    ys = sum([c*np.exp(-x**2/n**2) for (c, n) in zip(cs, ns)])
    return Normalize(ys)


res = load_json_data()
testCases = deserialize_object(res)

t0 = testCases[4]
t0.plot(0)


# # Visualization

# # Some Test Code

def test_cooling(plot_dE=True, use_ABM=False, T=0.5, plt_log=True, check_dE=False, **args):   
    b = BCSCooling(**args)
    solver = ABMEvolverAdapter if use_ABM else None
    h0 = HarmonicOscillator(w=1)
    h = HarmonicOscillator()
    da, db=b.divs    
    x = b.xyz[0]
    V = x**2/2
    H0 = b._get_H(mu_eff=0, V=0)
    H1 = b._get_H(mu_eff=0, V=V)
    U0, E0 = b.get_U_E(H0, transpose=True)
    U1, E1 = b.get_U_E(H1, transpose=True)
    #psi0 = h.get_wf(x)
    psi0 = U1[0]
    psi0 = Normalize(psi0, dx=b.dx)
    #psi = h0.get_wf(x, n=2)
    psi = random_gaussian_mixing(x) #U0[1]#
    psi = Normalize(psi, dx=b.dx)
    #b.erase_max_ks()
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    N0 = psi0.conj().dot(psi0)
    ts, psiss = b.solve([psi], T=T, rtol=1e-5, atol=1e-6, V=V, solver=solver, method='BDF')
    E0, _ = b.get_E_Ns([psi0], V=V)
    Es = [b.get_E_Ns([_psi], V=V)[0] for _psi in psiss[0]]   
    plt.plot(x, Prob(psiss[0][0]), "+", label='init')
    plt.plot(x, Prob(psiss[0][-1]), '--',label="final")
    plt.plot(x, Prob(psi0), label='Ground')
    plt.legend()
    plt.subplot(122)
    if plt_log:
        plt.semilogy(ts[0][:-2], (Es[:-2] - E0)/abs(E0), label="E")
    else:
        plt.plot(ts[0][:-2], (Es[:-2] - E0)/abs(E0), label="E")
    if plot_dE:
        dE_dt= [-1*b.get_dE_dt([_psi], V=V) for _psi in psiss[0]]
        plt.plot(ts[0][:-2], dE_dt[:-2], label='-dE/dt')
        plt.axhline(0, linestyle='dashed')
    plt.legend()
    plt.axhline(0, linestyle='dashed')
    plt.show()    
    return psiss[0][-1]


# %%time 
args = dict(N=128, dx=0.1, beta_0=1, beta_K=0, beta_V=30, beta_D=0, beta_Y=0, T=5, divs=(1, 1),plt_log=True, use_ABM=True)
psi = test_cooling(plot_dE=False, **args)


def Cooling(plot_dE=True, use_ABM=False, T=0.5, **args):
    b = BCSCooling(**args)
    solver = ABMEvolverAdapter if use_ABM else None
    h0 = HarmonicOscillator(w=1)
    h = HarmonicOscillator()
    da, db=b.divs    
    x = b.xyz[0]
    V =  x**2/2
    psi_0 = Normalize(np.sqrt(1.0 + 0.2*x*np.exp(-x**2)))
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    ts, psiss = b.solve([psi_0], T=T, rtol=1e-5, atol=1e-6, V=V,solver=solver, method='BDF')
    psi0 = h0.get_wf(x=x, n=0)
    N0 = psi0.conj().dot(psi0)
    Ns = [_psi.conj().dot(_psi) for _psi in psiss[0]]
    E0, _ = b.get_E_Ns([psi0], V=V)
    Es = [b.get_E_Ns([_psi], V=V)[0] for _psi in psiss[0]]
    dE_dt= [-1*b.get_dE_dt([_psi], V=V) for _psi in psiss[0]]
    plt.plot(x, Prob(psi0), 'o', label='Ground')
    plt.plot(x, Prob(psiss[0][0]), "+", label='init')
    plt.plot(x, Prob(psiss[0][-1]), '--',label="final")
    plt.legend()
    plt.subplot(1,2,2)
    plt.semilogy(ts[0][:-2], (Es[:-2] - E0)/abs(E0), label="E")
    if plot_dE:
        plt.plot(ts[0][:-2], dE_dt[:-2], label='-dE/dt')
        plt.axhline(0, linestyle='dashed')
    plt.legend()
    plt.axhline(0, linestyle='dashed')
    plt.show()    
    return psiss[0][-1]


# %%time 
args = dict(N=128, dx=0.1, beta_0=-1j, beta_K=0, beta_V=1, beta_D=0, beta_Y=0, T=5, divs=(1, 1), use_ABM=True)
psi = Cooling(plot_dE=False, **args)

# %%time 
args = dict(N=128, dx=0.1, beta_0=-1j, beta_K=0, beta_V=0, beta_D=0, beta_Y=0, T=5, divs=(1, 1), use_ABM=True, check_dE=False)
psi = Cooling(plot_dE=False, **args)

args = dict(N=4, g=1)
egs = [BCSCooling(beta_0=-1j, beta_V=0.0, beta_K=0.0, **args),
       BCSCooling(beta_0=0.0, beta_V=0.0, beta_K=1.0, **args),
       BCSCooling(beta_0=1.0, beta_V=0.0, beta_K=1.0, **args),      
       BCSCooling(beta_0=0.0, beta_V=1.0, beta_K=0.0, **args),
       BCSCooling(beta_0=1.0, beta_V=1.0, beta_K=0.0, **args),
       BCSCooling(beta_0=0.0, beta_V=1.0, beta_K=1.0, **args),
       BCSCooling(beta_0=1.0, beta_V=1.0, beta_K=1.0, **args)]
labels = ['Imaginary Time',
          'K', 'H+K',
          'V', 'H+V',
          'V+K', 'H+V+K']
eg = egs[0]
psi0 = 2*(np.random.random(eg.Nxyz[0]) + 1j*np.random.random(eg.Nxyz[0]) - 0.5 - 0.5j)
V = np.array(psi0)*0
x=egs[0].xyz[0]
#psi0 = 0*x + 1.5 + 1.5*np.exp(-x**2/2)
psi_ground = 0*psi0 + np.sqrt((abs(psi0)**2).mean())
E0, N0 = eg.get_E_Ns([psi_ground], V=V)
Es = [[] for _n in range(len(egs))]
psis = [psi0.copy() for _n in range(len(egs))]
t_max = 3.0
Nstep = 4
Ndata = int(np.round(t_max/eg.dt/Nstep))
ts = np.arange(Ndata)*Nstep*eg.dt
for _n in range(Ndata):
    for n, eg in enumerate(egs):
        ps = [psis[n]]
        ps = eg.step(psis=ps, n=Nstep, V=V)
        psis[n] = ps[0]
        E, N = eg.get_E_Ns(psis=ps, V=V) 
        Es[n].append(E/E0 - 1.0)
Es = np.asarray(Es)

plt.semilogy(ts, Es[1], c='C0', ls=':', label=labels[1])
plt.semilogy(ts, Es[2], c='C0', ls='-', label=labels[2])
plt.semilogy(ts, Es[3], c='C1', ls=':', label=labels[3])
plt.semilogy(ts, Es[4], c='C1', ls='-', label=labels[4])
plt.semilogy(ts, Es[5], c='C2', ls=':', label=labels[5])
plt.semilogy(ts, Es[6], c='C2', ls='-', label=labels[6])
plt.semilogy(ts, Es[0], c='k', ls='-', label=labels[0], scaley=False)
plt.xlabel("t")
plt.ylabel("E-E0")
plt.legend()


