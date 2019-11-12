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
from mmf_hfb.CoolingCaseTests import TestCase, Prob, Normalize, dict_to_complex, deserialize_object, load_json_data, random_gaussian_mixing


# +
def get_init_states(N=128, dx=0.1):
    b = BCSCooling(N=N, dx=dx)
    h = HarmonicOscillator()
    x = b.xyz[0]
    V = x**2/2
    H0 = b._get_H(mu_eff=0, V=0)
    H1 = b._get_H(mu_eff=0, V=V)
    U0, E0 = b.get_U_E(H0, transpose=True)
    U1, E1 = b.get_U_E(H1, transpose=True)
    psi_random = Normalize(np.random.random(N), dx=dx)
    psi_standing_wave=Normalize(U0[1],dx=dx)
    psi_gaussian_mixing = random_gaussian_mixing(x, dx=dx)
    psi_uniform = Normalize(U0[0], dx=dx)
    psi_interaction = Normalize(np.exp(-x**2/2.0)*np.exp(1j*x), dx=dx)
    return dict(RN=psi_random, ST=psi_standing_wave, GM=psi_gaussian_mixing, UN=psi_uniform, IN=psi_interaction)

def get_potentials(x):
    V0 = 0*x
    V_HO = x**2/2
    V_PO = V0 + np.random.random()*V_HO + abs(x**2)*np.random.random()
    return dict(V0=V0, HO=V_HO, PO=V_PO)


# -

psis_init = get_init_states()
# for key in psis_init:
#     plt.plot(Prob(psis_init[key]), label=key)
#     plt.legend()

# # Load Data from Files

res = load_json_data()
testCases = deserialize_object(res)

t0 = testCases[4]


# # Visualization

# +
def beta_filter(t, expr=None):
    t = t.b
    beta_0, beta_V, beta_K, beta_D, beta_Y =t.beta_0, t.beta_V, t.beta_K, t.beta_D, t.beta_Y 
    if expr is None:
        if t.beta_V >0 and t.beta_K==0 and t.beta_D==0 and t.beta_Y==0:
            return True
        return False
    else:
        return eval(expr)

def V_filter(t, expr=None):
    V = t.V_key
    if expr is None:
        if V == 'HO':
            return True
        return True
    else:
        return eval(expr)

def t_filter(t, expr=None):
    if expr is None:
        return True
    else:
        return eval(expr)

def filter(ts,t_expr=None, V_expr=None, beta_expr=None):
    filtered_ts = []
    for t in ts:
        if not beta_filter(t, expr=beta_expr):
            continue
        if not V_filter(t, expr=V_expr):
            continue
        if not t_filter(t, expr=t_expr):
            continue
            
        filtered_ts.append(t)
    return filtered_ts


# -

ts = filter(testCases, t_expr='t.g==1', V_expr='V=="PO"', beta_expr="beta_V>0 and beta_K==0 and beta_D==0 and beta_Y==0")
len(ts)


def plotCase(t):
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    t.plot(0)
    plt.subplot(122)
    E0, Es, Ts, Tws = t0.E0, t.Es, t0.physical_time, t0.wall_time
    Es = np.array(Es) - E0
    Es = Es/E0
    plt.semilogy(Tws, Es, '--')
    #plt.plot(Tws, Es, 'o')


# +
# # %pylab inline --no-import-all
# from gpe.imports import *
# import gpe.exact_solutions;reload(gpe.exact_solutions)
# from gpe.exact_solutions import BrightSoliton
# Nx = 128
# Lx = 10.0
# v = 2.0
# args = dict(Nx=Nx, Lx=Lx, v=v, sigma=1)
# s0 = BrightSoliton(v_x=0, **args)
# plt.plot(s0.x, s0.psi0)
# -

# ## Interaction

# +
# b = BCSCooling(N=128, dx=0.1, beta_0=-1j, g=-1)
# h0 = HarmonicOscillator(w=1)
# h = HarmonicOscillator()
# da, db=b.divs    
# x = b.xyz[0]
# V = x*0
# psi_0 = psis_init['IN'] #np.exp(-x**2/2.0)*np.exp(1j*x)
# plt.figure(figsize=(18,5))
# plt.subplot(131)
# plt.plot(x, V)
# plt.subplot(132)
# ts, psiss = b.solve([psi_0], T=30, rtol=1e-5, atol=1e-6, V=V, method='BDF')
# psi0 = psiss[0][-1]
# E0, _ = b.get_E_Ns([psi0], V=V)
# Es = [b.get_E_Ns([_psi], V=V)[0] for _psi in psiss[0]]   
# plt.plot(x, Normalize(psiss[0][0]), "--", label='init')
# plt.plot(x, Normalize(psiss[0][-1]), '-',label="final")
# plt.legend()
# plt.subplot(133)
# plt.semilogy(ts[0][:-1], (Es[:-1] - E0)/abs(E0), label="E")
# plt.legend()
# plt.show()
# -

# # Some Test Code

N=128
dx=0.2
b = BCSCooling(N=N, dx=dx)
h = HarmonicOscillator()
x = b.xyz[0]
psi_init= psis_init['IN']
Vs = get_potentials(x)
V=Vs['HO']
args = dict(N=N, dx=dx, eps=1e-1, T_ground_state=20, V=V, V_key='0', g=-1, psi_init=psi_init, use_abm=False, check_dE=False)
t=TestCase(ground_state_eps=1e-1, beta_0=-1j, **args)

plt.plot(x, Normalize(psi_init), "--", label='init')
plt.plot(x, Normalize(t.psi_ground), '-',label="final")

plt.figure(figsize=(18,5))
t.b.beta_V= 50
t.b.beta_K = 100
t.run(T=5, plot=True, plot_log=False)

# ## Batch Data

t.b.beta_V = 10
plt.figure(figsize=(18,5))
for beta_K in np.linspace(10, 100, 10):
    print("--------------------------------------")
    print(f"beta_K={beta_K}")
    t.b.beta_K= beta_K
    t.run(T=5, plot=False, plot_log=False)

plt.figure(figsize=(18,5))
t.b.beta_K = 0
for beta_V in np.linspace(10, 100, 10):
    print("--------------------------------------")
    print(f"beta_V={beta_V}")
    t.b.beta_V= beta_V
    t.run(T=5, plot=False, plot_log=False)


# # Cooling

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


def test_cooling(use_ABM=False, psi_key="ST", T=0.5,plot_dE=True,  plt_log=True, **args):   
    b = BCSCooling(**args)
    solver = ABMEvolverAdapter if use_ABM else None
    h0 = HarmonicOscillator(w=1)
    h = HarmonicOscillator()
    da, db=b.divs    
    x = b.xyz[0]
    V = x*0
    H0 = b._get_H(mu_eff=0, V=0)
    H1 = b._get_H(mu_eff=0, V=V)
    U0, E0 = b.get_U_E(H0, transpose=True)
    U1, E1 = b.get_U_E(H1, transpose=True)
    #psi0 = h.get_wf(x)
    psi0 = U1[0]
    psi0 = Normalize(psi0, dx=b.dx)
    # psi = h0.get_wf(x, n=2)
    psi = U0[1]# random_gaussian_mixing(x) #
    psi = Normalize(psi, dx=b.dx)
    #b.erase_max_ks()
    plt.figure(figsize=(18,5))
    plt.subplot(131)
    plt.plot(x, V)
    plt.subplot(132)
    N0 = psi0.conj().dot(psi0)
    ts, psiss = b.solve([psi], T=T, rtol=1e-5, atol=1e-6, V=V, solver=solver, method='BDF')
    psis = psiss[0]
    E0, _ = b.get_E_Ns([psi0], V=V)
    Es = [b.get_E_Ns([_psi], V=V)[0] for _psi in psiss[0]]   
    plt.plot(x, Prob(psiss[0][0]), "+", label='init')
    plt.plot(x, Prob(psiss[0][-1]), '--',label="final")
    plt.plot(x, Prob(psi0), label='Ground')
    plt.legend()
    plt.subplot(133)
    if plt_log:
        plt.semilogy(ts[0][:-1], (Es[:-1] - E0)/abs(E0), label="E")
    else:
        plt.plot(ts[0][:-1], (Es[:-1] - E0)/abs(E0), label="E")
    if plot_dE:
        dE_dt= [-1*b.get_dE_dt([_psi], V=V) for _psi in psiss[0]]
        plt.plot(ts[0][:-1], dE_dt[:-1], label='-dE/dt')
        plt.axhline(0, linestyle='dashed')
    plt.legend()
    plt.axhline(0, linestyle='dashed')
    plt.show()
    print((Es[-1] - E0)/abs(E0), (Es[0] - E0)/abs(E0))
    return psiss[0][-1]


args = dict(N=128, dx=0.1, beta_0=-1j, g=1, plt_log=False, check_dE=False, use_ABM=False)
psi = test_cooling(plot_dE=False, T=15, **args)

# +
from IPython.display import clear_output
from importlib import reload
import mmf_hfb.quantum_friction as quantum_friction;reload(quantum_friction)
from mmf_hfb.quantum_friction import StateBase
# plt.figure(figsize=(10,5))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
for Nx in [256]:
    s = StateBase(Nxyz=(Nx,), beta_0=-1j)
    s.g = -1
    r2 = sum(_x**2 for _x in s.xyz)
    psi_0 = Normalize(s.zero + np.exp(-r2/2.0)*np.exp(1j*s.xyz[0]))
    ts, psis = s.solve(psi_0, T=50, rtol=1e-5, atol=1e-6, method='BDF')
    psi0 = psis[-1]
    E0, N0 = s.get_E_N(psi0)
    Es = [s.get_E_N(_psi)[0] for _psi in psis]
    line, = ax1.semilogy(ts[:-2], (Es[:-2] - E0)/abs(E0), label=f"Nx={Nx}")
    plt.sca(ax2)
    s.plot(psi0, c=line.get_c(), alpha=0.5)

plt.sca(ax1)
plt.legend()
plt.xlabel('t')
plt.ylabel('abs((E-E0)/E0)')
# -


