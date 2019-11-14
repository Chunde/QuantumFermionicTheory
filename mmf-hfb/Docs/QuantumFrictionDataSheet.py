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
from IPython.display import display, clear_output
from mmf_hfb.CoolingCaseTests import TestCase, Prob, Normalize, dict_to_complex, deserialize_object, load_json_data, random_gaussian_mixing


def get_init_states(N=128, dx=0.1):
    b = BCSCooling(N=N, dx=dx)
    h = HarmonicOscillator()
    x = b.xyz[0]
    V = x**2/2
    H0 = b._get_H(mu_eff=0, V=0)
    H1 = b._get_H(mu_eff=0, V=V)
    U0, E0 = b.get_U_E(H0, transpose=True)
    U1, E1 = b.get_U_E(H1, transpose=True)
    psi_standing_wave=Normalize(U0[1],dx=dx)
    psi_gaussian_mixing = random_gaussian_mixing(x, dx=dx)
    psi_uniform = Normalize(U0[0], dx=dx)
    psi_bright_soliton = Normalize(np.exp(-x**2/2.0)*np.exp(1j*x), dx=dx)
    return dict(ST=psi_standing_wave, GM=psi_gaussian_mixing, UN=psi_uniform, BS=psi_bright_soliton)
def PN(psi):
    return Prob(Normalize(psi))
def get_potentials(x):
    V0 = 0*x
    V_HO = x**2/2
    V_PO = V0 + np.random.random()*V_HO + abs(x**2)*np.random.random()
    return dict(V0=V0, HO=V_HO, PO=V_PO)


# # Test Code
# * code used to check results generated by the code in [CoolingCaseTests.py](CoolingCaseTests.py)

N=128
beta_H = 3
dx=0.2
g=-1
V_key = 'HO'
psis_init = get_init_states()
psi_init= psis_init['UN']
b = BCSCooling(N=N, dx=dx)
h = HarmonicOscillator()
x = b.xyz[0]
Vs = get_potentials(x)
args = dict(N=N, dx=dx, eps=1e-1, T_ground_state=5, V=Vs[V_key], V_key=V_key, g=g, psi_init=psi_init, use_abm=False, check_dE=False)
t=TestCase(ground_state_eps=1e-1, beta_0=1, beta_H=beta_H, **args)

plt.plot(x, PN(psi_init), "--", label='init')
plt.plot(x, PN(t.psi_ground), '-',label="final")
plt.show()


def test_wall_time(beta_V=0, beta_K=0, beta_Y=0):
    plt.figure(figsize=(10, 10))
    t.b.beta_V= beta_V
    t.b.beta_K = beta_K
    t.b.keta_Y = beta_Y
    for beta_H in [1, 2, 3, 4]:
        Es = []
        Tsw = []
        t.b.beta_H=beta_H
        Ts = np.linspace(0, 2, 20)/beta_H
        for T in Ts:
            t.run(T=T, plot=False, plot_log=False)
            Es.append(t.Es[-1])
            Tsw.append(t.wall_time[-1])
        plt.loglog(Tsw, (np.array(Es) - t.E0)/t.E0, label=f"beta_H={beta_H}")
    plt.xlabel("Wall Time")
    plt.ylabel("(E-E0)/E0")
    plt.legend()
    clear_output()


test_wall_time(beta_V=100, beta_K=10, beta_Y=0)

test_wall_time(beta_V=100, beta_K=0, beta_Y=0)

test_wall_time(beta_K=50)

from mmf_setup.set_path import hgroot
from IPython.display import clear_output
from importlib import reload
from mmf_hfb.quantum_friction import StateBase

s = StateBase(Nxyz=(32, 32), beta_0=-1.0j, beta_V=0.0, beta_K=0.0)
#s = StateBase(Nxyz=(32, 32), beta_0=-1j)
x, y = s.xyz
x0 = 0.5
phase = ((x-x0) + 1j*y)*((x+x0) - 1j*y)
psi0 = 1.0*np.exp(1j*np.angle(phase))
ts, psis = s.solve(psi0, T=2.0, rtol=1e-5, atol=1e-6)
display(s.plot(psis[-1]))

s = BCSCooling(N=32, dx=0.1, beta_0=-1.0j, beta_V=0.0, beta_K=0.0, dim=2)
x, y = s.xyz
x0 = 0.5
phase = ((x-x0) + 1j*y)*((x+x0) - 1j*y)
psi0 = 1.0*np.exp(1j*np.angle(phase))
ts, psis = s.solve([psi0], T=2.0, rtol=1e-5, atol=1e-6)

s.plot(psis[0][-1])


