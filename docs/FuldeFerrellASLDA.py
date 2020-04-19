# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import *
from scipy.optimize import brentq
import mmf_hfb.class_factory as cf
reload(cf)
from mmf_hfb.class_factory import ClassFactory, FunctionalType, KernelType, Solvers
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import inspect
from os.path import join
import json
import glob
from json import dumps
import operator
import numpy as np


# +
currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, join(currentdir, '..','Projects','FuldeFerrellState'))
currentdir = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),"..","Projects","FuldeFerrellState","data")
# currentdir = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),"..","mmf-hfb","mmf_hfb","data")

import fulde_ferrell_state_agent as ffa
reload(ffa)
import fulde_ferrell_state_plot as ffp
reload(ffp)
# -

# # Solution Check

mu_eff=5
dmu_eff=0.5
delta=1.5
dim=2
LDA = ClassFactory(
            className="LDA",
            functionalType=FunctionalType.SLDA,
            kernelType=KernelType.HOM)
lda = LDA(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta, T=0, dim=dim)
lda.C = lda._get_C(mus_eff=(mu_eff,mu_eff), delta=delta)


def f(dq):
    return (lda._get_C(
        mus_eff=(mu_eff + dmu_eff, mu_eff - dmu_eff), delta=delta, dq=dq) - lda.C)
def get_C(dmu_eff, delta, dq=0):
    return lda._get_C(mus_eff=(mu_eff + dmu_eff,mu_eff-dmu_eff), delta=delta, dq=dq)


dmu_effs = np.linspace(0, delta, 5)
ds = np.linspace(0.001, 1.2*delta, 25)


# [0.4684362992740691, None, 0.9697323232323233]

# dmu0=0.51
# dq0, delta0=0.4684362992740691, 0.9697323232323233

def get_f(dmu=0, dq=0, delta=0):
    return get_C(dmu_eff=dmu, dq=dq, delta=delta) - lda.C


# ds = np.linspace(0.0001, delta, 10)
# fs = [get_f(dmu=dmu0, dq=dq0, delta=d) for d in ds]

plt.plot(ds, fs)
plt.axhline(0, ls='dashed')
plt.axvline(delta, ls='dashed')
plt.axvline(delta0, ls='dashed', c='red')

# ## Check dq effect

ret2=[]
dqs=np.linspace(0, 0.2, 3)
for dq in dqs:
    Cs = [get_C(dmu_eff=0, delta=d, dq=dq) for d in ds]
    ret2.append(Cs)


# for i in range(len(ret2)):
#     plt.plot(ds, ret2[i], label=f"q={dqs[i]}")
# plt.legend()

# # Solver
# Here we need to solve the dq for given $\Delta$ with fixed $\tilde{C}$

# get_C(dmu_eff=0, delta=1), lda.C

# delta = 1
# lda.C= get_C(dmu_eff=0, delta=delta)
# def f_q(dq):
#     return get_C(dmu_eff=0.1, delta=delta, dq=dq) - lda.C
# dqs=np.linspace(offset, 1+offset, 20)
# fs = [f_q(dq) for dq in dqs]
# plt.plot(dqs, fs)
# plt.axhline(0, linestyle='dashed')

# delta=1
# lda.C= get_C(dmu_eff=0, delta=delta)
# dmu_effs = np.linspace(0, 0.5, 3)
# for dmu_eff in dmu_effs:
#     def f_d(d):
#         return get_C(dmu_eff=dmu_eff, delta=d) - lda.C
#     ds=np.linspace(0.1, 1.2, 20)
#     fs = [f_d(d) for d in ds]
#     plt.plot(ds, fs)
# plt.axhline(0, linestyle='dashed')
# plt.axvline(delta, linestyle='dashed')

# get_C(dmu_eff=0.3, delta=1), get_C(dmu_eff=1.5, delta=1)

# # Visualize Data

def filter_state(mu, dmu, delta, C, dim):
    if dim != 2:
        return True
    #return False
    #if g != -2.8:
    #    return True
    #return False
    #if g != -3.2:
    #    return True
    if delta != .4:
         return True
#     if delta > 0.6:
#     if delta != 0.5:
#         return True
#     if dmu < 0.35:  
#           return True
#     if dmu > 0.36:
#          return True
    #if not np.allclose(dmu, 0.35, rtol=0.01):
    #    return True
    #print(dmu)
    return False


plt.figure(figsize(16,8))
ffp.PlotStates(current_dir=currentdir, two_plot=False,
               filter_fun=filter_state, plot_legend=True, ls='-+',print_file_name=True)

# plt.figure(figsize(16,10))
ffp.PlotCurrentPressure(current_dir=currentdir, filter_fun=filter_state,alignLowerBranches=False,
                        showLegend=False, FFState_only=False, print_file_name=False)

# # Plot the Diagram
# * Check the particle density, pressure, and $d\mu$ etc to see if a configuration is a FF state $\Delta$

output = ffa.label_states(current_dir=currentdir, raw_data=False, verbosity=False)
clear_output()

output_ff = []
for item in output:
    if item['state']:
        output_ff.append(item)

item_max = output_ff[0]
for item in output_ff:
    print(item['pf'] - item['pn'])
    if (item['pf'] - item['pn'])>(item_max['pf'] - item_max['pn']):
        item_max = item

item_max

with open("output_tmp.json", 'w') as wf:
            json.dump(output, wf)

plt.figure(figsize(16,16))
ffp.PlotPhaseDiagram(output=output)

# # Check a FF State
# * Find the first configeration that yields a FF State

np.linspace(1, 2, 2)

for dic in output:
    if dic['state']:
        data=dic['data']
    break

# * Check the source state file

15.940959103690169-15.94095910535347

# * Restructe the data

# +
ret=data
dim, mu_eff, dmu_eff, delta, C=ret['dim'], ret['mu_eff'], ret['dmu_eff'], ret['delta'], ret['C']
p0 = ret['p0']
a_inv = 4.0*np.pi*C  # inverse scattering length
data1, data2 = ret['data']
data1.extend(data2)
dqs1, dqs2, ds1, ds2= [], [], [], []
j1, j2, ja1, ja2, jb1, jb2, P1, P2 = [], [], [], [], [], [], [], []
for data_ in data1:
    d, q, p, j, j_a, j_b = (
        data_['d'], data_['q'], data_['p'], data_['j'], data_['ja'], data_['jb'])
    ds1.append(d)
    dqs1.append(q)
    j1.append(j)
    ja1.append(j_a)
    jb1.append(j_b)
    P1.append(p)

bFFState = False
if len(P1) > 0:
    index1, value = max(enumerate(P1), key=operator.itemgetter(1))
    ground_state_data = data1[index1]
    n_a, n_b = ground_state_data['na'], ground_state_data['nb']
    mu_a, mu_b = ground_state_data['mu_a'], ground_state_data['mu_b']
    dq = ground_state_data['q']
    mu, dmu = (mu_a + mu_b)/2.0, (mu_a - mu_b)/2.0
    print(f"na={n_a}, nb={n_b}, PF={value}, PN={p0}")
    if (value > p0) and (
        not np.allclose(
            n_a, n_b, rtol=1e-9) and (
                ground_state_data["q"]>0.0001 and ground_state_data["d"]>0.001)):
        bFFState = True
if bFFState:
    print("This is a FF state")
dic = dict(
    mu_eff=mu_eff, dmu_eff=dmu_eff,
    mu=mu, dmu=dmu, np=n_a + n_b, na=n_a,
    nb=n_b, ai=a_inv, C=C, delta=delta, state=bFFState)
# -

# ## Extract the parameters

delta=ground_state_data['d']
mu_a=mu + dmu
mu_b=mu - dmu
mus=(mu_a, mu_b)
mu_a_eff=mu_eff + dmu_eff
mu_b_eff=mu_eff - dmu_eff
mus_eff=(mu_a_eff, mu_b_eff)
mu_eff, dmu_eff, mu, dmu, C, delta, dq

# ## Compare pressure for different states

from mmf_hfb.fulde_ferrell_state_agent import fulde_ferrell_state_agent
args = dict(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=1, T=0, dim=3, k_c=50, verbosity=False, C=C)
lda = class_factory("LDA", (ffa.fulde_ferrell_state_agent,),  functionalType=FunctionalType.ASLDA, kernelType=KernelType.HOM, args=args)

mu_a_eff_, mu_b_eff_ = lda.get_mus_eff(mus=mus, delta=delta, dq=dq, verbosity=False)
mu_eff_=(mu_a_eff_ + mu_b_eff_)/2
dmu_eff_ = (mu_a_eff_-mu_b_eff_)/2

# ### Compare pressures with same bare mus

lda.get_ns_e_p(mus=mus,delta=delta, dq=dq, verbosity=False, solver=Solvers.BROYDEN1)

lda.get_ns_e_p(mus=mus,delta=None, verbosity=False, solver=Solvers.BROYDEN1)

lda.get_ns_e_p(mus=mus,delta=0, verbosity=False, solver=Solvers.BROYDEN1)

# ### FFState Pressure

lda.get_ns_mus_e_p(mus_eff=(mu_a_eff, mu_b_eff), delta=delta, dq=dq)[3]

# ### Sysmetric Superfuild State Pressure

lda.delta=delta
lda.get_ns_mus_e_p(mus_eff=(mu_a_eff, mu_b_eff), delta=None)[3]

# ### Normal State Pressure

lda.get_ns_mus_e_p(mus_eff=(mu_a_eff, mu_b_eff), delta=0)[3]

# # Playground


# +
from phase_diagram_generator import FFStateAgent

mu_eff = 10
dmu_eff = 1
delta = 1.2
dim = 2
k_c = 150
args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
        T=0, dim=dim, k_c=k_c, verbosity=False)
lda = ClassFactory(
    "LDA", (FFStateAgent,),
    functionalType=FunctionalType.BDG,
    kernelType=KernelType.HOM, args=args)
lda.C = lda._get_C(mus_eff=(mu_eff, mu_eff),delta=delta)


# -

def f(delta, dq):
    return lda._get_C(
        mus_eff=(mu_eff + dmu_eff, mu_eff - dmu_eff),
        delta=delta, dq=dq) - lda.C


dqs = np.linspace(0, delta/2, 10)
fs = [f(delta=0.1, dq=dq) for dq in dqs]

plt.plot(dqs, fs)
plt.axhline(0, ls='dashed')
plt.axvline(0, ls='dashed')

dq0, delta0= 0.11076732169336657, 0.0537
def g(dq):
    return f(delta=delta0, dq=dq)


g(dq=dq0)

# # ZoomIn Search Algorithm

import operator
def zoom_in_search(delta, dq):
    dq1, dq2 = dq0*0.9, dq0*1.1
    p1, p2 = None, None
    for i in range(20):
        dqs = np.linspace(dq1, dq2, 10)
        gs = np.array([g(dq) for dq in dqs])
        index, value = min(enumerate(gs), key=operator.itemgetter(1))
        
        plt.figure(figsize=(16,8))

        plt.plot(dqs, gs)
        plt.title(f"{i}:p1={p1}, p2={p2}")
        plt.show()
        clear_output(wait=True)
        
        if np.all(gs > 0):
            if index == 0:  # range expaned more to the left
                dq1 = dq1*0.9
                if p2 is None:
                    p2 = dqs[0]
                continue
            if index == len(dqs) - 1:
                dq2 = dq2*1.1
                if p1 is None:
                    p1 = dqs[-1]
                continue
            dq1, dq2 = dqs[index - 1], dqs[index + 1]
            p1, p2 = dq1, dq2
            continue

        delta1 = brentq(g, p1, dqs[index])
        delta2 = brentq(g, dqs[index], p2)
        print(delta1, delta2)
        break   


zoom_in_search(delta=delta0, dq=dq0)

# ## Plot $g(\mu,\delta\mu, \Delta)$

plt.figure(figsize(16, 8))
fontsize=20
mu = 10
ds = np.linspace(0.0001, 1.2*mu, 128)
dmus = np.linspace(0, mu, 5)
for dmu in dmus:
    gs = [ff.get_g(mu=mu, dmu=dmu, delta=d) for d in ds]
    plt.plot(ds/mu, gs, label=r"$\delta \mu/e_F$"+f"={dmu/mu}")
plt.legend(prop={'size': 18})
plt.xlabel(r"$\Delta/e_F$",fontsize=fontsize)
plt.ylabel("g",fontsize=fontsize)
plt.title(r"$g(\mu,\delta\mu, \Delta)$",fontsize=fontsize)
plt.savefig("g_as_a_function_delta_mu_dmu.pdf", bbox_inches='tight')

# # Draft

# +
L = 10
def f(x):
    return (2/L)**0.5*np.sin(np.pi*x/L)
def g(x):
    return f(x)**2

def f1(x):
    return np.pi**(-0.25)*np.exp(-x**2/2+1j*x)
def g1(x):
    v = f1(x)
    return v.conj()*v


# -

import scipy.integrate
import scipy as sp
sp.integrate.quad(g1, -L, L)


