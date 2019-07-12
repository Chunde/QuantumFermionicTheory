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
from scipy.optimize import brentq

from mmf_hfb.ClassFactory import ClassFactory, FunctionalType, KernelType, Solvers
from mmf_hfb.ParallelHelper import PoolHelper
import numpy as np
import warnings
warnings.filterwarnings("ignore")

mu_eff=10
dmu_eff=0.5
delta=1
dim=3
LDA = ClassFactory(
            className="LDA",
            functionalType=FunctionalType.SLDA,
            kernelType=KernelType.HOM)
lda = LDA(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta, T=0, dim=dim)
lda.C = lda._get_C(mus_eff=(mu_eff,mu_eff), delta=delta)


def get_C(dmu_eff, delta, dq=0):
    return lda._get_C(mus_eff=(mu_eff + dmu_eff,mu_eff-dmu_eff), delta=delta, dq=dq)


dmu_effs = np.linspace(0, delta, 5)
ds = np.linspace(0.001, 1.2*delta, 40)

rets = []
for dmu_eff in dmu_effs:
    Cs = [get_C(dmu_eff=dmu_eff, delta=d) for d in ds]
    rets.append(Cs)

for i in range(len(dmu_effs)):
    plt.plot(ds, rets[i], label=f"d$\mu$={dmu_effs[i]}")
    plt.axvline(dmu_effs[i],linestyle='dashed')
plt.legend()
plt.xlabel(f"\Delta")
plt.ylabel("C")

ret1=[]
dqs=np.linspace(0, 0.2, 20)
for dq in dqs:
    Cs = [get_C(dmu_eff=0.2, delta=d, dq=dq) for d in ds]
    ret1.append(Cs)

plt.figure(figsize(16,8))
for i in range(len(dqs)):
    plt.plot(ds, ret1[i], label=f"d$\mu$={dqs[i]}")
plt.legend()

ret2=[]
dqs=np.linspace(0, 0.2, 5)
for dq in dqs:
    Cs = [get_C(dmu_eff=0, delta=d, dq=dq) for d in ds]
    ret2.append(Cs)

for i in range(len(ret2)):
    plt.plot(ds, ret2[i], label=f"q={dqs[i]}")
plt.legend()


# # Solver
# Here we need to solve the dq for given $\Delta$ with fixed $\tilde{C}$

def f(dq):
    return get_C(dmu_eff=0.1, delta=0.003538, dq=dq) - lda.C


lda.C= get_C(dmu_eff=0, delta=0.2)
dqs=np.linspace(0, 0.1, 40)
fs = [f(dq) for dq in dqs]

plt.plot(dqs, fs)
plt.axhline(0, linestyle='dashed')

dqs

brentq(f, 0, 0.02)

# # Visualize Data

import os
import inspect
from os.path import join
def filter(mu, dmu, delta, g, dim):
    if dim != 3:
        return True
    #return False
    #if g != -2.8:
    #    return True
    #return False
    #if g != -3.2:
    #    return True
    if delta != 1:
        return True
    if not np.allclose(dmu,0.6):
        return True
    return False
currentdir = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),"..","mmf_hfb","data")

import json
import glob
from json import dumps
def plotStates(twoPlot=False):
    pattern = join(currentdir,"FFState_[()d_0-9]*.json")
    files = glob.glob(pattern)
    if twoPlot:
        plt.figure(figsize=(16,16))
    else:
        plt.figure(figsize=(12,12))
    style =['o','-','+']
    gs = set()
    for file in files:
        if os.path.exists(file):
            with open(file,'r') as rf:
                ret = json.load(rf)
                dim, mu, dmu, delta, g=ret['dim'], ret['mu'], ret['dmu'], ret['delta'], ret['g']
                gs.add(g)
                if filter(mu=mu, dmu=dmu, delta=delta, g=g, dim=dim):
                        continue
                print(file)
                datas = ret['data']
                dqs1, dqs2, ds1, ds2 = [],[],[],[]
                for data in datas:
                    dq1, dq2, d = data
                    if dq1 is not None:
                        dqs1.append(dq1)
                        ds1.append(d)
                    if dq2 is not None:
                        dqs2.append(dq2)
                        ds2.append(d)
                if twoPlot:
                    plt.subplot(211)
                if len(ds1) < len(ds2):
                    if len(ds1) > 0:
                        plt.plot(ds1, dqs1, style[dim-1], label=f"$\Delta=${delta}, $\mu$={mu:.2}, $d\mu=${dmu:.2}, g={g:.2}")
                else:
                    if len(ds2) > 0:
                        plt.plot(ds2, dqs2, style[dim-1], label=f"$\Delta=${delta}, $\mu$={mu:.2}, $d\mu=${dmu:.2}, g={g:.2}")
                if twoPlot:
                    plt.subplot(212)
                if len(ds1) < len(ds2):
                    if len(ds2) > 0:
                        plt.plot(ds2, dqs2, style[dim-1], label=f"$\Delta=${delta}, $\mu$={mu:.2}, $d\mu=${dmu:.2}, g={g:.2}")
                else:
                    if len(ds1)> 0:
                        plt.plot(ds1, dqs1, style[dim-1], label=f"$\Delta=${delta}, $\mu$={mu:.2}, $d\mu=${dmu:.2}, g={g:.2}")
                #break
    print(gs)   
    if twoPlot:
        plt.subplot(211)
        plt.xlabel(f"$\Delta$")
        plt.ylabel(f"$\delta q$")
        plt.title(f"Lower Branch")
        plt.legend()
        plt.subplot(212)
        plt.xlabel(f"$\Delta$")
        plt.ylabel(f"$\delta q$")
        plt.title(f"Upper Branch")
        plt.legend()
    else:
        plt.xlabel(f"$\Delta$")
        plt.ylabel(f"$\delta q$")
        plt.legend()
