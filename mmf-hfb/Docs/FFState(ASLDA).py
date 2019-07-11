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


lda.C

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

ret1=[]
dqs=np.linspace(0, 0.2, 10)
for dq in dqs:
    Cs = [get_C(dmu_eff=0.2, delta=d, dq=dq) for d in ds]
    ret1.append(Cs)

for i in range(len(dqs)):
    plt.plot(ds, ret1[i], label=f"d$\mu$={dqs[i]}")
plt.legend()

ret2=[]
dqs=np.linspace(0, 0.2, 5)
for dq in dqs:
    Cs = [get_C(dmu_eff=0, delta=d, dq=dq) for d in ds]
    ret2.append(Cs)

for i in range(len(dqs)):
    plt.plot(ds, ret2[i], label=f"d$\mu$={dqs[i]}")
plt.legend()


# # Solver
# Here we need to solve the dq for given $\Delta$ with fixed $\tilde{C}$

def f(dq):
    return get_C(dmu_eff=0.05, delta=0.1, dq=dq) - lda.C


dqs=np.linspace(0, 0.5, 20)
fs = [f(dq) for dq in dqs]

plt.plot(dqs, fs)
plt.axhline(0, linestyle='dashed')

brentq(f, 0, 0.4)


