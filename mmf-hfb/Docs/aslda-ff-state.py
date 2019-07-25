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

# # ASLDA FF State

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import * 

from mmf_hfb.ClassFactory import ClassFactory, FunctionalType, KernelType, Solvers
from mmf_hfb.ParallelHelper import PoolHelper
import numpy as np
import warnings
warnings.filterwarnings("ignore")

mu=10
dmu=0.5
delta=1
dim=3
LDA = ClassFactory(
            className="LDA",
            functionalType=FunctionalType.SLDA,
            kernelType=KernelType.HOM)
lda = LDA(mu_eff=mu, dmu_eff=dmu, delta=delta, T=0, dim=dim)
lda.C = lda._get_C(mus=(mu, mu), delta=delta)

lda.C

ns, e, p = lda.get_ns_e_p(mus=(mu + 1, mu - 1), delta=0.8, solver=Solvers.BROYDEN1, update_C=False, verbosity=True)

lda.get_ns_mus_e_p(mus_eff=(10.643740524431747,2.206170614085716), delta=1, update_C=False, verbosity=True)


def f(delta, dq):
    print(f"dq={dq}")
    return lda.C - lda._get_C(mus=(mu + dmu, mu - dmu), delta=delta, dq=dq, verbosity=False)


dqs = np.linspace(0, 0.018, 10)
fs = [f(delta=0.5*delta, dq=dq) for dq in dqs]
clear_output()

plt.plot(dqs, fs)
plt.axhline(0)
#plt.ylim(-0.01, 0.01)

dmus = np.linspace(0, 0.3 * mu, 10)
def c_thread(dmu):
    print(dmu)
    ds = []
    Cs = []
    ds_ = np.linspace(0.01, 3*delta, 100) 
    for d in ds_:
        try:
            C = lda._get_C(mus=(mu + dmu,mu - dmu), delta=d, dq=0, verbosity=False)
            ds.append(d)
            Cs.append(C)
            print(d, C)
        except:
            continue
    plt.plot(ds, Cs)
    return (ds, Cs)


rets = []
for dmu in dmus:
    rets.append(c_thread(dmu=dmu))
clear_output()


def plots(i):
    plt.plot(rets[i][0],rets[i][1], label=f"$d\mu$={dmus[i]}")
    plt.axvline(dmus[i])


for i in range(7):
    plots(i)
plt.legend()
#plt.ylim(-0.1, 0.1)


