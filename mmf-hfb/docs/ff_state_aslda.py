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
import mmf_hfb.ClassFactory as cf
reload(cf)
from mmf_hfb.ClassFactory import ClassFactory, FunctionalType, KernelType, Solvers
import warnings
warnings.filterwarnings("ignore")
import os
import inspect
from os.path import join
import json
import glob
from json import dumps
import operator
import mmf_hfb.FFStateAgent as ffa
reload(ffa)
import mmf_hfb.FFStatePlot as ffp
reload(ffp)
import numpy as np
currentdir = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),"..","mmf_hfb","data")

mu_eff=10
dmu_eff=0.5
delta=5
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
ds = np.linspace(0.001, 1.2*delta, 50)

rets = []
for dmu_eff in dmu_effs:
    Cs = [get_C(dmu_eff=dmu_eff, delta=d) for d in ds]
    rets.append(Cs)

plt.figure(figsize(16,8))
for i in range(len(dmu_effs)):
    plt.plot(ds, rets[i], label=f"d$\mu$={dmu_effs[i]}")
plt.axvline(dmu_effs[i],linestyle='dashed')
plt.legend()
plt.xlabel(f"$\Delta$")
plt.ylabel("C")

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

get_C(dmu_eff=0, delta=1), lda.C

delta = 1
lda.C= get_C(dmu_eff=0, delta=delta)
def f_q(dq):
    return get_C(dmu_eff=0.1, delta=delta, dq=dq) - lda.C
dqs=np.linspace(offset, 1+offset, 20)
fs = [f_q(dq) for dq in dqs]
plt.plot(dqs, fs)
plt.axhline(0, linestyle='dashed')

delta=1
lda.C= get_C(dmu_eff=0, delta=delta)
dmu_effs = np.linspace(0, 0.5, 3)
for dmu_eff in dmu_effs:
    def f_d(d):
        return get_C(dmu_eff=dmu_eff, delta=d) - lda.C
    ds=np.linspace(0.1, 1.2, 20)
    fs = [f_d(d) for d in ds]
    plt.plot(ds, fs)
plt.axhline(0, linestyle='dashed')
plt.axvline(delta, linestyle='dashed')

get_C(dmu_eff=0.3, delta=1), get_C(dmu_eff=1.5, delta=1)


# # Visualize Data

def filter_state(mu, dmu, delta, C, dim):
    if dim != 3:
        return True
    #return False
    #if g != -2.8:
    #    return True
    #return False
    #if g != -3.2:
    #    return True
    if delta != 1.5:
        return True
    #if dmu > 0.71 or dmu < 0.7:  
        #return True
    #if not np.allclose(dmu, 0.35, rtol=0.01):
    #    return True
    print(dmu)
    return False


plt.figure(figsize(8,8))
ffp.PlotStates(two_plot=False, filter_fun=filter_state, print_file_name= True)

plt.figure(figsize(16,10))
ffp.PlotCurrentPressure(filter_fun=filter_state, showLegend=True, FFState_only=False, print_file_name=True)

# # Plot the Diagram
# * Check the particle density, pressure, and $d\mu$ etc to see if a configuration is a FF state $\Delta$

output = ffa.label_states(raw_data=True)

plt.figure(figsize(16,8))
ffp.PlotPhaseDiagram(output=output)

# # Check a FF State
# * Find the first configeration that yields a FF State

for dic in output:
    if dic['state']:
        data=dic['data']
    break

# * Check the source state file

dic['file'], dic['C']

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

from mmf_hfb.FFStateAgent import FFStateAgent
args = dict(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=1, T=0, dim=3, k_c=50, verbosity=False, C=C)
lda = ClassFactory("LDA", (ffa.FFStateAgent,),  functionalType=FunctionalType.ASLDA, kernelType=KernelType.HOM, args=args)

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



