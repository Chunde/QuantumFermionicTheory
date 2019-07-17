# -*- coding: utf-8 -*-
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

get_C(dmu_eff=0, delta=1)

offset = 0.
lda.C= get_C(dmu_eff=0, delta=3)
def f(dq):
    return get_C(dmu_eff=1.3336666666666666, delta=3.001, dq=dq) - lda.C
dqs=np.linspace(offset, 1+offset, 20)
fs = [f(dq) for dq in dqs]
plt.plot(dqs, fs)
plt.axhline(0, linestyle='dashed')

brentq(f, 0, 0.4)

# # Visualize Data

import os
import inspect
from os.path import join
import json
import glob
from json import dumps
currentdir = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),"..","mmf_hfb","data")


# +
def filter(mu, dmu, delta, C, dim):
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
    if dmu > 0.6 or dmu < 0.58:
        return True
    #if not np.allclose(dmu,0.6):
    #    return True
    return False

def plotStates(twoPlot=False):
    pattern = join(currentdir,"FFState_[()d_0-9]*.json")
    files = glob.glob(pattern)
    if twoPlot:
        plt.figure(figsize=(16,16))
    else:
        plt.figure(figsize=(12,12))
    style =['o','-','+']
    Cs = set()
    for file in files:
        if os.path.exists(file):
            with open(file,'r') as rf:
                ret = json.load(rf)
                dim, mu, dmu, delta, C=ret['dim'], ret['mu_eff'], ret['dmu_eff'], ret['delta'], ret['C']
                if filter(mu=mu, dmu=dmu, delta=delta, C=C, dim=dim):
                        continue
                Cs.add(C)

                #print(file)
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
                        plt.plot(ds1, dqs1, style[dim-1], label=f"$\Delta=${delta}, $\mu$={mu}, $d\mu=${dmu:.2}, C={C:.2}")
                else:
                    if len(ds2) > 0:
                        plt.plot(ds2, dqs2, style[dim-1], label=f"$\Delta=${delta}, $\mu$={mu}, $d\mu=${dmu:.2}, C={C:.2}")
                if twoPlot:
                    plt.subplot(212)
                if len(ds1) < len(ds2):
                    if len(ds2) > 0:
                        plt.plot(ds2, dqs2, style[dim-1], label=f"$\Delta=${delta}, $\mu$={mu}, $d\mu=${dmu:.2}, C={C:.2}")
                else:
                    if len(ds1)> 0:
                        plt.plot(ds1, dqs1, style[dim-1], label=f"$\Delta=${delta}, $\mu$={mu}, $d\mu=${dmu:.2}, C={C:.2}")
                #break
    print(Cs)   
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


# -

plotStates(False)

import os
from mmf_hfb.FFStateAgent import FFStateAgent
import inspect
from os.path import join
import json
import glob
import operator
import warnings
warnings.filterwarnings("ignore")
from json import dumps
def filter1(mu, dmu, delta, C, dim):
    if dim != 3:
        return True
    
    #return False
    #if g != -3.2:
    #    return True
    if delta != 1:
        return True
    
    #if dmu< 1.5 or dmu > 1.8:
    #    return True
    #if not np.allclose(dmu,0.6):
    #    return True
    return False
currentdir = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),"..","mmf_hfb","data")
def PlotCurrentPressure(alignLowerBranches=False, alignUpperBranches=False, showLegend=False, FFState_only=True):
    pattern = join(currentdir, "FFState_J_P[()d_0-9]*")
    files=glob.glob(pattern)
    plt.figure(figsize=(20,20))
    Cs = set()
    for file in files[0:]:
        if os.path.exists(file):
            with open(file,'r') as rf:
                ret = json.load(rf)
                dim, mu_eff, dmu_eff, delta, C, p0=ret['dim'], ret['mu_eff'], ret['dmu_eff'], ret['delta'],ret['C'], ret['p0']
                mus_eff = (mu_eff + dmu_eff, mu_eff - dmu_eff)
                if filter1(mu=mu_eff, dmu=dmu_eff, delta=delta, C=C, dim=dim):
                    continue 
                label = f"$\Delta=${delta},$\mu=${float(mu_eff):.2},$d\mu=${float(dmu_eff):.2}"
                k_c = None
                if 'k_c' in ret:
                    k_c = ret['k_c']
                #print(C)
                #print(file)
                data1, data2 = ret['data']
                

                dqs1, dqs2, ds1, ds2, j1, j2, ja1, ja2, jb1, jb2, P1, P2 = [],[],[],[],[],[],[],[],[],[],[],[]
                for data in data1:
                    d, q, p, j, j_a, j_b = data['d'],data['q'],data['p'],data['j'],data['ja'],data['jb']
                    ds1.append(d)
                    dqs1.append(q)
                    j1.append(j)
                    ja1.append(j_a)
                    jb1.append(j_b)
                    P1.append(p)
                for data in data2:
                    d, q, p, j, j_a, j_b = data['d'],data['q'],data['p'],data['j'], data['ja'], data['jb']
                    ds2.append(d)
                    dqs2.append(q)
                    j2.append(j)
                    ja2.append(j_a)
                    jb2.append(j_b)
                    P2.append(p)

               
                
                
                def plot_P(P, Data, ds, align):
                    if len(P)==0:
                        return (0, False)
                    if align:
                        P = np.array(P)
                        P_ = P- P.min()
                    else:
                        P_=P
                    index, value = max(enumerate(P), key=operator.itemgetter(1))
                    data = Data[index]
                    n_a, n_b =  data['na'], data['nb']
                    state = "FF" if value > p0 and not np.allclose(n_a, n_b) else "NS"                  
                    #plt.axvline(ds1[index])
                    # plt.axhline(p0, color='r', linestyle='dashed')
                    # print(f"Delta={delta}, dmu={dmu_eff}, Normal Pressure={p0}，FFState Pressue={value}")
                    if FFState_only:
                        if state== "FF":
                        #print(index, Data[index])
                            plt.plot(ds, P_,"+", label=f"$\Delta=${delta},$\mu=${float(mu_eff):.2},$d\mu=${float(dmu_eff):.2},State:{state}")
                    else:
                        plt.plot(ds, P_,"+", label=f"$\Delta=${delta},$\mu=${float(mu_eff):.2},$d\mu=${float(dmu_eff):.2},State:{state}")
                    return (index, state=="FF")
                
                plt.subplot(323)
                index1, ffs1 = plot_P(P1, data1, ds1, alignLowerBranches)
                plt.subplot(324)
                index2, ffs2 = plot_P(P2, data2, ds2, alignUpperBranches)
                plt.subplot(325)
                #plt.plot(ds1, j1, label=f"$j_p, \Delta=${delta},$\mu=${mu},$d\mu=${dmu}")
                if FFState_only:
                    if ffs1:
                        plt.plot(ds1, ja1, "+",label=label)
                else:
                    plt.plot(ds1, ja1, "+",label=label)
                #plt.plot(ds1, jb1, label=f"j_b")
                #plt.axvline(ds1[index1])
                plt.subplot(326)
                #plt.plot(ds2, j2, label=f"$j_p, \Delta=${delta},$\mu=${mu},$d\mu=${dmu}")
                if FFState_only:
                    if ffs2:
                        plt.plot(ds2, ja2, "+",label=label)
                else:
                    plt.plot(ds2, ja2, "+",label=label)
                #plt.plot(ds2, jb2, "+",label=f"j_b")
                #if len(ds2) > 0:
                    #plt.axvline(ds2[index2])
                plt.axhline(0)
                plt.subplot(321)
                if FFState_only:
                    if ffs1:
                        plt.plot(ds1, dqs1,"+", label=label)
                else:
                    plt.plot(ds1, dqs1,"+", label=label)
                plt.subplot(322)
                if FFState_only:
                    if ffs2:
                        plt.plot(ds2, dqs2,"+", label=label)
                else:
                    plt.plot(ds2, dqs2,"+", label=label)
                #break
    # print(Cs)    
    for i in range(1,7):
        plt.subplot(3,2,i)
        if showLegend:
                plt.legend()
        if i == 1:
            plt.title(f"Lower Branch")
            plt.ylabel("$\delta q$")
        if i == 2:
            plt.title(f"Upper Branch")
            plt.ylabel("$\delta q$")
        if i == 3 or i == 4:
            plt.ylabel("$Pressure$")
        if i == 5 or i == 6:
            plt.ylabel("$Current$")
        plt.xlabel("$\Delta$")

PlotCurrentPressure(showLegend=True, FFState_only=True)


# # Plot the Diagram
# * Check the particle density, pressure, and $d\mu$ etc to see if a configuration is a FF state $\Delta$

def FindFFState(verbose=False):
        currentdir = join(
            os.path.dirname(
                os.path.abspath(
                    inspect.getfile(
                        inspect.currentframe()))), "..", "mmf_hfb", "data")
        output = []
        
        pattern = join(currentdir, "FFState_J_P[()d_0-9]*")
        files=glob.glob(pattern)
        for file in files[0:]:
            if os.path.exists(file):
                with open(file, 'r') as rf:
                    ret = json.load(rf)
                    dim, mu_eff, dmu_eff, delta, C=ret['dim'], ret['mu_eff'], ret['dmu_eff'], ret['delta'], ret['C']
                    p0 = ret['p0']
                    a_inv = 4.0*np.pi*C

                    if verbose:
                        print(file)
                    data1, data2 = ret['data']

                    data1.extend(data2)

                    dqs1, dqs2, ds1, ds2= [], [], [], []
                    j1, j2, ja1, ja2, jb1, jb2, P1, P2 = [], [], [], [], [], [], [], []
                    for data in data1:
                        d, q, p, j, j_a, j_b = data['d'], data['q'], data['p'], data['j'], data['ja'], data['jb']
                        ds1.append(d)
                        dqs1.append(q)
                        j1.append(j)
                        ja1.append(j_a)
                        jb1.append(j_b)
                        P1.append(p)
                   
                    bFFState = False
                    if len(P1) > 0:
                        index1, value = max(enumerate(P1), key=operator.itemgetter(1))
                        data = data1[index1]
                        n_a, n_b = data['na'], data['nb']
                        mu_a, mu_b = data['mu_a'], data['mu_b']
                        mu, dmu = (mu_a + mu_b)/2.0, (mu_a - mu_b)/2.0
                        if verbose:
                            print(f"na={n_a}, nb={n_b}, PF={value}, PN={p0}")
                        if (value > p0) and (
                            not np.allclose(
                                n_a, n_b, rtol=1e-9) and data["q"]>0.0001 and data["d"]>0.001):
                            bFFState = True
                    if bFFState and verbose:
                        print(f"FFState: {bFFState} |<-------------")
                    dic = dict(mu_eff=mu_eff, dmu_eff=dmu_eff, d
                        mu=mu, dmu=dmu, np=n_a + n_b, na=n_a,
                        nb=n_b, ai=a_inv, C=C, delta=delta, state=bFFState)
                    if verbose:
                        print(dic)
                    output.append(dic)
                    if verbose:
                        print("-----------------------------------")
        return output


output = FindFFState()

xs = []
ys = []
xs2 = []
ys2 = []
ys3 = []
states = []
k_F = (2 * 0.59060550703283853378393810185221521748413488992993 * 10)**(0.5)
for dic in output:  
    #if dic['delta'] !=0.5:
    #    continue
    n = dic['na'] + dic['nb']#dic['np']#
    mu = dic['mu']
    dmu = dic['dmu']
    delta = dic['delta']
    k_F = (2.0*mu)**0.5
    dn = dic['na'] - dic['nb']
    ai = dic['ai']
    C= dic['C']
    #xs.append(-1/(g*n**(1.0/3)))
    #xs.append(-ai/(n**(1.0/3)))
    xs.append(-ai/k_F)
    #ys.append(dic['dmu']/dic['delta'])
    ys.append(dn/n)  # polorization
    xs2.append(delta)
    ys2.append(dmu/delta)
    ys3.append(dic['dmu_eff'])
    states.append(dic['state'])
N=len(xs)
colors = []
area = []
for i in range(len(states)):
    s=states[i]
    if s:
        #print(xs[i],ys[i])
        #print(output[i])
        colors.append('red')        
        area.append(15)
    else:
        colors.append('blue')
        area.append(1)
plt.figure(figsize(16,16))
plt.subplot(221)
plt.scatter(xs, ys,  s=area, c=colors)
#plt.ylabel(r"$\delta \mu/\Delta_{\delta \mu=0}$")
plt.ylabel(r"$\delta n/n$", fontsize=16)
#plt.xlabel(r"$-1/gn^{1/3}$")
plt.xlabel(r"$-1/ak_F$", fontsize=16)
plt.subplot(222)
plt.scatter(xs, ys2, s=area, c=colors)
plt.ylabel(r"$\delta\mu/\Delta$", fontsize=16)
plt.xlabel(r"$-1/ak_F$", fontsize=16)
plt.subplot(223)
plt.scatter(xs2, ys3, s=area, c=colors)
plt.ylabel(r"$\delta\mu_{eff}$", fontsize=16)
plt.xlabel(r"$\Delta$", fontsize=16)




