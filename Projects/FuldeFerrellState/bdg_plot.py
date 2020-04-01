from mmf_hfb.fulde_ferrell_state import FFState
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
import operator
import warnings
import inspect
import json
import glob
import os

warnings.filterwarnings("ignore")


def PlotStates(current_dir=None, filter_fun=None, two_plot=False):
    if current_dir is None:
        current_dir = join(
            os.path.dirname(
                os.path.abspath(
                    inspect.getfile(
                        inspect.currentframe()))), "..", "mmf_hfb", "data(BdG)")
    pattern = join(current_dir, "FFState_[()d_0-9]*.json")
    files = glob.glob(pattern)
    
    style =['o', '-', '+']
    gs = set()
    for file in files:
        if os.path.exists(file):
            with open(file, 'r') as rf:
                ret = json.load(rf)
                dim, mu, dmu, delta, g=ret['dim'], ret['mu'], ret['dmu'], ret['delta'], ret['g']
                gs.add(g)
                if filter_fun is not None and filter_fun(mu=mu, dmu=dmu, delta=delta, g=g, dim=dim):
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
                if two_plot:
                    plt.subplot(211)
                if len(ds1) < len(ds2):
                    if len(ds1) > 0:
                        plt.plot(
                            ds1, dqs1, style[dim-1],
                            label=f"$\Delta=${delta}, $\mu$={mu:.2}, $d\mu=${dmu:.2}, g={g:.2}")
                else:
                    if len(ds2) > 0:
                        plt.plot(
                            ds2, dqs2, style[dim-1],
                            label=f"$\Delta=${delta}, $\mu$={mu:.2}, $d\mu=${dmu:.2}, g={g:.2}")
                if two_plot:
                    plt.subplot(212)
                if len(ds1) < len(ds2):
                    if len(ds2) > 0:
                        plt.plot(
                            ds2, dqs2, style[dim-1],
                            label=f"$\Delta=${delta}, $\mu$={mu:.2}, $d\mu=${dmu:.2}, g={g:.2}")
                else:
                    if len(ds1)> 0:
                        plt.plot(
                            ds1, dqs1, style[dim-1],
                            label=f"$\Delta=${delta}, $\mu$={mu:.2}, $d\mu=${dmu:.2}, g={g:.2}")
    print(gs)   
    if two_plot:
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


def PlotCurrentPressure(
    current_dir=None, filter_fun=None, alignLowerBranches=True,
    alignUpperBranches=True, showLegend=False):
    if current_dir is None:
        current_dir = join(
            os.path.dirname(
                os.path.abspath(
                    inspect.getfile(
                        inspect.currentframe()))), "..", "mmf_hfb", "data(BdG)")
    pattern = join(current_dir, "FFState_J_P[()d_0-9]*")
    files=glob.glob(pattern)
    gs = set()
    for file in files[0:]:
        if os.path.exists(file):
            with open(file, 'r') as rf:
                ret = json.load(rf)
                dim, mu, dmu, delta, g=ret['dim'], ret['mu'], ret['dmu'], ret['delta'], ret['g']
                if filter_fun is not None and filter_fun(mu=mu, dmu=dmu, delta=delta, g=g, dim=dim):
                    continue 
                k_c = None
                if 'k_c' in ret:
                    k_c = ret['k_c']
                gs.add(g)
                #print(file)
                ff = FFState(
                    mu=mu, dmu=dmu, delta=delta, dim=dim, g=g, k_c=k_c, fix_g=True)
                mu_eff, dmu_eff = mu, dmu
                n_a, n_b = ff.get_densities(mu=mu_eff, dmu=dmu_eff)
                p0 = ff.get_pressure(
                    mu=None, dmu=None, mu_eff=mu_eff, dmu_eff=dmu_eff, delta=0).n
                data1, data2 = ret['data']
                

                dqs1, dqs2, ds1, ds2, j1, j2, ja1, ja2, jb1, jb2, P1, P2 = [],[],[],[],[],[],[],[],[],[],[],[]
                for data in data1:
                    d, q, p, j, j_a, j_b = data['d'], data['q'], data['p'], data['j'], data['ja'], data['jb']
                    ds1.append(d)
                    dqs1.append(q)
                    j1.append(j)
                    ja1.append(j_a)
                    jb1.append(j_b)
                    P1.append(p)
                for data in data2:
                    d, q, p, j, j_a, j_b = data['d'], data['q'], data['p'], data['j'], data['ja'], data['jb']
                    ds2.append(d)
                    dqs2.append(q)
                    j2.append(j)
                    ja2.append(j_a)
                    jb2.append(j_b)
                    P2.append(p)

                plt.subplot(321)
                plt.plot(ds1, dqs1, "+", label=f"$\Delta=${delta},$\mu=${mu:.2},$d\mu=${dmu:.2}")
                plt.subplot(322)
                plt.plot(ds2, dqs2, "+", label=f"$\Delta=${delta},$\mu=${mu:.2},$d\mu=${dmu:.2}")
                plt.subplot(323)
                if len(P1) > 0:
                    if alignLowerBranches:
                        P1 = np.array(P1)
                        P1_ = P1- P1.min()
                    else:
                        P1_=P1
                    index1, value = max(enumerate(P1), key=operator.itemgetter(1))
                    data = data1[index1]
                    n_a, n_b = ff.get_densities(mu=mu_eff, dmu=dmu_eff, delta=data["d"], dq=data["q"])
                    state = "FF" if value > p0 and not np.allclose(n_a.n, n_b.n) else "NS"
                    plt.plot(
                        ds1, P1_, "+",
                        label=f"$\Delta=${delta},$\mu=${mu:.2},$d\mu=${dmu:.2},State:{state}")
                    plt.axhline(p0, color='r', linestyle='dashed')
                    print(
                        f"Delta={delta}, dmu={dmu}, Normal Presure={p0}，FFState Presure={value}")
                    if state== "FF":
                        print(index1, data1[index1])
                plt.subplot(324)
                if len(P2) > 0:
                    if alignUpperBranches:
                        P2 = np.array(P2)
                        P2_ = P2 - P2.min()
                    else:
                        P2_ = P2
                    index2, value = max(enumerate(P2), key=operator.itemgetter(1))
                    data = data2[index2]
                    n_a, n_b = ff.get_densities(
                        mu=mu_eff, dmu=dmu_eff, delta=data["d"], dq=data["q"])
                    state = "FF" if value > p0 and not np.allclose(n_a.n, n_b.n) and data['q']>0.0001 else "NS"
                    plt.plot(
                        ds2, P2_, "+",
                        label=f"$\Delta=${delta},$\mu=${mu:.2},$d\mu=${dmu:.2},State:{state}")
                    plt.axhline(p0,color='r', linestyle='dashed')
                    print(f"Delta={delta}, dmu={dmu}, Normal Pressure={p0}，FFState Pressure={value}")
                plt.subplot(325)
                plt.plot(ds1, ja1, "+",label=f"j_a")
                plt.subplot(326)
                plt.plot(ds2, ja2, "+",label=f"j_a")
                if len(ds2) > 0:
                    plt.axvline(ds2[index2])
                    plt.axhline(0)
        
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
