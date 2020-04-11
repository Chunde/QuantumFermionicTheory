"""
This code file provide methods to plot Fulde Ferrell state data
The three functions are used to plot results from different files:
-------------------------
PlotState(...) is used to plot the q vs delta graphs that used
    to determine a state is a FF state of normal state or superfluid
    state. It takes the files from generated by the function
    FFStateAgent.search_states(...)

PlotCurrentPressure(...) plot the current, pressure for a state used
    to determine the stable state, it takes files generated by the 
    function FFStateAgent.compute_pressure_current(...)

PlotPhaseDiagram plot the phase diagram based on the files generated by
    function FFStateAgent.compute_pressure_current(...)

"""
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
import operator
import warnings
import inspect
import json
import glob
import os
import sys
currentdir = os.path.dirname(
            os.path.abspath(
                inspect.getfile(inspect.currentframe())))
sys.path.insert(0, join(
    currentdir, '..','Projects', 'fulde_ferrell_state'))
from fulde_ferrell_state_agent import label_states

warnings.filterwarnings("ignore")


def PlotStates(
        filter_fun, current_dir=None, two_plot=False,
        plot_legend=True, print_file_name=False, ls=None):
    """
        plot all state in q vs delta
        Parameters:
        -----------------
        filter_fun: a filter used to filter out files satisfy certain condition
        current_dir: the folder contains all state files used to plot
        two_plot: if true, the lower and upper branches will be ploted in
            separate subplots
        print_file_name: if True, full paths of
        those selected files will be printed
    """
    if current_dir is None:
        current_dir = join(
            os.path.dirname(
                os.path.abspath(
                    inspect.getfile(
                        inspect.currentframe()))), "..", "mmf_hfb", "data")
    pattern = join(current_dir, "FFState_[()d_0-9]*.json")
    files = glob.glob(pattern)
    style = ['o', '+', '+']
    
    Cs = set()
    for file in files:
        if os.path.exists(file):
            with open(file, 'r') as rf:
                ret = json.load(rf)
                dim, mu, dmu, delta, C = (
                    ret['dim'], ret['mu_eff'],
                    ret['dmu_eff'], ret['delta'], ret['C'])
                if filter_fun(mu=mu, dmu=dmu, delta=delta, C=C, dim=dim):
                    continue
                Cs.add(C)
                if print_file_name:
                    print(file)
                datas = ret['data']
                if ls is None:
                    ls = style[dim - 1]
                dqs1, dqs2, ds1, ds2 = [], [], [], []
                for data in datas:
                    dq1, dq2, d = data
                    if dq1 is not None:
                        dqs1.append(dq1)
                        ds1.append(d)
                    if dq2 is not None:
                        dqs2.append(dq2)
                        ds2.append(d)
                if len(ds1) < 2 and len(ds2) < 2:
                    continue
                if two_plot:
                    plt.subplot(211)
                label = (
                    f"$\Delta=${delta}, $\mu$={mu},"
                    + f" $d\mu=${dmu:.3}, C={C:.2}")
                if len(ds1) < len(ds2):
                    if len(ds1) > 0:
                        plt.plot(ds1, dqs1, ls, label=label)
                else:
                    if len(ds2) > 0:
                        plt.plot(ds2, dqs2, ls, label=label)
                if two_plot:
                    plt.subplot(212)
                if len(ds1) < len(ds2):
                    if len(ds2) > 0:
                        plt.plot(ds2, dqs2, ls, label=label)
                else:
                    if len(ds1) > 0:
                        plt.plot(ds1, dqs1, ls, label=label)
    if two_plot:
        plt.subplot(211)
        plt.xlabel(f"$\Delta$")
        plt.ylabel(f"$\delta q$")
        plt.title(f"Lower Branch")
        if plot_legend:
            plt.legend()
        plt.subplot(212)
        plt.xlabel(f"$\Delta$")
        plt.ylabel(f"$\delta q$")
        plt.title(f"Upper Branch")
        if plot_legend:
            plt.legend()
    else:
        plt.xlabel(f"$\Delta$")
        plt.ylabel(f"$\delta q$")
        if plot_legend:
            plt.legend()
    plt.show()


def PlotCurrentPressure(
    filter_fun, current_dir=None, alignLowerBranches=False,
        alignUpperBranches=False, showLegend=False,
        FFState_only=True, print_file_name=False):
    if current_dir is None:
        current_dir = join(
            os.path.dirname(
                os.path.abspath(
                    inspect.getfile(
                        inspect.currentframe()))), "..", "mmf_hfb", "data")
    pattern = join(current_dir, "FFState_J_P[()d_0-9]*")
    files = glob.glob(pattern)
    for file in files[0:]:
        if os.path.exists(file):
            with open(file, 'r') as rf:
                ret = json.load(rf)
                dim, mu_eff, dmu_eff, delta, C, p0 = (
                    ret['dim'], ret['mu_eff'], ret['dmu_eff'],
                    ret['delta'], ret['C'], ret['p0'])
                if filter_fun(
                    mu=mu_eff, dmu=dmu_eff,
                    delta=delta, C=C, dim=dim):
                    continue
                if print_file_name:
                    print(file)
                label = (
                        f"$\Delta=${delta},$\mu=${float(mu_eff):.2},"
                        + f"$d\mu=${float(dmu_eff):.2}")

                data1, data2 = ret['data']
                dqs1, dqs2, ds1, ds2, j1, j2, ja1, ja2, jb1, jb2, P1, P2 =(
                    [], [], [], [], [], [], [], [], [], [], [], [])
                for data in data1:
                    d, q, p, j, j_a, j_b = (
                        data['d'], data['q'], data['p'],
                        data['j'], data['ja'], data['jb'])
                    ds1.append(d)
                    dqs1.append(q)
                    j1.append(j)
                    ja1.append(j_a)
                    jb1.append(j_b)
                    P1.append(p)
                for data in data2:
                    d, q, p, j, j_a, j_b = (
                        data['d'], data['q'], data['p'],
                        data['j'], data['ja'], data['jb'])
                    ds2.append(d)
                    dqs2.append(q)
                    j2.append(j)
                    ja2.append(j_a)
                    jb2.append(j_b)
                    P2.append(p)

                def plot_P(P, Data, ds, align):
                    if len(P) == 0:
                        return (0, False)
                    if align:
                        P = np.array(P)
                        P_ = P - P.min()
                    else:
                        P_ = P
                    index, value = max(
                        enumerate(P), key=operator.itemgetter(1))
                    data = Data[index]
                    n_a, n_b = data['na'], data['nb']
                    state = (
                        "FF" if value > p0 and not np.allclose(
                            n_a, n_b) else "NS")
                    if FFState_only:
                        if state == "FF":
                            plt.plot(
                                ds, P_, "+",
                                label=f"$\Delta=${delta},"
                                + f" $\mu=${float(mu_eff):.2},"
                                + f" $d\mu=${float(dmu_eff):.2}, State:{state}")
                    else:
                        plt.plot(
                            ds, P_, "+",
                            label=f"$\Delta=${delta},$\mu=${float(mu_eff):.2},"
                            + f"$d\mu=${float(dmu_eff):.2},State:{state}")
                    plt.axhline(p0, linestyle='dashed')
                    return (index, state == "FF")
                
                ffs1, ffs2 = False, False
                if len(ds1) > 2:
                    plt.subplot(323)
                    _, ffs1 = plot_P(P1, data1, ds1, alignLowerBranches)

                    plt.subplot(325)
                    if FFState_only:
                        if ffs1:
                            plt.plot(ds1, ja1, "+", label=label)
                    else:
                        plt.plot(ds1, ja1, "+", label=label)
                    plt.subplot(321)
                    if FFState_only:
                        if ffs1:
                            plt.plot(ds1, dqs1, "+", label=label)
                    else:
                        plt.plot(ds1, dqs1, "+", label=label)

                if len(ds2) > 2:
                    plt.subplot(324)
                    _, ffs2 = plot_P(P2, data2, ds2, alignUpperBranches)
                    plt.subplot(326)
                    if FFState_only:
                        if ffs2:
                            plt.plot(ds2, ja2, "+", label=label)
                    else:
                        plt.plot(ds2, ja2, "+", label=label)
                    plt.axhline(0)
                    plt.subplot(322)
                    if FFState_only:
                        if ffs2:
                            plt.plot(ds2, dqs2, "+", label=label)
                    else:
                        plt.plot(ds2, dqs2, "+", label=label)
    for i in range(1, 7):
        plt.subplot(3, 2, i)
        if showLegend:
            if (len(ds2) > 1 and i % 2 == 0) or (len(ds1) > 1 and i % 2 == 1):
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
    plt.show()


def PlotPhaseDiagram(output=None, raw_data=False):
    """
    plot the phase diagram
    Para:
    ------------------
    output: the list the come from the LableStates in FFStateAgent
    raw_data: if True, the return result will include the original
        data from each files(in json format)
    """
    if output is None:
        output = label_states(raw_data=raw_data)
    xs, xs2, ys, ys2, ys3, ys4, states = [], [], [], [], [], [], []
    for dic in output:
        n = dic['na'] + dic['nb']
        mu, dmu, delta = dic['mu'], dic['dmu'], dic['delta']
        k_F = (2.0*mu)**0.5
        dn = dic['na'] - dic['nb']
        ai = dic['ai']
        xs.append(-ai/k_F)
        ys.append(dn/n)  # polarization
        xs2.append(delta)
        ys2.append(dmu/delta)
        ys3.append(dic['dmu_eff'])
        ys4.append(dic['dmu_eff']/delta)
        states.append(dic['state'])
    colors, area = [], []
    for i in range(len(states)):
        s = states[i]
        if s:
            colors.append('red')
            area.append(15)
        else:
            colors.append('blue')
            area.append(1)
    plt.subplot(221)
    plt.scatter(xs, ys, s=area, c=colors)
    plt.ylabel(r"$\delta n/n$", fontsize=16)
    plt.xlabel(r"$-1/ak_F$", fontsize=16)
    plt.subplot(222)
    plt.scatter(xs, ys2, s=area, c=colors)
    plt.ylabel(r"$\delta\mu/\Delta$", fontsize=16)
    plt.xlabel(r"$-1/ak_F$", fontsize=16)
    plt.subplot(223)
    plt.scatter(xs2, ys3, s=area, c=colors)
    plt.ylabel(r"$\delta\mu_{eff}$", fontsize=16)
    plt.xlabel(r"$\Delta$", fontsize=16)
    plt.subplot(224)
    plt.scatter(xs, ys4, s=area, c=colors)
    plt.ylabel(r"$\delta\mu_{eff}/\Delta$", fontsize=16)
    plt.xlabel(r"$-1/ak_F$", fontsize=16)
    plt.show()


if __name__ == "__main__":
    def filter_state(mu, dmu, delta, C, dim):
        if dim != 3:
            return True

        if delta != .5:
            return True

        if not np.allclose(dmu, 0.33, rtol=0.01):
            return True
        print(dmu)
        return False
    PlotCurrentPressure(
        filter_fun=filter_state, showLegend=True,
        FFState_only=False, print_file_name=True)
