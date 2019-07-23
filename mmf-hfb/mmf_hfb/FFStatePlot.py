from mmf_hfb.FFStateAgent import label_states
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


def PlotStates(filter_fun, current_dir=None, two_plot=False, print_file_name=False):
    """
        plot all state in q vs delta
        Parameters:
        -----------------
        filter_fun: a filter used to filter out files satisfy certain condition
        current_dir: the folder contains all state files used to plot
        two_plot: if true, the lower and upper branches will be ploted in
            separate subplots
        print_file_name: if True, full paths of those selected files will be printed
    """
    if current_dir is None:
        current_dir = join(
            os.path.dirname(
                os.path.abspath(
                    inspect.getfile(
                        inspect.currentframe()))), "..", "mmf_hfb", "data")
    pattern = join(current_dir, "FFState_[()d_0-9]*.json")
    files = glob.glob(pattern)
    style =['o', '-', '+']
    Cs = set()
    for file in files:
        if os.path.exists(file):
            with open(file, 'r') as rf:
                ret = json.load(rf)
                dim, mu, dmu, delta, C=(
                    ret['dim'], ret['mu_eff'], ret['dmu_eff'], ret['delta'], ret['C'])
                if filter_fun(mu=mu, dmu=dmu, delta=delta, C=C, dim=dim):
                    continue
                Cs.add(C)
                if print_file_name:
                    print(file)
                datas = ret['data']
                dqs1, dqs2, ds1, ds2 = [], [], [], []
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
                label = f"$\Delta=${delta}, $\mu$={mu}, $d\mu=${dmu:.2}, C={C:.2}"
                if len(ds1) < len(ds2):
                    if len(ds1) > 0:
                        plt.plot(ds1, dqs1, style[dim-1], label=label)
                else:
                    if len(ds2) > 0:
                        plt.plot(ds2, dqs2, style[dim-1], label=label)
                if two_plot:
                    plt.subplot(212)
                if len(ds1) < len(ds2):
                    if len(ds2) > 0:
                        plt.plot(ds2, dqs2, style[dim-1], label=label)
                else:
                    if len(ds1)> 0:
                        plt.plot(ds1, dqs1, style[dim-1], label=label)
    print(Cs)
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
    files=glob.glob(pattern)
    for file in files[0:]:
        if os.path.exists(file):
            with open(file, 'r') as rf:
                ret = json.load(rf)
                dim, mu_eff, dmu_eff, delta, C, p0=(
                    ret['dim'], ret['mu_eff'], ret['dmu_eff'],
                    ret['delta'], ret['C'], ret['p0'])
                if filter_fun(mu=mu_eff, dmu=dmu_eff, delta=delta, C=C, dim=dim):
                    continue
                if print_file_name:
                    print(file)
                label = (
                    f"$\Delta=${delta},$\mu=${float(mu_eff):.2},"
                        +f"$d\mu=${float(dmu_eff):.2}")

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
                    if len(P)==0:
                        return (0, False)
                    if align:
                        P = np.array(P)
                        P_ = P- P.min()
                    else:
                        P_=P
                    index, value = max(enumerate(P), key=operator.itemgetter(1))
                    data = Data[index]
                    n_a, n_b = data['na'], data['nb']
                    state = "FF" if value > p0 and not np.allclose(n_a, n_b) else "NS"
                    if FFState_only:
                        if state== "FF":
                            plt.plot(
                                ds, P_, "+",
                                label=f"$\Delta=${delta}, $\mu=${float(mu_eff):.2},"
                                + f" $d\mu=${float(dmu_eff):.2}, State:{state}")
                    else:
                        plt.plot(
                            ds, P_, "+",
                            label=f"$\Delta=${delta},$\mu=${float(mu_eff):.2},"
                            + f"$d\mu=${float(dmu_eff):.2},State:{state}")
                    return (index, state=="FF")
                
                plt.subplot(323)
                index1, ffs1 = plot_P(P1, data1, ds1, alignLowerBranches)
                plt.subplot(324)
                index2, ffs2 = plot_P(P2, data2, ds2, alignUpperBranches)
                plt.subplot(325)
                if FFState_only:
                    if ffs1:
                        plt.plot(ds1, ja1, "+", label=label)
                else:
                    plt.plot(ds1, ja1, "+", label=label)
               
                plt.subplot(326)
                if FFState_only:
                    if ffs2:
                        plt.plot(ds2, ja2, "+", label=label)
                else:
                    plt.plot(ds2, ja2, "+", label=label)
               
                plt.axhline(0)
                plt.subplot(321)
                if FFState_only:
                    if ffs1:
                        plt.plot(ds1, dqs1, "+", label=label)
                else:
                    plt.plot(ds1, dqs1, "+", label=label)
                plt.subplot(322)
                if FFState_only:
                    if ffs2:
                        plt.plot(ds2, dqs2, "+", label=label)
                else:
                    plt.plot(ds2, dqs2, "+", label=label)
    for i in range(1, 7):
        plt.subplot(3, 2, i)
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
        output = LabelStates(raw_data=raw_data)
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
        s=states[i]
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
    from mmf_hfb.ClassFactory import ClassFactory, FunctionalType, KernelType, Solvers
    from mmf_hfb.FFStateAgent import FFStateAgent
    output = LabelStates(raw_data=True)
    for dic in output:
        if dic['state']:
            data=dic['data']
            break
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

    delta=ground_state_data['d']
    mu_a=mu+dmu
    mu_b=mu-dmu
    mus=(mu, dmu)
    mu_a_eff=mu_eff + dmu_eff
    mu_b_eff=mu_eff - dmu_eff
    mus_eff=(mu_a_eff, mu_b_eff)

    args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=1,
        T=0, dim=3, k_c=50, verbosity=False, C=C)
    lda = ClassFactory(
        "LDA", (FFStateAgent,),
        functionalType=FunctionalType.ASLDA,
        kernelType=KernelType.HOM, args=args)
    
    # mu_a_eff_, mu_b_eff_ = lda.get_mus_eff(mus=mus, delta=delta, dq=dq)
    # mu_eff_=(mu_a_eff_ + mu_b_eff_)/2
    # dmu_eff_ = (mu_a_eff_-mu_b_eff_)/2
    # res0 = lda.get_ns_e_p(mus=mus, delta=delta, dq=dq, verbosity=False, solver=Solvers.BROYDEN1)
    # res1 = lda.get_ns_mus_e_p(mus_eff=(mu_a_eff, mu_b_eff), delta=delta, dq=dq)
    # lda.delta = delta
    # res2 = lda.get_ns_mus_e_p(mus_eff=(mu_a_eff, mu_b_eff), delta=None)
    # print(res1)
    # print(res2)

    res3 = lda.get_ns_e_p(mus=mus, delta=None, verbosity=False, fix_delta=False, solver=Solvers.BROYDEN1)
    print(res3)
