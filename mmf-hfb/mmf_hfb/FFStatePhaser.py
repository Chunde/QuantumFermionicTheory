import os
import numpy as np
import inspect
from os.path import join
import json
import glob
import operator
import warnings
warnings.filterwarnings("ignore")
from json import dumps
from mmf_hfb import tf_completion as tf
from mmf_hfb.FuldeFerrelState import FFState
from mmf_hfb.FFStateFinder import FFStateFinder

import os
import inspect
from os.path import join
currentdir = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), "data")


def filter(mu, dmu, delta, g, dim):
    if dim != 3:
        return True
    #return False
    #if g != -2.8:
    #    return True
    #return False
    #if g != -3.2:
    #    return True
    #if delta != 3.5:
    #    return True
    #if dmu != 0.21:
    #    return True
    return False


def PlotCurrentPressure():
    output = []
    pattern = join(currentdir, "FFState_J_P[()d_0-9]*")
    files=glob.glob(pattern)
    for file in files[0:]:
        if os.path.exists(file):
            with open(file,'r') as rf:
                ret = json.load(rf)
                dim, mu, dmu, delta, g=ret['dim'], ret['mu'], ret['dmu'], ret['delta'],ret['g']
                if filter(mu=mu, dmu=dmu, delta=delta, g=g, dim=dim):
                    continue 
                k_c = None
                if 'k_c' in ret:
                    k_c = ret['k_c']
                print(file)
                ff = FFState(mu=mu, dmu=dmu, delta=delta, dim=dim, g=g, k_c=k_c, fix_g=True)
                mu_eff, dmu_eff = mu, dmu
                ns = ff.get_densities(mu=mu_eff, dmu=dmu_eff)
                print(ns)
                p0 = ff.get_pressure(mu=None, dmu=None, mu_eff=mu_eff, dmu_eff=dmu_eff, delta=0)
                p1 = ff.get_pressure(mu=None, dmu=None, mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta)
                print(f"p0={p0},p1={p1}")
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
                bFFState = False
                if len(P1) > 0:
                    index1, value = max(enumerate(P1), key=operator.itemgetter(1))
                    data = data1[index1]
                    n_a, n_b = ff.get_densities(mu=mu_eff, dmu=dmu_eff, delta=data["d"], dq=data["q"])
                    print(f"na={n_a.n}, nb={n_b.n}, PF={value}, PN={p0.n}")
                    if value > p0 and not np.allclose(n_a.n, n_b.n):
                        bFFState = True
                if len(P2) > 0:
                    index2, value = max(enumerate(P2), key=operator.itemgetter(1))
                    data = data2[index2]
                    n_a, n_b = ff.get_densities(mu=mu_eff, dmu=dmu_eff, delta=data["d"], dq=data["q"])
                    print(f"na={n_a.n}, nb={n_b.n}, PF={value}, PN={p0.n}")
                    if value > p0 and not np.allclose(n_a.n, n_b.n):
                        bFFState = True
                if bFFState:
                    print(f"FFState: {bFFState} |<-------------")
                dic = dict(mu=mu, dmu=dmu, na=ns[0].n, nb=ns[1].n,  g=g, delta=delta, state=bFFState)
                print(dic)
                output.append(dic)
                print("-----------------------------------")
    return output

if __name__ == "__main__":
    PlotCurrentPressure()