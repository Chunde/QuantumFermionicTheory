from mmf_hfb.FuldeFerrelState import FFState
from mmf_hfb.FFStateFinder import FFStateFinder
from mmf_hfb.ParallelHelper import PoolHelper
from multiprocessing import Pool
import os
import operator
import inspect
from os.path import join
import json
import time
import glob
import numpy as np


class FFStateHelper(object):

    def compute_pressure_current_worker(jsonData_file):
        """Use the FF State file to compute their current and pressure"""
        jsonData, fileName = jsonData_file
        filetokens = fileName.split("_")
        output_fileName = "FFState_J_P_" + "_".join(filetokens[1:]) + ".json"
        dim = jsonData['dim']
        delta = jsonData['delta']
        mu = jsonData['mu']
        dmu = jsonData['dmu']
        data = jsonData['data']
        ff = FFStateFinder(
            mu=mu, dmu=dmu, delta=delta, g=jsonData['g'],
            dim=dim, prefix=f"{output_fileName}", timeStamp=False)
        if os.path.exists(ff._get_fileName()):
            return None
        print(f"Processing {ff._get_fileName()}")
        output1 = []
        output2 = []
        try:
            for item in data:
                dq1, dq2, d = item
                if dq1 is not None:
                    dic = {}
                    p1 = ff.get_pressure(delta=d, dq=dq1).n
                    ja, jb, jp, _ = ff.get_current(delta=d, dq=dq1)
                    ns = ff.get_densities(delta=d, dq=dq1)
                    dic['na']=ns[0].n
                    dic['nb']=ns[1].n
                    dic['d']=d
                    dic['q']=dq1
                    dic['p']=p1
                    dic['j']=jp.n
                    dic['ja']=ja.n
                    dic['jb']=jb.n
                    output1.append(dic)
                    print(dic)
                if dq2 is not None:
                    dic = {}
                    p2 = ff.get_pressure(delta=d, dq=dq2).n
                    ja, jb, jp, _ = ff.get_current(delta=d, dq=dq2)
                    ns = ff.get_densities(delta=d, dq=dq2)
                    dic['na']=ns[0].n
                    dic['nb']=ns[1].n
                    dic['d']=d
                    dic['q']=dq2
                    dic['p']=p2
                    dic['j']=jp.n
                    dic['ja']=ja.n
                    dic['jb']=jb.n
                    output2.append(dic)
                    print(dic)
            output =[output1, output2]
            ff.SaveToFile(output)
        except ValueError as e:
            print(f"Parsing file: {fileName}. Error:{e}")
        
    def FindFFState(filter, currentdir=None, lastStates=None, verbose=False):
        if currentdir is None:
            currentdir = join(
                os.path.dirname(
                    os.path.abspath(
                        inspect.getfile(
                            inspect.currentframe()))), "..", "mmf_hfb", "data")
        output = []
        fileSet = []
        if lastStates is not None:
            print("Incremental search...")
            output, fileSet = lastStates
        pattern = join(currentdir, "FFState_J_P[()d_0-9]*")
        files=glob.glob(pattern)
        for file in files[0:]:
            if os.path.exists(file):
                with open(file, 'r') as rf:
                    ret = json.load(rf)
                    dim, mu, dmu, delta, g=ret['dim'], ret['mu'], ret['dmu'], ret['delta'], ret['g']
                    if filter(mu=mu, dmu=dmu, delta=delta, g=g, dim=dim):
                        continue
                    if file in fileSet:
                        print(file)
                        continue
                    fileSet.append(file)
                    k_c = None
                    if 'k_c' in ret:
                        k_c = ret['k_c']
                    if verbose:
                        print(file)
                    ff = FFState(
                        mu=mu, dmu=dmu, delta=delta,
                        dim=dim, g=g, k_c=k_c, fix_g=True)
                    a_inv = ff.get_a_inv(mu=mu, dmu=0, delta=delta).n
                    mu_eff, dmu_eff = mu, dmu
                    ns = ff.get_densities(mu=mu_eff, dmu=0)
                    if verbose:
                        print(ns)
                    p0 = ff.get_pressure(
                        mu=None, dmu=None, mu_eff=mu_eff, dmu_eff=dmu_eff, delta=0)
                    # p1 = ff.get_pressure(
                    #   mu=None, dmu=None, mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta)
                    if verbose:
                        print(f"p0={p0},p1={p1}")
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
                        n_a, n_b = ff.get_densities(mu=mu_eff, dmu=dmu_eff, delta=data["d"], dq=data["q"])
                        if verbose:
                            print(f"na={n_a.n}, nb={n_b.n}, PF={value}, PN={p0.n}")
                        if value > p0 and (not np.allclose(  #  [Check] was wrong
                                n_a.n, n_b.n, rtol=1e-9) and data["q"]>0.0001 and data["d"]>0.001):
                            bFFState = True
                    #if len(P2) > 0:
                    #    index2, value = max(enumerate(P2), key=operator.itemgetter(1))
                    #    data = data2[index2]
                    #    n_a, n_b = ff.get_densities(mu=mu_eff, dmu=dmu_eff, delta=data["d"], dq=data["q"])
                    #    if verbose:
                    #        print(f"na={n_a.n}, nb={n_b.n}, PF={value}, PN={p0.n}")

                    #    if value > p0 and not np.allclose(n_a.n, n_b.n, rtol=1e-9) and data["q"]>0.0001 and data["d"]>0.001:
                    #        bFFState = True
                    if bFFState and verbose:
                        print(f"FFState: {bFFState} |<-------------")
                    dic = dict(
                        mu=mu, dmu=dmu, np=sum(ns).n, na=n_a.n,
                        nb=n_b.n, ai=a_inv, g=g, delta=delta, state=bFFState)
                    if verbose:
                        print(dic)
                    output.append(dic)
                    if verbose:
                        print("-----------------------------------")
        return (output, fileSet)

    def compute_pressure_current(root=None):
        """compute current and pressure"""
        currentdir = root
        if currentdir is None:
            currentdir = os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe())))
        pattern = join(currentdir, "data","FFState_[()_0-9]*.json")
        files = files=glob.glob(pattern)

        jsonObjects = []
        for file in files:
            if os.path.exists(file):
                with open(file, 'r') as rf:
                    jsonObjects.append(
                        (json.load(rf), os.path.splitext(os.path.basename(file))[0]))
        logic_cpu_count = os.cpu_count() - 1
        logic_cpu_count = 1 if logic_cpu_count < 1 else logic_cpu_count
        if False:  # Debugging
            for item in jsonObjects:
                FFStateHelper.compute_pressure_current_worker(item)
        with Pool(logic_cpu_count) as Pools:
            Pools.map(FFStateHelper.compute_pressure_current_worker, jsonObjects)

    def search_FFState_worker(dim_delta_mus):
        """worker thread"""
        dim, delta, _, dmu=dim_delta_mus
        ff = FFStateFinder(delta=delta, dim=dim, dmu=dmu)
        ff.run(dl=0.001, du=0.2501, dn=60, ql=0, qu=1)
        
    def SearchFFState(delta=0.1, mu=10, dmus=None, dim=1):
        """Search FF State"""
        if dmus is None:
            dmus = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
        logic_cpu_count = os.cpu_count() - 1
        logic_cpu_count = 1 if logic_cpu_count < 1 else logic_cpu_count
        dim_delta_mus_list = [(dim, delta, mu, dmu) for dmu in dmus]
        with Pool(logic_cpu_count) as Pools:
            Pools.map(FFStateHelper.search_FFState_worker, dim_delta_mus_list)

    def search_single_configuration_1d():
        dim = 1
        mu = 10
        delta = 0.21  # when set g, delta is useless
        dmu = 0.5
        g = None# -10
        ff = FFStateFinder(delta=delta, dim=dim, mu=mu, dmu=dmu, g=g)
        ff.run(dl=0.001, du=0.5, dn=100, ql=0, qu=0.04)

    def search_single_configuration_3d():
        e_F = 10
        mu0 = 0.59060550703283853378393810185221521748413488992993*e_F
        # delta0 = 0.68640205206984016444108204356564421137062514068346*e_F

        mu = mu0 # ~6
        delta = 1
        dmu = 0.2
        dl = 0.0001
        du = delta
        ff = FFStateFinder(delta=delta, dim=3, mu=mu, dmu=dmu)

        def q_upper_lim():
            return 1
            print("Finding q upper limit...")
            dqs = np.linspace(0, 2*delta, 50)
            gs = [ff._gc(mu=mu, dmu=dmu, delta=0.001, dq=dq) for dq in dqs]
            g0 = gs[-1]
            for i in reversed(range(len(dqs))):
                if gs[i]*g0 < 0:
                    print(dqs[i]*2)
                    return dqs[i]*2
            return delta
        ql = 0
        qu = q_upper_lim()
        print(
            f"Start search:mu={mu}, dmu={dmu}, delta={delta},lower delta={dl},"+
            f" upper delta={du}, lower dq={ql}, upper dq={qu}")
        ff.run(dl=dl, du=du, dn=100, ql=ql, qu=qu)
    
    def search_single_configuration_2d():
        e_F = 10
        mu0 = 0.5*e_F
        delta0 = 2.0**0.5*e_F  # ~14
        mu = mu0 # =5
        delta = 3
        dmu = 4.5
        dl = 0.0001
        du = 2*dmu
        ff = FFStateFinder(delta=delta, dim=2, mu=mu, dmu=dmu)

        def q_upper_lim():
            # return delta
            print("Finding q upper limit...")
            dqs = np.linspace(0, 2*delta, 50)
            gs = [ff._gc(mu=mu, dmu=dmu, delta=0.001, dq=dq) for dq in dqs]
            g0 = gs[-1]
            for i in reversed(range(len(dqs))):
                if gs[i] * g0 < 0:
                    print(dqs[i] * 2)
                    return dqs[i] * 2
            return delta
        ql = 0
        qu = q_upper_lim()
        print(
            f"Start search:mu={mu}, dmu={dmu}, delta={delta},"+
            f"lower delta={dl}, upper delta={du}, lower dq={ql}, upper dq={qu}")
        ff.run(dl=dl, du=du, dn=100, ql=ql, qu=qu)

    def sort_file(files=None, abs_file=False):
        # files = ["FFState_(3d_0.5_5.906055070328385_0.55)2019_05_22_21_33_30.json"]
        currentdir = join(os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))), "data")
        if files is None:
            pattern = join(currentdir, "FFState_[()d_0-9]*.json")
            files = glob.glob(pattern)
            abs_file = True

        if len(files) < 1:
            print("At least two files input")
            return
        for file in files:
            if not abs_file:
                file = join(currentdir, file)
            if os.path.exists(file):
                with open(file,'r+') as rf:
                    ret = json.load(rf)
                    data = FFStateFinder.sort_data(ret['data'])
                    ret['data']=data
                    rf.seek(0)
                    json.dump(ret, rf)
                    rf.truncate()  # truncate the rest of the old file.
                    print(f"{file} saved")

    def merge_files():
        files = ["FFState_(3d_2.4_10_2.85)2019_05_04_23_22_48.json", "FFState_(3d_2.4_10_2.85)2019_05_04_23_23_01.json"]
        if len(files) < 1:
            print("At least two files input")
            return
        currentdir = join(
            os.path.dirname(
                os.path.abspath(
                    inspect.getfile(inspect.currentframe()))), "data")
        ts = time.strftime("%Y_%m_%d_%H_%M_%S.json")
        
        datas = []
        for file in files:
            file = join(currentdir, file)
            if os.path.exists(file):
                with open(file, 'r') as rf:
                    datas.append(json.load(rf))
        if len(datas) < 1:
            return
        filetokens = files[0].split(")")
        output_fileName = "_".join([filetokens[0] + ")", ts])

        output = datas[0]
        for i in range(1, len(datas)):
            output["data"].extend(datas[i]["data"])
        with open(join(currentdir, output_fileName),'w') as wf:
                json.dump(output, wf)


def check_FF_State():
    def filter(mu, dmu, delta, g, dim):
        if dim != 3:
            return True
        #return False
        #if g != -2.8:
        #    return True
        #return False
        #if g != -3.2:
        #    return True
        if delta != 0.5:
            return True
        if dmu != 0.6:
            return True
        return False
    output = FFStateHelper.FindFFState(filter)


def diagram_worker(mu_dmu_delta_dim):
    mu, dmu, delta, dim = mu_dmu_delta_dim
    dl = 0.0001
    du = delta
    ff = FFStateFinder(delta=delta, dim=dim, mu=mu, dmu=dmu)

    def q_upper_lim():
        return 1
        print("Finding q upper limit...")
        dqs = np.linspace(0, 2*delta, 50)
        gs = [ff._gc(mu=mu, dmu=dmu, delta=0.001, dq=dq) for dq in dqs]
        g0 = gs[-1]
        for i in reversed(range(len(dqs))):
            if gs[i] * g0 < 0:
                print(dqs[i] * 2)
                return dqs[i] * 2
        return delta
    ql = 0
    qu = q_upper_lim()
    print(f"Start search:mu={mu}, dmu={dmu}, delta={delta},lower delta={dl}, upper delta={du}, lower dq={ql}, upper dq={qu}")
    ff.run(dl=dl, du=du, dn=100, ql=ql, qu=qu)


def ConstructDiagram(dim=3, delta=None):
    e_F=10
    mu0=0.59060550703283853378393810185221521748413488992993*e_F
    delta0=0.68640205206984016444108204356564421137062514068346*e_F

    mu=mu0 # ~6
    if delta is None:
        deltas = np.linspace(0, e_F, 41)[1:]
    else:
        deltas = [delta]

    for delta in deltas:
        dmus=np.linspace(0.5*delta, 0.8*delta, max(20 + 1, int(delta/0.25)))[1:]
        dmus=dmus + (dmus[1] - dmus[0])/2.0
        args=[(mu, dmu, delta, dim) for dmu in dmus]
        PoolHelper.run(diagram_worker, args)

            
if __name__ == "__main__":
    #check_FF_State()
    ## Sort file with discontinuity
    #FFStateHelper.sort_file()
    ## Merge files with the same configuration
    # FFStateHelper.merge_files()
    ## Method: change parameters manually
    # FFStateHelper.search_single_configuration_1d()
    #FFStateHelper.search_single_configuration_2d()
    #FFStateHelper.search_single_configuration_3d()
    ## Method 2: Thread pool
    #dmus = np.array([0.11, 0.12, 0.13, 0.14, 0.15, 0.16]) * 2 + 2
    # FFStateHelper.SearchFFState(delta=2.1, mu=10, dmus=dmus, dim=1)
    ## Compute the pressure and current
    ConstructDiagram(delta=0.2)
    FFStateHelper.compute_pressure_current()
 