from mmf_hfb import tf_completion as tf
from mmf_hfb.FuldeFerrelState import FFState
from scipy.optimize import brentq
from multiprocessing import Pool
#import warnings
#warnings.filterwarnings("ignore")
import sys
import os
import inspect
from os.path import join
import json
from json import dumps
import numpy as np
import time
import glob


class FFStateFinder():
    def __init__(self, dim=1, delta=0.1, mu=10.0, dmu=0.11,
                prefix="FFState_", timeStamp=True):
        print(f"dim={dim}\tdelta={delta}\tmu={mu}\tdmu={dmu}")
        self.dim = dim
        self.delta = delta
        self.mu = mu
        self.dmu = dmu
        if timeStamp:
            ts = time.strftime("%Y_%m_%d_%H_%M_%S.json")
            self.fileName = prefix + f"({dim}_{delta}_{mu}_{dmu})" + ts
        else:
            self.fileName = prefix


        self.ff = FFState(mu=mu, dmu=dmu, delta=delta, dim=1,
                         k_c=np.inf, fix_g=True, bStateSentinel=True)

    def _gc(self, delta, dq, update_mus=True):
        """compute the difference of a g_c[ using delta, dq] and fixed g_c"""
        mus_eff = (None, None)
        if update_mus:
            mus_eff = self.ff._get_effetive_mus(mu=self.mu,
                                                dmu=self.dmu,
                                                delta=delta,
                                                dq=dq,
                                                update_g=False)
        return self.ff.get_g(mu=mus_eff[0], dmu=mus_eff[1], 
                             delta=delta, dq=dq) - self.ff._g

    def get_mus_eff(self, delta, dq, mus_eff=None):
        """return effective mus"""
        return self.ff._get_effetive_mus(mu=self.mu, dmu=self.dmu,
                                         delta=delta, dq=dq, update_g=False)

    def get_pressure(self, delta=None, dq=0, mus_eff=None):
        """return the pressure"""
        if delta is None:
            delta = self.delta
        if mus_eff is None:
            mu_eff, dmu_eff = self.ff._get_effetive_mus(mu=self.mu, 
                                                        dmu=self.dmu, 
                                                        delta=delta, 
                                                        dq=dq, 
                                                        update_g=False)
        else:
            mu_eff, dmu_eff = mus_eff
        n_a, n_b = self.ff.get_densities(mu=mu_eff, dmu=dmu_eff, 
                                         delta=delta, dq=dq)
        energy_density = self.ff.get_energy_density(mu=mu_eff, 
                                                    dmu=dmu_eff, 
                                                    delta=delta, 
                                                    dq=dq, 
                                                    n_a=n_a, 
                                                    n_b=n_b)

        mu_a, mu_b = self.mu + self.dmu, self.mu - self.dmu
        pressure = mu_a * n_a + mu_b * n_b - energy_density
        if False:
            """Check if pressure is consistent"""
            rets = self.ff.get_ns_p_e_mus_1d(mu=self.mu, dmu=self.dmu,
                                                delta=delta, dq=dq, update_g=False)
            print(rets[3], pressure.n)
            assert np.allclose(rets[2], energy_density.n)
            assert np.allclose(rets[3], pressure.n)
        return pressure.n

    def get_current(self, delta=None, dq=0, mus_eff=None):
        """return the current"""
        if delta is None:
            delta = self.delta
        if mus_eff is None:
            mu_eff, dmu_eff = self.ff._get_effetive_mus(mu=self.mu,
                                                        dmu=self.dmu,
                                                        delta=delta,
                                                        dq=dq,
                                                        update_g=False)
        else:
            mu_eff, dmu_eff = mus_eff
        return self.ff.get_current(mu=mu_eff, dmu=dmu_eff, delta=delta, dq=dq).n

    def _get_fileName(self):
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        return join(currentdir, self.fileName)

    def SaveToFile(self, data):
        """Save states to persistent storage"""
        file = self._get_fileName()
        output = {}
        output["dim"]= self.dim
        output["delta"] = self.delta
        output["mu"] = self.mu
        output["dmu"] = self.dmu
        output["g"] = self.ff._g
        output["data"] = data
        with open(file,'w') as wf:
            json.dump(output, wf)

    def SearchFFStates(self, delta, lg=None, ug=None, 
                       ql=0, qu=0.04, dn=10,
               dx=0.0005, rtol=1e-8, raiseExcpetion=True):
        """
        ------
        lg: lower value guess
        ug: upper value guess
        ql: lower boundary
        qu: upper boundary
        dn : divisions number
        """
        def g(dq):
            return self._gc(delta=delta, dq=dq)
    
        def refine(a, b, v):
            return brentq(g, a, b)
        
        rets = []
        if lg is None and ug is None:
            dqs = np.linspace(ql, qu, dn)
            gs = [g(dq) for dq in dqs]
            g0, i0 = gs[0], 0
            if np.allclose(gs[0], 0, rtol=rtol):
                rets.append(gs[0])
                g0, i0= gs[1], 1
            for i in range(len(rets), len(gs)):
                if g0 * gs[i] < 0:
                    rets.append(refine(dqs[i0], dqs[i], dqs[i0]))
                g0, i0 = gs[i], i
        else:
            bExcept = False
            if lg is not None:
                try:
                    ret1 = brentq(g, lg - dx, lg + dx)
                    rets.append(ret1)
                    print(ret1)
                except:
                    bExcept = True
                    rets.append(None)
            else:
                rets.append(None)
            if ug is not None:
                try:
                    ret2 = brentq(g, ug - dx, ug + dx)
                    print(ret2)
                    rets.append(ret2)
                except:
                    bExcept = True
                    rets.append(None)
            else:
                rets.append(None)
            if bExcept and raiseExcpetion:
                raise ValueError('No solution found.')
            
        for _ in range(2-len(rets)):
            rets.append(None)
        return rets

    def run(self, dl=0.001, du=0.1001, ql=0, qu=0.04, dn=40):
        """
        dl: lower delta limit
        du: upper delta limit
        ql: lower dq limit
        qu: upper dq limit
        dn: delta divisions
        """
        lg, ug=None, None
        ds = np.linspace(dl, du, dn)
        rets = []
        dx = 0.001
        trails=[1, 2, 5, 0.01, 0.2, 0.5, 10, 20]
        for d in ds:
            for t in trails:
                try:
                    ret = self.SearchFFStates(delta=d, lg=lg, 
                                              ug=ug, ql=ql, 
                                              qu=qu, dn=40,
                                              dx=dx*t)
                    lg, ug = ret
                    ret.append(d)
                    rets.append(ret)
                    print(ret)
                    break
                except ValueError:
                    print("No solution, try...")
                    continue

            if t == trails[-1]:
                print("Retry without exception...")
                ret =[None, None]
                for t in trails:
                    ret0 = self.SearchFFStates(delta=d, lg=lg, ug=ug, 
                                               ql=ql, qu=qu,
                                               dn=40, dx=dx*t,
                                               raiseExcpetion=False)
                    lg, ug = ret0
                    if lg is None and ug is None:
                        continue
                    ret = ret0
                    ret.append(d)
                    rets.append(ret)
                    print(ret)
                    break
            if len(rets) > 0:
                q1, q2, d_ = rets[-1]
                if q1 is None and q2 is None:
                    print(f"Delta={d} has no solution, stop trying with other values")
                    break
            self.SaveToFile(rets)

def compute_pressure_current_worker(jsonData_file):
    """Use the FF State file to compute their current and pressure"""
    jsonData, fileName = jsonData_file
    filetokens = fileName.split("_")
    fileName = "FFState_J_P_" + "_".join(filetokens[1:]) + ".json"
    dim = jsonData['dim']
    delta = jsonData['delta']
    mu = jsonData['mu']
    dmu = jsonData['dmu']
    data = jsonData['data']
    ff = FFStateFinder(mu=mu, dmu=dmu, delta=delta, 
                       dim=dim, prefix=f"{fileName}", timeStamp=False)
    if os.path.exists(ff._get_fileName()):
        print(f"Skip file:{ff._get_fileName()}...")
        return None
    output1 = []
    output2 = []
    try:
        for item in data:
            dq1, dq2, d = item
            if dq1 is not None:
                dic = {}
                p1 = ff.get_pressure(delta=d, dq=dq1)
                ja, jb, jp, jm = ff.get_current(delta=d, dq=dq1)
                dic['d']=d
                dic['q']=dq1
                dic['p']=p
                dic['j']=jp
                dic['ja']=ja
                dic['jb']=jb
                output1.append(dic)
                print(dic)
            if dq2 is not None:
                dic = {}
                p2 = ff.get_pressure(delta=d, dq=dq2)
                ja, jb, jp, jm = ff.get_current(delta=d, dq=dq2)
                dic['d']=d
                dic['q']=dq2
                dic['p']=p2
                dic['j']=jp
                dic['ja']=ja
                dic['jb']=jb
                output2.append(dic)
                print(dic)
        output =[output1, output2]
        ff.SaveToFile(output)
    except ValueError as e:
        print(f"Parsing file: {fileName}. Error:{e}")
    except:
        print(f"Unknown error when parsing file: {fileName}")


def compute_pressure_current(root=None):
    """compute current and pressure"""
    currentdir = root
    if currentdir is None:
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    pattern = join(currentdir, "FFState*.json")
    files = files=glob.glob(pattern)

    jsonObjects = []
    for file in files:
        if os.path.exists(file):
            with open(file, 'r') as rf:
                jsonObjects.append((json.load(rf), os.path.splitext(os.path.basename(file))[0]))
    logic_cpu_count = os.cpu_count() - 1
    logic_cpu_count = 1 if logic_cpu_count < 1 else logic_cpu_count
    with Pool(logic_cpu_count) as Pools:
        Pools.map(compute_pressure_current_worker, jsonObjects)


def search_FFState_worker(dim_delta_mus):
    """worker thread"""
    dim, delta, mu, dmu=dim_delta_mus
    ff = FFStateFinder(delta=delta, dim=dim, dmu=dmu)
    ff.run(dl=0.001, du=0.2501, dn=40, ql=0, qu=1)
    

def SearchFFState(delta=0.1, mu=10, dmus=None, dim=1):
    """Search FF State"""
    if dmus is None:
        dmus = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
    logic_cpu_count = os.cpu_count() - 1
    logic_cpu_count = 1 if logic_cpu_count < 1 else logic_cpu_count
    dim_delta_mus_list = [(dim, delta, mu, dmu) for dmu in dmus]
    with Pool(logic_cpu_count) as Pools:
        Pools.map(search_FFState_worker, dim_delta_mus_list)


def SearchWithSingleConfiguration():
    dim = 1
    mu = 10
    delta = 0.2
    dmu = 1
    ff = FFStateFinder(delta=delta, dim=dim, mu=mu, dmu=dmu)
    ff.run(dl=0.3, du=.4, dn=100, ql=0, qu=0.2)

if __name__ == "__main__":
    
    ## Method: change parameters manually
    #SearchWithSingleConfiguration()
    ## Method 2: Thread pool
    #dmus = np.array([0.11, 0.12, 0.13, 0.14, 0.15, 0.16]) * 2 + 2
    #SearchFFState(delta=2.1, mu=10, dmus=dmus, dim=1)
    ## Compute the pressure and current
    compute_pressure_current()
    