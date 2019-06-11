from mmf_hfb.FuldeFerrelState import FFState
from scipy.optimize import brentq
#import warnings
#warnings.filterwarnings("ignore")
import os
import inspect
from os.path import join
import json
import numpy as np
import time


class FFStateFinder():
    def __init__(self, dim=1, delta=0.1, mu=10.0, dmu=0, g=None,
                   k_c=None, prefix="FFState_", timeStamp=True):
        self.dim = dim
        self.delta = delta
        self.mu_eff = mu  # mu_eff for 1d
        self.dmu_eff = dmu  # dmu_eff for 1d
        if timeStamp:
            ts = time.strftime("%Y_%m_%d_%H_%M_%S.json")
            self.fileName = prefix + f"({dim}d_{delta:.2f}_{mu:.2f}_{dmu:.2f})" + ts
        else:
            self.fileName = prefix
        if k_c is None:
            if dim ==1:
                k_c = np.inf
            elif dim == 2:
                k_c = 100
            else:
                 k_c = 50
        self.k_c = k_c
        self.ff = FFState(mu=mu, dmu=0, delta=delta, g=g, dim=dim,
                         k_c=k_c, fix_g=True, bStateSentinel=True)
        print(f"dim={dim}\tdelta={delta}\tmu={mu}\tdmu={dmu}\tg={self.ff.g}\tk_c={k_c}")

    def _gc(self, delta, mu=None, dmu=None, dq=0, update_mus=True):
        """compute the difference of a g_c[ using delta, dq] and fixed g_c"""
        if update_mus:
            mu, dmu = self.ff._get_effective_mus(mu=self.mu_eff, dmu=self.dmu_eff,
                                                delta=delta, dq=dq,
                                                update_g=False)
        return self.ff.get_g(mu=mu, dmu=dmu,
                             delta=delta, dq=dq) - self.ff._g 

    def get_mus_eff(self, delta, q=0, dq=0, mus_eff=None):
        """return effective mus"""
        return self.ff._get_effective_mus(mu=self.mu_eff, dmu=self.dmu_eff,
                                          delta=delta, q=q, dq=dq, update_g=False)

    def get_densities(self, mus_eff=None, delta=None, q=0, dq=0):
        """return the pressure"""
        
        #if delta is None:
        #    delta = self.delta
        if mus_eff is None:
            mu_eff, dmu_eff = self.mu_eff, self.dmu_eff
        else:
            mu_eff, dmu_eff = mus_eff
        return self.ff.get_densities(mu=mu_eff, dmu=dmu_eff, delta=delta, q=q, dq=dq)

    def get_pressure(self, mus_eff=None, delta=None, q=0, dq=0):
        """return the pressure"""
        
        if delta is None:
            delta = self.delta
        if mus_eff is None:
            mu_eff, dmu_eff = self.mu_eff, self.dmu_eff
        else:
            mu_eff, dmu_eff = mus_eff
        return self.ff.get_pressure(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta, q=q, dq=dq, use_kappa=False)
        
        n_a, n_b = self.ff.get_densities(mu=mu_eff, dmu=dmu_eff,
                                         delta=delta, dq=dq)
        energy_density = self.ff.get_energy_density(mu=mu_eff, dmu=dmu_eff,
                                                    delta=delta, dq=dq,
                                                    n_a=n_a, n_b=n_b)
        mu, dmu = self.ff._get_bare_mus(mu_eff=mu_eff, dmu_eff=dmu_eff,
                                        q=q, dq=dq, delta=delta)
        mu_a, mu_b = mu + dmu, mu - dmu
        pressure = mu_a * n_a + mu_b * n_b - energy_density
        if False:
            """Check if pressure is consistent"""
            rets = self.ff.get_ns_p_e_mus_1d(
                mu=self.mu_eff, dmu=self.dmu_eff,
                delta=delta, q=q, dq=dq, update_g=False)
            print(rets[3], pressure.n)
            assert np.allclose(rets[2], energy_density.n)
            assert np.allclose(rets[3], pressure.n)
        return pressure.n

    def get_current(self, delta=None, q=0, dq=0, mus_eff=None):
        """return the current"""
        if delta is None:
            delta = self.delta
        if mus_eff is None:
            mu_eff, dmu_eff = self.ff._get_effective_mus(
                mu=self.mu_eff, dmu=self.dmu_eff, delta=delta, q=q, dq=dq, update_g=False)
        else:
            mu_eff, dmu_eff = mus_eff
        return self.ff.get_current(mu=mu_eff, dmu=dmu_eff, delta=delta, q=q, dq=dq)

    def _get_fileName(self):
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        return join(currentdir, "data", self.fileName)

    def SaveToFile(self, data):
        """Save states to persistent storage"""
        file = self._get_fileName()
        output = {}
        output["dim"]= self.dim
        output["delta"] = self.delta
        output["mu"] = self.mu_eff
        output["dmu"] = self.dmu_eff
        output["g"] = self.ff._g
        output["k_c"] = self.k_c
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
                    print(rets[-1])
                g0, i0 = gs[i], i
        else:
            bExcept = False
            if lg is not None:
                try:
                    ret1 = brentq(g, lg - dx, lg + dx)
                    rets.append(ret1)
                except:
                    bExcept = True
                    rets.append(None)
            else:
                rets.append(None)
            if ug is not None:
                try:
                    ret2 = brentq(g, ug - dx, ug + dx)
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

    def sort_data(rets):
        """
            sort data to make lines as long and smooth as possible
        """
        rets_ = []
        def p(ret):
            v1, v2, _ = ret
            if v1 is None and v2 is None:
                return 0
            if v1 is None or v2 is None:
                return 1
            return 2

        for ret in rets:
            if p(ret) > 0:
                rets_.append(ret)
        rets = rets_
        bflip = False
        for i in range(1, len(rets)):
            v1, v2, _ = rets[i]
            v1_, v2_, _ = rets[i-1]
            bflip = False
            p1 = p(rets[i])
            p2 = p(rets[i-1])
            if p1 > p2:
                if v1_ is None:
                    if abs(v1 - v2_) < abs(v2 - v2_):
                        bflip = True
                        print("flipping data")
                if v2_ is None:
                    if abs(v1 - v1_) > abs(v2 - v1_):
                        bflip = True
                        print("flipping data")
            elif p1 < p2:
                if v1 is None:
                    if abs(v1_ - v2) < abs(v2_ - v2):
                        bflip = True
                        print("flipping data")
                if v2 is None:
                    if abs(v1_ - v1) > abs(v2_ - v1):
                        bflip = True
                        print("flipping data")
            elif p1 == p2:
                if (v1 is None) !=(  v1_ is None) or (v2 is None) != (v2_ is None):
                    bflip=True
                    print("flipping data")
            if bflip:
                rets[i] = [rets[i][1], rets[i][0],rets[i][2]]
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

        dx0 = 0.001
        dx = dx0
        trails=[1, 2, 5, 0.01, 0.2, 0.5, 10, 20, 0]

        def do_search():
            nonlocal lg
            nonlocal ug
            ret = self.SearchFFStates(delta=d, lg=lg, ug=ug, ql=ql, qu=qu, dn=40, dx=dx)
            lg, ug = ret
            ret.append(d)
            rets.append(ret)
            print(ret)

        for d in ds:
            retry = True
            print(dx)
            if dx != dx0 and dx !=0:
                try:
                    do_search()
                    retry = False
                    continue
                except ValueError:
                    print(f"retry[{dx}]...")
            if retry:
                for t in trails:
                    dx = dx0 * t
                    try:
                        do_search()
                        break
                    except ValueError:
                        print(f"No solution[{dx}], try...")
                        continue

                if t == trails[-1]:
                    print("Retry without exception...")
                    ret =[None, None]
                    for t in trails:
                        dx = dx0 * t
                        ret0 = self.SearchFFStates(delta=d, lg=lg, ug=ug, 
                                                    ql=ql, qu=qu,
                                                    dn=40, dx=dx,
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
                    print(f"Delta={d} has no solution, trying with other deltas")
                    del rets[-1]
                    lg, ug=None, None
                    if list(ds).index(d) > 10: # this is arbitrary
                        break
                    continue
            
            self.SaveToFile(rets)
        rets = FFStateFinder.sort_data(rets)
        self.SaveToFile(rets)
