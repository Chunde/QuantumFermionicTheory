from mmf_hfb.DataHelper import ff_state_sort_data
from mmf_hfb.ParallelHelper import PoolHelper
from mmf_hfb.ClassFactory import ClassFactory
from scipy.optimize import brentq
from os.path import join
import numpy as np
import inspect
import time
import json
import os


class FFStateAgent(object):
    """
    An agent class used for searching FF states
    """

    def __init__(
            self, mu_eff, dmu_eff, delta, dim, verbosity=True,
            prefix="FFState_", timeStamp=True, **args):
        assert dmu_eff < delta
        self.delta = delta
        self.mu_eff = mu_eff
        self.dmu_eff = dmu_eff
        self.verbosity=verbosity
        if timeStamp:
            ts = time.strftime("%Y_%m_%d_%H_%M_%S.json")
            self.fileName = (
                prefix + f"({dim}d_{delta:.2f}_{mu_eff:.2f}_{dmu_eff:.2f})" + ts)
        else:
            self.fileName = prefix

    def _get_fileName(self):
        currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
        return join(currentdir, "data", self.fileName)

    def SaveToFile(self, data):
        """Save states to persistent storage"""
        file = self._get_fileName()
        output = {}
        output["dim"]= self.dim
        output["delta"] = self.delta
        output["mu"] = self.mu_eff
        output["dmu"] = self.dmu_eff
        output["C"] = self.C
        output["k_c"] = self.k_c
        output["data"] = data
        with open(file, 'w') as wf:
            json.dump(output, wf)

    def SearchFFStates(
            self, delta, mu_eff, dmu_eff,
            guess_lower=None, guess_upper=None, q_lower=0, q_upper=0.04,
            q_N=10, dx=0.0005, rtol=1e-8, raiseExcpetion=True):
        """
        Paras:
        -------------
        delta: a fixed delta value for which one or two q values
            will be found to satisfy the gap equation.
        guess_lower{upper}  : the lower{upper} boundary guess for
            the solution to q.
        q_lower{upper}  : the lower{upper} boundary to redo search
            if not solution is found for given guessed boundary.
        """

        def f(dq):
            return self._get_C(
                mus_eff=(mu_eff + dmu_eff, mu_eff - dmu_eff),
                    delta=delta, dq=dq) - self.C


        def refine(a, b, v):
            return brentq(f, a, b)
        
        rets = []
        if guess_lower is None and guess_upper is None:
            dqs = np.linspace(q_lower, q_upper, q_N)
            gs = [f(dq) for dq in dqs]
            g0, i0 = gs[0], 0
            if np.allclose(gs[0], 0, rtol=rtol):
                rets.append(gs[0])
                g0, i0= gs[1], 1
            for i in range(len(rets), len(gs)):
                if g0*gs[i] < 0:
                    rets.append(refine(dqs[i0], dqs[i], dqs[i0]))
                g0, i0 = gs[i], i
        else:
            bExcept = False
            if guess_lower is not None:
                try:
                    ret1 = brentq(f, min(0, guess_lower - dx), guess_lower + dx)
                    rets.append(ret1)
                except:
                    bExcept = True
                    rets.append(None)
            else:
                rets.append(None)
            if guess_upper is not None:
                try:
                    ret2 = brentq(f, min(0, guess_upper - dx), guess_upper + dx)
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

    def print(self, str):
        """to support verbosity control level"""
        if self.verbosity:
            print(str)

    def Search(
            self, delta_N, mu_eff=None, dmu_eff=None, delta=None,
            delta_lower=0.001, delta_upper=1, q_lower=0, q_upper=0.2,
            q_N=40, auto_incremental=True):
        """
        Search possible states in ranges of delta and q
        Paras:
        -------------
        mu_eff     : effective mu used to fix C
        delta      : the delta used to fix C
        delta_lower: lower boundary for delta
        delta_upper: upper boundary for delta
        q_lower    : lower boundary for q
        q_upper    : upper boundary for q
        auto_incremental: if True, the search will continue as long
            as there are some more possible solution based on the 
            condition that the last delta in the  deltas yields valid
            states
        -------------
        Note:
            dmu_eff must not larger than delta, or no solution
            could be found[See notebook FFState(ASLDA).py]
        """
        rets = []
        dx = dx0 = 0.001
        trails=[1, 2, 4, 0.01, 0.25, 0.5, 8, 16, 0]
        deltas = np.linspace(delta_lower, delta_upper, delta_N)
        incremental_step = (delta_upper - delta_lower) / (delta_N + 1)
        if delta is None:
            delta = self.delta
        if mu_eff is None:
            mu_eff = self.mu_eff
        if dmu_eff is None:
            dmu_eff = self.dmu_eff
        self.C = self._get_C(mus_eff=(mu_eff, mu_eff), delta=delta)
        lg, ug=None, None
        print(
            f"Search: delta={delta},mu={mu_eff},dmu={dmu_eff},C={self.C},"
            + f"lower q={q_lower}, upper q={q_upper},"
            + f"lower delta={delta_lower}, upper delta={delta_upper},"
            + f"auto_incremental={auto_incremental}")
            
        def do_search(delta):
            nonlocal lg
            nonlocal ug
            ret = self.SearchFFStates(
                mu_eff=mu_eff, dmu_eff=dmu_eff,
                delta=delta, guess_lower=lg, guess_upper=ug,
                q_lower=q_lower, q_upper=q_upper, q_N=q_N, dx=dx)
            lg, ug = ret
            ret.append(delta)
            rets.append(ret)
            print(ret)
        while(True):
            for delta_ in deltas:
                print(delta_)
                retry = True
                if dx != dx0 and dx !=0:
                    try:
                        do_search(delta=delta_)
                        retry = False
                        continue
                    except ValueError:
                        self.print(f"retry[dx={dx}]")
                if retry:
                    for t in trails:
                        if t == 0:
                            break
                        dx = dx0*t
                        try:
                            do_search(delta=delta_)
                            break
                        except ValueError:
                            self.print(f"No solution[dx={dx}]")
                            continue

                    if t == trails[-1]:
                        self.print("Retry without exception")
                        ret =[None, None]
                        for t in trails:
                            dx = dx0*t
                            print(dx)
                            ret0 = self.SearchFFStates(
                                mu_eff=mu_eff, dmu_eff=dmu_eff,
                                delta=delta_, guess_lower=lg, guess_upper=ug,
                                q_lower=q_lower, q_upper=q_upper, q_N=q_N, dx=dx,
                                raiseExcpetion=False)
                            lg, ug = ret0
                            print(ret0)
                            if lg is None and ug is None:
                                continue
                            ret = ret0
                            ret.append(delta_)
                            rets.append(ret)
                            print(ret)
                            break
                if len(rets) > 0:
                    q1, q2, _ = rets[-1]
                    if q1 is None and q2 is None:
                        if delta_lower != deltas[0]:
                            auto_incremental = False
                            break
                        self.print(f"Delta={delta_} has no solution, try next delta")
                        del rets[-1]
                        lg, ug=None, None
                        if list(deltas).index(delta_) > min(10, len(deltas)//2):
                            auto_incremental = False
                            break
                        continue
                self.SaveToFile(rets)
            if auto_incremental:
                print("Append 20 more search points")
                deltas = np.linspace(1, 20, 20)*incremental_step + deltas[-1]
            else:
                break
        self.print("Search Done")
        rets = ff_state_sort_data(rets)
        self.SaveToFile(rets)


if __name__ == "__main__":
    mu_eff = 10
    dmu_eff = 0.1
    delta = 0.2
    args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
        T=0, dim=3, k_c=50, verbosity=False)
    lda = ClassFactory("LDA", (FFStateAgent,), args=args)
    lda.Search(
        delta_N=20, delta_lower=0.001, delta_upper=delta,
        q_lower=0, q_upper=dmu_eff, q_N=10)
