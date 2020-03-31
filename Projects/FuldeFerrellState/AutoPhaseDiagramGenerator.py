
import inspect
import time
import glob
import json
import os
import sys
import operator
import itertools
from os.path import join
import numpy as np
from scipy.optimize import brentq
from mmf_hfb.class_factory import FunctionalType, KernelType
from mmf_hfb.class_factory import ClassFactory, Solvers
from mmf_hfb.parallel_helper import PoolHelper

currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from DataHelper import ff_state_sort_data


class FFStateAgent(object):
    """
    An agent class used for searching FF states
    """

    def __init__(
            self, mu_eff, dmu_eff, delta, dim, verbosity=True,
            prefix="FFState_", timeStamp=True, **args):
        # assert dmu_eff < delta
        self.delta = delta
        self.mu_eff = mu_eff
        self.dmu_eff = dmu_eff
        self.verbosity = verbosity
        if timeStamp:
            ts = time.strftime("%Y_%m_%d_%H_%M_%S.json")
            self.fileName = (
                prefix
                + f"({dim}d_{delta:.2f}_{mu_eff:.2f}_{dmu_eff:.2f})" + ts)
        else:
            self.fileName = prefix

    def _get_fileName(self):
        currentdir = join(os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))), "data")
        os.makedirs(currentdir, exist_ok=True)
        return join(currentdir, self.fileName)

    def get_other_pressures(self, mus, delta, dq, C, mus_eff):
        self.C = C
        # P_ff = self.get_ns_e_p(
        #     mus=mus, delta=delta, dq=dq, verbosity=False,
        #     x0=mus_eff +(delta,), solver=Solvers.BROYDEN1)
        P_ss = self.get_ns_e_p(  # superfluid pressure
            mus=mus, delta=None, verbosity=False, solver=Solvers.BROYDEN1)
        P_ns = self.get_ns_e_p(  # normal state pressure
            mus=mus, delta=0, verbosity=False, solver=Solvers.BROYDEN1)
        return (P_ss[2], P_ns[2])

    def SaveToFile(self, data, extra_items=None):
        """
        Save states to persistent storage
        Note: extra_item should be a dict object
        """
        if len(data) == 0:
            return
        file = self._get_fileName()
        output = {}
        output["dim"] = self.dim
        output["delta"] = self.delta
        output["mu_eff"] = self.mu_eff
        output["dmu_eff"] = self.dmu_eff
        output["C"] = self.C
        output["k_c"] = self.k_c
        output['functional'] = self.functional
        output['kernel'] = self.kernel
        if extra_items is not None:
            output.update(extra_items)
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
        dx: the change range of the dq based on a given reference value
            dq = [old_dq - dx, old_dq + dx]
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
                g0, i0 = gs[1], 1
            for i in range(len(rets), len(gs)):
                if g0*gs[i] < 0:
                    rets.append(refine(dqs[i0], dqs[i], dqs[i0]))
                g0, i0 = gs[i], i
        else:
            bExcept = False
            if guess_lower is not None:
                try:
                    ret1 = brentq(
                        f, max(0, guess_lower - dx), guess_lower + dx)
                    rets.append(ret1)
                except ValueError:
                    bExcept = True
                    rets.append(None)
            else:
                rets.append(None)
            if guess_upper is not None:
                try:
                    ret2 = brentq(
                        f, max(0, guess_upper - dx), guess_upper + dx)
                    rets.append(ret2)
                except ValueError:
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
            q_N=40, auto_incremental=False, auto_save=True):
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
            could be found[See notebook ff_state_aslda.py]
        """
        rets = []
        dx = dx0 = 0.001
        trails = [1, 2, 4, 0.01, 0.25, 0.5, 8, 16, 0]
        deltas = np.linspace(delta_lower, delta_upper, delta_N)
        incremental_step = (delta_upper - delta_lower) / (delta_N + 1)
        if delta is None:
            delta = self.delta
        if mu_eff is None:
            mu_eff = self.mu_eff
        if dmu_eff is None:
            dmu_eff = self.dmu_eff
        self.C = self._get_C(mus_eff=(mu_eff, mu_eff), delta=delta)
        # if abs(delta_upper - delta) < abs(delta - delta_lower):
        #     deltas = deltas[::-1]  #
        lg, ug = None, None
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
                retry = True
                if dx != dx0 and dx != 0:
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
                        ret = [None, None]
                        for t in trails:
                            dx = dx0*t
                            print(f"dx={dx}")
                            ret0 = self.SearchFFStates(
                                mu_eff=mu_eff, dmu_eff=dmu_eff,
                                delta=delta_, guess_lower=lg, guess_upper=ug,
                                q_lower=q_lower, q_upper=q_upper,
                                q_N=q_N, dx=dx,
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
                        self.print(f"Delta={delta_} has no solution, try next")
                        del rets[-1]
                        lg, ug = None, None
                        if (
                            list(
                                deltas).index(
                                    delta_) > min(10, len(deltas)//2)):
                            auto_incremental = False
                            break
                        continue
                # add a pending flag indicating state searching is going on
                if auto_save:
                    self.SaveToFile(rets, extra_items={"pending": 0})
            if auto_incremental:
                print("Append 20 more search points")
                deltas = np.linspace(1, 20, 20)*incremental_step + deltas[-1]
            else:
                break
        self.print("Search Done")
        rets = ff_state_sort_data(rets)
        if auto_save:
            self.SaveToFile(rets)
        return rets


def search_states_worker(obj_mus_delta_dim_kc):
    obj, mu_eff, dmu_eff, delta, dim, k_c = obj_mus_delta_dim_kc
    args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
        T=0, dim=dim, k_c=k_c, verbosity=False)
    lda = ClassFactory(
        "LDA", (FFStateAgent,),
        functionalType=obj.functionalType,
        kernelType=obj.kernelType, args=args)
    # return lda.Search(
    #     delta_N=50, delta_lower=0.01, delta_upper=0.062,
    #     q_lower=0, q_upper=dmu_eff, q_N=10, auto_incremental=False)

    return lda.Search(
        delta_N=50, delta_lower=0.0001, delta_upper=delta,
        q_lower=0, q_upper=dmu_eff, q_N=10, auto_incremental=False)


def search_condidate_worker(obj_mus_delta_dim_kc):
    obj, mu_eff, dmu_eff, delta, dim, k_c = obj_mus_delta_dim_kc
    args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
        T=0, dim=dim, k_c=k_c, verbosity=False)
    lda = ClassFactory(
        "LDA", (FFStateAgent,),
        functionalType=obj.functionalType,
        kernelType=obj.kernelType, args=args)

    return lda.Search(
        delta_N=2, delta_lower=0.0001, delta_upper=0.01,
        q_lower=0, q_upper=dmu_eff, q_N=10,
        auto_incremental=False, auto_save=False)


class AutoPDG(object):

    def __init__(
            self, functionalType, kernelType, dim=2,
            mu0=10, k_c=200, data_dir=None, poolSize=9):
        self.mu_eff = mu0
        self.k_c = k_c
        self.dim = dim
        self.functionalType = functionalType
        self.kernelType = kernelType
        self.poolSize = poolSize
        self.timeStamp = self.get_time_stamp()
        if data_dir is None:
            data_dir = os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe())))

    def get_time_stamp(self):
        return time.strftime("%Y_%m_%d_%H_%M_%S.json")

    def get_delta_q_file(self, delta, dmu):
        #  pid = os.getpid()
        ts = self.get_time_stamp()
        prefix = "delta_q_"
        fileName = (
                prefix + f"({self.dim}d_{delta:.2f}"
                + f"_{self.mu_eff:.2f}_{dmu:.2f})" + ts)
        return fileName

    def offset_para(self, p, dx=0.1, factor=1):
        offset = np.array([-1, 0, 1])
        return dx*p*offset*factor + p

    def mix_para2(self, p1, p2):
        return list(itertools.product(p1, p2))

    def search_delta_q_diagram(self, seed_delta, seed_dmu):

        #  obj, mu_eff, dmu_eff, delta, dim, k_c
        dmus = self.offset_para(seed_dmu)
        deltas = self.offset_para(seed_delta)
        dmu_delta_ls = self.mix_para2(dmus, deltas)
        # if os.path.exists("debug.json"):
        #     with open("debug.json", 'r') as rf:
        #         res = json.load(rf)
        #         return ([(0.175, 0.25)], res[1])
        paras = [(
            self, self.mu_eff, dmu,
            delta, self.dim, self.k_c) for (dmu, delta) in dmu_delta_ls]
        paras = [(self, self.mu_eff, seed_dmu, seed_delta, self.dim, self.k_c)]
        res = PoolHelper.run(
            search_states_worker, paras, poolsize=self.poolSize)
        return (dmu_delta_ls, res)

    def compute_press(self, dmu, delta, delta_q):
        mu_eff, dmu_eff = self.mu_eff, dmu
        mus_eff = (self.mu_eff + dmu, self.mu_eff - dmu)
        args = dict(
                mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
                T=0, dim=self.dim, k_c=self.k_c, verbosity=False)
        lda = ClassFactory(
                "LDA", (FFStateAgent,),
                functionalType=self.functionalType,
                kernelType=self.kernelType, args=args)
        lda.C = lda._get_C(mus_eff=mus_eff, delta=delta)
        normal_pressure = lda.get_ns_mus_e_p(
            mus_eff=mus_eff, delta=0)[3]
        superfluid_pressure = lda.get_ns_mus_e_p(
            mus_eff=mus_eff, delta=delta)[3]
        temp_data = []
        ffs_pressures = []
        for (dq1, dq2, delta) in delta_q:
            if dq1 is not None:
                p1 = lda.get_ns_mus_e_p(
                    mus_eff=mus_eff, delta=delta, dq=dq1)[3]
                temp_data.append((delta, dq1, p1))
            if dq2 is not None:
                p2 = lda.get_ns_mus_e_p(
                    mus_eff=mus_eff, delta=delta, dq=dq2)[3]
                temp_data.append((delta, dq2, p2))
        max_item = (0, 0, 0)
        for item in temp_data:
            delta, dq, pressure = item
            if pressure > max_item[2]:
                max_item = item
        print(normal_pressure, superfluid_pressure, max_item)

    def search_next_step(self, dmu_deltas, delta_qs):
        # press_ls = []
        for id, dmu_delta in enumerate(dmu_deltas):
            delta_q = delta_qs[id]
            dmu, delta = dmu_delta
            ps = self.compute_press(
                dmu=dmu, delta=delta, delta_q=delta_q)

    def search_valid_conditate(self, seed_delta, seed_dmu):
        trail = 1
        trail_max = 10
        while(trail < trail_max):
            print(f"#{trail} Delta={seed_delta}, dmu={seed_dmu}")
            dmus = self.offset_para(seed_dmu, factor=trail)
            deltas = self.offset_para(seed_delta, factor=trail)
            dmu_delta_ls = self.mix_para2(dmus, deltas)
            paras = [(
                self, self.mu_eff, dmu,
                delta, self.dim, self.k_c) for (dmu, delta) in dmu_delta_ls]
            res = PoolHelper.run(
                search_condidate_worker, paras, poolsize=self.poolSize)

            direction_flags = []
            for re in res:
                n = len(re)
                if n == 0:
                    direction_flags.append(False)
                    continue
                r = re[0]
                flag = ((r[0] is not None) and (r[1] is not None))
                if not flag:
                    direction_flags.append(False)
                    continue
                if n == 2:
                    r1, r2 = re
                    if min(r1[:2]) < min(r2[:2]):
                        direction_flags.append(True)
                        continue
                direction_flags.append(False)
            if not any(np.array(direction_flags)):  # when not state is found
                # find the one with minimum change
                index = -1
                min_dq = np.inf
                for id, re in enumerate(res):
                    n = len(re)
                    if n == 0:
                        continue
                    r = re[0]
                    if not ((r[0] is not None) and (r[1] is not None)):
                        continue
                    if n == 2:
                        r1, r2 = re
                        dq = min(r1[:2]) - min(r2[:2])
                        if dq < min_dq:
                            min_dq = dq
                            index = id
                if index == -1:  # the seed delta and dmu are not good
                    raise ValueError(
                        f"The seed delta {seed_delta} and dmu {seed_dmu} are not good")
                # the current point is the best, that need more check
                if index == len(dmu_delta_ls)//2:
                    trail = trail + 1
                    continue  # try next trail

                seed_dmu, seed_delta = dmu_delta_ls[index]
                trail = 1  # reset the trail number for next step
                continue
            # return all candidate with flags indcating
            # if they are good start states
            print(dmu_delta_ls, direction_flags)
            return (dmu_delta_ls, direction_flags)
        raise ValueError(
            f"The seed delta {seed_delta} and dmu {seed_dmu} are not good")

    def run(self, seed_delta, seed_dmu):
        # First step: Find the delta-q diagram
        if False:
            delta_q_res = self.search_delta_q_diagram(
                seed_delta=seed_delta, seed_dmu=seed_dmu)
        else:
            delta_q_res = self.search_valid_conditate(
                seed_delta=seed_delta, seed_dmu=seed_dmu)
        dmu_deltas, delta_qs = delta_q_res
        # self.search_next_step(dmu_deltas, delta_qs)
        # with open('debug.json', 'w') as wf:
        #     json.dump(delta_q_res, wf)
        # print(delta_q_res)


if __name__ == "__main__":
    pdg = AutoPDG(
        functionalType=FunctionalType.BDG,
        kernelType=KernelType.HOM, k_c=50, dim=3)
    # pdg.run(seed_delta=0.25, seed_dmu=0.135)
    dmu, delta = 0.175, 0.35  # 0.175, 0.25
    pdg.run(seed_delta=delta, seed_dmu=dmu)
