"""
This is a class used to search Fulde Ferrell state. 
The class FFStateAgent is a class that will be fed
into to the 'class_factory' instance to create a new class
that will perform the search task.

Example:

def search_states_worker(mu_eff, dmu_eff, delta):
    args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
        T=0, dim=2, k_c=100, verbosity=False)
    lda = class_factory(
        "LDA", (FFStateAgent,),
        functionalType=FunctionalType.BDG,
        kernelType=KernelType.HOM, args=args)
    lda.Search(
        delta_N=100, delta_lower=0.0001, delta_upper=delta,
        q_lower=0, q_upper=dmu_eff, q_N=10, auto_incremental=False)
"""
import inspect
import time
import glob
import json
import os
import sys
import operator

from os.path import join
import numpy as np
from scipy.optimize import brentq
from mmf_hfb.class_factory import FunctionalType, KernelType
from mmf_hfb.class_factory import ClassFactory, Solvers
from mmf_hfb.parallel_helper import PoolHelper

currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from data_helper import ff_state_sort_data


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
        solver = None
        if self.functional != FunctionalType.BDG:
            solver = Solvers.BROYDEN1
        # P_ff = self.get_ns_e_p(
        #     mus=mus, delta=delta, dq=dq, verbosity=False,
        #     x0=mus_eff +(delta,), solver=Solvers.BROYDEN1)
        P_ss = self.get_ns_e_p(  # superfluid pressure
            mus=mus, delta=None, verbosity=False, solver=solver)
        P_ns = self.get_ns_e_p(  # normal state pressure
            mus=mus, delta=0, verbosity=False, solver=solver)
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
        output['functional'] = self.functional_index
        output['kernel'] = self.kernel_index
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
                    ret1 = brentq(f, max(0, guess_lower - dx), guess_lower + dx)
                    rets.append(ret1)
                except:
                    bExcept = True
                    rets.append(None)
            else:
                rets.append(None)
            if guess_upper is not None:
                try:
                    ret2 = brentq(
                        f, max(0, guess_upper - dx), guess_upper + dx)
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
            q_N=40, auto_incremental=False):
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
        if abs(delta_upper - delta) < abs(delta - delta_lower):
            deltas = deltas[::-1]  #
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
                self.SaveToFile(rets, extra_items={"pending": 0})
            if auto_incremental:
                print("Append 20 more search points")
                deltas = np.linspace(1, 20, 20)*incremental_step + deltas[-1]
            else:
                break
        self.print("Search Done")
        rets = ff_state_sort_data(rets)
        self.SaveToFile(rets)


def compute_pressure_current_worker(jsonData_file):
    """
    Use the FF State file to compute their current and pressure
    """
    jsonData, fileName = jsonData_file
    filetokens = fileName.split("_")
    output_fileName = "FFState_J_P_" + "_".join(filetokens[1:]) + ".json"
    dim = jsonData['dim']
    delta = jsonData['delta']
    mu_eff = jsonData['mu_eff']
    dmu_eff = jsonData['dmu_eff']
    data = jsonData['data']
    k_c = jsonData['k_c']
    C = jsonData['C']
    functional_index = jsonData['functional']
    if 'pending' in jsonData:
        print(f"Skip a unfinished file: {fileName}")
        return
    mus_eff = (mu_eff + dmu_eff, mu_eff - dmu_eff)

    args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
        T=0, dim=dim, k_c=k_c, verbosity=False,
        prefix=f"{output_fileName}", timeStamp=False)
    lda = ClassFactory(
        "LDA", (FFStateAgent,),
        functionalType=ClassFactory(functionalIndex=functional_index),
        kernelType=KernelType.HOM, args=args)
    C_ = lda._get_C(mus_eff=mus_eff, delta=delta)
    assert np.allclose(C, C_, rtol=1e-16)  # verify the C value
    lda.C = C
    if os.path.exists(lda._get_fileName()):
        return None
    normal_pressure = lda.get_ns_mus_e_p(mus_eff=mus_eff, delta=0)[3]

    print(f"Processing {lda._get_fileName()}")
    output1 = []
    output2 = []

    def append_item(delta, dq, output):
        if dq is not None:
            dic = {}
            ns, mus, e, p = lda.get_ns_mus_e_p(mus_eff=mus_eff, delta=d, dq=dq)
            ja, jb, jp, _ = lda.get_current(mus_eff=mus_eff, delta=d, dq=dq)
            dic['na'] = ns[0]  # particle density a
            dic['nb'] = ns[1]  # particle density b
            dic['d'] = d  # delta satisfies the gap equation
            dic['q'] = dq  # the q value
            dic['e'] = e  # energy density
            dic['p'] = p  # pressure
            dic['j'] = jp.n  # current sum
            dic['ja'] = ja.n  # current a
            dic['jb'] = jb.n  # current b
            dic['mu_a'] = mus[0]  # bare mu_a
            dic['mu_b'] = mus[1]  # bare mu_b
            output.append(dic)
            print(dic)
    try:
        for item in data:
            dq1, dq2, d = item
            append_item(delta=d, dq=dq1, output=output1)
            append_item(delta=d, dq=dq2, output=output2)
        output = [output1, output2]
        lda.SaveToFile(output, extra_items={"p0": normal_pressure})
    except ValueError as e:
        print(f"Parsing file: {fileName}. Error:{e}")


def compute_pressure_current(root=None):
    """compute current and pressure"""
    currentdir = root
    if currentdir is None:
        currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
    pattern = join(currentdir, "data", "FFState_[()_0-9]*.json")
    files = glob.glob(pattern)

    jsonObjects = []
    for file in files:
        if os.path.exists(file):
            with open(file, 'r') as rf:
                jsonObjects.append(
                    (json.load(rf), os.path.splitext(
                        os.path.basename(file))[0]))

    if False:  # Debugging
        for item in jsonObjects:
            compute_pressure_current_worker(item)
    else:
        PoolHelper.run(compute_pressure_current_worker, jsonObjects)


def search_states_worker(mus_delta):
    mu_eff, dmu_eff, delta = mus_delta
    args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
        T=0, dim=2, k_c=100, verbosity=False)
    lda = ClassFactory(
        "LDA", (FFStateAgent,),
        functionalType=FunctionalType.BDG,
        kernelType=KernelType.HOM, args=args)
    lda.Search(
        delta_N=100, delta_lower=0.0001, delta_upper=delta,
        q_lower=0, q_upper=dmu_eff, q_N=10, auto_incremental=False)


def search_states(mu_eff=None, delta=1):
    e_F = 10
    # mu0 = 0.59060550703283853378393810185221521748413488992993*e_F
    mu0 = 0.5*e_F
    # delta0 = 0.68640205206984016444108204356564421137062514068346*e_F
    if mu_eff is None:
        mu_eff = mu0
    """compute current and pressure"""
    dmus = np.linspace(0.01*delta, 0.5*delta, 10)
    mus_deltas = [(mu_eff, dmu, delta) for dmu in dmus]
    if False:  # Debugging
        for item in mus_deltas:
            search_states_worker(item)
    else:
        PoolHelper.run(search_states_worker, mus_deltas, poolsize=8)


def label_states(current_dir=None, raw_data=False, print_file=False, verbosity=False):
    """
    check all the pressure and current output files for
    different configuration(mus_eff, delta), determine if
    each of them is FF state or not.
    --------------
    return a list of information include state tag flag
    can be used for plotting.
    Para:
    --------------
    current_dir: specifies the dir where to find the
        pressure and current output files
    raw_data: if this is True, the original data will
        also be included in the return list items

    """
    if current_dir is None:
        current_dir = join(
            os.path.dirname(
                os.path.abspath(
                    inspect.getfile(
                        inspect.currentframe()))), "data")
    output = []
    pattern = join(current_dir, "FFState_J_P[()d_0-9]*")
    files = glob.glob(pattern)

    for file in files[0:]:
        if os.path.exists(file):
            if print_file:
                print(file)
            with open(file, 'r') as rf:
                ret = json.load(rf)
                # if ret['delta'] != 2.0:
                #     continue
                dim, mu_eff, dmu_eff, delta, C = (
                    ret['dim'], ret['mu_eff'], ret['dmu_eff'],
                    ret['delta'], ret['C'])
                functional_index = ret['functional'] if ('functional' in ret) else 0

                p0 = ret['p0']
                a_inv = 4.0*np.pi*C  # inverse scattering length

                if verbosity:
                    print(file)
                data1, data2 = ret['data']

                data1.extend(data2)

                dqs1, ds1, = [], []
                j1, ja1, jb1, P1 = [], [], [], []
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

                bFFState = False
                if len(P1) > 0:
                    index1, value = max(enumerate(P1), key=operator.itemgetter(1))
                    data = data1[index1]
                    n_a, n_b = data['na'], data['nb']
                    mu_a, mu_b = data['mu_a'], data['mu_b']
                    mu, dmu = (mu_a + mu_b)/2.0, (mu_a - mu_b)/2.0
                    #  create a lda instance to compute all types of pressure
                    args = dict(
                        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
                        T=0, dim=dim, k_c=ret['k_c'], verbosity=False, C=C)
                    lda = ClassFactory(
                            "LDA", (FFStateAgent,),
                            functionalType=ClassFactory(
                                functionalIndex=functional_index),
                                kernelType=KernelType.HOM, args=args)
                    
                    if verbosity:
                        print(f"na={n_a}, nb={n_b}, PF={value}, PN={p0}")
                    if (
                        not np.allclose(
                            n_a, n_b, rtol=1e-9) and (
                                data["q"] > 0.0001 and data["d"] > 0.0001)):
                        pressures = lda.get_other_pressures(
                            mus_eff=(mu_eff + dmu_eff, mu_eff - dmu_eff),
                            mus=(mu+dmu, mu-dmu), delta=data["d"],
                            dq=data['q'], C=C)
                        if data['p'] > pressures[1] and data['p'] > pressures[0]:
                            bFFState = True
                if bFFState and verbosity:
                    print(f"FFState: {bFFState} |<-------------")
                dic = dict(
                    mu_eff=mu_eff, dmu_eff=dmu_eff,
                    mu=mu, dmu=dmu, np=n_a + n_b, na=n_a, d=data['d'], dq=data['q'],
                    nb=n_b, ai=a_inv, C=C, delta=delta, pf=data['p'],
                    ps=pressures[0], pn=pressures[1], state=bFFState)
                if verbosity:
                    print(dic)
                if raw_data:
                    dic['data'] = ret
                    dic['file'] = file
                output.append(dic)
                if verbosity:
                    print("-----------------------------------")
    return output


if __name__ == "__main__":
    # ds = np.linspace(1.1, 1.5, 10)
    # for delta in ds:
    # search_states(delta=0.5)
    # compute_pressure_current()
    label_states(raw_data=True)
