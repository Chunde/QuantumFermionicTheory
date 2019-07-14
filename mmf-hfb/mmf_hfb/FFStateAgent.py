from mmf_hfb.ClassFactory import ClassFactory, FunctionalType, KernelType
from mmf_hfb.DataHelper import ff_state_sort_data
from mmf_hfb.ParallelHelper import PoolHelper
from scipy.optimize import brentq
from os.path import join
import numpy as np
import inspect
import time
import glob
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

    def get_ns_e_p(self, mus_eff=None, delta=None, q=0, dq=0):
        """return the pressure"""
        if mus_eff is None:
            mus_eff = (self.mu_eff + self.dmu_eff, self.mu_eff - self.dmu_eff)
        ns, mus, e, p = self.get_ns_mus_e_p(mus_eff=mus_eff, delta=delta, q=q, dq=dq)
        return (ns, mus, e, p)

    def get_pressure(self, mus_eff, delta, q=0, dq=0):
        """return the pressure only"""
        return self.get_ns_e_p(mus_eff=mus_eff, delta=delta, q=q, dq=dq)[3]
        
    def SaveToFile(self, data, extra_items=None):
        """
        Save states to persistent storage
        Note: extra_item should be a dict object
        """
        if len(data) == 0:
            return
        file = self._get_fileName()
        output = {}
        output["dim"]= self.dim
        output["delta"] = self.delta
        output["mu_eff"] = self.mu_eff
        output["dmu_eff"] = self.dmu_eff
        output["C"] = self.C
        output["k_c"] = self.k_c
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
    mus_eff = (mu_eff + dmu_eff, mu_eff - dmu_eff)

    args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
        T=0, dim=dim, k_c=k_c, verbosity=False,
        prefix=f"{output_fileName}", timeStamp=False)
    lda = ClassFactory(
            "LDA", (FFStateAgent,),
            functionalType=FunctionalType.ASLDA,
            kernelType=KernelType.HOM, args=args)
    C_ = lda._get_C(mus_eff=mus_eff, delta=delta)
    assert np.allclose(C, C_, rtol=1e-16)  # verify the C value
    
    if os.path.exists(lda._get_fileName()):
        return None
    normal_pressure = lda.get_ns_e_p(mus_eff=mus_eff, delta=0)[3]

    print(f"Processing {lda._get_fileName()}")
    output1 = []
    output2 = []

    def append_item(delta, dq, output):
        if dq is not None:
            dic = {}
            ns, mus, e, p = lda.get_ns_e_p(delta=d, dq=dq)
            ja, jb, jp, _ = lda.get_current(mus_eff=mus_eff, delta=d, dq=dq)
            dic['na']=ns[0]  # particle density a
            dic['nb']=ns[1]  # particle density b
            dic['d']=d  # delta satisfies the gap equation
            dic['q']=dq  # the q value
            dic['e']=e  # energy density
            dic['p']=p  # pressure
            dic['j']=jp.n  # current sum
            dic['ja']=ja.n  # current a
            dic['jb']=jb.n  # current b
            dic['mu_a']=mus[0]  # bare mu_a
            dic['mu_b']=mus[1]  # bare mu_b
            output.append(dic)
            print(dic)
    try:
        for item in data:
            dq1, dq2, d = item
            append_item(delta=d, dq=dq1, output=output1)
            append_item(delta=d, dq=dq2, output=output2)
        output =[output1, output2]
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
    files = files=glob.glob(pattern)

    jsonObjects = []
    for file in files:
        if os.path.exists(file):
            with open(file, 'r') as rf:
                jsonObjects.append(
                    (json.load(rf), os.path.splitext(os.path.basename(file))[0]))
 
    if False:  # Debugging
        for item in jsonObjects:
            compute_pressure_current_worker(item)
    else:
        PoolHelper.run(compute_pressure_current_worker, jsonObjects)
 

def search_states_worker(mus_delta):
    mu_eff, dmu_eff, delta = mus_delta
    args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
        T=0, dim=3, k_c=50, verbosity=False)
    lda = ClassFactory(
        "LDA", (FFStateAgent,),
        functionalType=FunctionalType.ASLDA,
        kernelType=KernelType.HOM, args=args)
    lda.Search(
        delta_N=100, delta_lower=0.001, delta_upper=delta,
        q_lower=0, q_upper=dmu_eff, q_N=10)


def search_states(mu_eff=10, delta=1):
    """compute current and pressure"""
    dmus = np.linspace(0.001*delta, delta*0.999, 10)
    mus_deltas = [(mu_eff, dmu, delta) for dmu in dmus]
    if False:  # Debugging
        for item in mus_deltas:
            search_states_worker(item)
    else:
        PoolHelper.run(search_states_worker, mus_deltas, poolsize=5)


if __name__ == "__main__":
    #search_states(delta=2.5)
    compute_pressure_current()
    # mu_eff = 10
    # dmu_eff = 0.5
    # delta = 1
    # args = dict(
    #     mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
    #     T=0, dim=3, k_c=50, verbosity=False)
    # lda = ClassFactory(
    #     "LDA", (FFStateAgent,),
    #     functionalType=FunctionalType.ASLDA,
    #     kernelType=KernelType.HOM, args=args)
    # lda.Search(
    #     delta_N=100, delta_lower=0.001, delta_upper=delta,
    #     q_lower=0, q_upper=dmu_eff, q_N=10)
