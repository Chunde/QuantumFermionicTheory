import inspect
import time
import json
import os
import sys
from collections import deque
from os.path import join
import numpy as np
import operator
import glob
import itertools
from scipy.optimize import brentq
from mmf_hfb.class_factory import FunctionalType, KernelType
from mmf_hfb.class_factory import ClassFactory, Solvers
from mmf_hfb.parallel_helper import PoolHelper
from mmf_hfb import tf_completion as tf
from mmf_hfb.utils import JsonEncoderEx

currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from data_helper import ff_state_sort_data
tf.MAX_DIVISION = 150


def predict_joint_point(res):
    """
    predict the joint point for the two curves
    """
    if res is None or len(res) < 3:
        return None
    q1, q1_, d1 = res[-3]
    q2, q2_, d2 = res[-2]
    q3, q3_, d3 = res[-1]
    M = np.array([[d1**2, d1, 1], [d2**2, d2, 1], [d3**2, d3, 1]])
    y1 = np.array([q1, q2, q3])
    y2 = np.array([q1_, q2_, q3_])
    try:
        x1 = np.linalg.solve(M, y1)
        x2 = np.linalg.solve(M, y2)
        a, b, c = x2 - x1
        if (b**2 - 4*a*c) < 0:
            return None
        delta_p = (-b + (b**2 - 4*a*c)**0.5)/2.0/a
        delta_m = (-b - (b**2 - 4*a*c)**0.5)/2.0/a
    except ValueError:
        return None
    d4 = max(delta_p, delta_m)
    q4 = x1.dot([d4**2, d4, 1])
    return (d4, q4)  # common point


class FFStateAgent(object):
    """
        An agent class used to construct a new class that will
        have the functions related to Fulde Ferrell state search
        and pressure calculation, etc.
    """

    def __init__(
            self, mu_eff, dmu_eff, delta, dim, verbosity=True,
            prefix="FFState_", time_stamp=True, **args):
        """
        mu_eff: the effective mu=(mu_a + mu_b)/2, will be used to
            fix the C.
        verbosity: bool, to indicate if messages should be print to
            stdin. (To-Do:This should be upgraded to integer for message
            level control)
        prefix: is used for file name for saving the output
        time_stamp: used to make unique filename
        """
        self.delta = delta
        self.mu_eff = mu_eff
        self.dmu_eff = dmu_eff
        self.verbosity = verbosity
        if time_stamp:
            ts = time.strftime("%Y_%m_%d_%H_%M_%S.json")
            self.fileName = (
                prefix
                + f"({dim}d_{delta:.2f}_{mu_eff:.2f}_{dmu_eff:.2f})" + ts)
        else:
            self.fileName = prefix

    def get_file_name(self):
        currentdir = join(os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))), "data")
        os.makedirs(currentdir, exist_ok=True)
        return join(currentdir, self.fileName)

    def print(self, str, level=0):
        """to support verbosity control level"""
        if self.verbosity:
            print(str)

    def save_to_file(self, data, extra_items=None):
        """
        Save states to persistent storage
        Note: extra_item should be a dict object
        """
        if len(data) == 0:
            return
        file = self.get_file_name()
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
        try:
            with open(file, 'w') as wf:
                json.dump(output, wf, cls=JsonEncoderEx)
        except TypeError:
            print("Json Exception.")
            os.rename(file, f"{file}.__error__")

    def check_sign_flip(self, ls):
        """
        Check the sign flipping number,
        each flipping indicate a zero solution.
        This function will return max of 2 flipping
        as in this application two solution of q
        is the maximum number we can have.
        """
        sign_flip_num = 0
        if len(ls) > 2:
            for i in range(1, len(ls)):
                if ls[i]*ls[i-1] < 0:
                    sign_flip_num = sign_flip_num + 1
                    if sign_flip_num == 2:
                        return sign_flip_num
        return sign_flip_num

    def search_states(
            self, delta, mu_eff, dmu_eff,
            guess_lower=None, guess_upper=None, q_lower=0, q_upper=0.04,
            N_q=10, dx=0.0005, rtol=1e-8, raiseExcpetion=True):
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
            dqs = np.linspace(q_lower, q_upper, N_q)
            gs = [f(dq) for dq in dqs]
            g0, i0 = gs[0], 0
            if np.allclose(gs[0], 0, rtol=rtol):
                rets.append(gs[0])
                g0, i0 = gs[1], 1
            for i in range(len(rets), len(gs)):
                if g0*gs[i] < 0:
                    rets.append(refine(dqs[i0], dqs[i], dqs[i0]))
                g0, i0 = gs[i], i
                if len(rets) == 2:  # two solutions at max
                    break
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

    def smart_search_states(
            self, delta, mu_eff, dmu_eff, dxs=None,
            guess_lower=None, guess_upper=None, q_lower=0, q_upper=0.04,
            N_q=10, rtol=1e-8, raiseExcpetion=True):
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
            dqs = np.linspace(q_lower, q_upper, N_q)
            gs = []
            for dq in dqs:
                gs.append(f(dq))
                if self.check_sign_flip(gs) == 2:
                    break
            g0, i0 = gs[0], 0
            if np.allclose(gs[0], 0, rtol=rtol):
                rets.append(gs[0])
                g0, i0 = gs[1], 1
            for i in range(len(rets), len(gs)):
                if g0*gs[i] < 0:
                    rets.append(refine(dqs[i0], dqs[i], dqs[i0]))
                g0, i0 = gs[i], i
                if len(rets) == 2:  # two solutions at max
                    break
                qa, qb = self.zoom_in_search(
                        delta0=self.delta, mu_eff=mu_eff, dmu_eff=dmu_eff,
                        delta_pred=delta, dq_pred=(q_lower + q_upper)/2.0,
                        max_iter=10)
                print(f"ZoomIn:{qa},{qb}")
                rets = [qa, qb]
        else:
            bExcept = False
            if dxs is None:
                dxs = np.array(
                    [1, 2, 4, 0.01, 0.25, 0.5, 8, 16, 0])*0.001
            if guess_lower is not None:
                try:
                    for dx in dxs:
                        ret1 = brentq(
                            f, max(0, guess_lower - dx), guess_lower + dx)
                        rets.append(ret1)
                        break
                except ValueError:
                    bExcept = True
                    rets.append(None)
            else:
                rets.append(None)
            if guess_upper is not None:
                try:
                    for dx in dxs:
                        ret2 = brentq(
                            f, max(0, guess_upper - dx), guess_upper + dx)
                        rets.append(ret2)
                        break
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

    def search(
            self, N_delta, mu_eff=None, dmu_eff=None, delta=None,
            delta_lower=0.001, delta_upper=1, q_lower=0, q_upper=0.2,
            N_q=40, auto_incremental=False, auto_save=True, flip_order=False):
        """
        search the delta-q curves in a point to point
        method, can be used to search single solution
        curve. Search possible states in ranges of delta and q
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
        deltas = np.linspace(delta_lower, delta_upper, N_delta)
        incremental_step = (delta_upper - delta_lower) / (N_delta + 1)
        if delta is None:
            delta = self.delta
        if mu_eff is None:
            mu_eff = self.mu_eff
        if dmu_eff is None:
            dmu_eff = self.dmu_eff
        self.C = self._get_C(mus_eff=(mu_eff, mu_eff), delta=delta)
        if flip_order and (abs(
            delta_upper - delta) < abs(delta - delta_lower)):
            self.print("search order flipped...")
            deltas = deltas[::-1]
        lg, ug = None, None

        def do_search(delta):
            nonlocal lg
            nonlocal ug
            ret = self.search_states(
                mu_eff=mu_eff, dmu_eff=dmu_eff,
                delta=delta, guess_lower=lg, guess_upper=ug,
                q_lower=q_lower, q_upper=q_upper, N_q=N_q, dx=dx)
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
                            self.print(f"dx={dx}")
                            ret0 = self.search_states(
                                mu_eff=mu_eff, dmu_eff=dmu_eff,
                                delta=delta_, guess_lower=lg, guess_upper=ug,
                                q_lower=q_lower, q_upper=q_upper,
                                N_q=N_q, dx=dx,
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
                    self.save_to_file(rets, extra_items={"pending": 0})
            if auto_incremental:
                print("Append 20 more search points")
                deltas = np.linspace(1, 20, 20)*incremental_step + deltas[-1]
            else:
                break
        self.print("Search Done")
        rets = ff_state_sort_data(rets)
        if auto_save:
            self.save_to_file(rets)
        return rets

    def zoom_in_search(
            self, mu_eff, dmu_eff, delta_pred, dq_pred,
            max_iter=5, N_q=10, dqs0=None, gs0=None,
            delta0=None, fun=None):
        """
        The zoom-in search algorithm
        -----------------------------
        It will keep to find a region with an extremum
        and keep zooming to that region unless solutions
        are found.
        TODO: implement the case with just one solution
            and do some tests.
        """
        dq1, dq2 = dq_pred*0.5, dq_pred*1.1
        p1, p2, v1, v2 = None, None, None, None
        if fun is None:
            def g(dq):
                return self._get_C(
                    mus_eff=(mu_eff + dmu_eff, mu_eff - dmu_eff),
                    delta=delta_pred, dq=dq) - self.C
        else:
            g = fun

        search_trace = []

        def add_trace(dqs, gs):
            for (dq_, g_) in zip(dqs, gs):
                search_trace.append((dq_, g_))

        def get_trace():
            dq_gs = np.array(search_trace)
            dq_gs = dq_gs[dq_gs[:, 0].argsort()]  # dq_gs.sort(axis=0)
            return dq_gs.T

        dqs = [0.00001]
        if delta0 is not None:
            dqs.append(delta0)
        gs = np.array([g(dq) for dq in dqs])
        add_trace(dqs=dqs, gs=gs)
        for i in range(max_iter):
            if i == 0 and dqs0 is not None and gs0 is not None:
                dqs = dqs0
                gs = gs0
            else:
                dqs = np.linspace(dq1, dq2, N_q)
                gs = np.array([g(dq) for dq in dqs])
            add_trace(dqs=dqs, gs=gs)

            self.print(f"{i+1}/{max_iter}:gs={gs}", end='')
            if np.all(gs > 0):  # this is not compete
                index, _ = min(enumerate(gs), key=operator.itemgetter(1))
                if index == 0:  # range expanded more to the left
                    dq1 = dq1*0.9
                    if p2 is None:
                        p2 = dqs[0]
                    continue
                if index == len(dqs) - 1:
                    dq2 = dq2*1.1
                    if p1 is None:
                        p1 = dqs[-1]
                    continue
                dq1, dq2 = dqs[index - 1], dqs[index + 1]
                p1, p2 = dq1, dq2
                continue

            if np.all(gs < 0):  # this is not compete
                index, _ = max(enumerate(gs), key=operator.itemgetter(1))
                if index == 0:  # range expanded more to the left
                    dq1 = dq1*0.9
                    if p2 is None:
                        p2 = dqs[0]
                    continue
                if index == len(dqs) - 1:
                    dq2 = dq2*1.1
                    if p1 is None:
                        p1 = dqs[-1]
                    continue
                dq1, dq2 = dqs[index - 1], dqs[index + 1]
                p1, p2 = dq1, dq2
                continue
            N_flip = self.check_sign_flip(gs)
            if N_flip > 0:  # have solutions
                rets = []
                dqs, gs = get_trace()
                g0, i0 = gs[0], 0
                for i in range(1, len(gs)):
                    if g0*gs[i] < 0:
                        rets.append(brentq(g, dqs[i], dqs[i0]))
                    g0, i0 = gs[i], i
                    if len(rets) == 2:  # two solutions at max
                        break
                if len(rets) > 1:
                    v1, v2 = rets[:2]
                elif len(rets) == 1:
                    v1 = rets[0]
            break
        self.print(f"ZoomIn: {v1},{v2}")
        return (v1, v2)

    def smart_search(
            self, mu_eff=None, dmu_eff=None, delta=None,
            q_lower=0, q_upper=0.2, N_q=40, max_points=100,
            delta0=0.0001, delta1=None, N_delta=5, tol_y=0.01,
            tol_x=1e-6, auto_save=True, **args):
        """
        A solution search algorithm that will use adaptive
        method and a smarter zoom-in algorithm to find the
        delta-q curves that represent valid states.
        Argument
        --------
        delta0: the minimum value of delta to search with
        N_delta: the initial number of deltas, the actual num
            will change but no more that the value defined
            by 'max_points'
        tol_y: define the desired spacing in y direction(q)
        tol_x: define the desired spacing in x direction(delta)
        NOTE: To have smooth curves, set N_delta to a large
            number, such as 100.
        """
        delta = self.delta if delta is None else delta
        delta1 = delta if delta1 is None else delta1
        mu_eff = self.mu_eff if mu_eff is None else mu_eff
        dmu_eff = self.dmu_eff if dmu_eff is None else dmu_eff
        # before searching, the C should be fixed, it's
        # equivalent to fix g in other code.
        self.C = self._get_C(mus_eff=(mu_eff, mu_eff), delta=delta)

        def do_search(delta, qa=None, qb=None, qu=None, ql=None):
            # A search function the call wrap the search state.
            # It's a regular point to point search, no smartness,
            # the results of this method will be used to predict
            # the end points and as the zoom-in algorithm kicks in,
            # the rest of search will only use zoom-in search.
            if qu is None:
                qu = q_upper
            if qb is None:
                ql = q_lower
            try:
                ret = self.smart_search_states(
                    mu_eff=mu_eff, dmu_eff=dmu_eff,
                    delta=delta, guess_lower=qa, guess_upper=qb,
                    q_lower=ql, q_upper=qu, N_q=N_q)
                return ret
            except ValueError:
                return [None, None]

        deltas = np.linspace(delta0, delta1, N_delta)
        output, rets, = [], []
        # turn off right to left sweep as this code
        # is designed for two-solution side.
        right_to_left_swept = True
        last_good_delta = delta0
        last_bad_delta = delta
        delta_ref = delta
        # if qa, qb are None, this will force the function
        # lda.smart_search_states to scan through the range
        # defined by (ql, qu), so qa, qb, ql, qu can not be
        # None at the same time. see the smart_search_states
        # for more information. If the predicted_q is not None
        # that means the zoom-in search algorithm kicks in
        qa, qb, qu, ql, predicted_q = None, None, None, None, None
        done = False
        while(not done):
            for delta in deltas:
                if predicted_q is None:
                    # if predicted_q is None, no prediction has been made,
                    # then we just use the dumb method to search
                    qa, qb = do_search(delta=delta, qa=qa, qb=qb, qu=qu, ql=ql)
                else:
                    # else use the zoom-in algorithm to search
                    qa, qb = self.zoom_in_search(
                        delta0=self.delta,
                        delta_pred=delta, dq_pred=predicted_q,
                        mu_eff=mu_eff, dmu_eff=dmu_eff, max_iter=10)
                    # predicted_q = None  # reset the common q
                if not (qa is None or qb is None):
                    # if we have two solution
                    rets.append((qa, qb, delta))
                    # check if reach max points
                    if len(rets) > max_points:
                        done = True
                        break
                    # check if reach x limit, i.e., two set of solutions
                    # are too close, which may indicate end of search.
                    # This is not very strict, can be modified.
                    if len(rets) > 2:
                        _, _, d1 = rets[-1]
                        _, _, d2 = rets[-2]
                        if abs((d1 - d2)/(d1 + d2)) < tol_x:
                            print("Reach delta limit")
                            done = True
                            break
                    # store the valid delta
                    last_good_delta = delta
                    # check if to save to persist media
                    if auto_save:
                        print(f"Added {len(rets)} :{rets[-1]}")
                        self.save_to_file(rets, extra_items={"pending": 0})
                else:
                    # if have no of only one solution, not good
                    # then we need to go a step back to a better
                    # candidate delta that may sit in between the
                    # current failing delta and the last good one
                    new_delta = (last_good_delta + delta)/2
                    # make sure the new delta is different from the last
                    # good delta
                    if len(rets) == 0:
                        raise ValueError("Can't find any solution pair")

                    deltas = [new_delta]
                    # set the last bad delta to current one
                    last_bad_delta = delta
                    # used the last solutions as start point
                    # for the next search.
                    qa, qb, _ = rets[-1]
                    if predicted_q is not None:
                        # if predicted_q is valid, that may mean
                        # we are getting close the final joint point
                        # where the do_search method probably will be
                        # fail to work as solutions near the joint point
                        # are extremely close to each other, we have to
                        # zoom in to check the actual solutions.
                        qu, ql = max(qa, qb), min(qa, qb)
                        qa, qb = None, None  # because we want to scan
                    break
                # check if we are at the last item of deltas list
                if delta == deltas[-1]:
                    # if so, we need to check if the last pair of solution
                    # satisfies the dessired resolution defined by tol_y as
                    # we want to the final two point to as close as it can.
                    # the closeness is defined by tol_y x the spacing of the
                    # two solution in the first pair, by default it one out
                    # of one hundred of that spacing.
                    qa_, qb_, _ = rets[0]
                    if (abs(qa - qb) > abs(qa_ - qb_)*tol_y):
                        # if still too far away, we may need more deltas
                        self.print(
                            f"Acc:{abs(qa - qb)},{abs(qa_ - qb_)*tol_y}")
                        if abs(delta - last_bad_delta)/last_bad_delta < tol_y:
                            # if current delta is very close to the last bad
                            # delta, we will try to predict the final delta
                            # which will replace current last bad delta.
                            delta_qs = predict_joint_point(rets)
                            # sign define sweeping direction of delta
                            sign = -1.0 if right_to_left_swept else 1.0
                            if delta_qs is None:
                                # if prediction fails, due to not enough
                                # data points or bad data points. we simply
                                # change the last bad delta a bit larger or
                                # smaller depends on the direction of sweeping
                                last_bad_delta = last_bad_delta*(
                                    1 + 5.0*sign*tol_y)
                                self.print(f"Test bad delta:{last_bad_delta}")
                                deltas = [last_bad_delta]
                            else:
                                # we a prediction is made, we can try to search
                                # around the predicted region.
                                last_bad_delta, predicted_q = delta_qs
                                self.print(f"Predicted final point{delta_qs}")
                                q1, q2, last_good_dalta = rets[-1]
                                deltas = [(
                                    0.1*last_good_dalta + 0.9*last_bad_delta)]
                                qu, ql = max(q1, q2), min(q1, q2)
                                qa, qb = None, None  # because we want to scan
                        else:
                            # a valid delta may in the middle of the
                            # last bad delta and current delta
                            deltas = [(last_bad_delta + delta)/2]
                    else:
                        self.print("Searching completed")
                        deltas = []
                        done = True  # search end
                    break
            if done:
                output.extend(rets)
                rets = []
                # if we need to sweep delta from right to left
                # by default not. To-Do: update the code to support
                # single solution search to unify the code in the file
                if not right_to_left_swept:
                    done = False
                    right_to_left_swept = True
                    deltas = delta_ref - np.linspace(delta0, delta, N_delta)
                    qa, qb, qu, ql, predicted_q = None, None, None, None, None
        output = ff_state_sort_data(output)
        if auto_save:
            self.save_to_file(output)
        return output


def search_delta_q_worker(para):
    smart_search = True
    mu_eff, dmu_eff, delta, dim, k_c = para
    functionalType = FunctionalType.BDG
    kernelType = KernelType.HOM
    args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
        T=0, dim=dim, k_c=k_c, verbosity=False)
    # try:
    lda = ClassFactory(
        "LDA", (FFStateAgent,),
        functionalType=functionalType,
        kernelType=kernelType, args=args)
    if smart_search:
        return lda.smart_search(
            delta_lower=0.001, delta_upper=delta,
            q_lower=0, q_upper=dmu_eff, N_q=40, delta1=delta,
            N_delta=100)
    return lda.search(
        N_delta=50, delta_lower=0.0001, delta_upper=delta,
        q_lower=0, q_upper=dmu_eff, N_q=10,
        auto_incremental=False, flip_order=True)
    # except ValueError:
    #     return None


def smart_search_delta_q_worker(obj_mus_delta_dim_kc):
    """
    a worker method used to call the lda object with
    FF agent methods to search the delta-q curves.
    Only for two-solution curves, the one solution curve
    is not a potential Fulde Ferrell state.
    """
    obj, mu_eff, dmu_eff, delta, dim, k_c = obj_mus_delta_dim_kc
    args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
        T=0, dim=dim, k_c=k_c, verbosity=True)
    try:
        lda = ClassFactory(
            "LDA", (FFStateAgent,),
            functionalType=obj.functionalType,
            kernelType=obj.kernelType, args=args)
        return lda.smart_search(
            delta_lower=0.001, delta_upper=delta,
            q_lower=0, q_upper=dmu_eff, N_q=40, delta1=obj.delta1,
            N_delta=obj.N_delta)
    except:
        # rename the file to be incomplete
        file_name = lda.get_file_name()
        if os.path.exists(file_name):
            os.rename(file_name, file_name + ".error")
        return None


def search_condidate_worker(obj_mus_delta_dim_kc):
    obj, mu_eff, dmu_eff, delta, dim, k_c = obj_mus_delta_dim_kc
    if dmu_eff < 0 or mu_eff < 0 or delta < 0:
        return []
    args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
        T=0, dim=dim, k_c=k_c, verbosity=False)
    try:
        lda = ClassFactory(
            "LDA", (FFStateAgent,),
            functionalType=obj.functionalType,
            kernelType=obj.kernelType, args=args)

        return lda.search(
            N_delta=2, delta_lower=0.0001, delta_upper=0.01,
            q_lower=0, q_upper=dmu_eff, N_q=10,
            auto_save=False)
    except ValueError:
        return None


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
        prefix=f"{output_fileName}", time_stamp=False)
    lda = ClassFactory(
        "LDA", (FFStateAgent,),
        functionalType=ClassFactory(functionalIndex=functional_index),
        kernelType=KernelType.HOM, args=args)
    C_ = lda._get_C(mus_eff=mus_eff, delta=delta)
    assert np.allclose(C, C_, rtol=1e-16)  # verify the C value
    lda.C = C
    if os.path.exists(lda.get_file_name()):
        return None
    normal_pressure = lda.get_ns_mus_e_p(mus_eff=mus_eff, delta=0)[3]

    print(f"Processing {lda.get_file_name()}")
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
        lda.save_to_file(output, extra_items={"p0": normal_pressure})
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


class AutoPDG(object):

    def __init__(
            self, functionalType, kernelType, dim=2,
            mu0=10, k_c=200, data_dir=None, dx=0.05,
            poolSize=9):
        self.mu_eff = mu0
        self.k_c = k_c
        self.dim = dim
        self.dx = dx
        self.functionalType = functionalType
        self.kernelType = kernelType
        self.poolSize = poolSize
        self.time_stamp = self.get_time_stamp()
        self.condidate_trace = []
        if data_dir is None:
            data_dir = os.path.dirname(
                os.path.abspath(inspect.getfile(
                    inspect.currentframe())))

    def get_time_stamp(self):
        """return a time stamp in file naming format"""
        return time.strftime("%Y_%m_%d_%H_%M_%S")

    def is_candidate(self, res):
        """
        check if a pair of (dmu, and delta) is good candidate
        """
        n = len(res)
        if n == 0:
            return False
        r = res[0]
        flag = ((r[0] is not None) and (r[1] is not None))
        if not flag:
            return False
        if n == 2:
            r = res[1]
            flag = ((r[0] is not None) and (r[1] is not None))
            if not flag:
                return False
            r1, r2 = res
            if min(r1[:2]) < min(r2[:2]):
                return True
        return False

    def save_to_file(self, output, file):
        with open(file, 'w') as wf:
            json.dump(output, wf)

    def compute_pressure_current_from_files(self, root=None):
        compute_pressure_current(root=root)

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

    def get_delta_q_file(self, delta, dmu):
        #  pid = os.getpid()
        ts = self.get_time_stamp()
        prefix = "delta_q_"
        fileName = (
                prefix + f"({self.dim}d_{delta:.2f}"
                + f"_{self.mu_eff:.2f}_{dmu:.2f})" + ts)
        return fileName

    def offset_para(self, p, factor=1, prop=False):
        """
        prop: specify if the offset is in proportion to p
        """
        offset = np.array([-1, 0, 1])
        if prop:
            return self.dx*p*offset*factor + p
        return self.dx*offset*factor + p

    def mix_para2(self, p1, p2):
        return list(itertools.product(p1, p2))

    def search_delta_q_diagram(self, seed_delta, seed_dmu, only_one=True):
        """"""
        print(f"delta={seed_delta},dmu={seed_dmu}")
        dmus = self.offset_para(seed_dmu)
        deltas = self.offset_para(seed_delta)
        dmu_delta_ls = self.mix_para2(dmus, deltas)
        self.N_delta = 100
        self.delta1 = seed_delta
        paras = [(
            self, self.mu_eff, dmu,
            delta, self.dim, self.k_c) for (dmu, delta) in dmu_delta_ls]
        if only_one:
            paras = [(
                self, self.mu_eff, seed_dmu,
                seed_delta, self.dim, self.k_c)]
        res = PoolHelper.run(
            smart_search_delta_q_worker, paras, poolsize=self.poolSize)
        return (dmu_delta_ls, res)

    def is_good_candidate(self, dmu_delta, trace_list,  dx=None):
        dmu, delta = dmu_delta
        if dmu <= 0 or delta <= 0:
            return False
        if dx is None:
            dx = self.dx
        if trace_list is None:
            trace_list = self.condidate_trace
        for (dmu_, delta_) in trace_list:
            dis = ((dmu - dmu_)**2 + (delta - delta_)**2)**0.5
            if dis < dx:
                return False
            return True

    def search_valid_conditate(self, seed_delta, seed_dmu):
        trail = 1
        trail_max = 10
        while(trail < trail_max):
            print(f"#{trail} Delta={seed_delta}, dmu={seed_dmu}")
            dmus = self.offset_para(seed_dmu, factor=trail, prop=True)
            deltas = self.offset_para(seed_delta, factor=trail, prop=True)
            dmu_delta_ls = self.mix_para2(dmus, deltas)
            paras = [(
                self, self.mu_eff, dmu,
                delta, self.dim, self.k_c) for (dmu, delta) in dmu_delta_ls]
            res = PoolHelper.run(
                search_condidate_worker, paras, poolsize=self.poolSize)

            direction_flags = [self.is_candidate(re) for re in res]
            # filter out points inside the region being searched
            for (index, dmu_delta) in enumerate(dmu_delta_ls):
                if not direction_flags[index]:
                    continue
                if self.is_good_candidate(
                        dmu_delta=dmu_delta, trace_list=self.condidate_trace):
                    continue
                direction_flags[index] = False

            if not any(np.array(direction_flags)):  # when no state is found
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
                    return (dmu_delta_ls, direction_flags)
                # the current point is the best, that need more check
                if index == len(dmu_delta_ls)//2:
                    trail = trail + 1
                    continue  # try next trail

                seed_dmu, seed_delta = dmu_delta_ls[index]
                trail = 1  # reset the trail number for next step
                continue
            # if they are good start states
            print(dmu_delta_ls, direction_flags)
            return (dmu_delta_ls, direction_flags)
        return (dmu_delta_ls, direction_flags)

    def scan_valid_parameter_space(self, dmu_delta_flags):
        """"
        start with a valid candidate point, search the entire region
        """
        output_file = f"valid_region_{self.get_time_stamp()}.json"
        dmu_deltas, flags = dmu_delta_flags
        output = []
        valid_point_queue = deque()

        while(True):
            for index, flag in enumerate(flags):
                if flag:
                    output.append(dmu_deltas[index])
            flags[len(flags)//2] = False
            for index, flag in enumerate(flags):
                if flag and self.is_good_candidate(
                        dmu_delta=dmu_deltas[index],
                        trace_list=valid_point_queue):
                    valid_point_queue.append(dmu_deltas[index])
            self.save_to_file(output=output, file=output_file)
            if len(valid_point_queue) == 0:
                break
            seed_dmu, seed_delta = valid_point_queue.popleft()
            dmu_delta_flags = self.search_valid_conditate(
                seed_delta=seed_delta, seed_dmu=seed_dmu)

    def run(self, seed_delta, seed_dmu):
        self.condidate_trace = []
        res = self.search_valid_conditate(
            seed_delta=seed_delta, seed_dmu=seed_dmu)
        self.scan_valid_parameter_space(res)


def search_delta_q_manager(delta):
    """"
    search the delta-q curves in a point to point
    method, can be used to search single solution
    curve
    """
    mu = 10
    step = 0.1
    N = int(delta/0.1)
    dmus = np.array(list(range(N)))*step + step
    # dmus = np.linspace(0.01, delta, 10)
    paras = [(mu, dmu, delta, 2, 150) for dmu in dmus]
    PoolHelper.run(
        search_delta_q_worker, paras, poolsize=5)


def PDG():
    pdg = AutoPDG(
        functionalType=FunctionalType.BDG,
        kernelType=KernelType.HOM, k_c=150, dim=2)
    # Bug case: delta=0.6,dmu=0.45, single solution test
    # single solution bug case:
    # delta, dmu = 0.2, 0.14141782308472947
    delta, dmu = 2.7, 2.65 # no initial solution bug
    # delta, dmu = 1.5, 1.75
    # dmu, delta = 0.35349820923398134, .5  # 0.175, 0.25 # for 3D
    pdg.search_delta_q_diagram(seed_delta=delta, seed_dmu=dmu)
    # pdg.compute_pressure_current_from_files()
    # pdg.run(seed_delta=delta, seed_dmu=dmu)


if __name__ == "__main__":
    # search_delta_q_manager(delta=1.5)
    PDG()
    compute_pressure_current()
