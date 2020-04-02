
import inspect
import time
import glob
import json
import os
import sys
import numpy as np
from scipy.optimize import brentq
from mmf_hfb.class_factory import FunctionalType, KernelType
from mmf_hfb.class_factory import ClassFactory
from mmf_hfb.parallel_helper import PoolHelper
currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from fulde_ferrell_state import FFState
from fulde_ferrell_state_worker_thread import fulde_ferrell_state_solve_thread


def FFVortex(mus=None, delta=None, L=8, N=32, R=None, k_c=None, N1=10, N2=10):
    mu_a, mu_b = mus
    mu, dmu = (mu_a + mu_b)/2, (mu_a - mu_b)/2
    dx = L/N
    if R is None:
        R = L/2
    k_F = np.sqrt(2*mu)
    args = dict(mu=mu, dmu=0, delta=delta, dim=2, k_c=k_c)
    f = FFState(fix_g=True, **args)
    rs = np.linspace(0.0001, 1, N1)
    rs = np.append(rs, np.linspace(1.01, R, N2))
    paras = [(f, mu, dmu, delta, r) for r in rs]
    ds = PoolHelper.run(fulde_ferrell_state_solve_thread, paras=paras)        
    #ds = [f.solve(mu=mu, dmu=dmu, dq=0.5/_r, a=0.001, b=2*delta) for _r in rs]
    for i in range(len(ds)):
        ps = [f.get_pressure(
            mu_eff=mu, dmu_eff=dmu, delta=d, dq=0.5/r,
            use_kappa=False).n for r, d in zip(rs, ds)]
        ps0 = [f.get_pressure(
            mu_eff=mu, dmu_eff=dmu, delta=1e-8, q=0, dq=0,
            use_kappa=False).n for r, d in zip(rs, ds)]
    na = np.array([])
    nb = np.array([])
    for i in range(len(rs)):
        na_, nb_ = f.get_densities(delta=ds[i], dq=0.5/rs[i], mu=mu, dmu=dmu)
        na = np.append(na, na_.n)
        nb = np.append(nb, nb_.n)  
    n_p = na + nb  
    n_m = na - nb
    j_a = []
    j_b = []   
    js = [f.get_current(mu=mu, dmu=dmu, delta=d, dq=0.5/r) for r, d in zip(rs, ds)]
    for j in js:
        j_a.append(j[0].n)
        j_b.append(j[1].n)
    j_a, j_b = np.array(j_a), np.array(j_b)
    j_p, j_m = -(j_a + j_b), j_a - j_b
    return (rs/dx, ds, ps, ps0, n_p, n_m, j_a, j_b)


class FFStateAgent(object):
    """
    An agent class used for searching FF states
    """

    def __init__(
            self, mu_eff, dmu_eff, delta, dim, k_c, **args):
        self.delta = delta
        self.mu_eff = mu_eff
        self.dmu_eff = dmu_eff
        self.k_c = k_c

    def solve(
            self, mu=None, dmu=None, q=0, dq=0,
            a=None, b=None, throwException=False, **args):
        """
        On problem with brentq is that it requires very smooth function with a
        and b having different sign of values, this can fail frequently if our
        integration is not with high accuracy. Should be solved in the future.
        """
        assert (mu is None) == (dmu is None)
        if mu is None:
            mu_a, mu_b = self.mus
            mu, dmu = (mu_a + mu_b)/2.0, (mu_a - mu_b)/2.0

        if a is None:
            a = self.delta*0.1
        if b is None:
            b = self.delta*2

        def f(delta):
            return self._get_C(
                mus_eff=(mu + dmu, mu - dmu),
                delta=delta, dq=dq) - self.C

        self._delta = None  # a another possible solution
        if throwException:
            delta = brentq(f, a, b)
        else:
            try:
                delta = brentq(f, a, b)
            except ValueError:
                offset = 0
                if not np.allclose(abs(dmu), 0):
                    offset = min(abs(dq/dmu), 100)
                ds = np.linspace(
                    0.01, max(a, b)*(2 + offset),
                    min(100, int((2 + offset)*10)))

                assert len(ds) <= 100
                f0 = f(ds[-1])
                index0 = -1
                delta = 0
                for i in reversed(range(0, len(ds)-1)):
                    f_ = f(ds[i])
                    if f0*f_ < 0:
                        delta = brentq(f, ds[index0], ds[i])
                        if f_*f(ds[0]) < 0:  # another solution
                            delta_ = brentq(f, ds[0], ds[i])
                            self._delta = delta_

                            p_ = self.get_pressure(
                                mu_eff=mu, dmu_eff=dmu,
                                delta=delta_, q=q, dq=dq)
                            p = self.get_pressure(
                                mu_eff=mu, dmu_eff=dmu,
                                delta=delta, q=q, dq=dq)
                            print(
                                f"q={dq}: Delta={delta_}/{delta}"
                                + f",Pressue={p_}/{p}")
                            if p_ > p:
                                self._delta = delta
                                delta = delta_
                        break
                    else:
                        f0 = f_
                        index0 = i
                if (
                    delta == 0 and (f(
                        0.999*self.delta)*f(
                            1.001*self.delta) < 0)):
                    delta = brentq(f, 0.999*self.delta, 1.001*self.delta)
        return delta


def create_ffs_lda(
        mu, dmu, delta, k_c, dim=2,
        functionalType=FunctionalType.BDG, **args):
    """return a FF state object"""
    args = dict(mu_eff=mu, dmu_eff=dmu, delta=delta, T=0, dim=dim, k_c=k_c)
    lda = ClassFactory(
        "LDA", (FFStateAgent,),
        functionalType=functionalType,
        kernelType=KernelType.HOM, args=args)
    lda.C = lda._get_C(mus_eff=(mu + dmu, mu - dmu), delta=delta)
    return lda


def FFVortexFunctional(
        mus=None, delta=None, k_c=None, N=None, L=None, dim=2,
        functionalType=FunctionalType.BDG, N1=10, N2=10):
    """
        A function to compute parameter for a vortex structure
        NOTE: when dq( or 1/r) larger than k_c, that would be
        problematic.
    """
    mu_a, mu_b = mus
    mu, dmu = (mu_a + mu_b)/2, (mu_a - mu_b)/2
    dx = L/N
    R = L/2
    args = dict(mu=mu, dmu=0, delta=delta, dim=dim, k_c=k_c)
    f = create_ffs_lda(functionalType=functionalType, **args)
    rs = np.linspace(0.01, 1, N1)
    rs = np.append(rs, np.linspace(1.01, R, N2))
    # paras = [(f, mu, dmu, delta, r) for r in rs]
    # ds = PoolHelper.run(fulde_ferrell_state_solve_thread, paras=paras)        
    ds = [f.solve(mu=mu, dmu=dmu, dq=0.5/_r, a=0.001, b=2*delta) for _r in rs]
    print(ds)
    for i in range(len(ds)):
        ps = [f.get_pressure(
            mu_eff=mu, dmu_eff=dmu, delta=d, dq=0.5/r,
            use_kappa=False) for r, d in zip(rs, ds)]
        ps0 = [f.get_pressure(
            mu_eff=mu, dmu_eff=dmu, delta=1e-8, q=0, dq=0,
            use_kappa=False) for r, d in zip(rs, ds)]
    na = np.array([])
    nb = np.array([])
    for i in range(len(rs)):
        res = f.get_densities(
            delta=ds[i], dq=0.5/rs[i], mus_eff=(mu + dmu, mu - dmu))
        na = np.append(na, res.n_a)
        nb = np.append(nb, res.n_b)
    n_p = na + nb
    n_m = na - nb
    j_a = []
    j_b = []   
    js = [f.get_current(
        mus_eff=(
            mu + dmu, mu - dmu), delta=d, dq=0.5/r) for r, d in zip(rs, ds)]
    for j in js:
        j_a.append(j[0].n)
        j_b.append(j[1].n)
    j_a, j_b = np.array(j_a), np.array(j_b)
    return (rs/dx, ds, ps, ps0, n_p, n_m, j_a, j_b)


if __name__ == "__main__":
    mu = 10
    dmu = 4.5
    delta = 7.5
    mus = (mu + dmu, mu - dmu)
    FFVortexFunctional(mus=mus, delta=delta, k_c=14, L=8, N=32)
