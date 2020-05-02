
import inspect
import json
import os
import sys
import numpy as np
from scipy.optimize import brentq
from mmf_hfb.class_factory import FunctionalType, KernelType
from mmf_hfb.class_factory import ClassFactory, Solvers
from mmf_hfb import hfb, homogeneous
from mmf_hfb.utils import clockwise
from IPython.display import clear_output
import matplotlib.pyplot as plt
from mmfutils.plot import imcontourf
from collections import namedtuple
currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from fulde_ferrell_state import FFState


def to_list(c):
    """convert complex to two-component list"""
    return [c.real, c.imag]


def res_to_json(res):
    """
    convert the res to dict object for json file
    """
    output = {}
    try:
        output['j_a'] = to_list(res.j_a)
        output['j_b'] = to_list(res.j_b)
        output['tau_a'] = to_list(res.tau_a)
        output['tau_b'] = to_list(res.tau_b)
        output['n_a'] = to_list(res.n_a)
        output['n_b'] = to_list(res.n_b)
        output['nu'] = to_list(res.nu)
    except:
        return None
    return output


def json_to_res(res):
    """from json format to named structure"""
    def to_c(key):
        res_r, res_i = res[key]
        return np.array(res_r) + 1j*np.array(res_i)

    Densities = namedtuple(
            'Densities', ['n_a', 'n_b', 'tau_a', 'tau_b', 'nu', 'j_a', 'j_b'])
    return Densities(
        n_a=to_c("n_a"), n_b=to_c("n_b"),
        tau_a=to_c("tau_a"), tau_b=to_c("tau_b"),
        nu=to_c("nu"),
        j_a=to_c("j_a"), j_b=to_c("j_b"))


class PlotBase(object):
    dim = None
    xyz = None
    Delta = None

    def plot(self, fig=None, res=None):

        if self.dim == 1:
            x = self.xyz[0]
            if fig is None:
                fig = plt.figure(figsize=(20, 10))
            plt.subplot(233)
            plt.plot(x, abs(self.Delta))
            plt.title(r'$|\Delta|$')

            if res is not None:
                plt.subplot(231)
                plt.plot(x, (res.n_a+res.n_b).real)
                plt.title(r'$n_+$')

                plt.subplot(232)
                plt.plot(x, (res.n_a - res.n_b).real)
                plt.title(r'$n_-$')

                plt.subplot(234)
                j_a = res.j_a[0]
                j_b = res.j_b[0]
                j_p = j_a + j_b
                plt.plot(x, j_a)
                plt.title(r'$j_a$')

                plt.subplot(235)
                plt.plot(x, j_b)
                plt.title(r'$j_b$')

                plt.subplot(236)
                plt.plot(x, j_p)
                plt.title(r'$j_+$')
        else:
            x, y = self.xyz[:2]
            if fig is None:
                fig = plt.figure(figsize=(20, 10))
            plt.subplot(233)
            if self.dim == 2:
                imcontourf(x, y, abs(self.Delta), aspect=1)
            elif self.dim == 3:
                imcontourf(x, y,  np.sum(abs(self.Delta), axis=2))
            plt.title(r'$|\Delta|$')
            plt.colorbar()

            if res is not None:
                plt.subplot(231)
                imcontourf(x, y, (res.n_a+res.n_b).real, aspect=1)
                plt.title(r'$n_+$')
                plt.colorbar()

                plt.subplot(232)
                imcontourf(x, y, (res.n_a-res.n_b).real, aspect=1)
                plt.title(r'$n_-$')
                plt.colorbar()

                plt.subplot(234)
                j_a = res.j_a[0] + 1j*res.j_a[1]
                j_b = res.j_b[0] + 1j*res.j_b[1]
                j_p = j_a + j_b
                imcontourf(x, y, abs(j_a), aspect=1)
                plt.title(r'$j_a$')
                plt.colorbar()
                plt.quiver(x.ravel(), y.ravel(), j_a.real, j_a.imag)

                plt.subplot(235)
                imcontourf(x, y, abs(j_b), aspect=1)
                plt.title(r'$j_b$'); plt.colorbar()
                plt.quiver(x.ravel(), y.ravel(), j_b.real, j_b.imag)

                plt.subplot(236)
                imcontourf(x, y, abs(j_p), aspect=1)
                plt.title(r'$j_+$')
                plt.colorbar()
                plt.quiver(x.ravel(), y.ravel(), j_p.real, j_p.imag)
        return fig


def plot_2D(self, fig=None, res=None, fontsize=36):
    """
    plot the 2D bcs vortices
    """
    x, y = self.xyz[:2]
    # res = self.res if res is None else res
    if fig is None:
        fig = plt.figure(figsize=(20, 10))
    plt.subplot(233)
    if self.dim == 2:
        imcontourf(x, y, abs(self.Delta), aspect=1)
    elif self.dim == 3:
        imcontourf(x, y,  np.sum(abs(self.Delta), axis=2))
    plt.title(r'$|\Delta|$', fontsize=fontsize)
    plt.colorbar()

    if res is not None:
        plt.subplot(231)
        imcontourf(x, y, (res.n_a+res.n_b).real, aspect=1)
        plt.title(r'$n_+$', fontsize=fontsize)
        plt.colorbar()

        plt.subplot(232)
        imcontourf(x, y, (res.n_a-res.n_b).real, aspect=1)
        plt.title(r'$n_-$', fontsize=fontsize)
        plt.colorbar()

        plt.subplot(234)

        j_a = res.j_a[0] + 1j*res.j_a[1]
        j_b = res.j_b[0] + 1j*res.j_b[1]
        j_p = j_a + j_b
        # j_m = j_a - j_b
        # utheta = np.exp(1j*np.angle(x + 1j*y))
        imcontourf(x, y, abs(j_a), aspect=1)
        plt.title(r'$j_a$', fontsize=fontsize)
        plt.colorbar()
        plt.quiver(x.ravel(), y.ravel(), j_a.real, j_a.imag)

        plt.subplot(235)
        imcontourf(x, y, abs(j_b), aspect=1)
        plt.title(r'$j_b$', fontsize=fontsize); plt.colorbar()
        plt.quiver(x.ravel(), y.ravel(), j_b.real, j_b.imag)

        plt.subplot(236)
        imcontourf(x, y, abs(j_p), aspect=1)
        plt.title(r'$j_+$',fontsize=fontsize); plt.colorbar()
        plt.quiver(x.ravel(), y.ravel(), j_p.real, j_p.imag)


def plot_all(
        vs=[], hs=[], mu=10, fontsize=14,
        xlim=12, ls='-', one_col=False, dx=None, dx_text="dx", **args):
    """
    plot bcs and homogeneouse vortex data
    -----------
    vs: bcs vortices list
    hs: homogeneous results list
    one_col: bool,  if true, stack plots in single column
    """
    for v in vs:
        mu = sum(v.mus)/2
        if dx is None:
            dx = v.dxyz[0]
        r = np.sqrt(sum(_x**2 for _x in v.xyz))
        k_F = np.sqrt(2*mu)
        plt.subplot(511) if one_col else plt.subplot(321) 
        plt.plot(r.ravel()/dx, abs(v.Delta).ravel()/mu, '+', label="BCS")
        plt.ylabel(r'$\Delta/E_F$', fontsize=fontsize)
        plt.xlim(0, xlim)
        plt.subplot(512) if one_col else plt.subplot(323)  
        res = v.res
        plt.plot(
            r.ravel()/dx, abs(res.n_a + res.n_b).ravel()/k_F, '+', label="BCS")
        plt.ylabel(r"$n_p/k_F$", fontsize=fontsize)
        plt.xlim(0, xlim)
        plt.subplot(513) if one_col else plt.subplot(324)
        plt.plot(
            r.ravel()/dx, abs(res.n_a - res.n_b).ravel()/k_F, '+', label="BCS")
        plt.ylabel(r"$n_m/k_F$", fontsize=fontsize)
        plt.xlim(0, xlim)
        plt.ylim(-1, 1)
        x, y = v.xyz
        r_vec = x+1j*y
        j_a_ = res.j_a[0] + 1j*res.j_a[1]
        j_a_ = clockwise(r_vec, j_a_)*np.abs(j_a_)
        j_b_ = res.j_b[0] + 1j*res.j_b[1]
        j_b_ = clockwise(r_vec, j_b_)*np.abs(j_b_)
        plt.subplot(514) if one_col else plt.subplot(325)
        plt.plot(r.ravel()/dx, j_a_.ravel(), '+', label="BCS")
        plt.ylabel(r"$j_a$", fontsize=fontsize)
        plt.xlim(0, xlim)
        plt.subplot(515) if one_col else plt.subplot(326)
        plt.plot(r.ravel()/dx, j_b_.ravel(), '+', label="BCS")
        plt.ylabel(r"$j_b$", fontsize=fontsize)
        plt.xlim(0, xlim)
    # homogeneous part
    for res_h in hs:
        dx_, rs, ds, ds_ex = res_h.dx, res_h.rs, res_h.ds, res_h.ds_ex
        if dx is None:
            dx = dx_
        n_p, n_m, j_a, j_b = res_h.n_p, res_h.n_m, res_h.j_a, res_h.j_b
        k_F = np.sqrt(2*mu)
        rs_ = []
        ds_ = []
        for (r, d) in ds_ex:
            rs_.append(r)
            ds_.append(d)
        rs = np.array(rs)/dx
        plt.subplot(511) if one_col else plt.subplot(321)
        l, = plt.plot(rs, np.array(ds)/mu, ls, label="Homogeneous")
        if len(rs_) > 0:
            plt.plot(rs_, ds_, '--', c=l.get_c())
        plt.legend()
        plt.ylabel(r'$\Delta/E_F$', fontsize=fontsize)
        plt.subplot(512) if one_col else plt.subplot(323)
        plt.plot(rs, n_p/k_F, ls, label="Homogeneous")
        plt.ylabel(r"$n_p/k_F$", fontsize=fontsize)
        plt.legend()
        plt.subplot(513) if one_col else plt.subplot(324)
        plt.plot(rs, n_m/k_F, ls, label="Homogeneous")
        plt.ylabel(r"$n_m/k_F$", fontsize=fontsize)
        plt.legend()
        plt.subplot(514) if one_col else plt.subplot(325)
        plt.plot(rs, j_a, ls, label="Homogeneous")
        plt.ylabel(r"$j_a$", fontsize=fontsize)
        if not one_col:
            plt.xlabel(r"$r/$"+f"{dx_text}", fontsize=fontsize)
        plt.axhline(0, linestyle='dashed')
        plt.legend()
        plt.subplot(515) if one_col else plt.subplot(326)
        plt.plot(rs, j_b,  ls, label="Homogeneous")
        plt.axhline(0, linestyle='dashed')
        plt.xlabel(r"$r/$"+f"{dx_text}", fontsize=fontsize)
        plt.ylabel(r"$j_b$", fontsize=fontsize)
        plt.legend()


def FFVortex(
        mus=None, delta=None, L=8, N=32,
        R=None, k_c=None, N1=10, N2=10,
        ds=None, pressure_flag=False):
    """
        a function to solve gap equations for different q in a vortex.
        then the results deltas are used to compute densities, pressues
        and so on. The results can be compared to the results from a
        vortex in a box.
        NOTE: this can be discared as another function 'FFVortexFunctional'
        can produce the same result.
        NOTE: The k_c is not going to change the result as g will change
            in way that keep the gap and densities unchanged.
    """
    mu_a, mu_b = mus
    mu, dmu = (mu_a + mu_b)/2, (mu_a - mu_b)/2
    dx = L/N
    if R is None:
        R = L/2
    # k_F = np.sqrt(2*mu)
    args = dict(mu=mu, dmu=0, delta=delta, dim=2, k_c=k_c)
    f = FFState(fix_g=True, **args)
    #  for r close to the vortex core, some more points
    rs = np.linspace(0.1, 1, N1)
    if N2 > 0 and R > 1.1:
        rs = np.append(rs, np.linspace(1.1, R, N2))
    ds_ex = []
    if ds is None:
        ds = []
        for _r in rs:
            d = f.solve(
                mu=mu, dmu=dmu, dq=0.5/_r, a=0.001, b=2*delta)
            ds.append(d)
            if f.delta_ex is not None:
                ds_ex.append((_r, f.delta_ex))
        # ds = [f.solve(
        #     mu=mu, dmu=dmu, dq=0.5/_r, a=0.001, b=2*delta) for _r in rs]
    ps, pn = [], []
    if pressure_flag:
        for i in range(len(ds)):
            ps = [f.get_pressure(
                mu_eff=mu, dmu_eff=dmu, delta=d, dq=0.5/r,
                use_kappa=False).n for r, d in zip(rs, ds)]
            pn = [f.get_pressure(
                mu_eff=mu, dmu_eff=dmu, delta=1e-8, q=0, dq=0,
                use_kappa=False).n for r, d in zip(rs, ds)]
    na = np.array([])
    nb = np.array([])
    for i in range(len(rs)):
        na_, nb_ = f.get_densities(
            delta=ds[i], dq=0.5/rs[i], mu=mu, dmu=dmu)
        na = np.append(na, na_.n)
        nb = np.append(nb, nb_.n)
    n_p = na + nb
    n_m = na - nb
    j_a = []
    j_b = []
    js = [f.get_current(
        mu=mu, dmu=dmu, delta=d, dq=0.5/r) for r, d in zip(rs, ds)]
    for j in js:
        j_a.append(j[0].n)
        j_b.append(j[1].n)
    j_a, j_b = np.array(j_a), np.array(j_b)
    Results = namedtuple(
        'Results',
        ['dx', 'rs', 'ds', 'ds_ex', 'dqs', 'p_s', 'p_n', 'n_p', 'n_m', 'j_a', 'j_b'])
    return Results(
        dx=dx, rs=rs, dqs=0.5/rs, ds=ds, ds_ex=ds_ex, p_s=ps, p_n=pn,
        n_p=n_p, n_m=n_m, j_a=j_a, j_b=j_b)


class Vortex(hfb.BCS):
    """BCS Vortex class"""
    barrier_width = 0.2
    barrier_height = 100.0

    def __init__(self, Nxyz=(32, 32), Lxyz=(3.2, 3.2), **kw):
        self.R = min(Lxyz)/2
        hfb.BCS.__init__(self, Nxyz=Nxyz, Lxyz=Lxyz, **kw)

    def get_Vext(self):
        r = np.sqrt(sum([_x**2 for _x in self.xyz[:2]]))
        # R0 = self.barrier_width*self.R
        # V = self.barrier_height*mstep(r - self.R + R0, R0)
        V = self.barrier_height*np.where((r - self.R) > 0, 1, 0)
        return (V, V)


class VortexState(Vortex, PlotBase):
    """
    BCS Vortex class with plotting
    """
    def __init__(self, mu, dmu, delta, N_twist=1, g=None, **kw):
        Vortex.__init__(self, **kw)
        self.delta = delta
        self.N_twist = N_twist
        self.mus = (mu+dmu, mu-dmu)
        self.g = self.get_g(mu=mu, delta=delta) if g is None else g
        self.init_delta()

    def init_delta(self):
        x, y = self.xyz[:2]
        self.Delta = self.delta*(x+1j*y)

    def get_g(self, mu=1.0, delta=0.2):
        assert self.dim == len(self.Nxyz)
        mus_eff = (mu, mu)
        E_c = self.E_max if self.E_c is None else self.E_c
        self.k_c = (2*E_c)**0.5
        # The follow line may cause issue as its results includes
        # all state even we set a cutoff. Need to double check
        h = homogeneous.Homogeneous(
            Nxyz=self.Nxyz, Lxyz=self.Lxyz, dim=self.dim, k_c=self.k_c)
        # h = homogeneous.Homogeneous(dim=2, k_c=self.k_c)
        res = h.get_densities(mus_eff=mus_eff, delta=delta)
        g = delta/res.nu
        h = homogeneous.Homogeneous(dim=self.dim)
        self.k_c_g = h.set_kc_with_g(mus_eff=mus_eff, delta=delta, g=g)
        return g

    def solve(self, tol=0.05, plot=True):
        err = 100
        fig = None
        while err > tol:
            res = self.get_densities(
                mus_eff=self.mus, delta=self.Delta,
                N_twist=self.N_twist)
            self.res = res
            Delta0, self.Delta = self.Delta, self.g*res.nu
            err = abs(Delta0 - self.Delta).max()
            if plot and display:
                plt.clf()
                fig = self.plot(fig=fig, res=res)
                plt.suptitle(f"err={err}")
                display(fig)
                clear_output(wait=True)


class FFStateAgent(object):
    """
    An agent class used for searching FF states
    such a class can not be used alone, it has to
    be a part of a new class that is constructed
    from multiple classes using the function
    'ClassFactory' defined in the class_factory.py
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
    Example:
    -------- 
    if __name__ == "__main__":
        mu, dmu = 10, 4.5
        mus, delta, L, N, R, k_c = (mu + dmu, mu - dmu), 7.5, 8, 32, 4, 50
        res_aslda = FFVortexFunctional(
            mus=mus, delta=delta, L=L, N=N,
            functionalType=FunctionalType.ASLDA, N1=5, N2=5, dim=3)
    """
    mu_a, mu_b = mus
    mu, dmu = (mu_a + mu_b)/2, (mu_a - mu_b)/2
    dx = L/N
    R = L/2
    args = dict(mu=mu, dmu=0, delta=delta, dim=dim, k_c=k_c)
    f = create_ffs_lda(functionalType=functionalType, **args)
    rs = np.linspace(0.01, 1, N1)
    rs = np.append(rs, np.linspace(1.01, R, N2))
    ds = [f.solve(
        mu=mu, dmu=dmu, dq=0.5/_r, a=0.001, b=2*delta) for _r in rs]
    for i in range(len(ds)):
        ps = [f.get_pressure(  # superfluid state pressure
            mu_eff=mu, dmu_eff=dmu, delta=d, dq=0.5/r,
            use_kappa=False) for r, d in zip(rs, ds)]
        ps0 = [f.get_pressure(  # normal state pressure
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


class ExteralPotentailAgent(object):
    """
    To embed the get_Vext function to the new class
    created by the ClassFactory function, we need to
    add an new class that provides the function, such
    a class is called agent class.
    """
    barrier_width = 0.2
    barrier_height = 100.0

    def __init__(self, R, **args):
        self.R = R

    def get_Vext(self, **args):
        r = np.sqrt(sum([_x**2 for _x in self.xyz[:2]]))
        # R0 = self.barrier_width*self.R
        # the smooth step function mstep will
        # cause artifact for functional vortices
        # V = self.barrier_height*mstep(r - self.R + R0, R0)
        V = self.barrier_height*np.where((r - self.R) > 0, 1, 0)
        return (V, V)


class VortexFunctional(PlotBase):
    """a BCS with functional vortex class
    """
    def __init__(
            self, mu_eff, dmu_eff, delta,
            functionalType=FunctionalType.BDG,
            kernelType=KernelType.BCS,
            Nxyz=(32, 32), Lxyz=(3.2, 3.2),
            N_twist=1, k_c=None, **kw):
        self.delta = delta
        self.N_twist = N_twist
        self.mus = (mu_eff + dmu_eff, mu_eff - dmu_eff)

        args = dict(
            mu_eff=mu, dmu_eff=dmu, delta=delta, Nxyz=Nxyz, Lxyz=Lxyz,
            T=0, dim=len(Nxyz), k_c=k_c, R=min(Lxyz)/2, verbosity=False, **kw)
        lda = ClassFactory(
            "LDA", functionalType=functionalType, kernelType=kernelType,
            AgentClass=(ExteralPotentailAgent,), args=args)
        lda.C = lda._get_C(mus_eff=(mu_eff, mu_eff), delta=delta)
        self.lda = lda
        self.xyz = lda.xyz
        self.dxyz = lda.dxyz
        self.dim = lda.dim
        self.R = lda.R
        x, y = self.xyz[:2]
        self.Delta = delta*(x+1j*y)

    def solve(self, rtol=0.05, plot=True):
        err = 1.0
        fig = None
        if False:
            args = dict(
                mus=self.mus, delta=self.Delta, dim=self.lda.dim,
                k_c=self.lda.k_c, E_c=self.lda.E_c, fix_delta=False,
                verbosity=False, rtol=rtol, solver=Solvers.BROYDEN1)
            delta, mu_a_eff, mu_b_eff = self.lda.solve(**args)
        else:
            mu_a, mu_b = self.mus
            Vs = self.lda.get_Vs()
            V_a, V_b = Vs
            mu_a_eff, mu_b_eff = mu_a + V_a, mu_b + V_b
            args = dict(E_c=self.lda.E_c)

            delta = self.Delta

            while(err > rtol):
                res = self.lda.get_densities(
                    mus_eff=self.mus, delta=delta, Vs=Vs, **args)
                ns, taus = (res.n_a, res.n_b), (res.tau_a, res.tau_b)
                nu = res.nu
                args.update(ns=ns)
                V_a, V_b = self.lda.get_Vs(
                    delta=delta, ns=ns, taus=taus, nu=nu)
                mu_a_eff_, mu_b_eff_ = mu_a - V_a, mu_b - V_b
                g_eff = self.lda.get_effective_g(
                    mus_eff=(mu_a_eff_, mu_b_eff_), dim=self.dim, **args)
                delta_ = g_eff*nu
                self.res = res
                err = abs(delta_ - self.Delta).max()
                self.Delta = delta_
                if display:
                    plt.clf()
                    fig = self.plot(fig=fig, res=res)
                    plt.suptitle(f"err={err}")
                    display(fig)
                    clear_output(wait=True)
                delta, mu_a_eff, mu_b_eff = delta_, mu_a_eff_, mu_b_eff_
