# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Vortex in 2D

# + {"init_cell": true}
import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import *  # Conveniences like clear_output
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import operator
import inspect
# -

from mmf_hfb import hfb, homogeneous;reload(hfb)
from mmfutils.math.special import mstep
from mmfutils.plot import imcontourf
from os.path import join
currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, join(currentdir, '..','Projects','FuldeFerrellState'))
from FuldeFerrellState import FFState
from FFStateFinder import FFStateFinder
from FFStateSolveThread import fulde_ferrell_state_solve_thread


# The angle from $\vec{a}=a_x+ia_y$ to $\vec{b}=b_x+ib_y$ is the argument of the conjugate of a times b (rotating b backwards by the angle of a, scale not considered)

def clockwise(r, v):
    """return the angle between two vectors, which take order into account"""
    dot = r.conj()*v
    angs = np.arctan2(dot.imag,dot.real)
    return np.sign(angs)


# Here we generate some vortices.  These are regularized by fixing the coupling constant $g$ so that the homogeneous system in the box and on the lattice gives a fixed value of $\Delta$ at the specified chemical potential.  The self-consistent solution is found by simple iterations.

class PlotBase(object):
    dim = None
    xyz = None
    Delta = None
    
    def plot(self, fig=None, res=None):
        x, y = self.xyz[:2]
        if fig is None:
            fig = plt.figure(figsize=(20, 10))
        plt.subplot(233)
        if self.dim == 2:
            imcontourf(x, y, abs(self.Delta), aspect=1)
        elif self.dim == 3:
            imcontourf(x, y,  np.sum(abs(self.Delta), axis=2))
        plt.title(r'$|\Delta|$'); plt.colorbar()

        if res is not None:
            plt.subplot(231)
            imcontourf(x, y, (res.n_a+res.n_b).real, aspect=1)
            plt.title(r'$n_+$'); plt.colorbar()

            plt.subplot(232)
            imcontourf(x, y, (res.n_a-res.n_b).real, aspect=1)
            plt.title(r'$n_-$'); plt.colorbar()

            plt.subplot(234)

            j_a = res.j_a[0] + 1j*res.j_a[1]
            j_b = res.j_b[0] + 1j*res.j_b[1]
            j_p = j_a + j_b
            j_m = j_a - j_b
            utheta = np.exp(1j*np.angle(x + 1j*y))
            imcontourf(x, y, abs(j_a), aspect=1)
            plt.title(r'$j_a$'); plt.colorbar()
            plt.quiver(x.ravel(), y.ravel(), j_a.real, j_a.imag)

            plt.subplot(235)
            imcontourf(x, y, abs(j_b), aspect=1)
            plt.title(r'$J_b$'); plt.colorbar()
            plt.quiver(x.ravel(), y.ravel(), j_b.real, j_b.imag)

            plt.subplot(236)
            imcontourf(x, y, abs(j_p), aspect=1)
            plt.title(r'$J_+$'); plt.colorbar()
            plt.quiver(x.ravel(), y.ravel(), j_p.real, j_p.imag)
        return fig


# ## Simple BCS Vortex

# +
class Vortex(hfb.BCS):
    barrier_width = 0.2
    barrier_height = 100.0

    def __init__(self, Nxyz=(32, 32), Lxyz=(3.2, 3.2), **kw):
        self.R = min(Lxyz)/2
        hfb.BCS.__init__(self, Nxyz=Nxyz, Lxyz=Lxyz, **kw)

    def get_Vext(self):
        r = np.sqrt(sum([_x**2 for _x in self.xyz[:2]]))
        R0 = self.barrier_width*self.R
        V = self.barrier_height*mstep(r - self.R + R0, R0)
        return (V, V)


class VortexState(Vortex, PlotBase):
    def __init__(self, mu, dmu, delta, N_twist=1, g=None, **kw):
        Vortex.__init__(self, **kw)
        self.delta = delta
        self.N_twist = N_twist
        self.mus = (mu+dmu, mu-dmu)
        self.g = self.get_g(mu=mu, delta=delta) if g is None else g
        x, y = self.xyz[:2]
        self.Delta = delta*(x+1j*y)

    def get_g(self, mu=1.0, delta=0.2):
        mus_eff = (mu, mu)
        E_c = self.E_max if self.E_c is None else self.E_c
        # self.k_c = (2*E_c)**0.5
        h = homogeneous.Homogeneous(Nxyz=self.Nxyz, Lxyz=self.Lxyz, dim=2)
        res = h.get_densities(mus_eff=mus_eff, delta=delta)
        g = delta/res.nu
        h = homogeneous.Homogeneous(dim=2)
        self.k_c = h.set_kc_with_g(mus_eff=mus_eff, delta=delta, g=g)
        return g

    def solve(self, tol=0.05, plot=True):
        err = 1.0
        fig = None
        with NoInterrupt() as interrupted:
            display(self.plot())
            clear_output(wait=True)

            while not interrupted and err > tol:
                res = self.get_densities(mus_eff=self.mus, delta=self.Delta,
                                         N_twist=self.N_twist)
                self.res = res
                Delta0, self.Delta = self.Delta, self.g*res.nu  # ....
                err = abs(Delta0 - self.Delta).max()
                if display:
                    plt.clf()
                    fig = self.plot(fig=fig, res=res)
                    plt.suptitle(f"err={err}")
                    display(fig)
                    clear_output(wait=True)



# -

# ## ASLDA Vortex
# * a functional vortex class

# +
from mmf_hfb.class_factory import ClassFactory, FunctionalType, KernelType, Solvers

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
        R0 = self.barrier_width*self.R
        # the smooth step function mstep will cause artifact for functional vortices
        # V = self.barrier_height*mstep(r - self.R + R0, R0)
        V = self.barrier_height*np.where( (r - self.R) >0, 1, 0)
        return (V, V)


class VortexFunctional(PlotBase):
    
    def __init__(
            self, mu_eff, dmu_eff, delta,
            functionalType=FunctionalType.BDG,
            kernelType=KernelType.BCS,
            Nxyz=(32, 32), Lxyz=(3.2, 3.2),
            N_twist=1, k_c=None, **kw):
        self.delta = delta
        self.N_twist = N_twist
        self.mus_eff = (mu_eff + dmu_eff, mu_eff - dmu_eff)

        args = dict(
            mu_eff=mu, dmu_eff=dmu, delta=delta, Nxyz=Nxyz, Lxyz=Lxyz,
            T=0, dim=len(Nxyz), k_c=k_c, R=min(Lxyz)/2, verbosity=False, **kw)
        lda = ClassFactory(
            "LDA", functionalType=functionalType, kernelType=kernelType,
            AgentClass=(ExteralPotentailAgent,), args=args)
        lda.C = lda._get_C(mus_eff=(mu_eff, mu_eff), delta=delta)
        self.lda = lda
        self.xyz = lda.xyz
        self.dim = lda.dim
        x, y = self.xyz[:2]
        self.Delta = delta*(x+1j*y)
    
    def solve(self, rtol=0.05, plot=True):
        err = 1.0
        fig = None
        
        if False:
            args = dict(
                mus=self.mus_eff, delta=self.Delta, dim=self.lda.dim,
                k_c=self.lda.k_c, E_c=self.lda.E_c, fix_delta=False,
                verbosity=False, rtol=rtol, solver=Solvers.BROYDEN1)
            delta, mu_a_eff, mu_b_eff = self.lda.solve(**args)
        else:
            mu_a, mu_b = self.mus_eff
            Vs = self.lda.get_Vs()
            V_a, V_b  = Vs
            mu_a_eff, mu_b_eff = mu_a + V_a, mu_b + V_b
            args = dict(E_c=self.lda.E_c)
                
            delta = self.Delta
            with NoInterrupt() as interrupted:
                display(self.plot())
                clear_output(wait=True)
                while(not interrupted and err > rtol):  # use simple iteration if no solver is specified
                    res = self.lda.get_densities(mus_eff=self.mus_eff, delta=delta,Vs=Vs, **args)
                    ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
                    args.update(ns=ns)
                    V_a, V_b = self.lda.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu)
                    mu_a_eff_, mu_b_eff_ = mu_a - V_a, mu_b - V_b
                    g_eff = self.lda.get_effective_g(mus_eff=(mu_a_eff_, mu_b_eff_), dim=self.dim, **args)
                    delta_ =  g_eff*nu
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


# -

# ### A Vortex with BDG Functional

mu = 10
dmu=4.5
delta = 7.5
k_c = 14
E_c = k_c**2/2
v = VortexFunctional(
    functionalType=FunctionalType.BDG,
    mu_eff=mu, dmu_eff=dmu, delta=delta,
    Nxyz=(32, 32), Lxyz=(4,4), E_c=E_c)
v.solve(plot=True)

# ### A Vortex With ASLDA Functional

v1 = VortexFunctional(
    functionalType=FunctionalType.ASLDA,
    mu_eff=mu, dmu_eff=dmu, delta=delta,
    Nxyz=(32, 32), Lxyz=(4,4), E_c=E_c)
v1.solve(plot=True)

# ## Homogeneous

# +
import warnings
warnings.filterwarnings("ignore")
fontsize = 18
from mmf_hfb.parallel_helper import PoolHelper


def FFVortex(bcs_vortex, mus=None, delta=None, kc=None, N1=10):
    mu_a, mu_b=bcs_vortex.mus
    if delta is None:
        delta = bcs_vortex.delta
    mu, dmu = (mu_a + mu_b)/2, (mu_a - mu_b)/2
    N = bcs_vortex.Nxyz[0]
    L = bcs_vortex.Lxyz[0]
    dx = L/N
   
    k_c = bcs_vortex.k_c if kc is None else kc  #  np.sqrt(2)*np.pi*N/L 
    print(f"k_c={k_c},bcs_kc={v0.k_c}")
    k_F = np.sqrt(2*mu)
    args = dict(mu=mu, dmu=0, delta=delta, dim=2, k_c=k_c)
    f = FFState(fix_g=True, **args)
    rs = np.linspace(0.0001,1, N1)
    rs = np.append(rs, np.linspace(1.01, bcs_vortex.R, 10))
    paras = [(f, mu, dmu, delta, r) for r in rs]
    ds = PoolHelper.run(fulde_ferrell_state_solve_thread, paras=paras)        
    #ds = [f.solve(mu=mu, dmu=dmu, dq=0.5/_r, a=0.001, b=2*delta) for _r in rs]
    for i in range(len(ds)):
        ps = [f.get_pressure(mu_eff=mu, dmu_eff=dmu, delta=d, dq=0.5/r, use_kappa=False).n for r, d in zip(rs,ds)]
        ps0 = [f.get_pressure(mu_eff=mu, dmu_eff=dmu, delta=1e-8,q=0, dq=0, use_kappa=False).n for r, d in zip(rs,ds)]
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
    js = [f.get_current(mu=mu, dmu=dmu, delta=d,dq=0.5/r) for r, d in zip(rs,ds)]
    for j in js:
        j_a.append(j[0].n)
        j_b.append(j[1].n)
    j_a, j_b = np.array(j_a), np.array(j_b)
    j_p, j_m = -(j_a + j_b), j_a - j_b
    return (rs/dx, ds, ps, ps0, n_p, n_m, j_a, j_b)


# -

def plot_all(v, res_h=None, mu=10, dx=1, fontsize=14):
    plt.figure(figsize(16,8))
    if v is not None:
        mu = sum(v.mus)/2
        dx = v.dxyz[0]
        r = np.sqrt(sum(_x**2 for _x in v.xyz))
        k_F = np.sqrt(2*mu)
        plt.figure(figsize(16,8))
        plt.subplot(321)
        plt.plot(r.ravel()/dx, abs(v.Delta).ravel()/mu, '+', label="BCS")
        plt.ylabel(r'$\Delta/E_F$', fontsize=fontsize)
        plt.xlim(0, v.R/dx)
        plt.subplot(323)  
        res = v.get_densities(mus_eff=v.mus, delta=v.Delta)
        plt.plot(r.ravel()/dx, abs(res.n_a + res.n_b).ravel()/k_F, '+', label="BCS")
        plt.ylabel(r"$n_p/k_F$", fontsize=fontsize)
        plt.xlim(0, v.R/dx)
        plt.subplot(324)
        plt.plot(r.ravel()/dx, abs(res.n_a - res.n_b).ravel()/k_F, '+', label="BCS")
        plt.ylabel(r"$n_m/k_F$", fontsize=fontsize)
        plt.xlim(0, v.R/dx)
        #plt.ylim(-1, 1)
        x, y = v.xyz
        r_vec = x+1j*y
        j_a_ = res.j_a[0] + 1j*res.j_a[1]
        j_a_ = clockwise(r_vec, j_a_)*np.abs(j_a_)
        j_b_ = res.j_b[0] + 1j*res.j_b[1]
        j_b_ = clockwise(r_vec, j_b_)*np.abs(j_b_) 
        j_p_, j_m_ = j_a_ + j_b_, j_a_ - j_b_
        plt.subplot(325)
        plt.plot(r.ravel()/dx, j_a_.ravel(), '+', label="BCS"),plt.ylabel(r"$j_a$", fontsize=fontsize)
        plt.subplot(326)
        plt.plot(r.ravel()/dx, j_b_.ravel(), '+', label="BCS"), plt.ylabel(r"$j_b$", fontsize=fontsize)
        plt.xlim(0, v.R/dx)
    
    # homogeneous part
    if res_h is None:
        return
    rs_, ds, ps, ps0, n_p, n_m, j_a, j_b = res_h
    k_F = np.sqrt(2*mu)
    plt.subplot(321)
    plt.plot(rs_, np.array(ds)/mu, '-', label="Homogeneous")
    plt.legend()
    plt.ylabel(r'$\Delta/E_F$', fontsize=fontsize)
#     plt.subplot(322)
#     plt.ylabel(r"Pressure/$E_F$", fontsize=fontsize)
#     plt.plot(rs_, ps, label="FF State/Superfluid State Pressure")
#     plt.plot(rs_, ps0, '-', label="Normal State pressure")
#     plt.legend()
    plt.subplot(323)  
    plt.plot(rs_, n_p/k_F, label="Homogeneous")
    plt.ylabel(r"$n_p/k_F$", fontsize=fontsize)
    plt.legend()
    plt.subplot(324)
    plt.plot(rs_, n_m/k_F, label="Homogeneous")
    plt.ylabel(r"$n_m/k_F$", fontsize=fontsize)#,plt.title("Density Difference")
    plt.legend()
    plt.subplot(325)
    plt.plot(rs_, j_a, '-', label="Homogeneous")
    plt.xlabel(f"r/dx", fontsize=fontsize), plt.ylabel(r"$j_a$", fontsize=fontsize)
    plt.axhline(0, linestyle='dashed')
    plt.legend()
    plt.subplot(326)
    plt.plot(rs_, -j_b, label="Homogeneous") # seems we have different sign
    plt.axhline(0, linestyle='dashed')
    plt.xlabel(f"r/dx", fontsize=fontsize), plt.ylabel(r"$j_b$", fontsize=fontsize)
    plt.legend()


mu = 10
dmu=4.5
delta = 7.5
v0 = VortexState(mu=mu, dmu=dmu, delta=delta, Nxyz=(32, 32), Lxyz=(8,8))
v0.solve(plot=True)
plt.savefig("strongly_polarized_vortx_2d_bcs_plot.pdf", bbox_inches='tight')

res0=FFVortex(v0)

plot_all(v0, res0)
plt.savefig("strongly_polarized_vortx_2d_bcs_hom_plot.pdf", bbox_inches='tight')

plot_all(v0, res0)

v1 = VortexState(mu=10, dmu=3.5, delta=7.5, Nxyz=(32, 32), Lxyz=(8,8))
v1.solve(plot=True)

res1 = FFVortex(v1)

plot_all(v1, res1)

v2.g

v2 = VortexState(mu=10, dmu=0, delta=7.5, Nxyz=(32, 32), Lxyz=(8,8))
v2.solve(plot=True)
plt.savefig("balanced_vortx_2d_bcs_plot.pdf", bbox_inches='tight')

res2 = FFVortex(v2, N1=10)

plot_all(v2, res2)
plt.savefig("balanced_vortx_radial_plot_bcs_hom.pdf", bbox_inches='tight')

v3 = VortexState(mu=10, dmu=2.5, delta=7.5, Nxyz=(32, 32), Lxyz=(8,8))
v3.solve(plot=True)
plt.savefig("balanced_vortx_2d_bcs_weakly_polarized_plot.pdf", bbox_inches='tight')

res3 = FFVortex(v3, N1=10)

plot_all(v3, res3)
plt.savefig("balanced_vortx_radial_weakly_polarized_plot_bcs_hom.pdf", bbox_inches='tight')

# +
mu_a, mu_b=v0.mus
delta = v0.delta
mu, dmu = (mu_a + mu_b)/2, (mu_a - mu_b)/2
N = v0.Nxyz[0]
L = v0.Lxyz[0]
r=0.65
dx = L/N

k_c = v0.k_c
k_F = np.sqrt(2*mu)
E_c=k_c**2/2
args = dict(mu=mu, dmu=0, delta=delta, dim=2, k_c=k_c)
f = FuldeFerrellState.FFState(fix_g=True, g=v0.g, **args)
ds=np.linspace(0.001, 3, 20)
gs = [f.f(mu=mu, dmu=dmu, dq=0.5/r, delta=d) for d in ds]
plt.plot(ds, gs, label=f"r=r{r}")
plt.legend()
plt.axhline(0, ls='dashed')
# -

# ## Compare to FF State
# * The FFVortex will compute FF State data with the same $\mu,d\mu$, and compare the results in plots
# * $\Delta$ should be continuous in a close loop, in rotation frame, the gap should satisfy:
# $$
# \Delta=\Delta e^{-2i\delta qx}
# $$
# with phase change in a loop by $2\pi$, which requires: $\delta q= \frac{1}{2r}$
#

# ### Symmetric case $\delta \mu/\Delta=0$

from mmf_hfb import FuldeFerrellState
def HomogeneousVortx(mu, dmu, delta, k_c=50):
    k_F = np.sqrt(2*mu)   
    E_c=k_c**2/2
    dx = 1
    args = dict(mu=mu, dmu=dmu, delta=delta, dim=3, k_c=k_c)
    f = FuldeFerrellState.FFState(fix_g=True, **args)
    rs = np.linspace(0.0001,1, 30)
    rs = np.append(rs, np.linspace(1.1, 4, 10))

    ds = [f.solve(mu=mu, dmu=dmu, dq=0.5/_r, a=0.001, b=2*delta) for _r in rs]
    ps = [f.get_pressure(mu_eff=mu, dmu_eff=dmu, delta=d, dq=0.5/r, use_kappa=False).n for r, d in zip(rs,ds)]
    ps0 = [f.get_pressure(mu_eff=mu, dmu_eff=dmu, delta=1e-12,q=0, dq=0, use_kappa=False).n for r, d in zip(rs,ds)]

    
    plt.figure(figsize(16,8))
    plt.subplot(321)
    plt.plot(rs/dx, np.array(ds)/mu, label="Homogeneous")
    plt.legend()
    plt.xlabel(f"r/d(lattice spacing)", fontsize=fontsize)
    plt.ylabel(r'$\Delta/E_F$', fontsize=fontsize)
    plt.subplot(322)
    plt.xlabel(f"r/d(lattice spacing)", fontsize=fontsize)
    plt.ylabel(r"Pressure/$E_F$", fontsize=fontsize)
    plt.plot(rs/dx, ps, label="FF State/Superfluid State Pressure")
    plt.plot(rs/dx, ps0,'--', label="Normal State pressure")
    plt.legend()

    na = np.array([])
    nb = np.array([])
    for i in range(len(rs)):
        na_, nb_ = f.get_densities(delta=ds[i], dq=0.5/rs[i], mu=mu, dmu=dmu)
        na = np.append(na, na_.n)
        nb = np.append(nb, nb_.n)   
    plt.subplot(323)
    n_p = na + nb
    plt.plot(rs/dx, n_p/k_F, label="Homogeneous")
    plt.xlabel(f"r/d(lattice spacing)", fontsize=fontsize), plt.ylabel(r"$n_p/k_F$", fontsize=fontsize)
    #plt.title("Total Density")
    plt.legend()
    plt.subplot(324)
    n_m = na - nb
    plt.plot(rs/dx, n_m/k_F, label="Homogeneous")
    plt.xlabel(f"r/d(lattice spacing)", fontsize=fontsize), plt.ylabel(r"$n_m/k_F$", fontsize=20)#,plt.title("Density Difference")
    plt.legend()

    ja = []
    jb = []
    js = [f.get_current(mu=mu, dmu=dmu, delta=d,dq=0.5/r) for r, d in zip(rs,ds)]
    for j in js:
        ja.append(j[0].n)
        jb.append(j[1].n)
    ja, jb = np.array(ja), np.array(jb)
    j_p, j_m = -(ja + jb), ja - jb
    plt.subplot(325)
    plt.plot(rs/dx, j_m, label="Homogeneous")
    plt.xlabel(f"r/d(lattice spacing)", fontsize=fontsize), plt.ylabel(r"$j_p$", fontsize=fontsize)#,plt.title("Total Current")
    plt.legend()
    plt.subplot(326)
    plt.plot(rs/dx, j_p, label="Homogeneous")
    plt.xlabel(f"r/d(lattice spacing)", fontsize=fontsize), plt.ylabel(r"$j_m$", fontsize=fontsize)#,plt.title("Current Difference")
    plt.ylim(0,15)
    plt.legend()
    clear_output()

mu=5
dmu=0
delta = 1.16220056179 * mu
HomogeneousVortx(mu=mu, dmu=dmu, delta=delta)

mu=5
delta = 1.16220056179 * mu
dmu=0.25 * delta
HomogeneousVortx(mu=mu, dmu=dmu, delta=delta)



