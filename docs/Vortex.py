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
from fulde_ferrell_state_finder import FFStateFinder
from fulde_ferrell_state_vortex import FFVortex, FFVortexFunctional, plot_2D, plot_all, PlotBase, VortexState
import warnings
warnings.filterwarnings("ignore")
fontsize = 18

# ### A Simple BCS Vortex

# mu = 10
# dmu=4.5
# delta = 7.5
# v_bcs = VortexState(mu=mu, dmu=dmu, delta=delta, Nxyz=(32, 32), Lxyz=(8,8))
# v_bcs.solve(plot=True)
# #plt.savefig("strongly_polarized_vortx_2d_bcs_plot.pdf", bbox_inches='tight')

# ### A Vortex with BDG Functional

# k_c = v_bcs.k_c
# E_c = k_c**2/2
# v_bdg = VortexFunctional(
#     functionalType=FunctionalType.BDG,
#     mu_eff=mu, dmu_eff=dmu, delta=delta,
#     Nxyz=(32, 32), Lxyz=(8,8), E_c=E_c)
# v_bdg.solve(plot=True)

# ### A Vortex With ASLDA Functional

# v_aslda = VortexFunctional(
#     functionalType=FunctionalType.ASLDA,
#     mu_eff=mu, dmu_eff=dmu, delta=delta,
#     Nxyz=(32, 32), Lxyz=(8,8), E_c=E_c)
# v_aslda.solve(plot=True)

# mu, dmu = 10, 4.5
# mus, delta, L, N, R, k_c =(mu + dmu, mu - dmu), 7.5, 8, 32, 4, 50

# v = v_bcs
# mus=v.mus
# delta=v.delta
# L=v.Lxyz[0]
# N=v.Nxyz[0]
# R=v.R
# k_c=v.k_c
# res_bcs=FFVortex(mus=mus, delta=delta, L=L, N=N, R=R, k_c=k_c, N1=5, N2=2)

# v = v_bcs
# res_bdg=FFVortexFunctional(mus=v.mus, delta=v.delta, L=v.Lxyz[0], N=v.Nxyz[0], N1=5, N2=5)

# res_aslda=FFVortexFunctional(mus=mus, delta=delta, L=L, N=N, functionalType=FunctionalType.ASLDA, N1=5, N2=5, dim=2)

# # Blanced Cases:

v_b = VortexState(mu=10, dmu=0, delta=7.5, Nxyz=(32, 32), Lxyz=(4,4), E_c=10)
v_b.solve(plot=True)

v_b1 = VortexState(mu=10, dmu=0, delta=7.5, Nxyz=(32, 32), Lxyz=(4,4))
v_b1.solve(plot=True)

plt.figure(figsize=(18, 10))
plot_all(vs=[v_b])

# # Plot for the Disseration

v1 = VortexState(mu=10, dmu=3.5, delta=7.5, Nxyz=(32, 32), Lxyz=(8,8))
v1.solve(plot=False)

v2 = VortexState(mu=10, dmu=0, delta=7.5, Nxyz=(32, 32), Lxyz=(8,8))
v2.solve(plot=False)
# plt.savefig("balanced_vortx_2d_bcs_plot.pdf", bbox_inches='tight')

v3 = VortexState(mu=10, dmu=4.5, delta=7.5, Nxyz=(32, 32), Lxyz=(8,8))
v3.solve(plot=False)
# plt.savefig("balanced_vortx_2d_bcs_weakly_polarized_plot.pdf", bbox_inches='tight')

v = v1
res1 = FFVortex(mus=v.mus, delta=v.delta, L=v.Lxyz[0], N=v.Nxyz[0], N1=10, N2=10)

v = v2
res2 = FFVortex(mus=v.mus, delta=v.delta, L=v.Lxyz[0], N=v.Nxyz[0], N1=10, N2=10)

v = v3
res3 = FFVortex(mus=v.mus, delta=v.delta, L=v.Lxyz[0], N=v.Nxyz[0], N1=10, N2=10)

v=v2
plot_2D(v, fig=None, res=v.res, fontsize=22)
plt.savefig("balanced_vortx_2d_bcs_plot.pdf", bbox_inches='tight') #balanced_vortx_2d_bcs_plot

v=v1
plot_2D(v, fig=None, res=v.res, fontsize=22)
plt.savefig("vortx_2d_bcs_weakly_polarized_plot.pdf", bbox_inches='tight') #balanced_vortx_2d_bcs_plot

plot_2D(v3, fig=None, res=v3.res, fontsize=22)
plt.savefig("strongly_polarized_vortx_2d_bcs_plot.pdf", bbox_inches='tight') #balanced_vortx_2d_bcs_plot

one_column = False
plt.figure(figsize=(6,16)) if one_column else plt.figure(figsize=(16, 10))
plot_all([v3], [res3], one_c=one_column, fontsize=18)
# plt.savefig("balanced_vortx_radial_plot_bcs_hom.pdf", bbox_inches='tight')
plt.savefig("strongly_polarized_vortx_2d_bcs_hom_plot.pdf", bbox_inches='tight')

plt.figure(figsize=(6,16)) if one_column else plt.figure(figsize=(16, 10))
plot_all([v1], [res1], one_c=one_column, fontsize=18)
# plt.savefig("balanced_vortx_radial_plot_bcs_hom.pdf", bbox_inches='tight')
plt.savefig("weakly_polarized_vortx_radial_plot_bcs_hom.pdf", bbox_inches='tight')

plt.figure(figsize=(6,16)) if one_column else plt.figure(figsize=(16, 10))
plot_all([v1], [res1], one_c=one_column, fontsize=18)
# plt.savefig("balanced_vortx_radial_plot_bcs_hom.pdf", bbox_inches='tight')
plt.savefig("balanced_vortx_radial_plot_bcs_hom.pdf", bbox_inches='tight')


def plot_col3( hs=[], vs=[], mu=10, dx=1, fontsize=14, xlim=12):
  
    # homogeneous part
    c = len(hs)
    for (id,v) in enumerate(vs):
        mu = sum(v.mus)/2
        dx = v.dxyz[0]
        r = np.sqrt(sum(_x**2 for _x in v.xyz))
        k_F = np.sqrt(2*mu)
        plt.subplot(5, c,id + 1)
        plt.plot(r.ravel()/dx, abs(v.Delta).ravel()/mu, '+', label="BCS")
        plt.xlim(0, xlim)
        plt.subplot(5, c,c + id + 1)
        res = v.res
        plt.plot(r.ravel()/dx, abs(res.n_a + res.n_b).ravel()/k_F, '+', label="BCS")
        plt.xlim(0, xlim)
        plt.subplot(5, c,2*c + id + 1)
        plt.plot(r.ravel()/dx, abs(res.n_a - res.n_b).ravel()/k_F, '+', label="BCS")
        plt.xlim(0, xlim)
        
        plt.ylim(-1, 1)
        x, y = v.xyz
        r_vec = x+1j*y
        j_a_ = res.j_a[0] + 1j*res.j_a[1]
        j_a_ = clockwise(r_vec, j_a_)*np.abs(j_a_)
        j_b_ = res.j_b[0] + 1j*res.j_b[1]
        j_b_ = clockwise(r_vec, j_b_)*np.abs(j_b_) 
        j_p_, j_m_ = j_a_ + j_b_, j_a_ - j_b_
        plt.subplot(5, c,3*c + id + 1)
        plt.plot(r.ravel()/dx, j_a_.ravel(), '+', label="BCS")
        plt.xlim(0, xlim)
        plt.subplot(5, c,4*c + id + 1)
        plt.plot(r.ravel()/dx, j_b_.ravel(), '+', label="BCS")
        plt.xlim(0, xlim)
        
    for (id, res_h) in enumerate(hs):
        rs, ds, ps, ps0, n_p, n_m, j_a, j_b = res_h
        k_F = np.sqrt(2*mu)
        plt.subplot(5, c,id + 1)
        plt.plot(rs, np.array(ds)/mu, '-', label="Homogeneous")
        plt.legend()
        if id == 0:
            plt.ylabel(r'$\Delta/E_F$', fontsize=fontsize)
    #     plt.subplot(322)
    #     plt.ylabel(r"Pressure/$E_F$", fontsize=fontsize)
    #     plt.plot(rs, ps, label="FF State/Superfluid State Pressure")
    #     plt.plot(rs, ps0, '-', label="Normal State pressure")
    #     plt.legend()
        plt.subplot(5,c,c+id + 1)
        plt.plot(rs, n_p/k_F, label="Homogeneous")
        if id == 0:
            plt.ylabel(r"$n_p/k_F$", fontsize=fontsize)
        plt.legend()
        plt.subplot(5,c,2*c+id + 1)
        plt.plot(rs, n_m/k_F, label="Homogeneous")
        if id == 0:
            plt.ylabel(r"$n_m/k_F$", fontsize=fontsize)#,plt.title("Density Difference")
        plt.legend()
        plt.subplot(5,c,3*c+id + 1)
        plt.plot(rs, j_a, '-', label="Homogeneous")
        if id == 0:
            plt.ylabel(r"$j_a$", fontsize=fontsize)
        
        plt.axhline(0, linestyle='dashed')
        plt.legend()
        plt.subplot(5,c,4*c+id + 1)
        plt.plot(rs, -j_b, label="Homogeneous") # seems we have different sign
        plt.axhline(0, linestyle='dashed')
        plt.xlabel(r"$r/dx$", fontsize=fontsize)
        if id == 0:
            plt.ylabel(r"$j_b$", fontsize=fontsize)
        plt.legend() #prop={'size': 16}


plt.figure(figsize=(15,20))
plot_col3(vs=[v2, v1, v3],hs=[res2, res1, res3], fontsize=18)
plt.savefig("three_column_bcs_hom_plots.pdf", bbox_inches='tight')

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


