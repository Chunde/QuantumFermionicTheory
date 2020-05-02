# ---
# jupyter:
#   jupytext:
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

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from IPython.display import clear_output, display
#from nbimports import *  # Conveniences like clear_output

from mmf_hfb import hfb, homogeneous
from mmfutils.math.special import mstep
from mmfutils.plot import imcontourf
import os
import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, join(currentdir, '..','Projects','FuldeFerrellState'))
from fulde_ferrell_state_vortex import FFVortex, VortexState, plot_2D, plot_all, res_to_json, json_to_res, to_list
import warnings
warnings.filterwarnings("ignore")
fontsize = 18

# +
import json
from collections import namedtuple
from mmf_hfb.utils import JsonEncoderEx

def get_error_ratio(mu, delta, k_c=200):
    """get the error ratio dn/n"""
    h2 = homogeneous.Homogeneous(dim=2)
    res = h2.get_densities((mu, mu), delta=delta, k_c=k_c)
    n = res.n_a + res.n_b
    k_xi = np.pi*2*np.sqrt(2*delta)
    k_F = np.sqrt(2*np.pi * n)
    err = k_xi**4/k_c**2/k_F**2/(2*np.pi)**4/4
    return err

def dic2namedtupe(dic, name='Results'):
    Results = namedtuple(name, sorted(dic))
    return Results(**dic)

def to_complex(ls):
    """a two component array to complex numbers"""
    return np.array(ls[0])+1j*np.array(ls[1])

def Vortex2D(
        mu, dmu, delta, N=32, L=5, E_c=None, k_c=20, N1=7, N2=5,
        plot=True, plot_2d=False, xlim=None, use_file=False, file_name=None, tol=0.05, **args):
    """
    compute BCS 2D box vortex and homogeneous results
    ------
    N1: number of points near the vortex
    N2: number of points outside the vortex
    """
    if file_name is None:
        file_name=f"Vortex2D_Data_{mu}_{dmu}_{delta}_{N}_{L}_{E_c}_{k_c}_{N1}_{N2}.json"
    if use_file:
        try:
            with open(join(currentdir, file_name), 'r',encoding='utf-8', errors='ignore') as rf:
                obj = json.load(rf)
                obj['v_res'] = json_to_res(obj['v_res'])
                obj['h_res'] = dic2namedtupe(obj['h_res'])
                obj['v_delta']=to_complex(obj['v_delta'])
                if plot:
                    plt.figure(figsize=(16,8))
                    v = VortexState(
                        mu=obj['mu'], dmu=obj['dmu'], delta=obj['delta'],
                        Nxyz=(obj['N'],)*2, Lxyz=(obj['L'],)*2)
                    v.res = obj['v_res']
                    v.Delta = obj['v_delta']
                    plot_all(vs=[v], hs=[obj['h_res']], ls='-o', xlim=xlim, **args)
                return obj
        except:
            use_file = False
            print("Load file failed.")
    k_c = np.pi*N/L
    err = get_error_ratio(mu=mu, delta=delta, k_c=k_c)
    print(f"Error Ratio:{err}")
    v = VortexState(mu=mu, dmu=dmu, delta=delta, Nxyz=(N, N), Lxyz=(L,L), E_c=None)
    v.solve(plot=plot_2d, tol=tol)
    h_res = FFVortex(mus=v.mus, delta=v.delta, L=L, N=N, N1=N1, N2=N2, k_c=k_c)
    if plot:
        plt.figure(figsize=(16,8))
        plot_all(vs=[v], hs=[h_res], ls='-o', xlim=xlim, **args)
    Results = namedtuple('Results', ['N','L', 'delta','mu','dmu','E_c','k_c', 'v_res', 'h_res', 'err'])
    if use_file == False:
        try:
            with open(join(currentdir, file_name), 'w') as wf:
                output = dict(
                    N=N, L=L, delta=delta, mu=mu, dmu=dmu, v_delta=to_list(v.Delta), 
                    E_c=E_c, k_c=k_c, v_res=res_to_json(v.res), h_res=h_res._asdict(), err=err)
                json.dump(output, wf, cls=JsonEncoderEx)
                print(f'File {file_name} saved.')
        except:
            print("Json Exception.")
    return Results(N=N, L=N, delta=delta, mu=mu, dmu=dmu,
                    E_c=E_c, k_c=k_c, v_res=v.res, h_res=h_res, err=err)


# -

res_s_0 = Vortex2D(mu=1, dmu=0, delta=5, N=32, L=5, k_c=20, N1=15, N2=5, use_file=True)

# import importlib
# import fulde_ferrell_state_vortex as ffsv; importlib.reload(ffsv)
# from fulde_ferrell_state_vortex import FFVortex, plot_2D, plot_all, VortexState
# from mmf_hfb import homogeneous as hg; importlib.reload(hg)

# # Code Check with 1D Case
#
# Here are how errors scale for $\Delta$ and total density as $k_c$ getting large
# $$
#   \delta_{UV}\Delta = \frac{v_0}{2}\overbrace{2}^{\pm k}\int_{k_\max}^{\infty}
#     \frac{\d{k}}{2\pi}\;\frac{\Delta}{\sqrt{\epsilon_+^2 + \Delta^2}} 
#   \approx v_0\int_{k_\max}^{\infty} \frac{\d{k}}{2\pi}\;\frac{2m\Delta}{\hbar^2k^2}
#   = \frac{v_0m\Delta}{\pi\hbar^2k_\max} + \frac{2v_0 m^2\mu_\mathrm{eff}}{3\pi\hbar^4k_\max^3},\\
#   \delta_{UV}n_+ = 2\int_{k_\max}^{\infty}\frac{\d{k}}{2\pi}
#     \left[1 - \frac{\epsilon_+}{\sqrt{\epsilon_+^2 + \abs{\Delta}^2}}\right]
#   \approx \int_{k_\max}^{\infty}\frac{\d{k}}{2\pi}
#     \frac{4m^2\abs{\Delta}^2}{\hbar^4k^4}
#   = \frac{2m^2\abs{\Delta}^2}{3\pi\hbar^4k_{\max}^3} 
#     + \frac{8m^3\mu_{\mathrm{eff}}\abs{\Delta}^2}{5\pi \hbar^6 k_\max^5}
# $$

# * Check the 1D error using a Box(compared to homogeneous code with large $k_c$)

# +
from mmf_hfb import hfb, homogeneous

def check_1d_bcs_error(N=32, L=8, mu=1, dmu=0, delta=5, N_twist=1, dim=1, T=0):
    mus = (mu + dmu, mu - dmu)
    args=dict(Nxyz=(N,)*dim, Lxyz=(L,)*dim, T=T)
    h_summation = homogeneous.Homogeneous(**args)
    h_intgral = homogeneous.Homogeneous(dim=dim)
    b = hfb.BCS(**args)
    # a homogeneous intance uses summation to get the densities
    res_s = h_summation.get_densities(mus, delta, N_twist=N_twist)
    # a homogeneous instance uses integral to get the densities
    res_i = h_intgral.get_densities(mus, delta, N_twist=N_twist, k_c=200)
    # a box BCS instance
    res_b = b.get_densities(mus, delta, N_twist=N_twist)
    n_b = res_b.n_a.mean() + res_b.n_b.mean()
    n_i = res_i.n_a + res_i.n_b
    n_s = res_s.n_a + res_s.n_b
    assert np.allclose(n_b, n_s)
    print(f"Error Ratio:{(n_i-n_b)*100/n_i}%")
    print(f"Box Density:{n_b}, Integral Density:{n_i}")


# -

check_1d_bcs_error()

# ## 1D FFVortex
# Here we verify the code by testing 1D box and homogeneous vortices.

# +
from mmf_hfb import hfb, homogeneous

class VortexState1D(VortexState):
    """
    1D box support FF state inherit from
    the 2D code
    """
    def __init__(self, N, L, dq=0, **kw):
        self.dq = dq
        VortexState.__init__(self, Nxyz=(N,), Lxyz=(L,), **kw)

    def init_delta(self):
        """
        set the initial pairing field, since it's dq
        in matrix form, it should be dq/2 for homogeneuos
        calculation.
        """
        x = self.xyz[0]
        self.Delta = self.delta*np.exp(-1j*self.dq*x)

    def get_Vext(self):
        """reset the external potential to zero"""
        return (0, 0)


# -

# * In the following cell$\delta q_{max}=\frac{\Delta}{2\mu}$ is the minmum $dq$ where the dispersion start to cross the zero(verified with Maple, need to double check).

mu = 1
delta = 5
L=10
N=32
dq_max = np.sqrt(delta/mu)/2
print(dq_max)
v1d_tmp = VortexState1D(L=L, N=N, mu=mu, dmu=0, delta=delta)

# ### Symmetric 1D Box vortex $\delta \mu=0$
# Construct a 1D vortex
# * the pairing field to have the form: $\Delta(x)=\Delta e^{-i dq x}$
# * compare the densities and current with the homogeneous case (d_{})

dmu = 0
dq = v1d_tmp.kxyz[0][3]
v_1d = VortexState1D(L=10, N=32, mu=mu, dmu=dmu, delta=delta, dq=dq)
v_1d.solve(tol=1e-5)

h = homogeneous.Homogeneous(dim=1)
res = h.get_densities(mus_eff=(mu, mu), delta=delta, dq=v_1d.dq/2)
print(f"n_p={res.n_a + res.n_b}, n_m={res.n_a - res.n_b}")
js = h.get_current(mus_eff=(mu, mu), delta=delta, dq=v_1d.dq/2, named=True)
#assert np.allclose(js.j_a.n, v_1d.res.j_a.mean(), rtol=0.1)
assert np.allclose(res.n_a, v_1d.res.n_a.mean(), rtol=0.1)

delta = 0.5
dmu=.75
v1d_tmp = VortexState1D(L=L, N=N, mu=mu, dmu=dmu, delta=delta)
dq = v1d_tmp.kxyz[0][2]
v_1d_polarized = VortexState1D(L=10, N=32, mu=mu, dmu=dmu, delta=delta, dq=dq)
v_1d_polarized.solve(tol=1e-4)

res = h.get_densities(mus_eff=(mu+dmu, mu-dmu), delta=delta, dq=dq/2)
print(f"n_p={res.n_a + res.n_b}, n_m={res.n_a - res.n_b}")
js = h.get_current(mus_eff=(mu, mu), delta=delta, dq=dq, named=True)
js.j_a.n, js.j_b.n

# # 2D Vortices
# Useful relations:
# $$
#   n_F = \frac{k_F^2}{2\pi}\\
# $$
# $$
#   \hbar k_\xi = 2\pi \sqrt{2m\Delta}, \qquad
#   \hbar k_F = \sqrt{2m\epsilon_F}\\
#   \frac{k_\xi}{k_F} = \sqrt{\frac{\Delta}{\epsilon}} = \sqrt{\eta}.
# $$
# * Here we derivate the 2D error as a function of the cuttoff when it is large.
# $$
#   \delta_{UV}\Delta = \frac{v_0}{2}\overbrace{2}^{\pm k}\int_{k_\max}^{\infty}
#     \frac{k\d{k}}{2\pi}\;\frac{\Delta}{\sqrt{\epsilon_+^2 + \Delta^2}} 
#   \approx v_0\int_{k_\max}^{\infty} \frac{\d{k}}{2\pi}\;\frac{2m\Delta}{\hbar^2k}
#   = \frac{v_0m\Delta}{\pi\hbar^2}\bigg[(ln(\infty) -ln(k)\bigg]\\
#   \delta_{UV}n_+ = \int_{k_\max}^{\infty}\frac{\d{k}}{2\pi}
#     \left[1 - \frac{\epsilon_+}{\sqrt{\epsilon_+^2 + \abs{\Delta}^2}}\right]
#   \approx \int_{k_\max}^{\infty}\frac{\d{k}}{2\pi}
#     \frac{4m^2\abs{\Delta}^2}{\hbar^4k^3}
#   = \frac{m^2\abs{\Delta}^2}{2\pi\hbar^4k_{\max}^2}
# $$
# $$
# g = \Delta/\nu
# $$
#

# ## Numerically Error Check
# * create a 2D homogeneous object, compare densities with other cuttoff to the case with a large cutoff=1000

h2 = homogeneous.Homogeneous(dim=2)
res0 = h2.get_densities((mu, mu), delta, N_twist=1, k_c=1000)
ks = np.linspace(5, 20, 100)
errs = []
n_p0 = 2*res0.n_a
dns = [n_p0 - 2*h2.get_densities((mu, mu), delta, N_twist=1, k_c=_k).n_a for _k in ks]
plt.loglog(ks, dns, label='Numerical')
plt.loglog(ks, delta**2/2/np.pi/ks**2, '--', label='Thoery')
plt.ylabel("Err")
plt.xlabel(r"$k_c$")
plt.legend()


# The particle density error ratio can be estimated in terms of the healing momentum $k_{\xi}$, Fermi momentum $k_F$ and the momentum cutoff $k_c$:
# $$
# \text{Error Ratio}=\frac{\delta n}{n} = \frac{k_{\xi}^4}{4(2\pi)^4k_c^2k_F^2}
# $$

def get_error_ratio(mu, delta, k_c=200):
    """get the error ratio dn/n"""
    h2 = homogeneous.Homogeneous(dim=2)
    res = h2.get_densities((mu, mu), delta=delta, k_c=k_c)
    n = res.n_a + res.n_b
    k_xi = np.pi*2*np.sqrt(2*delta)
    k_F = np.sqrt(2*np.pi * n)
    err = k_xi**4/k_c**2/k_F**2/(2*np.pi)**4/4
    return err


get_error_ratio(1, 5, k_c=20)

# +
from fulde_ferrell_state_vortex import VortexState
from fulde_ferrell_state import FFState

def numberical_parameter_estimator(mu=1, delta=5, k_c=20):
    ff = FFState(mu=mu, dmu=0, delta=delta, dim=2, k_c=k_c)
    E_c = (ff.hbar*k_c)**2/2/ff.m
    k_h = 2*np.pi * np.sqrt(2*ff.m*delta)/ff.hbar
    na, nb = ff.get_densities(mu=mu, dmu=0, delta=delta)
    n = (na + nb).n
    kF = np.sqrt(2*np.pi*n)
    print(f"k_F={kF}, k_xi={k_h}, k_c={k_c}, E_c={k_c**2/2}")
    dx = np.pi/k_c
    print(f"dx={dx}, healing lenght={2*np.pi/k_h}")
    N = 32
    L = dx*N
    print(f"{2*np.pi/k_c:.2f} < L < {L:.2f}")


# -

numberical_parameter_estimator()

# ## Homogeneous state
# * Here we check expectations of convergence.  If our code is converging, then dimensionless relationships should be universal.
# * As a check, there is an exact solution for $\Delta/e_F = \sqrt{2}$ with $\mu/e_F = 0.5$
# * healing_length $h_{\xi}= \hbar/\sqrt{2m\Delta}$
# * A "large" $\Delta = 10\mu$ has $\mu \approx e_F/5$, and $\eta \approx 2$.  Thus, we must have $k_c > max(1, \sqrt{2})k_F$
# * If $\hbar = \mu = m = 1$, then $k_F \approx \sqrt{10}$, so we must have $k_c > \sqrt{20}$

deltas = np.linspace(0.1, 20.0, 20)
mus = [2, 1.0, 0.5]
ax1 = plt.gca()
ax2 = plt.twinx()
for mu in mus:
    etas = []
    mu_eF = []
    for delta in deltas:
        ff = FFState(mu=mu, dmu=0, delta=delta, dim=2, k_c=20.0)
        na, nb = ff.get_densities(mu=mu, dmu=0, delta=delta)
        n = (na + nb).n
        kF = np.sqrt(2*np.pi * n)
        eF = (ff.hbar*kF)**2/2/ff.m
        etas.append(delta/eF)
        mu_eF.append(mu/eF)
    ax1.plot(deltas/mu, etas, label=r'$\mu=$'+f"{mu}")
    ax2.plot(deltas/mu, mu_eF, ':')
ax1.plot([2*np.sqrt(2)], [np.sqrt(2)], '+')
ax2.plot([2*np.sqrt(2)], [0.5], '+')
ax1.set(xlabel='$\Delta/\mu$', ylabel='$\Delta/e_F$ (solid)');
ax2.set(xlabel='$\Delta/\mu$', ylabel='$\mu/e_F$ (dotted)');
ax1.legend()

# * For a box with lenght $L$ and number of point $N$, the $k_c\approx20$, then from the homogeneous plots above, the reange of coupling $\Delta$ that works well within this cutoff is below 5

# +
import json
from collections import namedtuple
from mmf_hfb.utils import JsonEncoderEx

def dic2namedtupe(dic, name='Results'):
    Results = namedtuple(name, sorted(dic))
    return Results(**dic)

def to_complex(ls):
    """a two component array to complex numbers"""
    return np.array(ls[0])+1j*np.array(ls[1])

def Vortex2D(
        mu, dmu, delta, N=32, L=5, E_c=None, k_c=20, N1=7, N2=5,
        plot=True, plot_2d=False, xlim=None, use_file=False, file_name=None, tol=0.05, **args):
    """
    compute BCS 2D box vortex and homogeneous results
    ------
    N1: number of points near the vortex
    N2: number of points outside the vortex
    """
    if file_name is None:
        file_name=f"Vortex2D_Data_{mu}_{dmu}_{delta}_{N}_{L}_{E_c}_{k_c}_{N1}_{N2}.json"
    if use_file:
        try:
            with open(join(currentdir, file_name), 'r',encoding='utf-8', errors='ignore') as rf:
                obj = json.load(rf)
                obj['v_res'] = json_to_res(obj['v_res'])
                obj['h_res'] = dic2namedtupe(obj['h_res'])
                obj['v_delta']=to_complex(obj['v_delta'])
                if plot:
                    plt.figure(figsize=(16,8))
                    v = VortexState(
                        mu=obj['mu'], dmu=obj['dmu'], delta=obj['delta'],
                        Nxyz=(obj['N'],)*2, Lxyz=(obj['L'],)*2)
                    v.res = obj['v_res']
                    v.Delta = obj['v_delta']
                    plot_all(vs=[v], hs=[obj['h_res']], ls='-o', xlim=xlim, **args)
                return obj
        except:
            use_file = False
            print("Load file failed.")
    k_c = np.pi*N/L
    err = get_error_ratio(mu=mu, delta=delta, k_c=k_c)
    print(f"Error Ratio:{err}")
    v = VortexState(mu=mu, dmu=dmu, delta=delta, Nxyz=(N, N), Lxyz=(L,L), E_c=None)
    v.solve(plot=plot_2d, tol=tol)
    h_res = FFVortex(mus=v.mus, delta=v.delta, L=L, N=N, N1=N1, N2=N2, k_c=k_c)
    
    if use_file == False:
        try:
            with open(join(currentdir, file_name), 'w') as wf:
                output = dict(
                    N=N, L=L, delta=delta, mu=mu, dmu=dmu, v_delta=to_list(v.Delta), 
                    E_c=E_c, k_c=k_c, v_res=res_to_json(v.res), h_res=h_res._asdict(), err=err)
                json.dump(output, wf, cls=JsonEncoderEx)
                print(f'File {file_name} saved.')
        except:
            print("Json Exception.")
    if plot:
        plt.figure(figsize=(16,8))
        plot_all(vs=[v], hs=[h_res], ls='-o', xlim=xlim, **args)
    Results = namedtuple('Results', ['N','L', 'delta','mu','dmu','E_c','k_c', 'v_res', 'h_res', 'err'])
    return Results(N=N, L=N, delta=delta, mu=mu, dmu=dmu,
                    E_c=E_c, k_c=k_c, v_res=v.res, h_res=h_res, err=err)


# -

# ## Strong Coupling
# In this section, we exam the strong coupling regime, where $\Delta\gg\mu$

# ## Symetric case

len_xi = 1.0/(2*5)**0.5

res_s_0 = Vortex2D(mu=1, dmu=0, delta=5, N=32, L=5, k_c=20, N1=15, N2=5, use_file=True, dx=len_xi, dx_text=r"$h_{\xi}$")

res_s_0 = Vortex2D(mu=1, dmu=0, delta=5, N=32, L=5, k_c=20, N1=15, N2=5, use_file=True, )

# ## Asymmetric Cases
# * For Strong coupling, in the box case, we do not see polarization for $\delta mu$ from 1 to 2.
# * When $\delta\mu>2.5$ we may see some polarization, but the homogeneous code does find a finit gap. (need to check carefully).

res_s_1 = Vortex2D(mu=1, dmu=1, delta=5, N=32, L=5, k_c=20, N1=15, N2=5)

res_s_2 = Vortex2D(mu=1, dmu=1.5, delta=5, N=32, L=5, k_c=20, N1=15, N2=5)

res_s_3 = Vortex2D(mu=1, dmu=2, delta=5, N=32, L=5, k_c=20, N1=15, N2=5)

res_s_4 = Vortex2D(mu=1, dmu=3, delta=5, N=32, L=5, k_c=20, N1=10, N2=5, use_file=True)

# ## Gap Solution Double Check
# * result format of the homogeneous calculation h_res = (rs/dx, ds, ps, ps0, n_p, n_m, j_a, j_b)

from fulde_ferrell_state import FFState
from scipy.optimize import brentq
def check_delta(v_res, id):
    rs = v_res.h_res[0]  # rs
    deltas = v_res.h_res[1]  # delta
    v = v_res.v
    dx = v.dxyz[0]
    delta = v.delta
    dx = v.dxyz[0]
    mu_a, mu_b = v.mus
    mu, dmu= (mu_a + mu_b)/2, (mu_a - mu_b)/2
    r = rs[id]
    dq = 0.5/r
    ff = FFState(mu=mu, dmu=dmu, delta=delta, dim=2, k_c=20)
    def f(delta):
        return ff.f(mu=10, dmu=0, delta=delta, dq=dq)
    ds = np.linspace(0, 3*delta, 5)
    fs = [f(d) for d in ds]
    plt.plot(ds, fs, '-+')
    plt.axhline(0)
    plt.xlabel(r'$\Delta$')
    return brentq(f, 0, 10)


deltas_new = res_s_4.h_res[1]

deltas_new[4]= check_delta(res_s_4, 4)

# ## Medium Coupling
# * To be medium compling, we may let $\Delta \approx\mu$

res_m_0 = Vortex2D(mu=1, dmu=0, delta=1, N=32, L=5, k_c=20, N1=15, N2=5)

res_m_1 = Vortex2D(mu=1, dmu=1, delta=1, N=32, L=5, k_c=20, N1=15, N2=5)

res_m_2 = Vortex2D(mu=1, dmu=2, delta=1, N=32, L=5, k_c=20, N1=15, N2=5)

# ## Weak Coupling
# For weak copling, consider $\Delta <\mu$

res_w_0 = Vortex2D(mu=1, dmu=0, delta=0.75, N=32, L=5, k_c=20, N1=15, N2=5)

res_w_1 = Vortex2D(mu=1, dmu=0.3, delta=0.75, N=32, L=5, k_c=20, N1=15, N2=5)

res_w_2 = Vortex2D(mu=1, dmu=0.5, delta=0.75, N=32, L=5, k_c=20, N1=15, N2=5)

res_w_3 = Vortex2D(mu=1, dmu=1, delta=0.75, N=32, L=5, k_c=20, N1=15, N2=5)

# # Counterflow

res_c_1 = Vortex2D(mu=10, dmu=4.5, delta=7.5, N=32, L=5, k_c=20, N1=15, N2=5)

mu = 1
delta = 10.0
ff = FFState(mu=mu, dmu=0, delta=delta, dim=2, k_c=20.0)
na, nb = ff.get_densities(mu=mu, dmu=0, delta=5)
n = (na + nb).n
kF = np.sqrt(2*np.pi * n)
eF = (ff.hbar*kF)**2/2/ff.m
k_xi = np.sqrt(2*delta)
k_xi, kF

from fulde_ferrell_state import FFState
delta = 2.5
ff = FFState(mu=10, dmu=0, delta=delta, dim=2, k_c=20)
def f(delta):
    return ff.f(mu=10, dmu=0, delta=delta, dq=0.5)
ds = np.linspace(0.5*delta, delta, 5)
fs = [f(d) for d in ds]
plt.plot(ds, fs, '-+')
