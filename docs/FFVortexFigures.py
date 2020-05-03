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
import fulde_ferrell_state_vortex as ffsv
from fulde_ferrell_state_vortex import FFVortex, VortexState, plot_2D, plot_all, res_to_json, json_to_res, to_list
import warnings
warnings.filterwarnings("ignore")
fontsize = 18


def Vortex2D(**kw):
    return ffsv.Vortex2D(current_dir=currentdir, **kw)


res_s_4 = Vortex2D(mu=1, dmu=3, delta=5, N=32, L=5, k_c=20, N1=10, N2=5, use_file=True)

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
v_1d.solve(tol=1e-3)

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

# ## Critial $\delta q$

from sympy import *
init_printing(use_latex='mathjax')
k=symbols('k')
q = symbols('q')
mu = symbols('mu_+')
dmu=symbols('mu_-')
mu_a = (mu+dmu)/2
mu_b = (mu-dmu)/2
delta = symbols('Delta')
e_a = (k+q)**2/2 - mu_a
e_b = (k-q)**2/2 - mu_b
e_p = (e_a + e_b)/2
e_m = (e_a - e_b)/2
E=sqrt(e_p**2+delta**2)
w_p=e_m + E
w_m=e_m - E
dw_p =diff(w_p, k)
dw_m= diff(w_m, k)

simplify(w_p), simplify(w_m), simplify(dw_p), simplify(dw_m)

# * Simplified by hand:
# \begin{align}
# \omega_{\pm}&=-\frac{\mu_-}{2} + kq \pm\sqrt{4\Delta^2 + (k^2 + q^2 - \mu_+^2)^2}=0\\
# \frac{\partial \omega_{\pm}}{\partial k}&=\frac{\pm k(-2\mu_+ + 2k^2+2q^2) +2q\sqrt{4\Delta^2 + (k^2 + q^2 - \mu_+^2)^2}}{2q\sqrt{4\Delta^2 + (k^2 + q^2 - \mu_+^2)^2}}=0
# \end{align}
# * How to solve for $q$?

# \begin{align}
#  k &=\frac{1}{q}\left(\frac{\mu_-}{2}\mp\sqrt{4\Delta^2 + (k^2 + q^2 - \mu_+^2)^2}\right)\\
#  q&=\frac{\mp k(-\mu_+ + k^2+q^2)}{\sqrt{4\Delta^2 + (k^2 + q^2 - \mu_+^2)^2}}
# \end{align}

# +
"""to find the minum qs"""
mu = 1
dmu = 0
delta = 5

def wp_k(k, q):
    mu_k_q =  k**2+q**2 - mu
    return (-dmu + 2*(4*delta**2+(-mu + mu_k_q**2)**2)**0.5 - 2*k*q)  # igonre a factor of 2

def wm_k(k, q):
    mu_k_q =  k**2+q**2 - mu
    return (-dmu - (4*delta**2+(-mu + mu_k_q**2)**2)**0.5 - 2*k*q ) # igonre a factor of 2

def d_wp_k(k, q):
    mu_k_q = -2*mu + 2*(k**2+q**2)
    denom = (16*delta**2 + mu_k_q**2)**0.5
    dk = k*mu_k_q + q*denom
    return dk  # /denom

def d_wm_k(k, q):
    mu_k_q = -2*mu + 2*(k**2+q**2)
    denom = (16*delta**2 + mu_k_q**2)**0.5
    dk = -k*mu_k_q + q*denom
    return dk  # /denom    

def k_wp_k(k, q):
    mu_k_q =  k**2+q**2 - mu
    return (mu/2 - (4*delta**2 + mu_k_q**2)**0.5)/q

def q_wp_k(k, q):
    mu_k_q =  k**2+q**2 - mu
    return -k*mu_k_q/(4*delta**2 + mu_k_q**2)**0.5


# -

qs = np.linspace(0, k_F, 2)
ks = np.linspace(0, 5, 20)
for dq in qs:
    wp = wp_k(ks, dq)
    plt.plot(ks, wp)

# +
from scipy.optimize import fsolve
import math
k_F = (2*mu)**0.5

def equations(p):
    k, q = p
    return (wp_k(k, q), d_wp_k(k, q))


# -

fsolve(equations, (k_F, k_F))

x, y = fsolve(equations, (1, 1))
wp_k(x,y),d_wp_k(x,y)

# ## Strong Coupling
# In this section, we exam the strong coupling regime, where $\Delta\gg\mu$

from mmf_hfb.hfb import BCS
import numpy as np
def get_min_e(mu, dmu, delta, dq, N=32, L=5):
    """return the min energy in the spectrum"""
    mus = (mu + dmu, mu - dmu)
    b = BCS(Nxyz=(N, N), Lxyz=(L,L))
    H = b.get_H(mus_eff=mus, delta=delta, dq=dq)
    d = np.linalg.eigvals(H)
    return abs(d).min()

# +

    
    
# -

# ## Symetric case

len_xi = 1.0/(2*5)**0.5

res_s_0 = Vortex2D(mu=1, dmu=0, delta=5, N=32, L=5, k_c=20, N1=15, N2=5, use_file=True, dx=len_xi, dx_text=r"$h_{\xi}$")

h_res = res_s_0['v_res']

# +
from collections import namedtuple

def unpack_data(res):
    N = res['N']  # box point
    L = res['L']  # box size
    mu = res['mu']  # mu
    dmu =  res['dmu']  # dmu
    box_delta0 = res['delta']  # delta0 for used to fix g
    
    E_c = res['E_c']  # E_c for box
    k_c = res['k_c']  # k_c for homogeneous
    v_res = res['v_res']
    # box result
    box_delta = np.array(res['v_delta'])  # final converge pairing field
    box_n_a = np.array(v_res.n_a)  # n_a for the box
    box_n_b = np.array(v_res.n_b)  # n_b for the box
    box_j_a = np.array(v_res.j_a)  # j_a for the box
    box_j_b = np.array(v_res.j_b)  #  j_b for the box
    # construct a vortex instance
    v = VortexState(mu=mu, dmu=dmu, delta=box_delta0, Nxyz=(N,)*2, Lxyz=(L,)*2)
    v.Delta=box_delta
    v.res=v_res
    box_rs = np.sqrt(sum(_x**2 for _x in v.xyz))  # box rs
    # homogeneous result
    h_res = res['h_res']
    hom_dx = np.array(h_res.dx)  # homogeneous dx =N/L
    hom_rs = np.array(h_res.rs)
    hom_delta = np.array(h_res.ds)
    hom_n_p = np.array(h_res.n_p)
    hom_n_m = np.array(h_res.n_m)
    hom_n_a = (hom_n_p + hom_n_m)/2
    hom_n_b = (hom_n_p - hom_n_m)/2
    hom_j_a = np.array(h_res.j_a)
    hom_j_b = np.array(h_res.j_b)
    ds_ex = h_res.ds_ex
    hom_rs_ex =[]
    hom_delta_ex = []
    for (r, d) in ds_ex:
        hom_rs_ex.append(r)
        hom_delta_ex.append(d)
    hom_rs_ex = np.array(hom_rs_ex)
    hom_delta_ex = np.array(hom_delta_ex)
    Data = namedtuple("Data", ['v', 'N', 'L', 'mu', 'dmu', 'box_delta0', 
                               'E_c', 'k_c', 'box_xyz', 'box_rs', 'box_delta', 'box_n_a',
                               'box_n_b', 'box_j_a', 'box_j_b','hom_dx',
                              'hom_rs', 'hom_delta', 'hom_n_a', 'hom_n_b',
                              'hom_j_a', 'hom_j_b', 'hom_delta_ex', 'hom_rs_ex'])
    return Data(v=v, N=N, L=L, mu=mu, dmu=dmu, box_delta0=box_delta0, E_c=E_c,
               k_c=k_c, box_xyz=v.xyz, box_rs=box_rs, box_delta=box_delta, box_n_a=box_n_a, box_n_b=box_n_b,
               box_j_a=box_j_a, box_j_b=box_j_b, hom_dx=hom_dx, hom_rs=hom_rs,
               hom_delta=hom_delta, hom_n_a=hom_n_a, hom_n_b=hom_n_b, hom_j_a=hom_j_a,
               hom_j_b=hom_j_b, hom_rs_ex=hom_rs_ex, hom_delta_ex=hom_delta_ex)
 

# -

res = unpack_data(res_s_0)
dx = res.hom_dx
plt.plot(res.box_rs.ravel()/dx, abs(res.box_delta.ravel()))
plt.plot(res.hom_rs/dx, res.hom_delta, '-o')



res_s_0 = Vortex2D(mu=1, dmu=0, delta=5, N=32, L=5, k_c=20, N1=15, N2=5, use_file=True, )

# ## Asymmetric Cases
# * For Strong coupling, in the box case, we do not see polarization for $\delta mu$ from 1 to 2.
# * When $\delta\mu>2.5$ we may see some polarization, but the homogeneous code does find a finit gap. (need to check carefully).

res_s_1 = Vortex2D(mu=1, dmu=1, delta=5, N=32, L=5, k_c=20, N1=15, N2=5, use_file=True)

res_s_2 = Vortex2D(mu=1, dmu=1.5, delta=5, N=32, L=5, k_c=20, N1=15, N2=5, use_file=True)

res_s_3 = Vortex2D(mu=1, dmu=2, delta=5, N=32, L=5, k_c=20, N1=15, N2=5, use_file=True)

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

res_w_3 = Vortex2D(mu=1, dmu=1, delta=0.75, N=32, L=5, k_c=20, N1=15, N2=5, use_file=True)

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
