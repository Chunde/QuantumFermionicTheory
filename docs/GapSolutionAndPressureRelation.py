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
from nbimports import * 
import os
import sys
import inspect
from os.path import join
from mmf_hfb import tf_completion as tf
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, join(currentdir, '..','Projects','FuldeFerrellState'))
from fulde_ferrell_state import FFState
from fulde_ferrell_state_finder import FFStateFinder
from scipy.optimize import brentq
from mmfutils.plot import imcontourf

# # Gap Equation and Pressure
# * Understanding how solution of gap equation yield max pressure

mu=10
dmu=0
delta=2
g= None#-0.3526852951505492
k_c=50
dq = 0.1
ff = FFState(mu=mu, dmu=dmu, delta=delta, dim=3, g=g, k_c=k_c, fix_g=True)
mu_eff, dmu_eff = mu, dmu
n_a, n_b = ff.get_densities(mu=mu, dmu=dmu, delta=delta)
print(f"n_a={n_a.n}, n_b={n_b.n}, P={(n_a - n_b).n/(n_a + n_b).n}")
ds = np.linspace(0.001,3,20)
dmus = np.linspace(0, 1.5, 4)
plt.figure(figsize(10,6))
for dmu in dmus:
    ps0 = [ff.get_pressure(mu=mu, dmu=dmu, mu_eff=mu, dmu_eff=dmu, delta=d, dq=dq, use_kappa=False).n for d in ds]
    ps1 = [ff.get_pressure(mu=mu, dmu=dmu, mu_eff=mu, dmu_eff=dmu, delta=d, dq=dq, use_kappa=True).n for d in ds]

    delta_max = ff.solve(mu=mu, dmu=dmu, dq=dq)
    print(f"dmu={dmu},g={ff._g},Delta={delta_max}")
    plt.plot(ds, ps0, label=f"$d\mu=${dmu}")
    plt.plot(ds, ps1,"--", label=f"$d\mu=${dmu}")
    plt.axvline(delta_max)
plt.xlabel(f"$\Delta$")
plt.ylabel(f"Press")
plt.legend()
plt.axvline(delta)

for dmu in dmus:
    ps0 = [ff.get_energy_density(mu=mu, dmu=dmu,delta=d, dq=dq, use_kappa=False).n for d in ds]
    ps1 = [ff.get_energy_density(mu=mu, dmu=dmu,delta=d, dq=dq, use_kappa=True).n for d in ds]
    delta_max = ff.solve(mu=mu, dmu=dmu, dq=dq)
    print(f"dmu={dmu},g={ff._g},Delta={delta_max}")
    plt.plot(ds, ps0, label=f"$d\mu=${dmu}")
    plt.plot(ds, ps1,"--", label=f"$d\mu=${dmu}")
    plt.axvline(delta_max)
plt.xlabel(f"$\Delta$")
plt.ylabel(f"Press")
plt.legend()
plt.axvline(delta)

plt.figure(figsize(8,4))
def g(d):
    return ff.get_g(mu=mu, dmu=dmu, delta=d, dq=dq) - ff._g
gs = [g(d) for d in ds]
plt.plot(ds, gs, label=f"d$\mu$={dmu}")
plt.axhline(0)
plt.legend()


