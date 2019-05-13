# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import * 
from mmf_hfb import tf_completion as tf
from mmf_hfb.FuldeFerrelState import FFState
from scipy.optimize import brentq
from mmfutils.plot import imcontourf

# # Conditions for vortex
# * $\Delta$ should be continuous in a close loop, in rotation frame, the gap should satisfy:
# $$
# \Delta=\Delta e^{-2i\delta qx}
# $$
# with phase change in a loop by $2\pi$, which requires: $\delta q= \frac{1}{2r}$
# * the overall momentum should be proportional to 1/r too: $q=\frac{1}{r}$

mu_eff=10
dmu_eff= 0.21
dim = 1
delta=0.2
g=-2.8
ff = FFState(mu=mu_eff, dmu=dmu_eff, delta=delta, dim=dim, fix_g=True, g=g, bStateSentinel=False)
print(ff._get_effective_mus(mu=mu, dmu=dmu, dq=dq, delta=delta, update_g=False))

ff.get_densities(mu=mu_eff, dmu=dmu_eff, delta=0, dq=1)

# ## Densities

rs = np.linspace(3, 8.3, 30)
rs1 = np.linspace(8.5,15,30)
rs = np.concatenate((rs,rs1))
deltas = []
nas, nbs = [], []
ps = []
for r in rs:
    q = 1/ r
    dq = 0.5/ r
    args = dict(mu=mu_eff, dmu=dmu_eff, q=q, dq=dq)
    d = ff.solve(a=0.8 * delta, b=3* delta, **args)
    n_a, n_b = ff.get_densities(delta=d, k_c=np.inf, **args)
    nas.append(n_a.n)
    nbs.append(n_b.n)
    deltas.append(d)

plt.figure(figsize(16, 6))
plt.subplot(221)
plt.title(f"$\Delta$ vs r")
plt.ylabel(f"$\Delta$")
plt.plot(rs, deltas)
plt.subplot(222)
plt.title(f"$n_a, n_b$ vs r")
plt.ylabel(f"$n_a, n_b$")
plt.plot(rs, nas, label=f"$n_a$")
plt.plot(rs, nbs, label=f"$n_b$")
plt.legend()
plt.subplot(223)
plt.title(f"$n_+$ vs r")
plt.ylabel(f"$n_a+n_b$")
plt.plot(rs, np.array(nas)+np.array(nbs))
plt.subplot(224)
plt.title(f"$n_-$ vs r")
plt.ylabel(f"$n_a-n_b$")
plt.plot(rs, np.array(nas)-np.array(nbs))

# ## Pressure

import warnings
warnings.filterwarnings("ignore")
ps = [ff.get_pressure(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=d, dq=0.5/r, use_kappa=False) for r, d in zip(rs,deltas)]
ps0 = [ff.get_pressure(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=0, dq=0, use_kappa=False) for r, d in zip(rs,deltas)]

plt.plot(rs,ps, label="FF/Superfluid State Pressure")
plt.plot(rs,ps0, '--', label="Normal State Pressure")
plt.ylabel(f"Pressure")
plt.xlabel(f"r")
plt.legend()

# ## Currents

js = [ff.get_current(mu=mu_eff, dmu=dmu_eff, delta=d, q=1/r, dq=0.5/r) for r, d in zip(rs,deltas)]

jas, jbs=[], []
jps, jms=[], []
for j in js:
    jas.append(j[0].n)
    jbs.append(j[1].n)
    jps.append(j[2].n)
    jms.append(j[3].n)
plt.plot(rs, jas, label=f"$j_a$")
plt.plot(rs, jbs, label=f"$j_b$")
#plt.plot(rs, jps, label=f"$j_p$")
#plt.plot(rs, jms, label=f"$j_m$")
plt.legend()

# ## Play with solve routine

ds = np.linspace(0.8 * delta, 3* delta, 40)
r=5.507692307692308
dq = 0.5/r
def f(d):
    return ff._g - ff.get_g(delta=d, mu=mu_eff, dmu=dmu_eff, q=0, dq=dq)
gs = [ff.f(mu=mu_eff, dmu=dmu_eff, delta=d, q=q, dq=dq) for d in ds]
ps = [ff.get_pressure(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=d, dq=dq, use_kappa=False) for d in ds]
ps0 = [ff.get_pressure(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=0, dq=0, use_kappa=False) for d in ds]
clear_output()
plt.subplot(211)
plt.plot(ds, gs)
plt.axhline(0)
plt.subplot(212)
plt.plot(ds, ps, "--")
plt.plot(ds, ps0, "-")
args = dict(mu=mu_eff, dmu=dmu_eff, q=q, dq=dq)
ff.solve(a=0.8 * delta, b=3* delta, **args)


