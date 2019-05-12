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

mu_eff=10
dmu_eff= 0.21
delta=0.2
g=-2.8
q=1
dq=0.02630155299196228
dim = 1
ff = FFState(mu=mu_eff, dmu=dmu_eff, delta=delta, dim=dim, fix_g=True, g=g, bStateSentinel=True)
mu, dmu = ff._get_bare_mus(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta, q=q, dq=dq)
n_a, n_b, e, p, mus_eff = ff.get_ns_p_e_mus_1d(mu=mu, dmu=dmu, delta=delta,q=q, dq=dq, update_g=False)
# or compute effective mus first
mu_eff, dmu_eff = ff._get_effective_mus(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq, update_g=False)
n_a, n_b = ff.get_densities(mu=mu_eff, dmu=dmu_eff,  delta=delta,q=q, dq=dq)
j_a, j_b, j_p, j_m = ff.get_current(mu=mu_eff, dmu=dmu_eff, delta=0, q=q, dq=dq)
print(f"n_a={n_a.n}, n_b={n_b.n}, j_a={j_a.n}, j_b={j_b.n}, j_p={j_p.n}, j_m={j_m.n}")
# re-compute the effective mus as for normal state, delta=dq=0
mu_eff, dmu_eff = ff._get_effective_mus(mu=mu, dmu=dmu, delta=0, mus_eff=(mu_eff, dmu_eff), q=q, dq=dq, update_g=False)
p0 = ff.get_pressure(mu=mu, dmu=dmu, mu_eff=mu_eff, dmu_eff=dmu_eff, delta=0)
print(f"FF State Pressure={p}, Normal State Pressue={p0}")
if not np.allclose(n_a.n, n_b.n) and dq != 0 and p0 < p:
    print("The ground state is a FF State")

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
dq=0.02630155299196228
ff = FFState(mu=mu_eff, dmu=dmu_eff, delta=delta, dim=dim, fix_g=True, g=g, bStateSentinel=False)
print(ff._get_effective_mus(mu=mu, dmu=dmu, dq=dq, delta=delta, update_g=False))

# +
rs = np.linspace(5, 8.3, 40)
rs1 = np.linspace(8.5,15,30)
rs = np.concatenate((rs,rs1))
deltas = []
nas, nbs = [], []
ps = []
jas, jbs = [], []

for r in rs:
    q = 1/ r
    dq = 0.5/ r
    args = dict(mu=mu_eff, dmu=dmu_eff, q=q, dq=dq)
    d = ff.solve(a=0.8 * delta, b=3* delta, **args)
    n_a, n_b = ff.get_densities(delta=d, k_c=np.inf, **args)
    nas.append(n_a.n)
    nbs.append(n_b.n)
    deltas.append(d)
    print(r, d)
# -

plt.figure(figsize(16, 6))
plt.subplot(121)
plt.plot(rs, deltas)
plt.subplot(122)
plt.plot(rs, nas)
plt.plot(rs, nbs)

ps = [ff.get_pressure(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=d, dq=dq, use_kappa=False) for d in deltas]
ps0 = [ff.get_pressure(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=0, dq=0, use_kappa=False) for d in deltas]
clear_output()

plt.plot(rs,ps)
plt.plot(rs,ps0, '--')

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


