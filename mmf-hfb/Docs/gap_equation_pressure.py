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
from mmf_hfb.FFStateFinder import FFStateFinder
from scipy.optimize import brentq
from mmfutils.plot import imcontourf
clear_output()

mu=10
dmu=0
delta=2
g= None#-0.3526852951505492
k_c=50
dq = 0.04810320463378813
ff = FFState(mu=mu, dmu=dmu, delta=delta, dim=3, g=g, k_c=k_c, fix_g=True)
mu_eff, dmu_eff = mu, dmu
n_a, n_b = ff.get_densities(mu=mu, dmu=dmu, delta=delta)
print(f"n_a={n_a.n}, n_b={n_b.n}, P={(n_a - n_b).n/(n_a + n_b).n}")
ds = np.linspace(0.001,3,30)
dmus = np.linspace(0, delta, 5)
for dmu in dmus:
    ps = [ff.get_pressure(mu=mu, dmu=dmu, mu_eff=mu, dmu_eff=dmu, delta=d, dq=dq).n for d in ds]
    delta_max = ff.solve(mu=mu, dmu=dmu, dq=dq)
    print(f"g={ff._g},Delta={delta_max}")
    plt.plot(ds,ps, label=f"$d\mu=${dmu}")
    #plt.axvline(delta_max)
plt.legend()


