# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
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
import mmf_hfb.FFStateHelper as ffh
reload(ffh)
from mmf_hfb.FFStateHelper import FFStateHelper
from scipy.optimize import brentq
import operator
from mmfutils.plot import imcontourf
import mmf_hfb.BdGPlot as bp
import numpy as np
reload(bp)
clear_output()

# # A FF Ground State in BdG

e_F = 10
mu0 = 0.59060550703283853378393810185221521748413488992993 * e_F
#delta0 = 0.68640205206984016444108204356564421137062514068346 * e_F
mu = mu0 # ~6
delta0 = 1.5
dmu = 1.0875000000000001
ffs = FFState(mu=mu0, dmu=0, delta=delta0, dim=3, k_c=50, fix_g=True)

ffs.g

# ## A FF State Solution
# $\Delta= 0.12130404040404041, q=0.37325733405245315$
# * we can check if this is a solution by computing the g value

delta = 0.12130404040404041
q=0.37325733405245315
args=dict(mu=mu0, dmu=dmu, delta=delta, dq=q)
assert np.allclose(ffs.get_g(**args), ffs.g)

# * compute the density

ffs.get_densities(**args)

# ### Compute the pressure and current

ffs.get_pressure(**args)

ffs.get_current(**args)  # j_a, j_b, j_p, j_m

# ### Compute pressures for normal state and symetric/polorized state

ffs.get_pressure(mu=mu0, dmu=dmu, delta=0, dq=0)  # normal state pressure

ffs.get_pressure(mu=mu0, dmu=dmu, delta=None) # pressure without dq, gap equation will be solved if delta is not given

ffs.get_densities(mu=mu0, dmu=dmu, delta=None)  # check density

ffs.get_pressure(mu=mu0, dmu=0, delta=None) # symetric state pressure

ffs.get_densities(mu=mu0, dmu=0, delta=None) # check density

# # ASLDA FF State
# * In ASLDA case(here I used SLDA because the alpha terms in ASLDA seems not right), if we set the effective mus to the mus we used in the BdG case above, and set the D term to zero by setting the weight of D to zero(D0 is the weight), we should be able to get the same result as that of BdG.
# * By varying the D0 term, we may be able to see at which value of D0 does the FF state vanish.

import mmf_hfb.ClassFactory as cf; reload(cf)
from mmf_hfb.ClassFactory import ClassFactory, FunctionalType, KernelType, Solvers

args = dict(mu_eff=mu, dmu_eff=dmu, delta=delta0,T=0, dim=3, k_c=50, verbosity=False)
lda = ClassFactory("LDA",functionalType=FunctionalType.SLDA, kernelType=KernelType.HOM, args=args)
lda.C=lda._get_C(mus_eff=(mu+dmu, mu - dmu), delta=delta0)

# ## Check the pressure of the configuration

lda.D0=0
lda.get_pressure(mus_eff=(mu+dmu, mu-dmu), delta=delta, dq=q)

# ## Get the bare mus
# * To compare with normal state and symetric state, we need to use the same bare mus

mu_a, mu_b = lda.get_mus_bare(mus_eff=(mu+dmu, mu-dmu), delta=delta, dq=q)
print(mu_a, mu_b)

lda.get_pressure(mus=(mu_a, mu_b), delta=0)

lda.get_pressure(mus=(mu_a, mu_b), delta=None)

# ## Plot Pressures vs D0

Ds = np.linspace(0, 0.2, 10)
mu_a, mu_b = lda.get_mus_bare(mus_eff=(mu+dmu, mu-dmu), delta=delta, dq=q)
P0, P1, P2 = [],[],[]
for D0 in Ds:
    lda.D0=D0
    P0.append(lda.get_pressure(mus_eff=(mu+dmu, mu-dmu), delta=delta, dq=q))  # FF State
    P1.append(lda.get_pressure(mus=(mu_a, mu_b), delta=0)) # Normal State
    P2.append(lda.get_pressure(mus=(mu_a, mu_b), delta=None))  # Sysmetric State
clear_output()

plt.plot(Ds, P0, label="FF State")
plt.plot(Ds, P1, label="Normal State")
plt.plot(Ds, P2, label="Sysmetric State")
plt.legend()
print(P0[0], P1[0], P2[0])


def f_d(D0):
    return 


mu=5
delta = 1
args = dict(mu_eff=mu, dmu_eff=0, delta=delta,T=0, dim=3, k_c=50, verbosity=False)
lda = ClassFactory("LDA",functionalType=FunctionalType.SLDA, kernelType=KernelType.HOM, args=args)
def f(dmu=0, dq=0.1): 
    args = dict(mus_eff=(mu + dmu, mu - dmu),delta=delta, dq=dq)
    js = lda.get_current(**args)
    dens = lda.get_densities(**args)
    ns = dens.n_a + dens.n_b
    j0 = ns*dq
    j = js[3]
    P = abs(dens.n_a - dens.n_b)/(dens.n_a + dens.n_b)
    return abs(j/j0).n, P


dmus = np.linspace(0, 5, 100)
js = []
Ps = []
for dmu in dmus:
    j, P = f(dmu=dmu)
    js.append(j)
    Ps.append(P)

plt.plot(Ps, js)
plt.xlabel("P")
plt.ylabel(r"$\frac{J_s}{J}$")

plt.plot(dmus, js)


