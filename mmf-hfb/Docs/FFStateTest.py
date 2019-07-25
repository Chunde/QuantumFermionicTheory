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
import mmf_hfb.FFStateHelper as ffh
reload(ffh)
from mmf_hfb.FFStateHelper import FFStateHelper
from scipy.optimize import brentq
import operator
from mmfutils.plot import imcontourf
import mmf_hfb.BdGPlot as bp
reload(bp)
clear_output()

# # A FF Ground State in BdG

e_F = 10
mu0 = 0.59060550703283853378393810185221521748413488992993 * e_F
delta0 = 0.68640205206984016444108204356564421137062514068346 * e_F
mu = mu0 # ~6
delta = 1.5
dmu = 1.0875000000000001
ffs = FFState(mu=mu0, dmu=0, delta=delta, dim=3, k_c=50, fix_g=True)

ffs.g

# ## A FF State Solution
# $\Delta= 0.12130404040404041, q=0.37325733405245315$
# * we can check if this is a solution by computing the g value

delta_ = 0.12130404040404041
q=0.37325733405245315
args=dict(mu=mu0, dmu=dmu, delta=delta_, dq=q)
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

import mmf_hfb.ClassFactory as cf; reload(cf)
from mmf_hfb.ClassFactory import ClassFactory, FunctionalType, KernelType, Solvers

mu_eff = 10
dmu_eff = 0
args = dict(mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,T=0, dim=3, k_c=50, verbosity=False)
lda = ClassFactory("LDA",functionalType=FunctionalType.SLDA, kernelType=KernelType.HOM, args=args)

lda.get_pressure(mus_eff=(mu_eff+dmu_eff, mu_eff-dmu_eff), delta=delta)

lda.get_pressure(mus_eff=(mu_eff+dmu_eff, mu_eff-dmu_eff), delta=0)

lda._get_C(mus_eff=(mu_eff,mu_eff), delta=1)

lda.fix_C_BdG(mu=mu_eff,dmu=0, delta=1)

lda.C


