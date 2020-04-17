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

# # Fulde Ferrell State Playgound
# This notebook contain a solution to the FF State that has higher pressure than the superfluid state. Since we have two methods to implement the FF State class, one is using the ClassFactory method, which can be extented to other functionals, but the code struture is a bit hard to understand. Another method is simply use the old FuldeFerrell Stete Class. I suggest to use the FuldeFerrellState class.
# The found FF State has fixed $g$ or $C$ using ($\Delta_0$=0.5, and $\mu=10$, $\delta \mu=0.446$, note: when compute $g$ or $C$, I only use $\Delta_0$ and $\mu$), then I found a FF state at ($dq= 0.11176852950565815, delta=0.05368101205886646$)

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import *
from scipy.optimize import brentq
from mmf_hfb.class_factory import ClassFactory, FunctionalType, KernelType, Solvers
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import inspect
from os.path import join
import json
import glob
from json import dumps
import operator
import numpy as np

currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, join(currentdir, '..','Projects','FuldeFerrellState'))
from phase_diagram_generator import FFStateAgent

# # Use the FF State Class

from fulde_ferrell_state import FFState
mu_eff = 10
dmu_eff = 0.446
delta0 = 0.5
dim = 2
k_c = 150
ff = FFState(mu=mu_eff, dmu=dmu_eff, delta=delta0, dim=2, k_c=k_c, fix_g=True)

# ## check if the found state satisifies the gap equation

dq = 0.11176852950565815
delta = 0.05368101205886646
ff.f(mu=mu_eff, dmu=dmu_eff,dq=dq, delta=delta)

# ## Check densites
# * $n_a\ne n_b$

ff.get_densities(mu=mu_eff, dmu=dmu_eff, delta=delta, dq=dq)

# ## FF State pressure and currents

ff.get_pressure(mu=mu_eff, dmu=dmu_eff, delta=delta, dq=dq, use_kappa=False)

# * Current format: $j_a, j_b, j_+, j_-$, we may have a sign different for $j_b$, if filp, we will get total current eqaual to zero. I may be wrong.

ff.get_current(mu=mu_eff, dmu=dmu_eff, delta=delta, dq=dq)

# ### Superfluid pressure

ff.get_pressure(mu=mu_eff, dmu=dmu_eff, delta=delta0, use_kappa=False)

# * current are all zero for superfluid state

ff.get_current(mu=mu_eff, dmu=dmu_eff, delta=delta0)

# ### Normal state pressure

ff.get_pressure(mu=mu_eff, dmu=dmu_eff, delta=0)

ff.get_current(mu=mu_eff, dmu=dmu_eff, delta=0)

# ## Dicussion
# The normal pressure is higher than superfluid pressure, they are just two different solutions. The Fulde Ferrell State has higher pressure than superful

# # Use the ClassFactory Method
# * To use the FuldeFerrellState class, see the previous section

mu_eff = 10
dmu_eff = 0.446
mus_eff = (mu_eff + dmu_eff, mu_eff - dmu_eff)
delta0 = 0.5
dim = 2
k_c = 150
args = dict(
        mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta0,
        T=0, dim=dim, k_c=k_c, verbosity=False)
lda = ClassFactory(
    "LDA", (FFStateAgent,),
    functionalType=FunctionalType.BDG,
    kernelType=KernelType.HOM, args=args)
lda.C = lda._get_C(mus_eff=(mu_eff, mu_eff),delta=delta0)


def f(delta, dq):
    return lda._get_C(
        mus_eff=(mu_eff + dmu_eff, mu_eff - dmu_eff),
        delta=delta, dq=dq) - lda.C


# ## A FF Solution
# * A solution that has higher pressure than the superfluid state

dq = 0.11176852950565815
delta = 0.05368101205886646

# ## Check the $f$ value

f(delta=delta, dq=dq)

# ## FF State Pressure

p_ff=lda.get_ns_mus_e_p(mus_eff=mus_eff, delta=delta, dq=dq)[3]

# ## Superfluid State Pressure

p_sf=lda.get_ns_mus_e_p(mus_eff=mus_eff, delta=delta0, dq=0)[3]

# ## Normal State Pressure
# * Note: when $\Delta=0$, we may have different $C$ or $g$, so it is not the same system we have.

p_nm=lda.get_ns_mus_e_p(mus_eff=mus_eff, delta=0, dq=0)[3]

print(p_ff, p_sf, p_nm)

assert p_ff > p_nm
assert p_ff > p_sf

# ## Check the Currents

ja, jb, jp, jm = lda.get_current(mus_eff=mus_eff, delta=delta, dq=dq)

print(ja, jb, jp, jm)


