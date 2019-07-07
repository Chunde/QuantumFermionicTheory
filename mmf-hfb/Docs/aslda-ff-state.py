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

# # ASLDA FF State

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import * 

from mmf_hfb.ClassFactory import ClassFactory, FunctionalType, KernelType, Solvers
import numpy as np
import warnings
# warnings.filterwarnings("ignore")

mu=10
dmu=0
delta=1
dim=3
LDA = ClassFactory(
            className="LDA",
            functionalType=FunctionalType.SLDA,
            kernelType=KernelType.HOM)
lda = LDA(mu_eff=mu, dmu_eff=dmu, delta=delta, T=0, dim=dim)

ns, e, p = lda.get_ns_e_p(mus=(mu, dmu), delta=delta, solver=Solvers.BROYDEN1, update_C=True, verbosity=True)

lda.get_ns_e_p(mus=(mu, dmu), delta=delta, dq=0.05, solver=Solvers.BROYDEN1, update_C=False, verbosity=True)


