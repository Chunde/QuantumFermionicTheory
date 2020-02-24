# -*- coding: utf-8 -*-
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

import mmf_setup; mmf_setup.nbinit()
import matplotlib.pyplot as plt
from nbimports import *
import numpy as np
from mmf_hfb.ClassFactory import ClassFactory, FunctionalType, KernelType, Solvers
import mmf_hfb.FFStateAgent as ffa

# # Test $g(x)$
#
# * $g(x)$ is defined as:
#
# $$
# \mathcal{E}(n_a, n_b)=\frac{3}{5} \frac{\hbar^{2}}{2 m}\left(6 \pi^{2}\right)^{2 / 3}\left[n_{a} g(x)\right]^{5 / 3}
# $$
#
# The Thomas-Fermi energy density(only has kinetic term) for free fermi gas（FFG） is:
# $$
# \tau_{\mathrm{TF}}(r)=\frac{3 \hbar^{2}}{10 m}\left[3 \pi^{2}\right]^{2 / 3} n(r)^{5 / 3}
# $$
#
# If the FFG density if uniform(i.e not depends on $r$), then the RHS of the first equation is just the FFG energy density. So $g(x)$ is just the ratio of energy of an interaction fermi gas over FFG, i.e.:
# $$
# g(x) = \bigl[\frac{\mathcal{E}(n_a, n_b)}{\tau_{\mathrm{TF}}}\bigr]^{3/5}
# $$

delta=0
args = dict(mu_eff=10, dmu_eff=0, delta=1, T=0, dim=3, k_c=100, verbosity=False)
bdg = ClassFactory(
    "BDG", (ffa.FFStateAgent,), functionalType=FunctionalType.BDG,
    kernelType=KernelType.HOM, args=args)
slda = ClassFactory(
    "SLDA", (ffa.FFStateAgent,), functionalType=FunctionalType.ASLDA,
    kernelType=KernelType.HOM, args=args)
aslda = ClassFactory(
    "ASLDA", (ffa.FFStateAgent,), functionalType=FunctionalType.ASLDA,
    kernelType=KernelType.HOM, args=args)
def g(e, ns):
    g = (e/0.6/0.5/(6*np.pi**2)**(2.0/3))/(ns[0]**(5.0/3))
    g=g**0.6
    return g


mu_eff = 10
ldas = [bdg, aslda]
dmu_effs = np.linspace(0, mu_eff, 20)
for lda in ldas:
    nss=[]
    es = []
    gs=[]
    xs = []
    for dmu_eff in dmu_effs:
        ns, mus, e, p = lda.get_ns_mus_e_p(
            mus_eff=(mu_eff + dmu_eff, mu_eff - dmu_eff), delta=0, solver=Solvers.BROYDEN1)
        nss.append(ns)
        es.append(e)
        gs.append(g(e, ns))
        xs.append(ns[1]/ns[0])
    plt.plot(xs, gs, label=f"{type(lda).__name__}")
plt.axhline(1, linestyle='dashed')
plt.axvline(0, linestyle='dashed')
plt.ylabel(f"g(x)")
plt.xlabel(f"$n_b/n_a$")
plt.legend()
plt.show()

# # Test $\Delta/\mathcal{E}_f$


# # Bertsch parameter
#


