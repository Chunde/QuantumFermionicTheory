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

# # Check Critical Polarization At Unitary

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import * 

from mmf_hfb.ClassFactory import ClassFactory, FunctionalType, KernelType, Solvers


# * The latest code unifies the old code, which supports both Homogeneous and BCS, different functionals can be chosen.

def create_lda(mu, dmu, delta):
    LDA = ClassFactory(className="LDA", functionalType=FunctionalType.SLDA, kernelType=KernelType.HOM)
    lda = LDA(mu_eff=mu, dmu_eff=dmu, delta=delta, T=0, dim=3)
    lda.C = 0 # unitary case
    return lda


# +
def get_p(lda, mus_eff, delta=None):
    """return polarization"""
    if delta is None:
        delta = lda.solve_delta(mus_eff = mus_eff)
    res = lda.get_densities(mus_eff=mus_eff, delta=delta,taus_flag=False, nu_flag=False)
    ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
    p = min(ns)/max(ns)
    return p

def f(delta, mus_eff):
    """
    we fixed C=0(unitary)
    then compute new C' for given delta and mus_eff
    return dC=C' - C
    if C'==C, we have a solution 
    """
    res = lda.get_densities(mus_eff=mus_eff, delta=delta,taus_flag=False, nu_flag=False)
    ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
    return (lda._get_C(mus_eff=mus_eff, delta=delta, dq=0, ns=ns,taus=taus, nu=nu) - lda.C)


# -

# # In Unitary(3D)
# $$
# \frac{\Delta}{\mu}=1.162200561790012570995259741628790656202543181557689
# $$

mu=10
dmu = 0
delta = 11.62200561790012570995259741628790656202543181557689
lda = create_lda(mu=mu, dmu=dmu, delta=delta)

mus_eff = (mu + dmu, mu-dmu)
lda.solve_delta(mus_eff=mus_eff)

# ## No Polarization Case: $d\mu<\Delta$

dmus = np.linspace(0, mu, 10)
ds = []
for dmu in dmus:
    try:
        ds.append(lda.solve_delta(mus_eff=(mu+dmu, mu-dmu)))
    except:
        break

for i in range(len(ds)):
    dmu = dmus[i]
    print(dmu, get_p(lda, mus_eff=(mu+dmu, mu-dmu)))


# ## Check Solution

def Plot_C(dmu=2, a=0, b=1.5, n=20):
    mus_eff=(mu+dmu, mu-dmu)
    ds = np.linspace(a*delta,b*delta,n)
    fs = [f(d, mus_eff=mus_eff) for d in ds]
    plt.plot(ds, fs)
    plt.xlabel(f"$\Delta$")
    plt.ylabel("C")
    plt.axhline(0, linestyle='dashed')


Plot_C(dmu=2)

# ### Polarized Case $d\mu > \Delta$

Plot_C(dmu=11.7)




