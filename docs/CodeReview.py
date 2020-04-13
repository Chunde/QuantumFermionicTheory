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

# # Code Review

# In the notebook, I put several tests include homogeneous code , BCS code and Functional Code. First, the homogenous code is tested using analytically known results. Then BCS code is tested with homogeneous code. Finally, the functional code is tested with both BCS, and homogeneous code

# +
import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import *  # Conveniences like clear_output
import numpy as np
import os
import sys
import operator
import inspect
import matplotlib.pyplot as plt
from mmfutils.testing import allclose

from mmf_hfb.homogeneous import Homogeneous1D
from mmf_hfb import hfb, homogeneous
from mmf_hfb.class_factory import ClassFactory, FunctionalType, KernelType, Solvers
clear_output()


# -

# # Homogeneous

def BCS(mu_eff, delta=1.0):
    m = hbar = 1.0
    """Return `(E_N_E_2, lam)` for comparing with the exact Gaudin
    solution.

    Arguments
    ---------
    delta : float
       Pairing gap.  This is the gap in the energy spectrum.
    mu_eff : float
       Effective chemical potential including both the bare chemical
       potential and the self-energy correction arising from the
       Hartree term.

    Returns
    -------
    E_N_E_2 : float
       Energy per particle divided by the two-body binding energy
       abs(energy per particle) for 2 particles.
    lam : float
       Dimensionless interaction strength.
    """
    h = Homogeneous1D()
    v_0, ns, mu, e = h.get_BCS_v_n_e(mus_eff=(mu_eff,)*2, delta=delta)
    n = sum(ns)
    lam = m*v_0/n/hbar**2

    # Energy per-particle
    E_N = e/n

    # Energy per-particle for 2 particles
    E_2 = -m*v_0**2/4.0 / 2.0
    E_N_E_2 = E_N/abs(E_2)
    return E_N_E_2.n, lam.n


# ## Test the Homogeneous1D class for a known solution.
#

np.random.seed(2)
hbar, m, kF = 1.0 + np.random.random(3)
nF = kF/np.pi
eF = (hbar*kF)**2/2/m
C_unit = m/hbar**2/kF
mu = 0.28223521359748843*eF
delta = 0.411726229961806*eF
h = homogeneous.Homogeneous1D(m=m, hbar=hbar)
res = h.get_BCS_v_n_e(mus_eff=(mu,)*2, delta=delta)
assert allclose(res.v_0.n, 1./C_unit)
assert allclose(sum(res.ns), nF)

# ## Test the Homogeneous 2D class for a known solution.

np.random.seed(2)
hbar, m, kF = 1.0 + np.random.random(3)
nF = kF**2/2/np.pi
eF = (hbar*kF)**2/2/m
mu = 0.5*eF
delta = np.sqrt(2)*eF
h = homogeneous.Homogeneous2D(m=m, hbar=hbar)
res = h.get_densities(mus_eff=(mu,)*2, delta=delta)
assert allclose(res.n_a+res.n_b, nF)

# ## Test the Homogeneous3D class for a known solution

# +
np.random.seed(2)
hbar, m, kF = 1.0 + np.random.random(3)
xi = 0.59060550703283853378393810185221521748413488992993
nF = kF**3/3/np.pi**2
eF = (hbar*kF)**2/2/m
# E_FG = 3*nF*eF/5
mu = xi*eF
delta = 0.68640205206984016444108204356564421137062514068346*eF

h = homogeneous.Homogeneous3D(m=m, hbar=hbar)
res = h.get_densities(mus_eff=(mu,)*2, delta=delta, k_c=np.inf)
assert allclose(res.n_a+res.n_b, nF)


# -

# # Compare BCS to Homogeneous

def test_BCS(dim, NLx, T=0, N_twist=1):
    """Compare the BCS lattice class with the homogeneous results."""
    np.random.seed(1)
    hbar, m, kF = 1 + np.random.random(3)
    eF = (hbar*kF)**2/2/m
    mu = 0.28223521359748843*eF
    delta = 0.411726229961806*eF

    N, L, dx = NLx
    if dx is None:
        args = dict(Nxyz=(N,)*dim, Lxyz=(L,)*dim)
    elif L is None:
        args = dict(Nxyz=(N,)*dim, dx=dx)
    else:
        args = dict(Lxyz=(L,)*dim, dx=dx)

    args.update(T=T)

    h = homogeneous.Homogeneous(**args)
    b = hfb.BCS(**args)

    res_h = h.get_densities((mu, mu), delta, N_twist=N_twist)
    res_b = b.get_densities((mu, mu), delta, N_twist=N_twist)

    assert np.allclose(res_h.n_a, res_b.n_a.mean())
    assert np.allclose(res_h.n_b, res_b.n_b.mean())
    assert np.allclose(res_h.nu, res_b.nu.mean())


# N should be small, or for dim=3, it takes forever and may cause memory errors
NLx = (4, 10, None)
dims = [1, 2, 3]
for dim in dims:
    test_BCS(dim=dim, NLx=NLx)


# ## Functionals

def test_thermodynamics_ns_functionals(
        functional, kernel, delta=5.0, mu=10, dmu=1, dim=3, N=8, L=0.46, N_twist=1, k_c=200):
    dx = 1e-3  
    
    mu = mu
    args = dict(Nxyz=(N, )*dim, Lxyz=(L,)*dim, mu_eff=mu, dmu_eff=dmu,
        delta=delta, T=0, dim=dim, k_c=k_c)
    lda = ClassFactory(
        className="LDA",
        functionalType=functional,
        kernelType=kernel, args=args)

    h = homogeneous.Homogeneous(dim=dim)
    res = h.get_densities((mu+dmu, mu-dmu), delta=delta, k_c=k_c)
    def get_ns_e_p(mu, dmu, update_C=False, **args):
        ns, e, p = lda.get_ns_e_p(
            mus=(mu + dmu, mu - dmu), delta=delta, N_twist=N_twist, Laplacian_only=True,
            update_C=update_C, max_iter=32, solver=Solvers.BROYDEN1,
            verbosity=False, **args)
        return ns, e, p
    ns, _, _ = get_ns_e_p(mu=mu, dmu=dmu, update_C=True)
    ns1, e1, p1 = get_ns_e_p(mu=mu+dx, dmu=dmu)
    ns2, e2, p2 = get_ns_e_p(mu=mu-dx, dmu=dmu)
    n_p = (p1-p2)/2.0/dx
    mu_ = (e1-e2)/(np.sum(ns1) - np.sum(ns2))  # this is wrong
    print(res.n_a, res.n_b)
    print(np.mean(ns[0]), np.mean(ns[1]))
    print(np.max(n_p), np.max(sum(ns)))
    print(np.max(mu_), mu)
    if functional == FunctionalType.BDG:
        assert np.allclose(ns[0], res.n_a)
        assert np.allclose(ns[1], res.n_b)
    assert np.allclose(np.mean(n_p).real, sum(ns), rtol=1e-2)
    # should first fix one of the density then compute the partial derivative.
    
    # assert np.allclose(np.mean(mu_).real, mu, rtol=1e-2)  


# ## Thermodynamic
# * The reason why we can not simply use $\frac{E_1-E_2}{n_1 - n_2}$ to get $\mu$ is because the following relations
# \begin{align}
# \frac{d E}{d n}
# &=\frac{d E(n_a, n_b)}{d n}\\
# &=\frac{\partial E(n_a, n_b)}{\partial n_a}\frac{\partial n_a}{\partial n}+\frac{\partial E(n_a, n_b)}{\partial n_b}\frac{\partial n_b}{\partial n}\\
# &=\frac{1}{2}\left[\frac{\partial E(n_a, n_b)}{\partial n_a}\bigg|_{n_b}+\frac{\partial E(n_a, n_b)}{\partial n_b}\bigg|_{n_a}\right]\\
# &=\frac{\mu_a + \mu_b}{2}=\mu\tag{6}
# \end{align}
# * Since $n=n_+=n_a + n_b$, $n_-=n_a - n_b$, so $n_a = (n_++n_-)/2$, $n_b = (n_+-n_-)/2$, then $\frac{\partial n_a}{\partial n_+}=1/2$

# FunctionalType: BDG, SLDA, ASLDA
# KernelType: BCS, HOM (homogeneous)
test_thermodynamics_ns_functionals(
    functional=FunctionalType.BDG, kernel=KernelType.HOM,
    delta=5, mu=15, dmu=5.5, dim=1, k_c=200, N=4, N_twist=1)


