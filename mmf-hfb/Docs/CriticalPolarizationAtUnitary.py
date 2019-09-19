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

# # Check Critical Polarization At Unitary

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import *
import numpy as np

from mmf_hfb.ClassFactory import ClassFactory, FunctionalType, KernelType, Solvers


# * The latest code unifies the old code, which supports both Homogeneous and BCS, different functionals can be chosen. The factory function is defnied in file 'ClassFactory.py'

def create_lda(mu, dmu, delta):
    """
    Functional Type: BDG/SLDA/ASLDA
    Kernel Type:HOM/BCS
    """    
    LDA = ClassFactory(className="LDA", functionalType=FunctionalType.ASLDA, kernelType=KernelType.HOM)
    lda = LDA(mu_eff=mu, dmu_eff=dmu, delta=delta, T=0, dim=3)
    lda.C = 0 #lda._get_C(mus_eff=(mu+dmu, mu-dmu), delta=delta)  # unitary case
    return lda


# +
def get_p(lda, mus_eff, delta=None, dq=0):
    """return polarization"""
    if delta is None:
        delta = lda.solve_delta(mus_eff = mus_eff)
    res = lda.get_densities(mus_eff=mus_eff, delta=delta, dq=dq, taus_flag=False, nu_flag=False)
    na, nb= (res.n_a, res.n_b)
    p = (na-nb)/(na+nb)
    return p

def f(delta, mus_eff, dq=0):
    """
    we fixed C=0(unitary)
    then compute new C' for given delta and mus_eff
    return dC=C' - C
    if C'==C, we have a solution 
    """
    res = lda.get_densities(mus_eff=mus_eff, delta=delta, dq=dq,
                            taus_flag=False, nu_flag=False)
    ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
    return (lda._get_C(mus_eff=mus_eff, delta=delta, dq=dq, ns=ns,taus=taus, nu=nu) - lda.C)


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

# ## Check Solution
# $P_c$ is the polarization at the point where the FF solution to the gap equation begins with $\Delta_{FF} = 0$.  Here I solve for this and then compute the critical polarization.According to the paper, this should be 0.834 so something is wrong.  Please check that this is insensitive to regularization etc.

k0 = np.sqrt(2*mu)
def Plot_C(dmu=2, delta0=1e-8, n=20, a=0, b=1):
    mus_eff=(mu+dmu, mu-dmu)
    dqs = np.linspace(a, b, n)*k0
    fs = [f(delta0, mus_eff=mus_eff, dq=dq) for dq in dqs]
    plt.plot(dqs/k0, fs)
    plt.xlabel(f"$q$")
    plt.ylabel("C")
    plt.axhline(0, linestyle='dashed')
    min_index = 0
    min_value = fs[0]
    for i in range(1, len(fs)):
        if fs[i] < min_value:
            min_value = fs[i]
            min_index = i
    print(min_value)
    plt.axvline(dqs[min_index]/k0, linestyle='dashed')
    return dqs[min_index]


Plot_C(dmu=3.5, n=10)

dq = Plot_C(dmu=4.025, n=20, a=0.2, b=0.3)

dmu = 4.025
delta = 1e-8
get_p(lda, mus_eff=(mu + dmu, mu - dmu), delta=delta, dq=dq)

# # Formulation

# Here we consider integration of the BdG equations at $T=0$.  In particular, we identify when the integrands might have kinks in order to specify points of integration.  We start with the quasi-particle dispersion relationships which define the occupation numbers.  We allow for Fulde-Ferrell states with momentum $q$ along the $x$ axis. Kinks occur when these change sign:
#
# $$
#   \omega_{\pm} = \epsilon_{\pm} \pm E, \qquad
#   E = \sqrt{\epsilon_+^2+\abs{\Delta}^2},\\
#   \epsilon_{\pm} = \frac{\epsilon_a \pm \epsilon_b}{2}, \qquad
#   \epsilon_{a, b} = \frac{(p_x \pm q)^2 + p_\perp^2}{2m} - \mu_{a,b}.
# $$
#
# Simplifying, we have:
#
# $$
#   \epsilon_{-} = \frac{qp_x}{m} - \mu_{-}, \qquad
#   \epsilon_{+} = \frac{p_x^2 + p_\perp^2}{2m} - \Bigl(\overbrace{\mu_{+} - \frac{q^2}{2m}}^{\mu_q}\Bigr), 
#   \qquad
#   \mu_{\pm} = \frac{\mu_{a} \pm \mu_{b}}{2}.
# $$
#

# ## Derivatives

# $$
# \frac{\partial \epsilon_-}{\partial q}=\frac{p_x}{m} \qquad \frac{\partial \epsilon_+}{\partial q}=\frac{q}{m} \qquad
# \frac{\partial E}{\partial q}=\frac{\epsilon_+}{E}\frac{\partial \epsilon_+}{\partial q}=\frac{\epsilon_+}{E}\frac{q}{m}
# $$

# $$
# \frac{\partial \omega_+}{\partial q}=\frac{\partial \epsilon_+}{\partial q}+\frac{\partial E}{\partial q}=\frac{q}{m}-\frac{\epsilon_+}{E}\frac{q}{m}\\
# \frac{\partial \omega_-}{\partial q}=\frac{\partial \epsilon_-}{\partial q}-\frac{\partial E}{\partial q}=\frac{p_x}{m}-\frac{\epsilon_+}{E}\frac{q}{m}
# $$

# ## Indentities

# $$
# C=\frac{m}{4 \pi \hbar^{2} a}=\frac{1}{g}+\frac{1}{2} \int \frac{\mathrm{d}^{3} \mathbf{k}}{(2 \pi)^{3}} \frac{1}{\frac{h^{2} k^{2}}{2 m}+\mathrm{i} 0^{+}}=\frac{1}{g}+\Lambda\\
# \Lambda=\frac{m}{\hbar^{2}} \frac{k_{c}}{2 \pi^{2}}\left\{1-\frac{k_{0}}{2 k_{c}} \ln \frac{k_{c}+k_{0}}{k_{c}-k_{0}}\right\}=\frac{m}{\hbar^{2}} \frac{k_{c}}{2 \pi^{2}}\bigl{|}_{k_0/k_c\rightarrow 0}
# $$

# Assume
# $
# \epsilon_k^{\uparrow} =\epsilon_{-k}^{\downarrow}
# $, then $\omega_- = -\omega_+$
#
# $n_+$ is the total particle number, while $n_-$ is the number difference
#
# $$
# \begin{align}
# n_+ &= n(\epsilon_+)+n(\epsilon_-)= \int\frac{\d{k}}{2\pi}\left(1 - \frac{\epsilon^+_k}{E_k}\bigl(f(\omega_-) - f(\omega_+)\bigr)\right),\\
# n_- &= n(\epsilon_+)-n(\epsilon_-)= \int\frac{\d{k}}{2\pi}\bigl(f(\omega_+) - f(-\omega_-)\bigr),\\
# \Delta&= \frac{v}{2}\int \frac{\d{k}}{2\pi}\frac{\Delta}{E_k}\bigl(f(\omega_-)-f(\omega_+)\bigr),\\
# \frac{1}{v}&= \frac{1}{2}\int \frac{\d{k}}{2\pi}\frac{1}{E_k}\bigl(f(\omega_-)-f(\omega_+)\bigr).\\
# \Delta&=g\nu \qquad g=\frac{\Delta}{\nu} \qquad \frac{1}{g}=\frac{\nu}{\Delta}=\frac{1}{2}\int \frac{\d{k}}{2\pi}\frac{\Delta}{E_k}\bigl(f(\omega_-)-f(\omega_+)\bigr).\\
# \end{align}
# $$

# Then if $C$ is treated as a funtion of $q$：
# $$
# \begin{align}
# C(q,\mu)
# &=\frac{1}{g(q)}+\Lambda\\
# &=\frac{1}{2}\int \frac{\d{k}}{2\pi}\frac{\Delta}{E_k}\bigl(f(\omega_-)-f(\omega_+)\bigr) + \Lambda\\
# \end{align}
# $$
#

# To determine the extremum of $C(q,\mu)$, we just need to compute the derivative of it with $T=0$, so $f(x)$ is a step function and its derivative is just $\delta(x)$:
#
# $$
# \frac{\partial C}{\partial q}=\left\{\frac{\Delta}{2}\int \frac{\d{k}}{2\pi} 
# \frac{\partial}{\partial q}\left[\frac{f(\omega_-)-f(\omega_+)}{E_k}\right]\right\}\bigl{|}_{\mu}=0\\
# $$
# $$
# \begin{align}
# 0&=\int \frac{d^3k}{2\pi}\frac{\partial}{\partial q}\left[\frac{f(\omega_-)-f(\omega_+)}{E_k}\right]\bigl{|}_{\mu}\\
# &=\int d^3k \left[-\frac{f(\omega_-)-f(\omega_+)}{E_k^2}\frac{\partial E_k}{\partial q} + \frac{\delta(\omega_-)}{E_k}\frac{\partial \omega_-}{\partial q}+ \frac{\delta(\omega_+)}{E_k}\frac{\partial \omega_+}{\partial q} \right]\\
# &=-\int d^3k \frac{f(\omega_-)-f(\omega_+)}{E_k^2}\frac{\partial E_k}{\partial q}\\
# &=\int d^3k \frac{f(\omega_-)-f(\omega_+)}{mE_k^3}q\epsilon_+
# \end{align}
# $$
#
# The condition to have minimum for $C$ is:
# $$\int d^3k \frac{f(\omega_-)-f(\omega_+)}{E_k^3}\epsilon_+=0$$

# In above dervivate, the fact $\omega_+>0$ and $\omega_-<0$, which means $\delta(\omega_+)$ and $\delta(\omega_-)$ are zero and that enables us to simplify above calculation.
# * Be careful, the above statement in not true in gneneral when $\Delta$ is close to zero

from mmf_hfb import tf_completion as tf
reload(tf)
def dC(dmu, dq):
    args = dict(m_a=1, m_b=1, mu_a=mu + dmu, mu_b=mu-dmu, delta=delta, dim=3, hbar=1, T=0, k_c=1000, dq=dq)
    return tf.integrate_q(tf.dC_dq_integrand, **args).n


dqs = np.linspace(0, k0, 20)
dCs = [dC(dmu=6.775, dq=dq) for dq in dqs]

plt.plot(dqs, dCs)

# The above will minimizes $C(q,\mu)$ with minimized value $C_m$, the $P_c$ happens when $C_m$ is zero for a cetern $\mu$

# +
# %pylab inline --no-import-all
from ipywidgets import interact
m = mu = eF = 1.0
pF = np.sqrt(2*m*eF)
delta = 0.5*mu
p_max = 2*pF
p_x = np.linspace(-p_max, p_max, 50)
p_perp = 0

@interact(dq=(-2, 2, 0.1), dmu=(-1.0, 1.0, 0.1), delta=(0, 1, 0.1),q=(0, 2, 0.2))
def plot_regions(q=0, dq=0, dmu=0.4, delta=0.2):
    delta = np.abs(delta)
    mu_a = mu + dmu
    mu_b = mu - dmu
    e_a = ((p_x + q + dq)**2 + p_perp**2)/2/m - mu_a
    e_b = ((p_x + q - dq)**2 + p_perp**2)/2/m - mu_b
    e_p, e_m = (e_a + e_b)/2.0, (e_a-e_b)/2.0
    E = np.sqrt(e_p**2 + delta**2)
    w_p, w_m = e_m + E, e_m - E
    plt.plot(p_x, w_p)
    plt.plot(p_x, w_m)
    plt.axhline(0, linestyle='dashed')
# -


