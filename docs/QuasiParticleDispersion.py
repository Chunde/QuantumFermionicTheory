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

# # Quasi Particle Dispersion

import mmf_setup;mmf_setup.nbinit()

# $$
#   \omega_{\pm} = \epsilon_- \pm E, \qquad
#   E = \sqrt{\epsilon_+^2+\abs{\Delta}^2},\\
#   \epsilon_{\pm} = \frac{\epsilon_a \pm \epsilon_b}{2}, \qquad
#   \epsilon_{a, b} = \frac{(p_x \pm q)^2 }{2m} - \mu_{a,b}.
# $$
#
# Simplifying, we have:
#
# $$
#   \epsilon_{-} = \frac{qp_x}{m} - \mu_{-}, \qquad
#   \epsilon_{+} = \frac{p_x^2}{2m} - \Bigl(\overbrace{\mu_{+} - \frac{q^2}{2m}}^{\mu_q}\Bigr), 
#   \qquad
#   \mu_{\pm} = \frac{\mu_{a} \pm \mu_{b}}{2}.
#   \qquad
#    \mu_q = \mu - \frac{dq^2}{2m}
# $$

# +
# %pylab inline --no-import-all
from ipywidgets import interact
def f(E, T):
    """Fermi distribution function"""
    T = max(T, 1e-32)
    return 1./(1+np.exp(E/T))


@interact(delta=(0, 1, 0.1), 
          mu_eF=(0, 2, 0.1),
          dmu=(-0.4, 0.4, 0.01),
          T=(0, 0.1, 0.01),
          dq=(0, 1, 0.1)
         )
def go(delta=0.1, mu_eF=1.0, dmu=0.0, dq=0, T=0.02):
    plt.figure(figsize=(12, 8))

    k = np.linspace(0, 1.4, 1000)
    hbar = m = kF = 1.0
    eF = (hbar*kF)**2/2/m
    mu = mu_eF*eF
    #dmu = dmu_delta*delta
    mu_a, mu_b = mu + dmu, mu - dmu
    e_a, e_b = (hbar*k+dq)**2/2/m - mu_a, (hbar*k-dq)**2/2/m - mu_b
    e_p, e_m = (e_a + e_b)/2, (e_a - e_b)/2
    E = np.sqrt(e_p**2+abs(delta)**2)
    w_p, w_m = e_m + E, e_m - E
    
    # Occupation numbers
    f_p = 1 - e_p/E*(f(w_m, T) - f(w_p, T))
    f_m = f(w_p, T) - f(-w_m, T)
    f_a, f_b = (f_p+f_m)/2, (f_p-f_m)/2

    plt.subplot(211);plt.grid()
    plt.plot(k/kF, f_a, label='a')
    plt.plot(k/kF, f_b, '--', label='b',);plt.legend()
    plt.ylabel('n')
    plt.subplot(212);plt.grid()
    plt.plot(k/kF, w_p/eF, k/kF, w_m/eF)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('$k/k_F$')
    plt.ylabel(r'$\omega_{\pm}/\epsilon_F$')
    plt.axhline(0, c='y')


# -

from sympy import *
init_printing(use_latex='mathjax')

k=symbols('k')
q = symbols('q')
mu = symbols('mu_+')
dmu=symbols('mu_-')
mu_a = (mu+dmu)/2
mu_b = (mu-dmu)/2
delta = symbols('Delta')
e_a = (k+q)**2/2 - mu_a
e_b = (k-q)**2/2 - mu_b
e_p = (e_a + e_b)/2
e_m = (e_a - e_b)/2
E=sqrt(e_p**2+delta**2)
w_p=e_m + E
w_m=e_m - E
dw_p =diff(w_p, k)
dw_m= diff(w_m, k)



e_p, e_m

simplify(w_p)

simplify(w_p.evalf())

simplify(dw_p)

simplify(dw_m)

# # Bose and Fermi Statistics

# +
import numpy as np
mu = 1

def fb(e, T):
    return 1/(np.exp((e-mu)/T)-1)

def ff(e, T):
    return 1/(np.exp((e-mu)/T)+1)

Ts = np.linspace(0.0001, 2*mu, 3)
es = np.linspace(0, 2*mu, 50)
plt.figure(figsize=(16,5))
for T in Ts:
    fbs = [fb(e, T) for e in es]
    ffs = [ff(e, T) for e in es]
    #plt.plot(es, fbs)
    plt.subplot(121)
    plt.plot(es, ffs, '--')
    plt.subplot(122)
    plt.plot(es, fbs)
# -
# # Pressure vs $\delta \mu$


# +
import os
import sys
import inspect
from os.path import join
import warnings
from IPython.display import clear_output, display

warnings.filterwarnings("ignore")
currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, join(currentdir, '..','Projects'
                        ,'FuldeFerrellState'))
from fulde_ferrell_state import FFState
# -

mu = 10
dmu = 0
delta = 5
ff = FFState(dim=2, mu=mu, dmu=dmu, delta=delta, k_c=50)

dmus = np.linspace(0, 1.5*delta, 751)
pps = [ff.get_pressure(mu=mu, dmu=dmu, delta=delta).n for dmu in dmus]
pns = [ff.get_pressure(mu=mu, dmu=dmu, delta=0).n for dmu in dmus]
clear_output()

matplotlib.rcParams.update({'font.size': 18})
plt.figure(figsize=(16, 8))
plt.plot(dmus/mu, np.array(pps)/mu, '-', label="superfluid pressure")
plt.plot(dmus/mu, np.array(pns)/mu, label="normalfluid pressure")
plt.axvline(delta/mu, ls='dashed')
plt.xlabel(r"$\delta \mu /\mu$")
plt.ylabel(r"$P/\mu$")
plt.legend()
plt.savefig("clogston_limit_pressure.pdf", bbox_inches='tight')


