# -*- coding: utf-8 -*-
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

# # Construct FF States

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import * 
from mmf_hfb import tf_completion as tf
from mmf_hfb.FuldeFerrelState import FFState
import itertools
from mmfutils.plot import imcontourf
tf.MAX_DIVISION = 200
plt.figure(figsize(10,4))
clear_output()

# ## Mathematical Relations:
# \begin{align}
# k_a &= k + q + dq, \qquad k_b = k+q - dq ,\qquad
# \epsilon_a = \frac{\hbar^2}{2m}k_a^2 - \mu_a, \qquad \epsilon_b = \frac{\hbar^2}{2m}k_b^2 - \mu_b,\\
# E&=\sqrt{\epsilon_+^2+\abs{\Delta}^2},\qquad \omega_+= \epsilon_-+E, \qquad \omega_- = \epsilon_- - E\\
# \epsilon_+&= \frac{\hbar^2}{4m}(k_a^2+k_b^2) - \mu_+= \frac{\hbar^2}{2m}\left[(k+q)^2 + dq^2\right] - \mu_+\\
# \epsilon_-&= \frac{\hbar^2}{4m}(k_a^2-k_b^2) - \mu_-=\frac{\hbar^2}{m}(k +q)dq - \mu_-\tag{1}\\
# \end{align}
#

# * For a FF state, the ground state should have zero overall current:
# $$
# n_a (q + \delta q) + n_b(q-\delta q) = 0\tag{2}
# $$
# which means:
# $$
# \frac{n_a}{n_b}=\frac{\delta q - q}{\delta q +q}
# $$
# Let $n_a/n_b=r$, then
# $$
# \frac{\delta q}{q}=\frac{1 + r}{1-r}=\frac{n_a+n_b}{n_b-n_a}
# $$

# ## Stable conditions for $q$ and $\delta q$:
# ### A wrong Derivation:
# $$
# \newcommand{\E}{\mathcal{E}}
# \newcommand{\q}{\delta q}
# \newcommand{\d}{\partial}
# \E^0=\E^0(n_a, n_b) \qquad \frac{\d\E^0}{\d n_a} |_{n_b}=\mu_{a}^0=\E^0_{,a} \qquad \frac{\d\E^0}{\d n_b}|_{n_a}=\mu_b^0=\E^0_{,b}\\
# $$
# \begin{align}
# \E(n_a,n_b,\q)_{q=0}
# =\E^0(n_a,n_b) + n_a \frac{(\q)^2}{2m} + n_b\frac{(-\q)^2}{2m}=\E^0(n_a,n_b) + (n_a+n_b) \frac{\q^2}{2m}\\
# \end{align}
#
# When boost with $v =q/\hbar$
#
# \begin{align}
# \E(n_a,n_b,q,\q)
# =\E^0(n_a,n_b) + n_a \frac{(q+\q)^2}{2m}+ n_b \frac{(q-\q)^2}{2m}\\
# \end{align}

# $$
# n_a=n_a(\mu_a, \q) \qquad n_b=n_b(\mu_b, \q)\\
# $$
#
# $$
# P^0(\mu_a^0,\mu_b^0) = \mu_a^0 n_a + \mu_b^0n_b-\E^0
# $$
#
# $$
# \frac{\d\E}{\d n_a} |_{n_b}=\mu_{a}=\mu_a^0+ \frac{(q+\q)^2}{2m} \\
# \frac{\d\E}{\d n_b} |_{n_a}=\mu_{a}=\mu_b^0+ \frac{(q-\q)^2}{2m} 
# $$
#
# \begin{align}
# P(\mu_a, \mu_b, q,\q) &= \mu_a^0 n_a + \mu_b^0n_b-\E^0=P^0(\mu_a^0,\mu_b^0)\\
# &= \left[\mu_a - \frac{(q+\q)^2}{2m}\right] n_a + \left[\mu_b - \frac{(q-\q)^2}{2m}\right]n_b-\E^0
# \end{align}

# #### Fix $\q$ , $\mu_a$ and $\mu_b$
# * The densities $n_a$, $n_b$ do not depend on $q$, so $\E^0$
# $$
# d\mu_a = \frac{\d \mu_a^0}{\d n_a}dn_a + \frac{\d \mu_b^0}{\d n_b}dn_b + \frac{q+\q}{m}dq\\
# d\mu_b = \frac{\d \mu_b^0}{\d n_a}dn_a + \frac{\d \mu_b^0}{\d n_b}dn_b + \frac{q-\q}{m}dq\\
# $$

# * Find the conditions to maximize the pressure with respect to $q$ and $\q$ with fixed $\mu_a$ and $\mu_b$

# $$
# \frac{\d P}{\d q}|_{\q,\mu_a,\mu_b} = -\frac{q+\q}{m}n_a - \frac{q-\q}{m}n_b=0
# $$
# Means the overall current is zero

# #### Fix $q$, $\mu_a$ and $\mu_b$
# when $\q$ varies, the densities $n_a$, $n_b$ will also changed, then:
# $$
# \frac{\d P}{\d \q}|_{q,\mu_a,\mu_b} 
# = -\frac{q+\q}{m}n_a +  \left[\mu_a - \frac{(q^2+\q^2)}{2m}\right]\frac{\d n_a}{\d \q}  + \frac{q-\q}{m}n_b +  \left[\mu_b - \frac{(q^2+\q^2)}{2m}\right]\frac{\d n_b}{\d \q} - \frac{\d \E^0}{\d n_a}\frac{\d n_a}{\d \q} - \frac{\d \E^0}{\d n_b}\frac{\d n_b}{\d \q}\\
# = -\frac{q+\q}{m}n_a +  \mu^0_a\frac{\d n_a}{\d \q}  + \frac{q-\q}{m}n_b +  \mu^0_b\frac{\d n_b}{\d \q} - \mu^0_a\frac{\d n_a}{\d \q} - \mu^0_b\frac{\d n_b}{\d \q}=0
# $$

# That means $q=\q=0$?

# ### Another Derivation:
# * The energy depends on $n_a$, $n_b$, and $\q$, as the $\q$ will change $n_a$, $n_b$
# $$
# \E^0=\E^0(n_a, n_b,\q) \qquad \frac{\d\E^0}{\d n_a} |_{n_b, \q}=\mu_{a}^0=\E^0_{,a} \qquad \frac{\d\E^0}{\d n_b}|_{n_a,\q}=\mu_b^0=\E^0_{,b}\\
# $$
#
# $$
# n_a=n_a(\mu_a, \q) \qquad n_b=n_b(\mu_b, \q)\\
# $$
#
#
# When boost with $v =q/\hbar$
#
# \begin{align}
# \E(n_a,n_b,q,\q)
# =\E^0(n_a,n_b,\q) + (n_a+n_b) \frac{q^2}{2m}\\
# \end{align}

# \begin{align}
# P^0(\mu_a^0,\mu_b^0) &= \mu_a^0 n_a + \mu_b^0n_b-\E^0 \\
# \frac{\d\E}{\d n_a} |_{n_b, q,\q}&=\mu_{a}=\mu_a^0+ \frac{q^2}{2m} \\
# \frac{\d\E}{\d n_b} |_{n_a,q,\q}&=\mu_{a}=\mu_b^0+ \frac{q^2}{2m} \\
# P(\mu_a, \mu_b, q,\q) &= \mu_a^0 n_a + \mu_b^0n_b-\E^0=P^0(\mu_a^0,\mu_b^0)\\
# &= \left[\mu_a - \frac{q^2}{2m}\right] n_a + \left[\mu_b - \frac{q^2}{2m}\right]n_b-\E^0
# \end{align}

# * Compute the conditions for local Maxima:
# $$
# \frac{\d P}{\d q}|_{\q,\mu_a,\mu_b} = -\frac{q}{m}(n_a + n_b)=0
# $$
#
# Means the overall current is zero
# \begin{align}
# \frac{\d P}{\d \q}|_{q,\mu_a,\mu_b}
# &=\mu_a^0\frac{\d n_a}{\d \q} + \mu_b^0\frac{\d n_b}{\d \q} - \frac{\d \E^0}{\d n_a}\frac{\d n_a}{\d \q}- \frac{\d \E^0}{\d n_b}\frac{\d n_b}{\d \q} + \frac{\d \E^0}{\d \q}\\
# &=\mu_a^0\frac{\d n_a}{\d \q} + \mu_b^0\frac{\d n_b}{\d \q} - \mu_a^0 \frac{\d n_a}{\d \q}- \mu_b^0\frac{\d n_b}{\d \q} + \frac{d \E^0}{d \q}\\
# &=\frac{d \E^0}{d \q} = 0
# \end{align}
# * If the energy does not depend on the $\q$ explicitly, the condition holds automatically?

# ## 1D case
# * $d\mu > \Delta$ to have densities inbalance
# * set $bStateSentinel$ to True so the code will check if a valid solution to $\Delta$ is found.

# ### Sarma Phase

delta = 1
mu = 2 * delta
dmu = 1.2
ff = FFState(mu=mu, dmu=dmu, delta=delta, dim=1, k_c=100,fix_g=True, bStateSentinel=True)

# * Check for densities
# * $q$ and $\delta q$ can not be too large as there would be no solution to delta

# ### Compute the densities

ns = ff.get_densities(mu=mu, dmu=dmu)
na0, nb0 = ns[0].n, ns[1].n
print(na0, nb0)

ff.get_densities(mu=mu, dmu=dmu, q=0.02)

ff.get_densities(mu=mu, dmu=dmu, q=0, dq=0.1)

# ### Fix $d\mu$ and $g$
# * also fix $n_a/n_b$, while changing $\mu$

k = 0.12 # select a scaling factor, start with small number
r=na0/nb0
print(r)

q, dq = (nb0-na0)*k, (na0+nb0)*k
mus = np.linspace(-0.2,0.5,40) * 10 + mu
ss = [ff.check_superfluidity(mu=mu, dmu=dmu, q=q, dq=dq) for mu in mus]
plt.plot(mus,ss)

na0 * (q+dq) + nb0*(q-dq)

mus = np.linspace(0.7,2.5,20) 
ns = [ ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq) for mu in mus]

rs = [n[0].n/n[1].n for n in ns]
plt.axhline(r)
plt.axvline(mu)
plt.plot(mus, rs,'--', c='r')

# ### Solve the $\mu$ with the same density ratio

# +
from scipy.optimize import brentq

def f_mu(x):
    ns = ff.get_densities(mu=x, dmu=dmu, q=q, dq=dq)
    return ns[0].n/ns[1].n - r
mu1 = brentq(f_mu, 2, 2.5)
mu1
# -

# ### Check the current, should be zero

j=ff.get_current(mu=mu1, dmu=dmu, q=q, dq=dq).n
print(na0, nb0, j)
assert na0 != nb0
assert np.allclose(j, 0)

# ### Check Pressure
# * We just construct a FF ground state, now check the pressure

ff.get_pressure(mu=mu, dmu=dmu, q=q,dq=dq),ff.get_pressure(mu=mu, dmu=dmu, q=q,dq=dq, delta=0)

# * However, the normal state win in term of pressure
# * but we have a method to construct FF State, we can move to create a phase diagram

# ### Chek local pressure
# * We also need to check if that state is local maximum in press

qs = np.linspace(-1,1, 20) *0.1 * dq + dq
ps = [ff.get_pressure(mu=mu, dmu=dmu, q=q,dq=dq).n for dq in qs]
pn = [ff.get_pressure(mu=mu, dmu=dmu, q=q,dq=dq, delta=0).n for dq in qs]

plt.plot(qs, ps)
plt.plot(qs, pn)


# ## Check the range of  𝑑𝜇  that yields a solution to $\Delta$

dmus = np.linspace(-1,1,30) * 0.1 * dmu + dmu
states = [ff.check_superfluidity(mu=mu, dmu=dmu) for dmu in dmus]
plt.plot(dmus, states)
plt.axvline(dmu)

# ## Check how $g$ changes with $d\mu$

# $$
# \frac{1}{g} = -\frac{1}{2}\int \frac{d{k}}{2\pi}\frac{1}{E_k}\bigl(f(\omega_-)-f(\omega_+)\bigr).
# $$
# * So $g$ would not change until dmu exceeds the gap, where the occupancy change
# * Does not look exact

dmus = np.linspace(-1.2 * delta, 1.2 * delta, 40)
gs = [ff.get_g(mu=mu * 2, dmu=dmu, delta=delta, k_c=200) for dmu in dmus]
plt.plot(dmus, gs,'--')
plt.axvline(delta)
plt.axvline(-delta)

dmus = np.linspace(-1.2 * delta, 1.2 * delta, 40)
gs = [ff.get_g(mu=mu * 5, dmu=dmu, delta=delta, k_c=200) for dmu in dmus]
plt.plot(dmus, gs,'--')
plt.axvline(delta)
plt.axvline(-delta)

dmus = np.linspace(-1.2 * delta, 1.2 * delta, 40)
gs = [ff.get_g(mu=mu * 2, dmu=dmu, delta=0.5* delta, k_c=200) for dmu in dmus]
plt.plot(dmus, gs,'--')
plt.axvline(delta)
plt.axvline(-delta)

# ## Fixed the ratio of $q$ and $\delta q$
# * Change the $q$ and $\delta q$ with fix $\frac{q}{\delta q}$
# * Issue: can't find a solution with zero net current

dmu = 1.5
na, nb = ff.get_densities(mu=mu, dmu=dmu)
na, nb=na.n, nb.n
print(na, nb)
scale = 0.01
ks = np.linspace(-scale,scale, 40)
r = na/nb
dns = []
js = []
for k in ks:
    q, dq = (1 + r)*k, (1 - r)*k
    na_, nb_ = ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
    na_, nb_ = na_.n, nb_.n
    dn = np.sqrt((na - na_)**2 + (nb - nb_)**2)
    j = na_ * (q + dq) + nb_ * (q-dq)
    dns.append(dn)
    js.append(j)
plt.plot(ks, dns)
plt.plot(ks, js)
plt.axvline(0)

# ## Use Minimize Routine
# * Issue: different initial guess would yield different results

import scipy.optimize as optimize
ff = FFState(mu=mu, dmu=dmu, delta=delta,dim=1, k_c=200,fix_g=True)
def fun(para):
    q, dq=para
    return -ff.get_pressure(mu=mu, dmu=dmu, q=q, dq=dq).n


initial_guess = (0.5,.8)
result = optimize.minimize(fun, initial_guess)
if result.success:
    fitted_params = result.x
    print(fitted_params)
else:
    raise ValueError(result.message)

# ## Plots of $f_a$, $f_b$, $f_\nu$

# +
# %pylab inline --no-import-all
from ipywidgets import interact

def f(E, T):
    """Fermi distribution function"""
    T = max(T, 1e-12)
    return 1./(1+np.exp(E/T))


@interact(delta=(0, 1, 0.1), 
          mu_eF=(0, 2, 0.1),
          dmu=(-0.4, 0.4, 0.01),
          q=(-0.4, 0.4, 0.01),
          dq=(-0.4, 0.4, 0.01),
          T=(0, 0.1, 0.01))
def go(delta=0.1, mu_eF=1.0, dmu=0.0, q=0, dq=0, T=0.02):
    k = np.linspace(-2, 2, 100)
    hbar = m = kF = 1.0
    eF = (hbar*kF)**2/2/m
    mu = mu_eF*eF
    #dmu = dmu_delta*delta
    mu_a, mu_b = mu + dmu, mu - dmu
    e_a, e_b = (hbar*(k+q+dq))**2/2/m - mu_a, (hbar*(k+q-dq))**2/2/m - mu_b
    e_p, e_m = (e_a + e_b)/2, (e_a - e_b)/2
    E = np.sqrt(e_p**2+abs(delta)**2)
    w_p, w_m = e_m + E, e_m - E
    
    # Occupation numbers
    f_p = 1 - e_p/E*(f(w_m, T) - f(w_p, T))
    f_m = f(w_p, T) - f(-w_m, T)
    f_a, f_b = (f_p+f_m)/2, (f_p-f_m)/2
    f_nu = f(w_m, T) - f(w_p, T)
    plt.subplot(211);plt.grid()
    plt.plot(k/kF, f_a, label='a')
    plt.plot(k/kF, f_b, label='b')
    plt.plot(k/kF, f_nu, label=r'$\nu$');plt.legend()
    plt.ylabel('n')
    plt.subplot(212);plt.grid()
    plt.plot(k/kF, w_p/eF, k/kF, w_m/eF)
    plt.xlabel('$k/k_F$')
    plt.ylabel(r'$\omega_{\pm}/\epsilon_F$')
    plt.axhline(0, c='y')


# -

# ## Brutal Method

def case_1d(mu_delta=5, dmu_delta=3, q_delta = 0.5, dq_q=1, dq1_q=-3, dq2_q=3, delta=1, dim = 1, k_c=200):
    plt.figure(figsize(10,5))
    mu = mu_delta * delta
    dmu = dmu_delta * delta
    q = q_delta * delta
    dqs = np.linspace(dq1_q, dq2_q, 40) * q
    ff = FFState(mu=mu, dmu=dmu, delta=delta,q=q, dq=dq_q * q, dim=dim, k_c=k_c,fix_g=True)
    states =[ff.check_superfluidity(mu=mu, dmu=dmu, q=q, dq=dq) for dq in dqs]
    ps =[ff.get_pressure(mu=mu, dmu=dmu, q=q, dq=dq).n for dq in dqs]
    plt.subplot(211)
    plt.plot(dqs, states,'-', label='States')
    plt.legend()
    plt.title(f'$\mu=${mu},d$\mu$={dmu}, $\Delta$={delta}, q={q}', fontsize=16)
    plt.subplot(212)
    plt.plot(dqs, ps, 'o', label='SF Pressure')
    ps =[ff.get_pressure(mu=mu, dmu=dmu, q=q, delta=0, dq=dq).n for dq in dqs] # normal states
    plt.plot(dqs, ps,'--', label='NS Pressure')
    plt.axvline(q)  
    plt.xlabel(f'$\delta q$', fontsize=16)
    plt.legend()
    na, nb = ff.get_densities(mu=mu, dmu=dmu, q=q, dq=1.4)
    return (na.n, nb.n)


na,nb=case_1d(mu_delta=2, dmu_delta=0.8, q_delta = .55, dq_q=1)


