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

# # Construct FF States

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import * 
from mmf_hfb import tf_completion as tf
from mmf_hfb.FuldeFerrellState import FFState
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

# ### Plots from external data

# E:\Physics\quantum-fermion-theories\mmf-hfb\Docs\..\mmf_hfb\data(BdG)\FFState_J_P_(3d_0.20_5.91_0.14)2019_06_08_22_30_20.json
# E:\Physics\quantum-fermion-theories\mmf-hfb\Docs\..\mmf_hfb\data(BdG)\FFState_J_P_(3d_0.20_5.91_0.15)2019_06_08_22_32_21.json
# E:\Physics\quantum-fermion-theories\mmf-hfb\Docs\..\mmf_hfb\data(BdG)\FFState_J_P_(3d_0.20_5.91_0.15)2019_06_08_22_32_25.json
# E:\Physics\quantum-fermion-theories\mmf-hfb\Docs\..\mmf_hfb\data(BdG)\FFState_J_P_(3d_0.50_5.91_0.36)2019_06_08_00_49_31.json
# E:\Physics\quantum-fermion-theories\mmf-hfb\Docs\..\mmf_hfb\data(BdG)\FFState_J_P_(3d_0.75_5.91_0.54)2019_06_08_01_18_56.json
# E:\Physics\quantum-fermion-theories\mmf-hfb\Docs\..\mmf_hfb\data(BdG)\FFState_J_P_(3d_1.00_5.91_0.73)2019_06_08_01_44_25.json
# E:\Physics\quantum-fermion-theories\mmf-hfb\Docs\..\mmf_hfb\data(BdG)\FFState_J_P_(3d_1.25_5.91_0.91)2019_06_08_02_06_56.json
# E:\Physics\quantum-fermion-theories\mmf-hfb\Docs\..\mmf_hfb\data(BdG)\FFState_J_P_(3d_1.50_5.91_1.09)2019_06_08_02_27_12.json

import os
import inspect
from os.path import join
def filter(mu, dmu, delta, g, dim):
    if dim != 3:
        return True
    #return False
    #if g != -2.8:
    #    return True
    #return False
    #if g != -3.2:
    #    return True
    if delta != 1.5:
        return True
    if not np.allclose(dmu,1.09,rtol=1e-2):
        return True
    return False
currentdir = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),"..","mmf_hfb","data(BdG)")


plt.figure(figsize(12,12))
bp.PlotStates(filter_fun=filter)

# ## Compute Current and Pressure
# $$
# J=\int dk\left[(k+\delta q) f_a + (k - \delta q) f_b\right]
# $$

plt.figure(figsize(16, 16))
bp.PlotCurrentPressure(filter_fun=filter, alignLowerBranches=False, alignUpperBranches=False, showLegend=True)

# $$
# \frac{1}{g}=\frac{m}{4\pi\hbar^2 a}\\
# \delta\mu, \Delta
# $$

lastStates=None
output, fileSet=None, None


def filter1(mu, dmu, delta, g, dim):
    if dim != 3:
        return True
    #return False
    #if g != -2.8:
    #    return True
    #return False
    #if g != -3.2:
    #    return True
    #if delta != 0.5:
    #    return True
    #if dmu != 0.55:
    #    return True
    return False
if output is not None:
    lastStates=(output, fileSet)
output, fileSet = FFStateHelper.label_states(filter1,currentdir=currentdir, lastStates=lastStates, verbosity=False)
clear_output()

xs = []
ys = []
xs2 = []
ys2 = []
states = []
k_F = (2 * 0.59060550703283853378393810185221521748413488992993 * 10)**(0.5)
for dic in output:  
    #if dic['delta'] !=0.5:
    #    continue
    n = dic['na'] + dic['nb']#dic['np']#
    dn = dic['na'] - dic['nb']
    ai = dic['ai']
    g= dic['g']
    #xs.append(-1/(g*n**(1.0/3)))
    #xs.append(-ai/(n**(1.0/3)))
    xs.append(-ai/k_F)
    #ys.append(dic['dmu']/dic['delta'])
    ys.append(dn/n)
    xs2.append(dic['delta'])
    ys2.append(dic['dmu'])
    states.append(dic['state'])
N=len(xs)
colors = []
area = []
for i in range(len(states)):
    s=states[i]
    if s:
        #print(xs[i],ys[i])
        #print(output[i])
        colors.append('red')        
        area.append(15)
    else:
        colors.append('blue')
        area.append(1)
plt.figure(figsize(16,7))
plt.subplot(121)
plt.scatter(xs, ys,  s=area, c=colors)
#plt.ylabel(r"$\delta \mu/\Delta_{\delta \mu=0}$")
plt.ylabel(r"$\delta n/n_{\delta \mu=0}$")
#plt.xlabel(r"$-1/gn^{1/3}$")
plt.xlabel(r"$-1/ak_F$")
#plt.ylim(0.5,0.8)
#plt.xlim(-0.1,0.2)
plt.subplot(122)
plt.scatter(xs2, ys2,  s=area, c=colors)
plt.ylabel(r"$\delta \mu$")
plt.xlabel(r"$\Delta$")
#plt.ylim(0.,1.5)
plt.axhline(1.075, linestyle='dashed')
#plt.xlim(0,4)
plt.ylim(1,1.2)
plt.axvline(1.2)
plt.axhline(.25)

for ret in output:
    if ret is not None and ret['state']:
        print(ret['file'])

# ## Check range of $\Delta$
# ### 3D case
# * To compare with the phase diagram in [Leo 2010]
# [Leo 2010]: http://iopscience.iop.org/article/10.1088/0034-4885/73/7/076501/meta 'Imbalanced Feshbach-resonant Fermi gases'

e_F = 10
mu0 = 0.59060550703283853378393810185221521748413488992993 * e_F
delta0 = 0.68640205206984016444108204356564421137062514068346 * e_F
mu = mu0 # ~6
delta = 1.5
dmu = 1.0875000000000001
ff = FFStateFinder(delta=delta, dim=3, mu=mu, dmu=dmu,  k_c=50)
dqs = np.linspace(0.3,0.6, 20)
plt.figure(figsize(8,4))
ds = np.linspace(0.001, .5, 10)
gss=[]
for d in ds:
    gs = [ff._gc(mu=mu, dmu=dmu, delta=d, dq=dq) for dq in dqs]
    gss.append(gs)
    plt.plot(dqs, gs, label=f"{d}")
plt.axhline(0, linestyle='dashed')
#plt.ylim(0, -0.002)

for d in ds:
    gss.append(gs)
    plt.plot(dqs, gs, label=f"{d}")
plt.axhline(0, linestyle='dashed')


def f(dq):
    return ff._gc(mu=mu, dmu=dmu, delta=delta, dq=dq)
dq = brentq(f,0,0.02)
print(dq)
ff.get_densities(mus_eff=(mu,dmu), dq=dq)

# # How g changes with $\Delta$ and $\mu$

plt.figure(figsize(16,8))
dmus = np.linspace(0, mu, 10)
ds = np.linspace(0.0001, mu * 1.3, 200)
gss = []
for dmu in dmus:
    gs = [ff.ff.get_g(mu=mu, dmu=dmu, delta=d) for d in ds]
    gss.append(gs)

for gs in gss:
    plt.plot(ds, gs, label=f"$\delta \mu$={dmu}")
    plt.axvline(dmu)
plt.legend()
plt.xlabel(f"$\Delta$")
plt.ylabel("g")

plt.figure(figsize(16,8))
dmu = 1.6
dqs = np.linspace(0, 1, 10)
gss1 = []
ds = np.linspace(0.0001, mu * 1.3, 200)
for dq in dqs:
    gs = [ff.ff.get_g(mu=mu, dmu=dmu, delta=d, dq=dq) for d in ds]
    gss1.append(gs)

# +
for gs in gss1:
    plt.plot(ds, gs, label=f"$\delta \mu$={dmu}")
    plt.axvline(dmu)
    plt.axvline(dq, color='r', linestyle='dashed')

plt.legend()
plt.xlabel(f"$\Delta$")
plt.ylabel("g")
# -

# ## Revisit the unitary case
# * In [Leo 2010] scattering length is used in x axis
# * $$\tilde{C}=\frac{m}{4\pi\hbar^2 a}=\frac{1}{g}+\Lambda$$
# * To make it's more straightforward comparision, we fix $\tilde{C}$ here

e_F = 1
mu0 = 0.59060550703283853378393810185221521748413488992993 * e_F
delta0 = 0.68640205206984016444108204356564421137062514068346 * e_F
mu = mu0 # ~6
delta = delta0
dmu = 0
ff = FFStateFinder(delta=delta, dim=3, mu=mu, dmu=dmu,  k_c=50)
ks = np.linspace(0, 100, 100)
ais = [ff.ff.get_a_inv(mu=mu, delta=delta, dmu=0, k_c=k).n for k in ks]
plt.plot(ks, ais)

mu=10
dmu=0
delta=2
g= None#-0.3526852951505492
k_c=50
dq = 0.04810320463378813
ff = FFState(mu=mu, dmu=dmu, delta=delta, dim=3, g=g, k_c=k_c, fix_g=True)
print(f"g={ff._g}")
mu_eff, dmu_eff = mu, dmu
n_a, n_b = ff.get_densities(mu=mu_eff, dmu=dmu_eff, delta=delta)
print(f"n_a={n_a.n}, n_b={n_b.n}, P={(n_a - n_b).n/(n_a + n_b).n}")
ds = np.linspace(0.001,4, 40)
ps = [ff.get_pressure(mu=mu, dmu=dmu, mu_eff=mu_eff, dmu_eff=dmu_eff, delta=d, dq=dq, use_kappa=False).n for d in ds]
delta_max = ff.solve(mu=mu, dmu=dmu, dq=dq)
print(f"Delta={delta_max}")
plt.plot(ds,ps)
plt.axvline(delta_max)

m = hbar =  1.0
dq = 0.8041434967141077    
mu=10
delta=2.4
dmu = delta * 1.2
print(mu, delta) 
ff = FFState(delta=delta, dim=3, mu=mu, dmu=delta, fix_g=True, g=-0.3682966327988467, k_c=100)
na, nb = ff.get_densities(mu=mu, dmu= dmu, delta=delta, dq=dq)
P_ff = ff.get_pressure(mu=mu, dmu=dmu, delta= 1.3323919597989948, dq=dq).n
P_ns = ff.get_pressure(mu=mu, dmu=dmu, delta=0, dq=dq).n
print(f"Normal State Pressure:{P_ns}, FF State Presure:{P_ff}")
P = (na - nb).n/(na+nb).n
print(f"n_a={na.n}, n_b={nb.n}, P={P}")
a = m / 4/ np.pi/hbar**2 * ff._g
x = - 1/ kF/a
print(f"x={x}, y={P}")
if ff.fix_g:
    print(f"g={ff._g}")
else:
    print(ff._C)
if (P_ff > P_ns):
    print("This is a FF State")

mu=10
dmu= 0.27
delta=0.2
dq=0.02630155299196228
dim = 1
ff = FFState(mu=mu, dmu=dmu, delta=delta, dim=dim, fix_g=True, bStateSentinel=True)
# compute n_a, n_b, energy density and pressure in single function
n_a, n_b, e, p, mus_eff = ff.get_ns_p_e_mus_1d(mu=mu, dmu=dmu, dq=dq, update_g=True)
# or compute effective mus first
mu_eff, dmu_eff = ff._get_effective_mus(mu=mu, dmu=dmu, dq=dq, update_g=False)
n_a, n_b = ff.get_densities(mu=mu_eff, dmu=dmu_eff,  dq=dq)
j_a, j_b, j_p, j_m = ff.get_current(mu=mu_eff, dmu=dmu_eff, delta=0, dq=dq)
print(f"n_a={n_a.n}, n_b={n_b.n}, j_a={j_a.n}, j_b={j_b.n}, j_p={j_p.n}, j_m={j_m.n}")
# re-compute the effective mus as for normal state, delta=dq=0
mu_eff, dmu_eff = ff._get_effective_mus(mu=mu, dmu=dmu, delta=0, mus_eff=(mu_eff, dmu_eff), update_g=False)
p0 = ff.get_pressure(mu=mu, dmu=dmu, mu_eff=mu_eff, dmu_eff=dmu_eff, delta=0)
print(f"FF State Pressure={p}, Normal State Pressue={p0}")
if p0 < p:
    print("The ground state is a FF State")

# ## Mathematical Relations:
# \begin{align}
# k_a &= k + q + dq, \qquad k_b = k+q - dq ,\qquad
# \epsilon_a = \frac{\hbar^2}{2m}k_a^2 - \mu_a, \qquad \epsilon_b = \frac{\hbar^2}{2m}k_b^2 - \mu_b,\\
# E&=\sqrt{\epsilon_+^2+\abs{\Delta}^2},\qquad \omega_+= \epsilon_-+E, \qquad \omega_- = \epsilon_- - E\\
# \epsilon_+&= \frac{\hbar^2}{4m}(k_a^2+k_b^2) - \mu_+= \frac{\hbar^2}{2m}\left[(k+q)^2 + dq^2\right] - \mu_+\\
# \epsilon_-&= \frac{\hbar^2}{4m}(k_a^2-k_b^2) - \mu_-=\frac{\hbar^2}{m}(k +q)dq - \mu_-\tag{1}\\
# \end{align}

# ### Thermodynamics Derivation:
# * The energy depends on $n_a$, $n_b$, and $\q$, as the $\q$ will change $n_a$, $n_b$
# $$
# \E^0=\E^0(n_a, n_b,\q) \qquad \frac{\d\E^0}{\d n_a} |_{n_b, \q}=\mu_{a}^0=\E^0_{,a} \qquad \frac{\d\E^0}{\d n_b}|_{n_a,\q}=\mu_b^0=\E^0_{,b}\\
# $$
#
# $$
# n_a=n_a(\mu_a,\mu_b,\q) \qquad n_b=n_b(\mu_a, \mu_b, \q)\\
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
# * Means the overall current is zero( $q=0$)
#
# \begin{align}
# \frac{\d P}{\d \q}|_{q,\mu_a,\mu_b}
# &=\mu_a^0\frac{\d n_a}{\d \q} + \mu_b^0\frac{\d n_b}{\d \q} - \frac{\d \E^0}{\d n_a}\frac{\d n_a}{\d \q}- \frac{\d \E^0}{\d n_b}\frac{\d n_b}{\d \q} + \frac{\d \E^0}{\d \q}\\
# &=\mu_a^0\frac{\d n_a}{\d \q} + \mu_b^0\frac{\d n_b}{\d \q} - \mu_a^0 \frac{\d n_a}{\d \q}- \mu_b^0\frac{\d n_b}{\d \q} + \frac{\d \E^0}{\d \q}\\
# &=\frac{\d \E^0}{\d \q} = 0
# \end{align}
# * If the energy does not depend on the $\q$ explicitly, the condition holds automatically? But this not is true, as $\E^0$ is a function of $\q$

ff = FFState(mu=3, dmu=1, delta=1, dim=3, k_c=1000,fix_g=True, bStateSentinel=False)
kcs = np.linspace(100, 5000,30)
gs = [ff.get_g(delta=1, k_c = kc) for kc in kcs]
plt.plot(kcs, gs)

# ## Check how $g$ changes with $d\mu$

# $$
# \frac{1}{g} = -\frac{1}{2}\int \frac{d{k}}{2\pi}\frac{1}{E_k}\bigl(f(\omega_-)-f(\omega_+)\bigr).
# $$
# * So $g$ would not change until dmu exceeds the gap, where the occupancy change
# * Does not look exact

# ## Plots of $f_a$, $f_b$, $f_\nu$

# +
# %pylab inline --no-import-all
from ipywidgets import interact
def f(E, T):
    """Fermi distribution function"""
    T = max(T, 1e-12)
    return 1./(1+np.exp(E/T))


@interact(delta=(0, 1, 0.1), 
          mu=(0, 20, 1),
          dmu=(-1, 1, 0.05),
          dq=(-0.4, 0.4, 0.01),
          T=(0, 0.1, 0.01))
def go(delta=0.2, mu=10, dmu=0.2,dq=0, T=0.02):
    plt.figure(figsize=(15,10))
    q=0 
    k = np.linspace(0, 10, 500)
    hbar = m = kF = 1.0
    eF = (hbar*kF)**2/2/m
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
    plt.plot(k, f_a, label='a')
    plt.plot(k, f_b, label='b')
    plt.plot(k, f_p, '--',label='p')
    plt.plot(k, f_m, label='m')
    plt.legend()
    plt.ylabel('n')
    plt.subplot(212);plt.grid()
    plt.plot(k, w_p, k, w_m)
    plt.xlabel('$k/k_F$')
    plt.ylabel(r'$\omega_{\pm}/\epsilon_F$')
    plt.axhline(0, c='y')

# -


