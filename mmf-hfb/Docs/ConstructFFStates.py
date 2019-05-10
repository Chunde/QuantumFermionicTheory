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
from mmf_hfb.FFStateFinder import FFStateFinder
from scipy.optimize import brentq
from mmfutils.plot import imcontourf
clear_output()

# # 1D Phase Diagram

# * Pick $\mu$, $\delta \mu$, then vary $\delta q$ to have maxized pressure
# * Fix $g_c$ at very beginning, so the system has a constant
# * For any $\delta q$, make sure there is a solution to the gas equation
#     * if we change the $\delta q$, we need to update the effective $\mu$, $\delta \mu$

# ## Looking for FFStates
# * before compareing pressure, we should know all possible FF states for a given $\mu$, $\delta \mu$, and $\Delta$
# * here were found and plot all FF States
# * to determine if any of the state is a ground state, we need to find a max pressure.

# ### One FF Ground State
# * Here is a ground FF state in 1d, these numbers are from the plot in this section.
# * It's meta-state because it's slightly smaller than a nomral state pressure with the same $\mu$, $\delta \mu$
# * it's interesting to find that to fix g in different way will change the result a lot,
#     * if fix the g with dq, we get g=-3.077012439639267
#     * However, if fix g without dq(g=-2.7137980772945562), we get something else that will not be a ground FF State.
#     * So, it's better to fix g instead of picking a $\Delta$
# * Normal Pressure=26.800450217524567
# * FFState Pressue=26.801008398851206

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
p0 = ff.get_pressure(mu=mu, dmu=dmu, mu_eff=mu_eff, dmu_eff=dmu_eff, delta=0).n
print(f"FF State Pressure={p}, Normal State Pressue={p0}")
if p0 < p:
    print("The ground state is a FF State")


# ### Plots from external data

def filter(mu, dmu, delta, g, dim):
    if dim != 3:
        return True
    #return False
    #if g != -2.88:
    #    return True
    #return False
    #if g !=-2.6:
    #    return True
    if delta != 2.4:
        return True
    if dmu != 2.45:
        return True
    return False
currentdir = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),"..","mmf_hfb","data")


import os
import inspect
from os.path import join
import json
import glob
from json import dumps
pattern = join(currentdir,"FFState_[()d_0-9]*.json")
files = glob.glob(pattern)
plt.figure(figsize=(20,20))
style =['o','+','+']
gs = set()
for file in files:
    if os.path.exists(file):
        with open(file,'r') as rf:
            ret = json.load(rf)
            dim, mu, dmu, delta, g=ret['dim'], ret['mu'], ret['dmu'], ret['delta'], ret['g']
            gs.add(g)
            if filter(mu=mu, dmu=dmu, delta=delta, g=g, dim=dim):
                    continue
            print(file)
            datas = ret['data']
            dqs1, dqs2, ds1, ds2 = [],[],[],[]
            for data in datas:
                dq1, dq2, d = data
                if dq1 is not None:
                    dqs1.append(dq1)
                    ds1.append(d)
                if dq2 is not None:
                    dqs2.append(dq2)
                    ds2.append(d)
            plt.subplot(211)
            if len(ds1) < len(ds2):
                if len(ds1) > 0:
                    plt.plot(ds1, dqs1, style[dim-1], label=f"$\Delta=${delta}, $\mu$={mu}, $d\mu=${dmu}, g={g}")
            else:
                if len(ds2) > 0:
                    plt.plot(ds2, dqs2, style[dim-1], label=f"$\Delta=${delta}, $\mu$={mu}, $d\mu=${dmu}, g={g}")
            plt.subplot(212)
            if len(ds1) < len(ds2):
                if len(ds2) > 0:
                    plt.plot(ds2, dqs2, style[dim-1], label=f"$\Delta=${delta}, $\mu$={mu}, $d\mu=${dmu}, g={g}")
            else:
                if len(ds1)> 0:
                    plt.plot(ds1, dqs1, style[dim-1], label=f"$\Delta=${delta}, $\mu$={mu}, $d\mu=${dmu}, g={g}")
            #break
print(gs)    
plt.subplot(211)
plt.xlabel(f"$\Delta$")
plt.ylabel(f"$\delta q$")
plt.title(f"Lower Branch")
plt.legend()
plt.subplot(212)
plt.xlabel(f"$\Delta$")
plt.ylabel(f"$\delta q$")
plt.title(f"Upper Branch")
plt.legend()

# ## Compute Current and Pressure
# $$
# J=\int dk\left[(k+\delta q) f_a + (k - \delta q) f_b\right]
# $$

import os
import inspect
from os.path import join
import json
import glob
import operator
import warnings
warnings.filterwarnings("ignore")
from json import dumps
def PlotCurrentPressure(alignLowerBranches=True, alignUpperBranches=True, showLegend=False):
    pattern = join(currentdir, "FFState_J_P[()d_0-9]*")
    files=glob.glob(pattern)
    plt.figure(figsize=(20,20))
    gs = set()
    for file in files[0:]:
        if os.path.exists(file):
            with open(file,'r') as rf:
                ret = json.load(rf)
                dim, mu, dmu, delta, g=ret['dim'], ret['mu'], ret['dmu'], ret['delta'],ret['g']
                if filter(mu=mu, dmu=dmu, delta=delta, g=g, dim=dim):
                    continue                
                print(f"g={g}")
                k_c = None
                if 'k_c' in ret:
                    k_c = ret['k_c']
                    print(f"k_c={k_c}")
                gs.add(g)
                print(file)    
                ff = FFState(mu=mu, dmu=dmu, delta=delta, dim=dim, g=g, k_c=k_c, fix_g=True)
                mu_eff, dmu_eff = ff._get_effective_mus(mu=mu, dmu=dmu)
                n_a, n_b = ff.get_densities(mu=mu_eff, dmu=dmu_eff)
                print(f"n_a={n_a.n}, n_b={n_b.n}, P={(n_a - n_b).n/(n_a + n_b).n}")
                mu_eff, dmu_eff = ff._get_effective_mus(mu=mu, dmu=dmu, delta=0, mus_eff=(mu_eff, dmu_eff), update_g=False)
                p0 = ff.get_pressure(mu=mu, dmu=dmu, mu_eff=mu_eff, dmu_eff=dmu_eff, delta=0).n
                data1, data2 = ret['data']
                del data2[-1]

                dqs1, dqs2, ds1, ds2, j1, j2, ja1, ja2, jb1, jb2, P1, P2 = [],[],[],[],[],[],[],[],[],[],[],[]
                for data in data1:
                    d, q, p, j, j_a, j_b = data['d'],data['q'],data['p'],data['j'],data['ja'],data['jb']
                    ds1.append(d)
                    dqs1.append(q)
                    j1.append(j)
                    ja1.append(j_a)
                    jb1.append(j_b)
                    P1.append(p)
                for data in data2:
                    d, q, p, j, j_a, j_b = data['d'],data['q'],data['p'],data['j'], data['ja'], data['jb']
                    ds2.append(d)
                    dqs2.append(q)
                    j2.append(j)
                    ja2.append(j_a)
                    jb2.append(j_b)
                    P2.append(p)

                plt.subplot(321)
                plt.plot(ds1, dqs1,"+", label=f"$\Delta=${delta},$\mu=${mu},$d\mu=${dmu}")
                plt.subplot(322)
                plt.plot(ds2, dqs2,"+", label=f"$\Delta=${delta},$\mu=${mu},$d\mu=${dmu}")
                plt.subplot(323)
                if len(P1) > 0:
                    if alignLowerBranches:
                        P1 = np.array(P1)
                        P1_ = P1- P1.min()
                    else:
                        P1_=P1
                    index1, value = max(enumerate(P1), key=operator.itemgetter(1))
                    state = "FF" if value > p0 else "NS"
                    plt.plot(ds1, P1_, label=f"$\Delta=${delta},$\mu=${mu},$d\mu=${dmu},State:{state}")
                    plt.axvline(ds1[index1])
                    print(f"Normal Pressure={p0}，FFState Pressue={value}")
                    print(data1[index1])
                plt.subplot(324)
                if len(P2) > 0:
                    if alignUpperBranches:
                        P2 = np.array(P2)
                        P2_ = P2 - P2.min()
                    else:
                        P2_ = P2
                    index2, value = max(enumerate(P2), key=operator.itemgetter(1))
                    state = "FF" if value > p0 else "NS"
                    plt.plot(ds2, P2_, label=f"$\Delta=${delta},$\mu=${mu},$d\mu=${dmu},State:{state}")
                    plt.axvline(ds2[index2])
                    print(f"Normal Pressure={p0}，FFState Pressue={value}")
                    print(data2[index2])
                plt.subplot(325)
                plt.plot(ds1, j1, label=f"$j_p, \Delta=${delta},$\mu=${mu},$d\mu=${dmu}")
                plt.plot(ds1, ja1, label=f"j_a")
                plt.plot(ds1, jb1, label=f"j_b")
                plt.axvline(ds1[index1])
                plt.subplot(326)
                #plt.plot(ds2, j2, label=f"$j_p, \Delta=${delta},$\mu=${mu},$d\mu=${dmu}")
                plt.plot(ds2, ja2, "+",label=f"j_a")
                plt.plot(ds2, jb2, "+",label=f"j_b")
                if len(ds2) > 0:
                    plt.axvline(ds2[index2])
                    plt.axhline(0)
                #break
        
    for i in range(1,7):
        plt.subplot(3,2,i)
        if showLegend:
                plt.legend()
        if i == 1:
            plt.title(f"Lower Branch")
            plt.ylabel("$\delta q$")
        if i == 2:
            plt.title(f"Upper Branch")
            plt.ylabel("$\delta q$")
        if i == 3 or i == 4:
            plt.ylabel("$Pressure$")
        if i == 5 or i == 6:
            plt.ylabel("$Current$")
        plt.xlabel("$\Delta$")


PlotCurrentPressure(alignLowerBranches=True, alignUpperBranches=True, showLegend=True)

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
ds = np.linspace(0.001,2, 40)
ps = [ff.get_pressure(mu=mu, dmu=dmu, mu_eff=mu_eff, dmu_eff=dmu_eff, delta=d, dq=dq).n for d in ds]
delta_max = ff.solve(mu=mu, dmu=dmu, dq=dq)
print(f"Delta={delta_max}")
plt.plot(ds,ps)
plt.axvline(delta_max)

# ## Check range of $\Delta$
# ### 1D Case

g=-2.8 #-3.077012439639267
#g=None
ff = FFStateFinder(delta=2.4, dim=1, mu=10, dmu=0.8, g=g)
dqs = np.linspace(0.001, 2, 20)
gs = [ff._gc(delta=0.1, dq=dq) for dq in dqs]
plt.plot(dqs, gs)
plt.axhline(0)

# ### 3D case
# * To compare with the phase diagram in [Leo 2010]
# [Leo 2010]: http://iopscience.iop.org/article/10.1088/0034-4885/73/7/076501/meta 'Imbalanced Feshbach-resonant Fermi gases'

g = -10 #-3.077012439639267
g = None
delta = 2.4
dim = 3
mu = 10
dmu = 2.41
k_c = 50
ff = FFStateFinder(delta=delta, dim=dim, mu=mu, dmu=dmu, g=g, k_c=k_c)
dqs = np.linspace(0,1.5, 20)
plt.figure(figsize(8,4))
gs = [ff._gc(mu=mu, dmu=dmu, delta=2.6, dq=dq) for dq in dqs]
plt.plot(dqs, gs, label=f'd={d}')
plt.axhline(0)
plt.legend()

# ## Revisit the unitary case
# * In [Leo 2010] scattering length is used in x axis
# * $$\tilde{C}=\frac{m}{4\pi\hbar^2 a}=\frac{1}{g}+\Lambda$$
# * To make it's more straightforward comparision, we fix $\tilde{C}$ here

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

# ## Mathematical Relations:
# \begin{align}
# k_a &= k + q + dq, \qquad k_b = k+q - dq ,\qquad
# \epsilon_a = \frac{\hbar^2}{2m}k_a^2 - \mu_a, \qquad \epsilon_b = \frac{\hbar^2}{2m}k_b^2 - \mu_b,\\
# E&=\sqrt{\epsilon_+^2+\abs{\Delta}^2},\qquad \omega_+= \epsilon_-+E, \qquad \omega_- = \epsilon_- - E\\
# \epsilon_+&= \frac{\hbar^2}{4m}(k_a^2+k_b^2) - \mu_+= \frac{\hbar^2}{2m}\left[(k+q)^2 + dq^2\right] - \mu_+\\
# \epsilon_-&= \frac{\hbar^2}{4m}(k_a^2-k_b^2) - \mu_-=\frac{\hbar^2}{m}(k +q)dq - \mu_-\tag{1}\\
# \end{align}

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
# We can start with the regular way to define the gap:
# $$
# \Delta(x)=\Delta_0e^{-2i\vec{\delta}.\vec{x}}
# $$
#
# Then:
#
# $$
# H=\begin{pmatrix}
# \nabla-\mu_a & \Delta_0 e^{2i\delta x}\\
#  \Delta_0 e^{-2i\delta x} & -\nabla + \mu_b\\
# \end{pmatrix} \\
# H
# \begin{pmatrix}
# U\\
# V\\
# \end{pmatrix}=\omega\begin{pmatrix}
# U\\
# V\\
# \end{pmatrix}
# $$

# Let
# $$
# U=e^{+i\delta x}\tilde{U}\\
# V=e^{-i\delta x}\tilde{V}
# $$
# Then 
# $$
# \begin{pmatrix}
# \nabla-\mu_a & \Delta_0 e^{2i\delta x}\\
#  \Delta_0 e^{-2i\delta x} & -\nabla + \mu_b\\
# \end{pmatrix}
# \begin{pmatrix}
# e^{+i\delta x}\tilde{U}\\
# e^{-i\delta x}\tilde{V}
# \end{pmatrix}
# $$

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


