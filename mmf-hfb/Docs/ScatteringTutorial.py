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

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import *
import math

# # Attractive Well
# * $v_0=\frac{q^2}{2m}$

import operator
r0 = 1
def cot(x):
    return np.cos(x)/np.sin(x)
def k_prime(k, q):
    return (k**2 + q**2)**0.5
def delta(k, q):
    kp = k_prime(k, q)
    skr = np.sin(k*r0)
    skpr = np.sin(kp*r0)
    ckr = np.cos(k*r0)
    ckpr = np.cos(kp*r0)
    theta = np.arctan2(k*skpr*ckr - kp*skr*ckpr, kp*ckpr*ckr + k*skr*skpr)
   # return theta
    #if theta < 0:
    #    return theta + 2*np.pi
    return theta
def align_phase(xs):
    index, value = min(enumerate(xs), key=operator.itemgetter(1))
    if value < 0:       
        max = np.max(xs) - np.min(xs)
        n =  int(max/2/np.pi) + 1
        offset = n*2*np.pi
        for i in range(index + 1):
            xs[i]=xs[i] + offset
    return xs


q0 = 0.1*np.pi/r0
qs =np.linspace(0, 30, 31)*q0
ks = np.linspace(0, 3, 500)*np.pi
plt.figure(figsize(16,8))
for q in qs:
    ys = [delta(k, q) for k in ks]
    ys = np.array(align_phase(ys))/np.pi
    plt.plot(ks[1:], ys[1:])




