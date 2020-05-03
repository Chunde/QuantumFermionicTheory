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

import mmf_setup;mmf_setup.nbinit()

# $$
#   \int \frac{\d^2{k}}{(2\pi)^2} = \int \d{k_z}2\int_{0}^{k_c}\d{k_\perp} \frac{1}{4\pi^2} f
#   = \int \d{k_z}\int_{0}^{k_c}\d{k_\perp} \frac{1}{2\pi^2} f\\
#   \int \frac{\d^3{k}}{(2\pi)^3} = \int\d{k_z} 2\pi \int_{0}^{k_c}k_\perp\d{k_\perp} \frac{1}{8\pi^3} f
#   = \int \d{k_z} \int_{0}^{k_c}k_\perp\d{k_\perp} \frac{1}{4\pi^2} f\\
# $$

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
#   \qquad
#    \mu_q = \mu - \frac{dq^2}{2m}
# $$
#
# Critical points occur when $\omega_{\pm} = 0$ which gives the following conditions:
#
# $$
#   \epsilon_-^2 = \epsilon_+^2 + \abs{\Delta}^2\\
#   \left(\frac{qp_x}{m} - \mu_{-}\right)^2 = \left(\frac{p_x^2 + p_\perp^2}{2m} - \mu_q\right)^2 + \abs{\Delta}^2.
# $$
#
# This may be solved for $p_\perp$ giving the following critical points:
#
# $$
#   p_\perp^2 = 2m\left(\mu_q \pm \sqrt{\left(\frac{qp_x}{m} - \mu_{-}\right)^2 - \abs{\Delta}^2}\right) - p_x^2.
# $$

# +
# %pylab inline --no-import-all
from ipywidgets import interact
from mmfutils.plot import imcontourf
m = 1

mu = eF = 1.0
pF = np.sqrt(2*m*eF)
delta = 0.5*mu
p_max = 2*pF

@interact(dq=(-2, 2, 0.1), dmu=(-1.0, 1.0, 0.1), delta=(0, 1, 0.1),q=(0, 2, 0.2))
def plot_regions(q=0, dq=0, dmu=0.4, delta=0.2):
    delta = np.abs(delta)
    p_x, p_perp = np.meshgrid(np.linspace(-p_max, p_max, 500),
                              np.linspace(-p_max, p_max, 500),
                              indexing='ij', sparse=True)
    mu_a = mu + dmu
    mu_b = mu - dmu
    e_a = ((p_x + q + dq)**2 + p_perp**2)/2/m - mu_a
    e_b = ((p_x + q - dq)**2 + p_perp**2)/2/m - mu_b
    e_p, e_m = (e_a + e_b)/2.0, (e_a-e_b)/2.0
    E = np.sqrt(e_p**2 + delta**2)
    w_p, w_m = e_m + E, e_m - E
    
    # Analytic regions
    p_x_ = (p_x + q).ravel()
    mu_q = mu - dq**2/2/m
    p_perp1_ = np.ma.sqrt(2*m*(mu_q + np.sqrt((dq*p_x_/m - dmu)**2 - delta**2)) - p_x_**2
                         ).filled(np.nan)
    p_perp2_ = np.ma.sqrt(2*m*(mu_q - np.sqrt((dq*p_x_/m - dmu)**2 - delta**2)) - p_x_**2
                         ).filled(np.nan)
    
    p_x_special = np.ma.divide(m*(dmu - np.array([delta, -delta])), 
                               dq).filled(np.nan).tolist()
    P = [1,  0, -4*(m*mu_q + dq**2), 8*m*dq*dmu, 4*m**2*(delta**2 + mu_q**2 - dmu**2)]
    p_x_special.extend([p for p in np.roots(P) if p.imag == 0])
    p_x_special = (np.array(p_x_special) - q).tolist()
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    imcontourf(p_x, p_perp, np.sign(w_p))
    plt.plot(p_x, p_perp1_, p_x, p_perp2_)
    plt.vlines(p_x_special, 0, p_max); plt.xlim(-p_max, p_max)
    plt.xlabel('p_x');plt.ylabel('p_perp')
    plt.subplot(122)    
    imcontourf(p_x, p_perp, np.sign(-w_m))
    plt.plot(p_x, p_perp1_, p_x, p_perp2_)
    plt.vlines(p_x_special, 0, p_max); plt.xlim(-p_max, p_max)
    plt.xlabel('p_x');plt.ylabel('p_perp')


# -

mu = eF = 10.0
pF = np.sqrt(2*m*eF)
delta = 0.5*mu
p_max = 2*pF
plot_regions(q=1/3.0, dmu=6.5, delta=8.0)

# We thus have the correct regions.  The second task is to figure out where these regions start and end so we can indentify special points in $p_x$.  These occur when $\d{p_x}/\d{p_\perp} = 0$, or when the right-hand side of the following expression diverges.  One condition is simple, but the condition that $p_\perp = 0$ requires solving a quartic polynomial:
#
# $$
#   \frac{\d{p_\perp}}{\d{p_x}} = \frac{1}{2p_\perp}\left(\pm \frac{2\left(\frac{qp_x}{m} - \mu_{-}\right)q}{\sqrt{\left(\frac{qp_x}{m} - \mu_{-}\right)^2 - \abs{\Delta}^2}} - 2 p_x\right), \\
#   \left(\frac{qp_x}{m} - \mu_{-}\right)^2 - \abs{\Delta}^2 = 0,\\
#   p_x = \frac{m(\mu_{-} \pm \abs{\Delta})}{q}, \qquad
#   \left(\frac{qp_x}{m} - \mu_{-}\right)^2 - \abs{\Delta}^2 = \left(\frac{p_x^2}{2m} - \mu_q\right)^2.
# $$

# The last condition can be solve numerically from the following polynomial:
#
# $$
#   p_x^4 - 4\left(m\mu_q + q^2\right)p_x^2 + 8mq\mu_-p_x + 4m^2\left(\abs{\Delta}^2 + \mu_q^2 - \mu_-^2\right) = 0
# $$

# $$
#   \left(p_x^2 - 2m\mu_q\right)^2 + 4m^2\abs{\Delta}^2 - 4m^2\left(\frac{qp_x}{m} - 4m^2\mu_{-}\right)^2 = 0
# $$

# ### Large $q$

# We now consider the limit of large $q$ (close to the core of a vortex).  The last equation can be rearranged as:
#
# $$
#   4m^2\abs{\Delta}^2 = 4m^2\left(\frac{qp_x}{m} - 4m^2\mu_{-}\right)^2 - \left(p_x^2 - 2m\mu_q\right)^2.
# $$
#
# The rhs has a maximum when 
#
# $$
#   8m^2\left(\frac{qp_x}{m} - 4m^2\mu_{-}\right)\frac{q}{m} = 4p_x\left(p_x^2 - 2m\mu_q\right),
# $$
#
# which, for large $q$ requires:
#
# $$
#   p_x \approx \frac{4m^3\mu_{-}}{q}, \qquad
#   4m^2\abs{\Delta}^2 =  - \left(\frac{16m^6\mu_{-}^2}{q^2} - 2m\mu_q\right)^2.
# $$

p = np.linspace(-2,2,100)
plt.plot(p, (p-1.0)**2 - (p**2-1.0)**2)

# +
# %pylab inline --no-import-all
from mmf_hfb.integrate import dquad

import sympy
def f(y, x, x0=0, y0=0, np=np):
        return np.exp(-(x-x0)**2 - (y-y0)**2)
x, y = sympy.var(['x', 'y'], real=True)
f_xy = f(y, x, x0=0, y0=0, np=sympy)
f_xy.integrate((x, -np.inf, np.inf), (y, -np.inf, np.inf))
f_xy.integrate((x, -1, 1), (y, -1, 1))


x0 = -np.inf
x1 = np.inf
def y0_x(x):
    return -np.inf
def y1_x(x):
    return np.inf

points = [1e5]
def points_y_x(x):
    return points
dquad(f, x0, x1, y0_x, y1_x, points, points_y_x)

# +
_xs = []
def f(x, x0=100):
    global _xs
    _xs.append(x)
    return np.exp(-(x-x0)**2)
    
x = np.linspace(-10,10,100)
plt.plot(x, f(x, x0=0))

_xs = []
from scipy.integrate import quad
print(quad(f, -np.inf, np.inf))
xs = np.array(_xs)
# -

plt.plot(xs, f(xs), '+')

# +
from functools import partial
import numpy as np
import scipy.integrate
import scipy as sp
from uncertainties import ufloat
import math
import cmath
import numba
import warnings
def quad(func, a, b, points=None, **kw):
    if points is not None and np.any(np.isinf([a, b])):
        sign = 1
        if b < a:
            sign = -1
            a, b = b, a
        points = sorted([a, b] + [p for p in points if a < p and p < b])
        res = [sp.integrate.quad(func=func, a=_a, b=_b, **kw)
               for _a, _b in zip(points[:-1], points[1:])]
        return sign * sum(ufloat(*_r) for _r in res)
    else:
        return ufloat(*sp.integrate.quad(func=func, a=a, b=b, points=points, **kw))


def dquad(func, x0, x1, y0_x, y1_x, points_x=None, points_y_x=None, **kw):
    def inner_integrand(x):
        points = None
        if points_y_x is not None:
            points = points_y_x(x)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                return quad(partial(func, x), a=y0_x(x), b=y1_x(x),points=points, **kw).n
            except Warning as e:
                print(x, y0_x(x),y1_x(x), points, kw)
                
        return quad(partial(func, x), a=y0_x(x), b=y1_x(x),points=points, **kw).n
    
    return quad(inner_integrand, x0, x1, points=points_x, **kw)


# -

def integrate(f, mu_a, mu_b, delta, m_a, m_b, d=3,q=0, dq=0.0, hbar=1.0, T=0.0, k_0=0, k_c=None, limit=500):
    k_inf = np.inf if k_c is None else k_c
    delta = abs(delta)
    args = (mu_a, mu_b, delta, m_a, m_b, hbar, T)
    print(args)
    if d == 1:
        def integrand(k):
            k2_a = (k + q + dq)**2
            k2_b = (k + q - dq)**2
            return f(k2_a, k2_b, *args) / np.pi
    elif d == 2:
        def integrand(kx, kp):
            k2_a = (kx + q + dq)**2 + kp**2
            k2_b = (kx + q - dq)**2 + kp**2
            assert(kp>=0)
            return f(k2_a, k2_b, *args) / (2*np.pi**2)
    elif d == 3:
        def integrand(kx, kp):
            k2_a = (kx + q + dq)**2 + kp**2
            k2_b = (kx + q - dq)**2 + kp**2
            assert(kp>=0)
            return f(k2_a, k2_b, *args) * (kp/4/np.pi**2)
    else:
        raise ValueError(f"Only d=1, 2, or 3 supported (got d={d})")

    mu = (mu_a + mu_b)/2
    dmu = (mu_a - mu_b)/2
    minv = (1/m_a + 1/m_b)/2
    mu_q = mu - dq**2/2*minv
    assert m_a == m_b   # Need to re-derive for different masses
    m = 1./minv
    kF = np.sqrt(2*mu/minv)/hbar

    p_x_special = (np.ma.divide(m*(dmu - np.array([delta, -delta])),dq).filled(np.nan)- q).tolist()

    P = [1, 0, -4*(m*mu_q + dq**2),8*m*dq*dmu, 4*m**2*(delta**2 + mu_q**2 - dmu**2)]
    p_x_special.extend([p.real - q for p in np.roots(P)])
    points = sorted(set([x/hbar for x in p_x_special if not math.isnan(x)]))
    if d == 1:
        integrand = numba.cfunc(numba.float64(numba.float64))(integrand)
        integrand = sp.LowLevelCallable(integrand.ctypes)
        return quad(func=integrand, a=k_0, b=k_inf, points=points, limit=limit)

  
    def kp0(kx):
        kx2 = kx**2
        k_02 = k_0**2
        if kx2 < k_02:
            return math.sqrt(k_02 - kx2)
        else:
            return 0.0

    def kp1(kx):
        return math.sqrt(k_inf**2 - kx**2)

    def kp_special(kx):
        px = hbar*(kx + q)
        D = (dq*px/m - dmu)**2 - delta**2
        A = 2*m*mu_q - px**2
        return (cmath.sqrt(A + 2*m*cmath.sqrt(D)).real/hbar,cmath.sqrt(A - 2*m*cmath.sqrt(D)).real/hbar)
    
    return dquad(func=integrand,x0=-k_inf, x1=k_inf,y0_x=kp0, y1_x=kp1,points_x=points,points_y_x=kp_special,limit=limit)


from mmf_hfb import tf_completion as tf
args = {'T': 0, 'd': 3, 'delta': 1.0000370124897544, 'dq': 0, 'hbar': 1, 'k_c': 100, 'm_a': 1, 'm_b': 1, 'mu_a': 5.6400999999999994, 'mu_b': 4.3601, 'q': 0}
kappa = integrate(tf.kappa_integrand, limit= 500, **args)
kappa

# -4.358431136964881 0.0 499.98100371716555 (1.1908404580715668, 1.1908404580715668) {'limit': 200}

# +
q=0
dq = 0
_xs = []

def func(kx, kp):
    _xs.append(kp)
    args = (5.6400999999999994, 4.3601, 1.0000370124897544, 1, 1, 1, 0)
    k2_a = (kx + q + dq)**2 + kp**2
    k2_b = (kx + q - dq)**2 + kp**2
    return tf.kappa_integrand(k2_a, k2_b, *args) * (kp/4/np.pi**2)
quad(partial(func, -0.5083886444002128), a=0, b=499.99974154091944,points=(3.130811597522146, 3.130811597522146), limit=500)
# -

1.5379400874493416 + 1.5379400874493416

xs = np.array(_xs)
def f(y):
    return func(-4.358431136964881,y)
ys = f(xs)

x = np.linspace(0,5,1000)
y = f(x)
# %matplotlib inline
plt.plot(xs, ys, '+')
plt.plot(x,y)
plt.xlim(0,5)
plt.savefig('tmp.pdf')

fs = np.array(list(map(f,xs)))
# %matplotlib inline
#import mpld3
#mpld3.enable_notebook()

i=np.argsort(xs)
#plt.figure(figsize=(20, 5))
plt.plot(np.asarray(xs)[i], np.asarray(fs)[i],'+',ms=0.1)

# +
from mmf_hfb.FuldeFerrelState import FFState as FF
delta0 = 1
mu=5
dmu=0.64
d=1
q=1.0
dq=0.5
k_c=500
dx = 1e-7
ff = FF(dmu=dmu, mu=mu, delta=delta0, d=d, k_c=k_c, fix_g=True)

def get_P(mu, dmu):
    delta = ff.solve(mu=mu, dmu=dmu, q=q, dq=dq)
    return ff.get_pressure(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)

def get_E_n(mu, dmu):
    E = ff.get_energy_density(mu=mu, dmu=dmu, q=q, dq=dq)
    n = sum(ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq))
    return E, n
def get_density(q, dq):
    return ff.get_energy_density(mu=mu, dmu=dmu, q=q, dq=dq)
def get_ns(mu, dmu):
    return ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)


E1, n1 = get_E_n(mu=mu+dx, dmu=dmu)
E0, n0 = get_E_n(mu=mu-dx, dmu=dmu)

n_p = (get_P(mu+dx, dmu) - get_P(mu-dx, dmu))/2/dx
n_a, n_b = get_ns(mu, dmu)

n_a_ = (get_P(mu+dx/2, dmu+dx/2) - get_P(mu-dx/2, dmu - dx/2))/2/dx
n_b_ = (get_P(mu+dx/2, dmu-dx/2) - get_P(mu-dx/2, dmu + dx/2))/2/dx
print(f"n_a={n_a.n}\tNumerical  n_a={n_a_.n}")
print(f"n_b={n_b.n}\tNumerical  n_b={n_b_.n}")
print(f"n_p={n_a.n+n_b.n}\tNumerical  n_p={n_p.n}")
print(f"mu={mu}\tNumerical mu={(E1-E0)/(n1-n0)}")
# -

dxs = np.linspace(-1,1,20)
es = [ff.get_energy_density(mu=mu + dx, dmu=dmu, q=q, dq=dq).n for dx in dxs]
ns = [sum(ff.get_densities(mu=mu + dx, dmu=dmu, q=q, dq=dq)).n for dx in dxs]

plt.plot(dxs, es)
plt.plot(dxs, ns)

N = 32
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
es = []
for y_ in y:
    for x_ in x:
      es.append(get_density(x_,y_).n)  

from mmfutils.plot import imcontourf
a = np.asarray(es).reshape(N,N)
imcontourf(x,y,a.T)
plt.colorbar()

plt.pcolormesh (x,y,a)
plt.colorbar()


def cutoff_error(mu, dmu, d, k_c, q, dq,  dx = 1e-3):
    delta0 = 1

    ff = FF(dmu=dmu, mu=mu, delta=delta0, d=d, k_c=k_c, fix_g=True)
    
    def get_P(mu, dmu):
        delta = ff.solve(mu=mu, dmu=dmu, q=q, dq=dq)
        return ff.get_pressure(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)

    def get_E_n(mu, dmu):
        E = ff.get_energy_density(mu=mu, dmu=dmu, q=q, dq=dq)
        n = sum(ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq))
        return E, n

    def get_ns(mu, dmu):
        return ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
    
   
    E1, n1 = get_E_n(mu=mu+dx, dmu=dmu)
    E0, n0 = get_E_n(mu=mu-dx, dmu=dmu)
    mu_ = (E1 - E0)/(n1 - n0)
    n_p = (get_P(mu+dx, dmu) - get_P(mu-dx, dmu))/2/dx
    n_a, n_b = get_ns(mu, dmu)

    n_a_ = (get_P(mu+dx/2, dmu+dx/2) - get_P(mu-dx/2, dmu - dx/2))/2/dx
    n_b_ = (get_P(mu+dx/2, dmu-dx/2) - get_P(mu-dx/2, dmu + dx/2))/2/dx
    return n_a - n_a_

kcs = np.linspace(10, 1000, 10)
errs = [cutoff_error(mu = 10, dmu = 0.64, d = 3, k_c = kc, q = 0, dq = 0, dx = 0.001).n for kc in kcs]

errs

plt.plot(kcs, errs)


