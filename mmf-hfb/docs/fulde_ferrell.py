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

# + init_cell=true
import mmf_setup;mmf_setup.nbinit()
from mmfutils.contexts import NoInterrupt
try: from importlib import reload
except ImportError: pass
# %pylab inline --no-import-all
# -

# # Fulde-Ferrell (FF) States

# Here we work through some details about the Fulde Ferrell state.  The corresponding code is in the following files:
#
# * [`tf_completion.py`](tf_completion.py)
# * [`fulde_ferrell.py`](fulde_ferrell.py)
#

# The first file contains all the integrals:
#
# * `n_p_integrand`:
#
#   $$
#     \newcommand{\E}{\mathcal{E}}\newcommand{\e}{\epsilon}
#     \int \frac{\d^d{k}}{(2\pi)^d}\; \left(1 - \frac{\e_+}{E}\bigl(f_\beta(\omega_-) - f_\beta(\omega_+)\bigr)\right)
#   $$

# The idea of the FF state is that the gap has a form $\Delta(z) = e^{\I q z}\Delta$.  By rotating the basis, we can absorb the $q$ into the kinetic energy:
#
# $$
#   \mat{M} = \begin{pmatrix}
#     \frac{\hbar^2[(k_z-q)^2 + k_\perp^2]}{2m_a} - \mu_a & \Delta\\
#     \Delta & -\left(\frac{\hbar^2[(k_z+q)^2 + k_\perp^2]}{2m_b} - \mu_b\right)
#   \end{pmatrix}.
# $$
#
# The eigenvalues $\omega_{\pm}$ are zero when:
#
# $$
#   m_a = m_b = m, \qquad k^2 = k_z^2 + k_\perp^2, \qquad
#   \e_{\pm} = \frac{\e_a \pm \e_b}{2}, \\
#   \omega_{\pm} = \e_- \pm \sqrt{\e_+^2 + \abs{\Delta}^2} = 0 \quad 
#   \Rightarrow \quad \e_{-}^2 = \e_+^2 + \abs{\Delta}^2,\\
#   \e_+ = \frac{\hbar^2k^2}{2m} - \overbrace{\left(\mu_+ - \frac{\hbar^2q^2}{2m}\right)}^{\mu_q},\qquad
#   \e_- = \frac{qk_z\hbar^2}{m} - \mu_-,\\
#   k_\perp^2 = -k_z^2 + \frac{2m}{\hbar^2}\left(\mu_q \pm \sqrt{\left(\frac{qk_z\hbar^2}{m} - \mu_-\right)^2 - \abs{\Delta}^2}\right).
# $$
#
# All the real parts of this equation can be used as `points` for accurate integration.

# ## Momentum Integrals

# To regulate the theory, we will need to compute integrals up to a cubic cutoff $k < k_c$.  We shall assume in the code that $k_c$ is much larger than any of these `points`.

# ## Gap Equation

# Our code takes as inputs: ($\mu_+$, $\mu_-$, $q$, $\Delta$, $T$).  From these, we can compute $\mat{M}$ and integrate all of the states up to $k_c$ or beyond.  The gap equation has the form:
#
# $$
#   \tilde{C} = \frac{-\nu_c}{\Delta} + \Lambda_c = \frac{m}{4\pi \hbar^2 a_s}\\
#   = \lim_{k_c\rightarrow \infty}
#   \left(
#     - \int_{k<k_c}\frac{\d^d{k}}{(2\pi)^d}\;
#     \frac{f_\beta(\omega_-) - f_\beta(\omega_+)}{2\sqrt{\e_+^2 + \Delta^2}}
#     +
#     \overbrace{\frac{1}{2} \int_{k<k_c}\frac{\d^d{k}}{(2\pi)^d}\;\frac{1}{\e_+ + \I 0^+}}^{\Lambda_c}
#   \right)
# $$
#
# *Note: Computing this is a bit tricky. Numerically one cannot easily integrated the entire expression because of the poles at $\epsilon_+ = 0$, however, the second integral can be computed analytically for finite $k_c$.  Thus, the computation of this term can proceed as follows, which is done in the function `compute_C()`:*
#
# $$
#   \tilde{C} = 
#     - \int_{k<k_c}\frac{\d^d{k}}{(2\pi)^d}\;
#     \frac{f_\beta(\omega_-) - f_\beta(\omega_+)}{2\sqrt{\e_+^2 + \Delta^2}}
#   + \Lambda_c(\mu_q)
#   +  \int_{k\geq k_c}\frac{\d^d{k}}{(2\pi)^d}\;\left(
#     -\frac{f_\beta(\omega_-) - f_\beta(\omega_+)}{2\sqrt{\e_+^2 + \Delta^2}}
#     + \frac{1}{2\e_+}
#   \right).
# $$
#
# A physical theory holds $\tilde{C}$ fixed (or as a predefined function of densities).  Thus, this equation must be solved to figure out $\Delta$.  In the unitary limit, $\tilde{C} = 0$.  Solving this equation is tricky, because the rhs might not even be monotonic.  There might also only be the single solution $\Delta = 0$.
#
#
#
# As a result, one can construct states by fixing $\Delta$ and then computing the interaction at which that state is self-consistent.

# +
import numpy as np
import tf_completion as tf;reload(tf)

np.random.seed(1)
m, hbar, kF = 1 + np.random.random(3)
eF = (hbar*kF)**2/2/m
nF = kF**3/3/np.pi**2
mu = 0.59060550703283853378393810185221521748413488992993*eF
delta = 0.68640205206984016444108204356564421137062514068346*eF

dmu = 0.5*delta
args0 = dict(mu_a=mu+dmu, mu_b=mu-dmu, delta=delta, m_a=m, m_b=m, 
             hbar=hbar, d=3, T=0.0)

# Make a FF state with q
k_c = 10*kF
q = -0.8*kF
args = dict(args0, q=q, k_c=k_c)
tf.compute_C(**args), tf.integrate_q(tf.n_m_integrand, **args)

# -

from mmfutils.plot import imcontourf
kz = np.linspace(-2, 2, 100)[:, None]
kp = np.linspace(-2, 2, 100)[None, :]
e = hbar**2/2/m
e_a = e*((kz-q)**2 + kp**2) - args['mu_a']
e_b = e*((kz+q)**2 + kp**2) - args['mu_b']
e_p, e_m = (e_a + e_b)/2, (e_a - e_b)/2
E = np.sqrt(e_p**2+abs(delta)**2)
w_p, w_m = e_m + E, e_m - E
plt.figure(figsize=(10,5))
plt.subplot(121)
imcontourf(kz, kp, np.sign(w_p), vmin=-1, vmax=1);plt.colorbar()
plt.subplot(122)
imcontourf(kz, kp, np.sign(w_m), vmin=-1, vmax=1);plt.colorbar()

# +
ss = np.linspace(-2, 2, 50)
res = []
for s in ss:
    dmu = s*delta
    mu_a, mu_b = mu + dmu, mu - dmu
    args = dict(args0, mu_a=mu_a, mu_b=mu_b)
    n_p = tf.integrate(tf.n_p_integrand, **args)
    n_m = tf.integrate(tf.n_m_integrand, **args)
    res.append((n_p.n, n_m.n))

res = np.asarray(res)
plt.plot(ss, res);plt.xlabel('dmu/delta');plt.ylabel('n')

# +
from scipy.optimize import brentq
def get_delta(dmu=0, q=0, debug=False, **kw):
    def f(delta):        
        args = dict(args0)
        
        args['mu_a'], args['mu_b'] = mu + dmu, mu - dmu
        args.update(kw)        
        args['delta'] = delta
        C = tf.compute_C(q=q, **args)
        return C.n
    if debug:
        return f
    return brentq(f, 0.5, 1.0)

get_delta(dmu=0.5*delta, q=0), delta

# +
f = get_delta(dmu=0.2*delta, q=0.5, debug=True)

ss = np.linspace(0.3, 1.1, 20)
plt.plot(ss, [f(s) for s in ss])


# +
from scipy.optimize import brentq
def get_delta(dmu=0, q=0, debug=False, **kw):
    def f(delta):         
        args = dict(args0)
        args['mu_a'], args['mu_b'] = mu + dmu, mu - dmu
        args.update(kw)        
        args['delta'] = delta
        C = tf.integrate(tf.C_integrand, k_c=10.0, **args)
        return C.n
    if debug:
        return f
    return brentq(f, 0.0, 1.0)

ss = np.linspace(0.3, 1.1, 20)
res = []
for s in ss:
    f = get_delta(dmu=s*delta, debug=True)
    res.append(f(0.001))
plt.plot(ss, res);plt.xlabel('dmu/delta');plt.ylabel('C(0)')

# +
from scipy.optimize import brentq
def get_delta(dmu=0, q=0, debug=False, **kw):
    def f(delta):        
        args = dict(args0)
 
        args['mu_a'], args['mu_b'] = mu + dmu, mu - dmu
        args.update(kw)        
        args['delta'] = delta
        C = tf.compute_C(**args)
        return C.n
    if debug:
        return f
    return brentq(f, 0.0, 1.0)

ss = np.linspace(0.3, 1.1, 20)
res = []
for s in ss:
    f = get_delta(dmu=s*delta, debug=True)
    res.append(f(delta))
plt.plot(ss, res);plt.xlabel('dmu/delta');plt.ylabel('C(0)')

# +
ss = np.linspace(-4, 4, 100)
res = []
for s in ss:
    args = dict(args0,delta=s*delta)
    C = tf.compute_C(**args)
    res.append(C.n)

res = np.asarray(res)
plt.plot(ss*delta, res);plt.xlabel('delta');plt.ylabel('C')
# -

# # To Do

# * Check speed of `dblquad` vs nested calls to `quad`.

# # Playing with FuldeFerrelState class.

# %pylab inline --no-import-all
plt.rcParams['figure.figsize'] = (10, 5)
from uncertainties import unumpy as unp
from mmfutils.plot import imcontourf
from mmf_hfb import tf_completion, FuldeFerrelState;
from scipy.integrate import IntegrationWarning
reload(tf_completion);reload(FuldeFerrelState)
import warnings
warnings.simplefilter('error', IntegrationWarning)

# In the weak coupling regime, we expect to find FF states.  We start here:

# +
N = 32
L = 10.0
k_c = np.pi*N/L
dim = 1
delta = 0.1
mu = 10.0
dmu = 0.11

args = dict(mu=mu, dmu=dmu, delta=delta, k_c=k_c, dim=dim)
f = FuldeFerrelState.FFState(fix_g=True, **args)
f.get_g(delta=delta)
# -

deltas = np.linspace(0, 0.16, 80)[:, None]
dqs = np.linspace(0, 0.04, 80)[None, :]
res = np.vectorize(lambda _d, _dq: f.get_g(delta=_d, dq=_dq) - f._g)(deltas, dqs)
Ps = np.vectorize(lambda _d, _dq: f.get_pressure(delta=_d, dq=_dq))(deltas, dqs)

# Here is a plot tracing the solutions to the gap equation along with pressure.  In principle these can be combined to find the ground state at these chemical potentials.  I am a little worried that the 1D code has an error (no Hartree term.)

# +
plt.subplot(121)
imcontourf(deltas, dqs, np.sign(res))
plt.xlabel('Delta'); plt.ylabel('dq')
plt.colorbar(label='sign(g-g_0)')

plt.subplot(122)
imcontourf(deltas, dqs, unp.nominal_values(Ps))
plt.xlabel('Delta'); plt.ylabel('dq')
plt.colorbar(label='P')
# -

imcontourf(deltas, dqs, np.sign(res))
plt.xlabel('Delta'); plt.ylabel('dq')
plt.colorbar(label='sign(g-g_0)')
plt.contour(deltas.ravel(), dqs.ravel(), unp.nominal_values(Ps).T, 100)

deltas = np.linspace(0.001, delta, 40)
deltas = np.linspace(0, 0.2, 20)
z = [f.get_g(r=3.04, delta=_d) - f._g for _d in deltas]
plt.plot(deltas, z)
plt.axhline(0)

f._tf_args

# +
N = 32
L = 10.0
k_c = np.pi*N/L
mu = 10.0
dmu = 2.0
dmu = 6.5

delta = 5.0
delta = 8.0
args = dict(mu=mu, dmu=dmu, delta=delta, k_c=k_c)
f = FuldeFerrelState.FFState(fix_g=True, **args)
deltas = np.linspace(0.001, delta, 40)
deltas = np.linspace(0, 8.2, 20)
z = [f.get_g(r=3.04, delta=_d) - f._g for _d in deltas]
plt.plot(deltas, z)
plt.axhline(0)
# -

# ## Integration Error

args

# +
from mmf_hfb import tf_completion as tf
N = 32
L = 10.0
k_c = np.pi*N/L
mu = 10.0
dmu = 6.5
delta = 8.0
r = 1.0
q = 1./r
m = hbar = 1
mu_a, mu_b = mu + dmu, mu - dmu

_args = [mu_a, mu_b, delta, 1, 1, 1, 0]
#tf.integrate_q(tf.nu_delta_integrand, **args)
kz = -4.59
kp = np.linspace(0, k_c, 100)

k2_a = (kz+q)**2 + kp**2
k2_b = (kz-q)**2 + kp**2
#plt.plot(kp, tf.nu_delta_integrand(k2_a, k2_b, *_args))
pz = kz
mu_q = mu - q**2/2
D = (q*pz/m - dmu)**2 - delta**2
A = 2*m*mu_q - pz**2
print(np.sqrt(A + 2*m*np.sqrt(D))/hbar, k1)

px = kz
mu_q = mu - q**2/2/m
sqrt0 = (q*px/m - dmu)**2 - delta**2
sqrt1 = np.sqrt(sqrt0)
sqrt2 = 2*m*(mu_q + sqrt1) - px**2
sqrt3 = 2*m*(mu_q - sqrt1) - px**2
k1 = np.sqrt(sqrt2)/hbar
k2 = np.sqrt(sqrt3)/hbar
sqrt2, 
#k1, k2

# -

xs = []
sp.integrate.quad(f, -6.744561989371367+0.001, -1.6457536310736816-0.001)



#sp.integrate.quad(func, a=a, b=b)
all_points = [a] + points + [b]
for _a, _b in zip(all_points[:-1], all_points[1:]):
    print((_a, _b))
    print(sp.integrate.quad(func, a=_a, b=_b))


points

x1

#locals().update(sys._l1)
from mmf_hfb.integrate import quad, partial
x=x0
quad(partial(func, x), a=y0_x(x), b=y1_x(x), points=points, **kw)
0.0053154780215518005 - 0.005316021949098559



x_ = x0
a_=y0_x(x_)
b_=y1_x(x_)
xs = []
def f(x):
    xs.append(x)
    return partial(func, x_)(x)
quad(f, a=a_, b=b_, points=points, **kw)

fs = np.array(list(map(partial(func, x), xs)))
plt.plot(xs, fs, '+')
points

x0 = -4.6
x1 = -4.59
x = np.linspace(x0, x1, 100)
func(x0)

# %matplotlib notebook
fs = np.array(list(map(func, xs)))

#plt.figure(figsize=(20,2))
i = np.argsort(xs)
plt.plot(np.asarray(xs)[i], np.asarray(fs)[i], '+', ms=1)
#plt.vlines(points, -0.04, 0)
#plt.savefig('tmp.pdf')
# #!open tmp.pdf

plt.plot(deltas, z, '-+')

rs = np.linspace(0.01, L/2, 100)
ds = [f.solve(r=_r, a=0.001, b=2*delta) for _r in rs]

f.solve(r=0.01, a=0.001, b=2*delta)

plt.plot(rs, ds)

plt.plot(rs, ds)

# ## Homogeneous

from mmf_hfb import tf_completion, FuldeFerrelState;
from scipy.integrate import IntegrationWarning
reload(tf_completion);reload(FuldeFerrelState)
import warnings
warnings.simplefilter('error', IntegrationWarning)

# +
N = 32
L = 10.0
k_c = 20*np.pi*N/L
mu = 10.0
dmu = 2.0
dmu = 6.5

delta = 5.0
delta = 8.0
args = dict(mu=mu, dmu=dmu, delta=delta, k_c=k_c)
f = FuldeFerrelState.FFState(fix_g=True, **args)
# -

f.get_g(delta=0.1, mu=mu, dmu=dmu, q=q, dq=dq)

print(f.get_densities(q=0, dq=1.0, **args))

# +
print(f.get_densities(q=0, dq=0, **args))

q = 1.0
print(f.get_densities(q=q, dq=0, **args))
# -

q = 0
dq = 1.0
(f.get_pressure(mu=mu, dmu=dmu, q=q, dq=0), 
 f.get_pressure(mu=mu, dmu=dmu, q=q, dq=dq))

# +
P1 = (1, -1)
P2 = np.polymul(P1, P1)
P4 = np.polymul(P2, P2)
P8 = np.polymul(P4, P4)
P16 = np.polymul(P8, P8)
P32 = np.polymul(P16, P16)
x = np.linspace(1-0.001,1+0.001,100)
plt.plot(x, (np.polyval(P8, x)- (x-1)**8))


# -

deltas = np.linspace(0.001, delta, 40)
deltas = np.linspace(0, 8.2, 20)
z = [f.get_g(r=3.04, delta=_d) - f._g for _d in deltas]
plt.plot(deltas, z)
plt.axhline(0)
