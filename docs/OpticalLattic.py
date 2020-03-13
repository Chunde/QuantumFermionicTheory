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

import mmf_setup;mmf_setup.nbinit(hgroot=False)
from mmfutils.contexts import NoInterrupt
import hfb_dir_init
from importlib import reload 

# # <font color='green'>Optical Lattice</font>

# Prompted by a paper I had to review, here I consider pairing in a system where one component is trapped in an optical lattice with cell-length $L$.  The paper claimed ultra-high critical temperatures due to the modified dispersion.
#
# Here we solve the lattice problem explicitly by modeling the unit cell since we use periodic boundary conditions.  Note: simply solving the 1D problem in a periodic box is not the same as solving the periodic problem in infinite space because the 1D periodic problem neglects Bloch waves of the form:
#
# $$
#   \psi_{k_b}(x) = e^{\I k_b x} u(x), \qquad
#   u(x+L) = u(x), \qquad 
#   \abs{k_b} < \frac{\pi}{L},
# $$
#
# where $u(x)$ is periodic.
#
# One way to understand the need for the Bloch momentum is to note that a periodic box has discrete momenta $k_n = 2\pi n/L$ with spacing $\d{k} = 2\pi/L$.  The Bloch momenta are implement by shifting $k_n \rightarrow k_n + k_b$.  Thus, the range of Bloch momenta $\abs{k_b}| \d{k}/2$ exactly allows one to sample all points in the range $\d{k}$.
#
# These Bloch momenta are implemented by averaging the density matrix over the full range of $k_b$ (which we implement as a twist angle $\theta = k_b L$, $\abs{\theta} < \pi$).

# The Lattice potential admits an exact solution in terms of Mathier fuctions and the single-band model is appropriate when $h>2$:
#
# $$
#   \frac{\op{p}^2}{2m} + \frac{V_0}{2}\cos(2k_0x),
#   \qquad
#   L = \frac{\pi}{k_0}, \qquad
#   E_R = \frac{\hbar^2 k_0^2}{2m}, \qquad
#   4h^2 = \frac{V_0}{E_R}, \qquad
#   \lambda = \frac{E}{E_R},\\
#   E(k) = E_0 + 2t\cos(kL) +\cdots, \qquad
#   \frac{t}{E_R} = \frac{8\sqrt{2}}{\sqrt{\pi}}h^{3/2}e^{-4h}\left[1+\order(1/h)\right]
# $$
#
# http://iopscience.iop.org/article/10.1088/1361-6404/aa8d2c/meta

# The paper I am reviewing has $d\equiv L$ $t/E_F = 0.1$ and $dk_F = 20-50$.  In terms of these parameters we have:
#
# $$
#   \frac{t}{E_R} = \frac{t}{E_F}\left(\frac{k_F}{k_0}\right)^2 
#   = \frac{t}{E_F}\left(\frac{d k_F}{\pi}\right)^2
#   = \frac{2mt d^2}{\pi^2} < 0.1, \qquad
#   t d^2 < \frac{1}{2m},\\
#   \frac{t}{E_F}\left(d k_F\right)^2 < 1
# $$

# +
# %pylab inline --no-import-all
from IPython.display import display, clear_output
import bcs;reload(bcs)
import homogeneous;reload(homogeneous)
from bcs import BCS

class Lattice(BCS):
    """Adds optical lattice potential to species a with depth V0."""
    cells = 1.0
    t = 0.0007018621290128983
    E0 = -0.312433127299677
    power = 4
    V0 = -10.5

    def __init__(self, cells=1, N=2**5, L=10.0, 
                 mu_a=1.0, mu_b=1.0, v0=0.1,
                 V0=-10.5, power=2,
                 **kw):
        self.power = power
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.v0 = v0
        self.V0 = V0
        self.cells = cells
        BCS.__init__(self, L=cells*L, N=cells*N, **kw)

    def get_Vext(self):
        v_a = (-self.V0 * (1-((1+np.cos(2*np.pi * self.cells*self.x/self.L))/2)**self.power)
               )
        v_b = 0 * self.x
        return v_a, v_b

    def iterate(self, mudelta, N_twist=1, plot=False, **kw):
        mu_a, mu_b, delta = nudelta
        if np.isinf(N_twist):
            R = self.get_R_twist_average(mus=(mu_a, mu_b), delta=delta, **kw)
        else:
            R = self.get_R(mus=(mu_a, mu_b), delta=delta, N_twist=N_twist)
        na = np.diag(R)[:l.N]/l.dx
        nb = (1 - np.diag(R)[l.N:])/l.dx
        kappa = np.diag(R[:l.N, l.N:])/l.dx
        mu_a = self.mu_a + self.v0*nb
        mu_b = self.mu_b + self.v0*na
        delta = self.v0*kappa
        if plot:
            plt.clf()
            plt.plot(self.x, na)
            plt.plot(self.x, nb)
            plt.plot(self.x, delta)
            display(plt.gcf())
            print(delta.real.max(), na.real.max(), nb.real.max())
        else:
            display("{:.12f}, {:.12f}, {:.12f}".format(delta.real.max(), na.real.max(), nb.real.max()))
        clear_output(wait=True)
        return np.array((mu_a, mu_b, delta))

    def iterate_full(self, mudelta, 
                     na_avg=0.5, nb_avg=0.5, 
                     N_twist=1, plot=False, **kw):
        mu_a, mu_b, mu_a_eff, mu_b_eff, delta = mudelta
        mus = (mu_a_eff, mu_b_eff)
        if np.isinf(N_twist):
            R = self.get_R_twist_average(mus=mus, delta=delta, **kw)
        else:
            R = self.get_R(mus=mus, delta=delta, N_twist=N_twist)
        na = np.diag(R)[:l.N]/l.dx
        nb = (1 - np.diag(R)[l.N:])/l.dx

        mu_a = mu_a*(1 + (na_avg - na.mean()))
        mu_b = mu_b*(1 + (nb_avg - nb.mean()))

        kappa = np.diag(R[:l.N, l.N:])/l.dx
        mu_a_eff = mu_a + self.v0*nb
        mu_b_eff = mu_b + self.v0*na
        delta = self.v0*kappa
        if plot:
            plt.clf()
            plt.plot(self.x, na)
            plt.plot(self.x, nb)
            plt.plot(self.x, delta)
            display(plt.gcf())
            print(delta.real.max(), na.real.mean(), nb.real.mean())
        else:
            display("{:.12f}, {:.12f}, {:.12f}".format(
                delta.real.max(), na.real.mean(), nb.real.mean()))
            #display("{:.12f}, {:.12f}, {:.12f}, {:.12f}, {:.12f}".format(
            #    mu_a, mu_b, delta.real.max(), na.real.mean(), nb.real.mean()))
        clear_output(wait=True)
        return (mu_a, mu_b, mu_a_eff, mu_b_eff, delta)


# +
# Test - reproduce homogeneous results
import homogeneous
delta = 1.0
mu_eff = 1.0
v_0, n, mu, e_0 = homogeneous.get_BCS_v_n_e(delta=delta, mu_eff=mu_eff)

L = 0.46
N = 2**8
N_twist = 2**5
for b in [bcs.BCS(T=0, N=N, L=L),
          Lattice(T=0.0, N=N, L=L, v0=v_0, V0=0)]:
    R = b.get_R(mus=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    na = np.diag(R)[:N]/b.dx
    nb = (1 - np.diag(R)[N:])/b.dx
    kappa = np.diag(R[:N, N:])/b.dx
    print((n, na[0].real + nb[0].real), (delta, v_0*kappa[0].real))
# -

l = Lattice(T=0.0, N=N, L=L, v0=v_0, V0=0)
qT = (mu, mu) + (mu_eff*np.ones(l.N),)*2 + (np.ones(l.N)*delta,)
max_iteration = 5
with NoInterrupt() as interrupted:
    while max_iteration > 0:
        max_iteration -= 1
        qT = l.iterate_full(qT, plot=False, N_twist=np.inf,na_avg=n/2, nb_avg=n/2, abs_tol=1e-2)


