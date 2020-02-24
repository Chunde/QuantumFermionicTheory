# -*- coding: utf-8 -*-
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
# %pylab inline --no-import-all
from nbinit import *                # Conveniences like clear_output
# -

# # <font color='green'>Band Structure</font>

# We now consider the [Kronig–Penney model](https://en.wikipedia.org/wiki/Particle_in_a_one-dimensional_lattice#Kronig–Penney_model) for a potential barrier of width $b$ separating wells by height $V_0>0$ and period $L$.  The solution for the lowest band ($E<V_0$) is:
#
# $$
#   \cos k_B L = \cosh \kappa b\cos k (L-b)
#   + \frac{\kappa_0^2 - k^2}{2k\kappa_0}\sinh\kappa_0 b\sin k(L-b), \\
#   k = \frac{\sqrt{2mE}}{\hbar}, \qquad
#   \kappa_0 = \frac{\sqrt{2m(V_0-E)}}{\hbar}.
# $$
#
# There may be additional bands with $E<V_0$ but in general, higher bands will have $E>V_0$:
#
# $$
#   \cos k_B L = \cos k_0 b\cos k (L-b)
#   - \frac{k_0^2 + k^2}{2kk_0}\sin k_0 b\sin k(L-b), \qquad
#   k_0 = \frac{\sqrt{2m(E-V_0)}}{\hbar}.
# $$

# In the tight-binding limit, we expect the first form $E < V_0$ and the dispersion to have the form:
#
# $$
#   E(k_B) \approx E_0 + 2t(1-\cos k_B L).
# $$
# Define as for the optical lattice potential
# $$
#   k_0 = \frac{\pi}{L}, \qquad E_R = \frac{\hbar^2 k_0^2}{2m}.
# $$

# +
# %pylab inline --no-import-all
import scipy.optimize
import scipy as sp
import ad.admath
hbar = k0 = 1.0
m = 0.5
ER = (hbar*k0)**2/2/m
L = np.pi/k0

class KronigPenney(object):
    def __init__(self, V0, b_L):
        self.V0 = V0
        self.b_L = b_L

    def rhs(self, E_V0, d=0, np=np):
        V0 = self.V0
        E = E_V0*V0
        if d==1:
            E = ad.adfloat(E)
            np = ad.admath
        else:
            E = E + 0j
        b = self.b_L*L
        k = np.sqrt(2*m*E)/hbar
        kappa0 = np.sqrt(2*m*(V0-E))/hbar
        P = (kappa0**2 - k**2)/2/k/kappa0
        res = (np.cosh(kappa0*b)*np.cos(k*(L-b)) 
               + P*np.sinh(kappa0*b)*np.sin(k*(L-b)))
        if d==0:
            return res
        elif d==1:
            return res.d(E)

    def get_E0_V0(self, E_V0_max=1-1e-12):
        def f(E0_V0):
            return self.rhs(E0_V0) - 1
        return sp.optimize.brentq(f, 1e-12, E_V0_max)
    
    def get_t(self, E0_V0=None):
        if E0_V0 is None:
            E0_V0 = p.get_E0_V0()
        drhs = self.rhs(E0_V0, d=1)
        t_V0 = -0.5/drhs
        return t_V0

#p = KronigPenney(V0=35.6879098, b_L=0.1)
p = KronigPenney(V0=15.2196051, b_L=0.9)
#p = KronigPenney(V0=67.3944700, b_L=0.1)
kFL, DeltaEF = 20., 8.
kFL, DeltaEF = 55., 24. 

kF = kFL/L
EF = (hbar*kF)**2/2/m
Delta = DeltaEF*EF

E0_V0 = p.get_E0_V0()
print(E0_V0/p.V0)
t = p.get_t(E0_V0=E0_V0)
print(t)
print(2*m*t*L**2)
E_V0 = E0_V0 + np.linspace(0,1000*t/p.V0,1000)
kL = np.linspace(0, np.pi, 100)
E_unit = EF
plt.plot(np.arccos(p.rhs(E_V0)), (E_V0-E0_V0)*p.V0/E_unit)
plt.plot(kL, 2*t*(1-np.cos(kL))/E_unit)
# -

# In the revised paper, they have $2mtd^2 = 0.16$ and $Lk_F = 20-55$ which corresponds to:
# $$
#   \frac{t}{E_R} = \frac{0.16}{\pi^2} = 0.0162, \qquad
#   h \approx 1.69, \qquad
#   V_0 = 4h^2E_R = 4h^2E_F\left(\frac{\pi}{d k_F}\right)^2 = 0.28-0.037,\\
#   \frac{\Delta}{E_F} \approx 8 - 24, \qquad
#   \frac{\mu^{\uparrow}}{E_F} \approx 4 - 11, \qquad
#   \frac{\mu^{\downarrow}}{E_F} \approx -4 - -12.5.
# $$

# Here we present the band structure calculation for a periodic square well potential with period $L$ and barrier $V_0$ extending from $x=0$ to $x=a$.  To find the band structure, we use [Bloch's theorem](https://en.wikipedia.org/wiki/Bloch_wave) that the solution can be expressed as:
#
# $$
#   \psi(x) = e^{\I k_B x} u(x), \qquad u(x+L) = u(x).
# $$
#
# To formulate the solution, we work with the transfer matrix approach.  At any point $x$ in space, we decompose the wavefunction $u(x) = u_-(x) + u_+(x)$ into a left-moving piece $u_-(x)$ and a right-moving piece $u_+(x)$:
#
# $$
#   \vect{U}(x) = \begin{pmatrix}
#     u_+(x)\\
#     u_-(x)
#   \end{pmatrix}.
# $$
#
# We then define the transfer matrix $\mat{T}(x, x')$ such that
#
# $$
#   \vect{U}(x) = \mat{T}(x, x')\cdot\vect{U}(x').
# $$
#
# The transfer matrix over a region where the potential is constant $V(x) = V_0$ is:
#
# $$
#   \mat{T}(x,x') = \begin{pmatrix}
#     e^{\I k_0 (x-x')}\\
#     0 & e^{-\I k_0 (x-x')}
#   \end{pmatrix}, \qquad
#   \frac{\hbar^2 k_0^2}{2m} = E - V_0.
# $$
#
# A little more complicated is the transfer matrix from a region with $k=k$ to a region with $k=k_0$.  Across this jump we must make sure the wavefunction is $C^1$ continuous, thus we have:
#
# $$
#   u_+(-\epsilon) + u_-(-\epsilon) = u_+(\epsilon) + u_-(\epsilon), \qquad
#   \I k u_+(-\epsilon) - \I k u_-(-\epsilon) = \I k_0 u_+(\epsilon) - \I k_0 u_-(\epsilon).
# $$
#
# $$
#   u_+(-\epsilon) + u_-(-\epsilon) = u_+(\epsilon) + u_-(\epsilon), \qquad
#   \I k u_+(-\epsilon) - \I k u_-(-\epsilon) = \I k_0 u_+(\epsilon) - \I k_0 u_-(\epsilon),\\
#   \vect{U}(\epsilon) = \frac{1}{2}
#   \begin{pmatrix}
#     \overbrace{1+\frac{k}{k_0}}^{\alpha^{+}_{k/k_0}} & \overbrace{1-\frac{k}{k_0}}^{\alpha^{-}_{k/k_0}}\\
#     1-\frac{k}{k_0} & 1+\frac{k}{k_0}
#   \end{pmatrix}\cdot\vect{U}(-\epsilon).
# $$
#
# Likewise, for a transition from a region with $k=k_0$ to a region with $k=k$ we have:
#
# $$
#   \vect{U}(\epsilon) = \frac{1}{2}
#   \begin{pmatrix}
#     \overbrace{1+\frac{k_0}{k}}^{\alpha^{+}_{k_0/k}} & \overbrace{1-\frac{k_0}{k}}^{\alpha^{-}_{k_0/k}}\\
#     1-\frac{k_0}{k} & 1+\frac{k_0}{k}
#   \end{pmatrix}\cdot\vect{U}(-\epsilon).
# $$
#
# This can be simplified by noting that:
#
# $$
#   \alpha^{\pm}_{k_0/k} = \pm\frac{k_0}{k}\overbrace{\alpha^{\pm}_{k/k_0}}^{\alpha_{\pm}}.
# $$
#
# Thus, the full transfer matrix for our potential from $x=0$ to $x=L$ is:
#
# $$
#   \mat{T} = \frac{1}{4}\frac{k_0}{k}
#   \begin{pmatrix}
#     e^{\I k (L-a)}\\
#     0 & e^{-\I k (L-a)}
#   \end{pmatrix}
#   \cdot
#   \begin{pmatrix}
#     \alpha_{+} & - \alpha_{-}\\
#     -\alpha_{-} & \alpha_{+}
#   \end{pmatrix}
#   \cdot
#   \begin{pmatrix}
#     e^{\I k_0 a}\\
#     0 & e^{-\I k_0 a}
#   \end{pmatrix}
#   \cdot
#   \begin{pmatrix}
#     \alpha_{+} & \alpha_{-}\\
#     \alpha_{-} & \alpha_{+}
#   \end{pmatrix},\\
#   = \frac{1}{4}\frac{k_0}{k}
#   \begin{pmatrix}
#     e^{\I k (L-a)}\\
#     0 & e^{-\I k (L-a)}
#   \end{pmatrix}
#   \cdot
#   \begin{pmatrix}
#     e^{\I k_0 a}\alpha_{+}^2 - e^{-\I k_0 a}\alpha_{-}^2
#     & \alpha_{+}\alpha_{-}(e^{\I k_0 a}-e^{-\I k_0 a})\\
#     \alpha_{+}\alpha_{-}(-e^{\I k_0 a}+e^{-\I k_0 a})
#     & -e^{\I k_0 a}\alpha_{-}^2 + e^{-\I k_0 a}\alpha_{+}^2
#   \end{pmatrix},\\
#   = \frac{1}{4}\frac{k_0}{k}
#   \begin{pmatrix}
#     e^{\I k (L-a)}\\
#     0 & e^{-\I k (L-a)}
#   \end{pmatrix}
#   \cdot
#   \begin{pmatrix}
#     \overbrace{2\I\left(1+\frac{k^2}{k_0^2}\right)\sin(k_0a) + 4\frac{k}{k_0}\cos(k_0 a)}^{A}
#     & \overbrace{2\I\left(1-\frac{k^2}{k_0^2}\right)\sin(k_0 a)}^{B}\\
#     -2\I\left(1-\frac{k^2}{k_0^2}\right)\sin(k_0 a)
#     & -2\I\left(1+\frac{k^2}{k_0^2}\right)\sin(k_0a) + 4\frac{k}{k_0}\cos(k_0 a)
#   \end{pmatrix},\\
#   = \frac{1}{4}\frac{k_0}{k}
#   \begin{pmatrix}
#     e^{\I k (L-a)}\\
#     0 & e^{-\I k (L-a)}
#   \end{pmatrix}
#   \cdot
#   \begin{pmatrix}
#     A & B\\
#     B^* & A^*
#   \end{pmatrix}.  
# $$


