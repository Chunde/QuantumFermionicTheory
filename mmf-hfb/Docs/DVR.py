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

# # Defination of DVR

# Let $\phi_1(x), \phi_2(x)\dots \phi_n(x)$ be normlized and orthogonal basis in the Hilbert space $H$, $\{x_\alpha\}=(x_1, x_2, \dots, x_m)$ be a set of grid point in the configuration space of the system on which the coordinate system is based. Define the projector operator as:
#
# $$
# P=\sum_n{\ket{\phi_n}\bra{\phi_n}}\qquad \text{It may be easy to prove}:  P^2=P=P^{\dagger}
# $$
# Then let:
# $$
# \ket{\Delta_\alpha}=P\ket{x_\alpha}=\sum_n{\ket{\phi_n}\braket{\phi_n| x_\alpha}}=\sum_n{\phi_n^*(x_\alpha)}\ket{\phi_n}
# $$
# If these $(\ket{\Delta_1}, \ket{\Delta_2},\dots, \ket{\Delta_m})$ is complete in the subspace $S=PH$, and orthogonal, ie:
#
# $$
# \braket{\Delta_\alpha|\Delta_\beta}=N_{\alpha}\delta_{\alpha\beta}\qquad N_\alpha > 0 \tag{1}
# $$
#
# Then we say $(\ket{\Delta_1}, \ket{\Delta_2},\dots, \ket{\Delta_m})$ is the DVR set of the space $S$, we may also call $\ket{\Delta_{\alpha}}$ a DVR state, and each of such state is associated with a grid point, i.e: $x_{\alpha}$ as it's defined upon.

# ## Example

# Say we have three function in the basis: $\phi_1(x), \phi_2(x), \phi_3(x)$ accociated with a set of abscissa{$x_n$}={$x_1, x_2, x_3, x_4$}, they are orthogonal,ie:
# $$
# \braket{\phi_i|\phi_j}=\int_a^b{\phi^*_i(x)\phi_j(x)}dx=\delta_{ij} \qquad
# $$
# Then:
# $$
# \ket{\Delta_1}=\sum_{n=1}^3{\phi_n^*(x_1)}\ket{\phi_n}=\phi_1^*(x_1)\ket{\phi_1}+\phi_2^*(x_1)\ket{\phi_2}+\phi_3^*(x_1)\ket{\phi_3}\\
# \ket{\Delta_2}=\sum_{n=1}^3{\phi_n^*(x_2)}\ket{\phi_n}=\phi_1^*(x_2)\ket{\phi_1}+\phi_2^*(x_2)\ket{\phi_2}+\phi_3^*(x_2)\ket{\phi_3}\\
# \ket{\Delta_3}=\sum_{n=1}^3{\phi_n^*(x_3)}\ket{\phi_n}=\phi_1^*(x_3)\ket{\phi_1}+\phi_2^*(x_3)\ket{\phi_2}+\phi_3^*(x_3)\ket{\phi_3}\\
# \ket{\Delta_4}=\sum_{n=1}^3{\phi_n^*(x_4)}\ket{\phi_n}=\phi_1^*(x_4)\ket{\phi_1}+\phi_2^*(x_4)\ket{\phi_2}+\phi_3^*(x_4)\ket{\phi_3}\\
# $$
#
# With the condition (1):

# $$
# \braket{\Delta_i|\Delta_j}=\phi^*_1(x_i)\phi_1(x_j)+\phi^*_2(x_i)\phi_2(x_j)+\phi^*_3(x_i)\phi_3(x_j)=N_i\delta_{ij}
# $$
#
# where
# $$
# N_i=\sum_{n=1}^3{\phi_n^*(x_i)\phi_n(x_i)}
# $$

# Let:
# $$
# \mat{G}=\begin{pmatrix}
# \phi_1(x_1) & \phi_1(x_2) & \phi_1(x_3) &\phi_1(x_4)\\
# \phi_2(x_1) & \phi_2(x_2) & \phi_2(x_3) &\phi_2(x_4)\\
# \phi_3(x_1) & \phi_3(x_2) & \phi_3(x_3) &\phi_3(x_4)\\
# \end{pmatrix}
# $$
# Hence, we arrive:
# $$
# G^{\dagger}G=\mat{G}=\begin{pmatrix}
# N_1 & 0 & 0 & 0\\
# 0 & N_2 & 0 & 0\\
# 0 & 0 & N_3 & 0\\
# 0 & 0 & 0 & N_4
# \end{pmatrix}
# $$
#

# Now let consider the properity of $\braket{x|\Delta_i}$
# $$
# \Delta_i(x)=\braket{x|\Delta_i}=\psi^*_1(x_i)\psi_1(x)+\psi^*_2(x_i)\psi_2(x)+\psi^*_3(x_i)\psi_3(x)
# $$
#
# if evaluate the $\Delta_i(x)$ at those grid points, it can be found:
#
# $$
# \Delta_i(x_j)=N_i\delta_{ij}
# $$
#
# This is an interesting property, the DVR state $\ket{\Delta_i}$ is localized at it's own grid point $x_i$, which means, it's only non-zero at it's own grid point. In other words, the DVR states satisfy simultaneously two properties: $Orthogonality$ and $Interpolation.$

# ## Normalized DVR
# if define:
# $$
# \ket{F_{\alpha}}=\frac{1}{\sqrt{N_{\alpha}}}\ket{\Delta_{\alpha}}\qquad\\
# $$
#
# Then
#
# $$
# \braket{F_i|F_j}=\delta_{ij} \qquad (Normalized)
# $$

# ## Expansion of States
# For a general state $\ket{\psi}$ in the sub space $\mat{H}$, then it can be expanded exactly in the DVR basis:
#
# $$
# \ket{\psi}=\sum_{n=1}^m\ket{F_n}\braket{F_n|\psi}
# $$
#
# As:
# $$
# \ket{F_i}=\frac{1}{\sqrt{N_i}}\ket{\Delta_i}=\frac{1}{\sqrt{N_i}}P\ket{x_i}
# $$

# So:
# $$
# \braket{F_i|\psi}=\frac{1}{\sqrt{N_i}}\bra{x_i}P^{\dagger}\ket{\psi}=\frac{1}{\sqrt{N_i}}\bra{x_i}P\ket{\psi}
# $$

# Because we assume $\ket{\psi}$ is in the subspace spaned by the basis, then $P\ket{\psi}=\psi$ as it's being projected to the same space.
# So the result is:
# $$
# \braket{F_i|\psi}=\frac{1}{\sqrt{N_i}}\psi(x_i)\\
# \ket{\psi}=\sum_{n=1}^m\frac{1}{\sqrt{N_i}}\psi(x_i)\ket{F_n}
# $$

# This result shows that the expansion coefficient of a state is simply connect to its value at grid points.

# ## Scalar Product
# To compute integral $\braket{\phi|\psi}$, we insert the unitary relation into the integral to get:
# $$
# \braket{\phi|\psi}=\sum_i\braket{\phi|F_i}\braket{F_i|\psi}=\sum_i \frac{1}{N_i}\phi^*(x_i)\psi(x_i)
# $$

# # Direct Sinc-DVR

# $$
# x_\alpha=x_0+\alpha\Delta x, \qquad \alpha=...,-1,0, 1, 2,...\\
# \chi_\alpha(x)=\frac{(\Delta x)^{1/2}}{\pi}\frac{sin\frac{\pi}{\Delta x}(x-x_\alpha)}{x-x_\alpha}
# $$

dx=1
x0=0
def chi(x, alpha):
    x_alpha = x0+alpha*dx
    if x-x_alpha == 0:
        return dx**(-0.5)
    return dx**0.5/np.pi*np.sin(np.pi/dx*(x-x_alpha))/(x-x_alpha)


plt.figure(figsize(16,8))
xs = np.linspace(-10, 10, 501)
alphas = list(range(-5,5))
yss = []
for alpha in alphas:
    ys = [chi(x, alpha=alpha) for x in xs]
    yss.append(ys)
    plt.plot(xs,ys)

for i in range(1, len(yss)):
    print(np.dot(yss[i], yss[i-1]))

# # Sine-DVR

# $$
# \psi_j(x)=\begin{cases}
#     \sqrt{\frac{2}{L}}sin(\frac{j\pi(x-x_0)}{L}) & \text{for}  \qquad x_0\leq x\leq L,\\
#     0 & else
#   \end{cases}
# $$

L=np.pi *2
x0=0
xs = np.linspace(0, L, 100)
def psi(x, j):
    return (2.0/L)**0.5*np.sin(j*np.pi*(x-x0)/L)


for j in range(1,6):
    ys = psi(xs, j=j)
    plt.plot(xs, ys)
plt.axhline(0, linestyle='dashed')
plt.xlabel("x")
plt.ylabel(f"$\phi$")

ys1=psi(xs, j=1)
ys2=psi(xs, j=4)
np.dot(ys1, ys2)

# # Aurel's Code

# ## $U_{1\rightarrow 0}$
# * The U in Aurel's Matlab code lines 152-165 is said to be the coordinate transform matrices, but it's hard to understand

# $$
# a=\frac{cos(z_0)}{\sqrt{z_0}}\qquad b=\frac{sin(z_1)}{\sqrt{z_1}}\\
# \begin{align}
# U_{10}
# &=\frac{2\sqrt{z_0z_1}b}{(z_1^2-z_0^2)a}\\
# &=\frac{2z_0sin(z_1)}{(z_1^2-z_0^2)cos(z_0)}
# \end{align}
# $$

# Where $z_0$ is zeros in the case when angular momentum $\nu=0$, while $z_1$ is zeros when $\nu=1$

# ## $U_{0\rightarrow1}$

# $$
# a=\frac{sin(z_1)}{\sqrt{z_1}}\qquad b=-\frac{cos(z_0)}{\sqrt{z_0}}\\
# \begin{align}
# U_{01}
# &=\frac{2\sqrt{z_0z_1}b}{(z_0^2-z_1^2)a}\\
# &=\frac{2z_1cos(z_0)}{(z_1^2-z_0^2)sin(z_1)}
# \end{align}
# $$


