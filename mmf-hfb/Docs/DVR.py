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
# P=\sum_n{\ket{\phi_n}\bra{\phi_n}}
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
# Then we say $(\ket{\Delta_1}, \ket{\Delta_2},\dots, \ket{\Delta_m})$ is the DVR set of the space $S$

# ## Example

# Say we have three function in the basis: $\phi_1(x), \phi_2(x), \phi_3(x)$ accociated with a set of abscissa{$x_n$}={$x_1, x_2, x_3, x_4$}, they are orthogonal,ie:
# $$
# \braket{\phi_i|\phi_j}=\int_a^b{\phi^*_i(x)\phi_j(x)}dx=\delta_{ij} \qquad
# $$
# Then:
# $$
# \ket{\Delta_1}=\sum_{n=1}^3{\phi_n^*(x_1)}\ket{\phi_n}\\
# \ket{\Delta_2}=\sum_{n=1}^3{\phi_n^*(x_2)}\ket{\phi_n}\\
# \ket{\Delta_3}=\sum_{n=1}^3{\phi_n^*(x_3)}\ket{\phi_n}\\
# \ket{\Delta_4}=\sum_{n=1}^3{\phi_n^*(x_4)}\ket{\phi_n}\\
# $$
#
# With the condition (1):

# $$
# \braket{\Delta_i|\Delta_j}=\phi^*_1(x_i)\phi_1(x_j)+\phi^*_2(x_i)\phi_2(x_j)+\phi^*_3(x_i)\phi_3(x_j)=N_i\delta_{ij}
# $$
#
# where
# $$
# N_i=\sum_{n=1}^3{\phi_n^*(x_n)\phi_n(x_n)}
# $$

# Let:
# $$
# \mat{G}=\begin{pmatrix}
# \phi_1(x_1) & \phi_1(x_2) & \phi_1(x_3) &\phi_1(x_4)\\
# \phi_2(x_1) & \phi_2(x_2) & \phi_2(x_3) &\phi_2(x_4)\\
# \phi_3(x_1) & \phi_3(x_2) & \phi_3(x_3) &\phi_3(x_4)\\
# \end{pmatrix}
# $$
# Hence, we arive:
# $$
# G^{\dagger}G=\mat{G}=\begin{pmatrix}
# N_1 & 0 & 0 & 0\\
# 0 & N_2 & 0 & 0\\
# 0 & 0 & N_3 & 0\\
# 0 & 0 & 0 & N_4
# \end{pmatrix}
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
#     \sqrt{2/L}sin(j\pi(x-x_0)/L) & \text{for}  \qquad x_0\leq x\leq L,\\
#     0 & else
#   \end{cases}
#   $$

L=5
x0=0
xs = np.linspace(0, L, 100)
def psi(x, j):
    return (2.0/L)**0.5*np.sin(j*np.pi*(x-x0)/L)


ys = psi(xs, j=2)
plt.plot(xs, ys)
plt.axhline(0, linestyle='dashed')

ys1=psi(xs, j=1)
ys2=psi(xs, j=2)
np.dot(ys1, ys2)


