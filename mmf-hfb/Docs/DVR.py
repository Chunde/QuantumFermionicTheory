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
# If these $(\ket{\Delta_1}, \ket{\Delta_2},\dots, \ket{\Delta_n})$ is complete in the subspace $S=PH$, and orthogonal, ie:
#
# $$
# \braket{\Delta_\alpha|\Delta_\beta}=N_{\alpha}\delta_{\alpha\beta}\qquad N_\alpha > 0
# $$
#
# Then we say $(\ket{\Delta_1}, \ket{\Delta_2},\dots, \ket{\Delta_n})$ is the DVR set of the space $S$

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


