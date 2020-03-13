# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"init_cell": true}
import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbinit import *                # Conveniences like clear_output
# -

# ## BCS Code
# ### <font color='green'>Single-Particle Hamiltonian Matrix</font>

# $$
#   \mat{H} = \begin{pmatrix}
#     -\frac{\hbar^2 k^2}{2m} - \mu_a +V_a & -\Delta\\
#     -\Delta^\dagger & -\left(-\frac{\hbar^2 k^2}{2m} - \mu_b +V_b\right)
#   \end{pmatrix}, \qquad
#   \op{\Psi} = \begin{pmatrix}
#     \op{a}_{k}\\
#     \op{b}^\dagger_{-k}
#   \end{pmatrix}.
# $$
#
# The application of the kinetic energy would be as follows:
#
# $$
#   \DeclareMathOperator{\FFT}{FFT}
#   \psi(x) = 
#   -\frac{\hbar^2\nabla_x^2\psi(x)}{2m} 
#   = \FFT^{-1}\Bigl(\frac{\hbar^2k^2}{2m}\overbrace{\FFT(\psi)}^{\int \d{x}\; e^{-\I k x}\psi(x)}\Bigr)
#   = \int\frac{\d{k}}{(2\pi)}\; e^{\I k x} \Bigl(\frac{\hbar^2k^2}{2m} \int\d{y}\; e^{-\I k y}\psi(y) \Bigr)\\
#   = \int\frac{\d{k}}{(2\pi)}\d{y}\; \frac{e^{\I k (x-y)}\hbar^2k^2}{2m} \psi(y).
# $$
#
# $$
#   \FFT[\psi(x)] = \int\d{x}\; e^{-\I k x}\psi(x) = \frac{L}{N}\sum_{n} e^{-\I k x_n}\psi(x_n),\\
#   \FFT^{-1}(\psi_k) = \int\frac{\d{k}}{(2\pi)}\; e^{\I k x}\psi_k = \frac{1}{L}\sum_{m} e^{\I k_m x}\psi_{k_m}.
# $$
#
# When doing both the FFT and the IFFT, the factors of $L$ cancel and we are left with an overall factor of $1/N$.  This can be split into two factors of $1/\sqrt{N}$:
#
# $$
#   \FFT^{-1}\Bigl(f(k)\FFT(\psi)\Bigr) = \mat{U}^{-1}\cdot \diag(f_k) \cdot \mat{U} \cdot \psi
#   = \sum_{xy}(U_{xk}^* f_k U_{ky}\psi_y).
# $$
#
# Thus, the kinetic energy matrix is:
#
# $$
#   \mat{K}_{xy} = \frac{1}{N}\frac{e^{\I k (x-y)}\hbar^2k^2}{2m}.
# $$

# $$
#   U_{kx} = \frac{1}{\sqrt{N}}e^{-\I k x}, \qquad
#   \mat{U}^{-1} = \mat{U}^\dagger.
# $$

# Even all the above looks right to me, to get the code level, I have to struggle a lot to fully appreciate how it works.
# To better understand the way how the kenectic matrix is presented, let dig a bit more to the detail of how linear aglbra works here, let say the kenitic operator is $\vec{T})$, from above derivation, we got:
#
# $$
# \vec{T}\psi(x)=\frac{1}{N}\sum_{ky}{e^{ikx-iky}f(k)}\psi(y)
# $$
#
# the varibles $x$ and $k$ are spatial and wave-vector points. Pick one specific spatial position $x_1$, then
#
# \begin{align}
# \vec{T}\psi(x)|_{x=x1}
# &=\frac{1}{N}\sum_{ky}{e^{ikx_1-iky}f(k)}\phi(y)\\
# &=\frac{1}{N}\sum_{k}{e^{ikx_1}f(k)}\left[e^{-iky_1}\psi(y_1)+e^{-iky_2}\psi(y_2)+...+ e^{-iky_N}\psi(y_N)\right]
# \end{align}
#
# for different values of $x={x_1,x_2,...x_N}$, we have:
# \begin{align}
# \vec{T}\psi(x)|_{x=x_1}=\frac{1}{N}\sum_{k}{e^{ikx_1}f(k)}\left[e^{-iky_1}\psi(y_1)+e^{-iky_2}\psi(y_2)+...+ e^{-iky_N}\psi(y_N)\right]\\
# \vec{T}\psi(x)|_{x=x_2}=\frac{1}{N}\sum_{k}{e^{ikx_2}f(k)}\left[e^{-iky_1}\psi(y_1)+e^{-iky_2}\psi(y_2)+...+ e^{-iky_N}\psi(y_N)\right]\\
# \vdots\\
# \vec{T}\psi(x)|_{x=x_N}=\frac{1}{N}\sum_{k}{e^{ikx_N}f(k)}\left[e^{-iky_1}\psi(y_1)+e^{-iky_2}\psi(y_2)+...+ e^{-iky_N}\psi(y_N)\right]\\
# \end{align}
# The above result is can be put as a matrix times a column vector:

# $$
#   \begin{pmatrix}
#     \vec{T}\psi(x)|_{x=x_1}\\
#     \vec{T}\psi(x)|_{x=x_2}\\
#     \vdots\\
#     \vec{T}\psi(x)|_{x=x_N}\\
#   \end{pmatrix} = \frac{1}{N}\begin{pmatrix}
#     \sum_{k}{e^{ikx_1}e^{-iky_1}f(k)} & \sum_{k}{e^{ikx_1}e^{-iky_2}f(k)}&...&\sum_{k}{e^{ikx_1}e^{-iky_N}f(k)}\\
#     \sum_{k}{e^{ikx_2}e^{-iky_1}f(k)} & \sum_{k}{e^{ikx_2}e^{-iky_2}f(k)}&...&\sum_{k}{e^{ikx_2}e^{-iky_N}f(k)}\\
#     \vdots\\
#     \sum_{k}{e^{ikx_N}e^{-iky_1}f(k)} & \sum_{k}{e^{ikx_N}e^{-iky_2}f(k)}&...&\sum_{k}{e^{ikx_N}e^{-iky_N}f(k)}\\
#   \end{pmatrix}
#  \begin{pmatrix}
#     \psi(y_1)\\
#     \psi(y_2)\\
#     \vdots\\
#     \psi(y_N)\\
#   \end{pmatrix}.
# $$

# The RHS matrix can be further decompose into three matrices:
# $$
#  \begin{pmatrix}
#     \sum_{k}{e^{ikx_1}e^{-iky_1}f(k)} & \sum_{k}{e^{ikx_1}e^{-iky_2}f(k)}&...&\sum_{k}{e^{ikx_1}e^{-iky_N}f(k)}\\
#     \sum_{k}{e^{ikx_2}e^{-iky_1}f(k)} & \sum_{k}{e^{ikx_2}e^{-iky_2}f(k)}&...&\sum_{k}{e^{ikx_2}e^{-iky_N}f(k)}\\
#     \vdots\\
#     \sum_{k}{e^{ikx_N}e^{-iky_1}f(k)} & \sum_{k}{e^{ikx_N}e^{-iky_2}f(k)}&...&\sum_{k}{e^{ikx_N}e^{-iky_N}f(k)}\\
#   \end{pmatrix}=
#    \begin{pmatrix}
#     e^{ik_1x_1} & e^{ik_2x_1}&...&e^{ik_Nx_1}\\
#     e^{ik_1x_2} & e^{ik_2x_2}&...&e^{ik_Nx_2}\\
#     \vdots\\
#     e^{ik_1x_N} & e^{ik_2x_N}&...&e^{ik_Nx_N}\\
#   \end{pmatrix}
#   \begin{pmatrix}
#     f(k_1) & 0&...&0\\
#     0 & f(k_2)&...&0\\
#     \vdots\\
#     0 & 0&...&f(k_N)\\
#   \end{pmatrix}
#  \begin{pmatrix}
#     e^{-ik_1y_1} & e^{-ik_1y_2}&...&e^{-ik_1y_N}\\
#     e^{-ik_2y_1} & e^{-ik_2y_2}&...&e^{-ik_2y_N}\\
#     \vdots\\
#     e^{-ik_Ny_1} & e^{-ik_Ny_2}&...&e^{-ik_Ny_N}\\
#   \end{pmatrix}
# $$
#

# Now I convince myself that the kinetic matrix can be put as:
# \begin{align}
# T&=U^\dagger\times diag[f(k)]\times U\\
# U&=\frac{1}{\sqrt{N}}e^{-kx}\\
# U&=\frac{1}{\sqrt{N}}
# \begin{pmatrix}
#     e^{-ik_1x_1} & e^{-ik_1x_2}&...&e^{-ik_1x_N}\\
#     e^{-ik_2x_1} & e^{-ik_2x_2}&...&e^{-ik_2x_N}\\
#     \vdots\\
#     e^{-ik_Nx_1} & e^{-ik_Nx_2}&...&e^{-ik_Nx_N}\\
#   \end{pmatrix}
# \end{align}

# Or expanded as a row vector times a column vector:
# $$
# U = e^{-i\begin{pmatrix}
#    k_1\\k_2\\\vdots\\k_N\\
#   \end{pmatrix}\times  
#   \begin{pmatrix}
#    x_1&x_2&...&x_N\\
#   \end{pmatrix}
#   }
# $$




