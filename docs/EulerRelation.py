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

# # Euler Relation for Pauli Matrices

# * Pauli Matrices:
#
# \begin{align}
#   \sigma_1 = \sigma_x &=
#     \begin{pmatrix}
#       0&1\\
#       1&0
#     \end{pmatrix} \\
#   \sigma_2 = \sigma_y &=
#     \begin{pmatrix}
#       0&-i\\
#       i&0
#     \end{pmatrix} \\
#   \sigma_3 = \sigma_z &=
#     \begin{pmatrix}
#       1&0\\
#       0&-1
#     \end{pmatrix} \,.
# \end{align}
# $$
# \sigma_x^2=\sigma_y^2=\sigma_z^2=-i\sigma_x\sigma_y\sigma_z=
# \begin{pmatrix}
# 1&0\\
# 0&1\\
# \end{pmatrix}
# $$

# * A very useful relation
#
# \begin{align}
# e^{ix\sigma}
# &=1+ix\sigma-\frac{x^2}{2!} - \frac{ix^3\sigma}{3!} + \frac{x^4}{4!}+ i\frac{x^5\sigma}{5!}+\dots\\
# &=\left[1 -\frac{x^2}{2!}  + \frac{x^4}{4!}+\dots \right] + i\left[ x -  \frac{x^3}{3!} + \frac{x^5}{5!} + \dots\right]\sigma\\
# &=cos(x)+i\sigma sin(x)
# \end{align}

# * Rewrite the off-diagnal terms
#
# $$
# \begin{pmatrix}
# 0 & e^{i\theta}\\
# e^{-i\theta}& 0\\
# \end{pmatrix}=\begin{pmatrix}
# 0 & cos(\theta) + i sin(\theta)\\
# cos(\theta)-isin(\theta)& 0\\
# \end{pmatrix}=cos(\theta)\sigma_x - sin(\theta)\sigma_y
# $$

# * The RHS of the last expression can be put as:
#
# $$
# cos(\theta)\sigma_x - sin(\theta)\sigma_y=e^{i\theta\sigma_z /2} \sigma_x e^{-i\theta\sigma_z /2}
# $$

# ## One example
# In BdG theory, if we define the gap:
# $$
# \Delta(x)=\Delta_0e^{-2i\delta x}
# $$
#
# Then:
#
# $$
# H=\begin{pmatrix}
# \frac{k^2}{2m}-\mu_a & \Delta_0 e^{2i\delta x}\\
#  \Delta_0 e^{-2i\delta x} & -\frac{k^2}{2m} + \mu_b\\
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

# $$
# H=\frac{k^2}{2m} \sigma_z + \begin{pmatrix}-\mu_a &0\\0&\mu_b\end{pmatrix} +\Delta_0 e^{i\delta x \sigma_z}\sigma_x e^{-i \delta x \sigma_z}
# $$
# Define a unitary operator U:
#
# $$
# U=e^{i\delta x \sigma_z}
# $$
# Then we apply the $U$ to the $H$:

# $$
# \tilde{H}=U^T H U\\
# $$
# The the kinetic term will change to (with $x\rightarrow\delta x$):
#

# $$
# \left[\left(1 -\frac{x^2}{2!}  + \frac{x^4}{4!}+\dots \right) - i\left( x - \frac{x^3}{3!} + \frac{x^5}{5!} + \dots\right)\sigma_z \right]
# \frac{k^2}{2m}\sigma_z \left[\left(1 -\frac{x^2}{2!}  + \frac{x^4}{4!}+\dots \right) + i\left( x -  \frac{x^3}{3!} + \frac{x^5}{5!} + \dots\right)\sigma_z\right]$$


