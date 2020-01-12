# -*- coding: utf-8 -*-
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

# # Introduction
# There are many way to represent a function $f$, for example, we can expand the function in terms of sin and cos function, which is just the Fourier series representation of this function, i.e.:
# $$
# f(x)=\sum_{n=0}^{\infty}{a_n sin\left(\frac{2\pi nx}{L}\right) + b_n cos\left(\frac{2\pi nx}{L}\right)} \qquad n=0, 1, 2\dots
# $$
# where $L$ is the range where the function $f(x)$ is defined. The continuous version is just the Fourier transform a the function:
# $$
# f(x)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} f(k)e^{ikx}dk
# $$
# In the Fourier series representation, the functions $sin\left(\frac{2\pi nx}{L}\right)$ and $cos\left(\frac{2\pi nx}{L}\right)$ are the basis functions used to expand the function $f(x)$, and all the sin and cos function forms a basis function set. It's clear that these functions are mutual orthogonal and complete in the parameter space $S$ where the function $f(x)$ lives. In principle we can pick any basis set if any functions can expand any functions accurately in the same space $S$. In general,  the number of basis function can be infinite, in some condition, to use finite number basis function can express function inside a space good enough within a desired accuracy.The method to express a function using a finite size basis set is called finite basis representation(FBR)
# In quantum physics, the matrix element of an operator $\mathbb{O}$ can be represented in Dirac notation as:
# $$
# \mathbb{O}_{ij}=\braket{i|\mathbb{O}|j}
# $$
# where $i, j$ are the $i_{th}$ and $j_{th}$ the basis states, and $\braket{x|i}=\psi_i(x)$ is the $i_{th}$ basis functions, then each matrix element can be computed:
# $$
# \mathbb{O}_{ij}=\iint \braket{i|x}\braket{x|\mathbb{O}|y}\braket{y|j}dx dy\\
# =\iint \psi^*_i(x)\braket{x|\mathbb{O}|y}\psi_j(y)dx dy\\
# $$
# where $\braket{x|\mathbb{O}|y}$ is the representation in the real space. For example, if $\mathbb{O}$ is the external potential operator. In real space is just $V(x-y)$. It can be seen that it needs to integrate to get a matrix element, this can be computationally expensive. Which is one of the drawback of the FBR.
#
# In numerical calculation, the function $f(x)$ is often presented as a $N$ dimensional vector $\left[f(x_0), f(x_1), \dots (f(x_0)\right]$ where $x_0, x_1, \dots$ are equal spacing grid points in the range where the function is well defined.

# # 2D Harmonic Oscillator in Polar System
# In polar coordinates, the Del operator $\nabla^2$ is defined as:
# $$
# \begin{align}
# \nabla^2
# &=\frac{1}{r} \frac{\partial}{\partial r}\left(r \frac{\partial f}{\partial r}\right)+\frac{1}{r^{2}} \frac{\partial^{2} f}{\partial \theta^{2}}\\
# &=\frac{\partial^2 f}{\partial r^2}+\frac{1}{r} \frac{\partial f}{\partial r}+\frac{1}{r^{2}} \frac{\partial^{2} f}{\partial \theta^{2}}
# \end{align}
# $$

# Then the Shrodinger Equation for this system can be written as:
# $$
# \left(-\frac{\hbar^2\nabla^2}{2M}+\frac{M\omega^2r^2}{2}\right)\Psi(r,\theta)=E\Psi(r,\theta)\\
# \left(-\frac{\partial^2}{2M\partial r^2}-\frac{1}{2Mr} \frac{\partial}{\partial r}-\frac{1}{2Mr^2} \frac{\partial^{2} }{\partial \theta^{2}}+\frac{M\omega^2r^2}{2}\right)\Psi(r,\theta)=E\Psi(r,\theta)\\
# \left(-\frac{\partial^2}{\partial r^2}-\frac{1}{r} \frac{\partial}{\partial r}-\frac{1}{r^2} \frac{\partial^{2} }{\partial \theta^{2}}+M^2\omega^2r^2\right)\Psi(r,\theta)=2ME\Psi(r,\theta)
# $$
# By assuming that the solution is separatable $\Psi(r,\theta)=R(r)\psi(\theta)$, we can solve the angular part fairly easily:
# $$
# \left(-\frac{\partial^2 R(r)}{\partial r^2}\phi(\theta)-\frac{1}{r} \frac{\partial R(r)}{\partial r}\phi(\theta)-\frac{1}{r^2} \frac{\partial^2 \phi(\theta) }{\partial \theta^{2}}R(r)+M^2\omega^2r^2 R(r)\phi(\theta)\right)=2MER(r)\phi(\theta)
# $$
# Divide both side by $R(r)\phi(\theta)$ to get:
# $$
# \left(-\frac{\partial^2 R(r)}{R(r)\partial r^2}-\frac{1}{rR(r)} \frac{\partial R(r)}{\partial r}-\frac{1}{r^2} \frac{\partial^2 \phi(\theta) }{\phi(\theta)\partial \theta^{2}}+M^2\omega^2r^2 \right)=2ME
# $$
# that means
# $$
# \psi(\theta)=e^{im\theta}\qquad m=0,1,2\dots
# $$
# The the radia part can be rearraged when substitude the angular solution into the Scrodinger equation:
#

# $$
# r^2R''+rR'+ \left(2r^2ME-m^2-M^2\omega^2r^4\right)R=0
# $$
# The equation can be simplfied by setting $M=\omega=1$
# $$
# r^2R''+rR'+ \left(2r^2E-m^2-r^4\right)R=0
# $$


