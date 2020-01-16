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

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import *

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
#
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

#

# To define a DVR basis, let $\mathcal{H}$ be the Hilbert space, and $\op{P}$ be the projecting operator sends $\mathcal{H}$ to a subspace $\mathcal{S}$, i.e.
# $$
# \mathcal{S}=\op{P}\mathcal{H}
# $$
# The subspace is the practical space we are going to study, or put another way, it is the space where we try to approximate $\mathcal{H}$. For more formal discussion, see \cite{littlejohn2002general}\cite{martinazzodiscrete}. Let $M$ the configuration space of $\mathcal{H}$, for example, for a system without spin, the configuration space in 3D space is just  $\mathbb{R}^3$. In the configuration space, define a set of grid points $\{x_i\}$. Similar to the grid representation, where $\{x_i\}$ may be just the collection of all grid points represent a function $g(x)$ over the space $M$. In DVR case, these grid points do not have to be equally spaced. 

# In DVR case, these grid points do not have to be equally spaced.  The grid points set and the projector $P$ define a DVR set if they satisfy the following properties:
# $$
# \ket{\Delta_{\alpha}}=\op{P}\ket{x_{\alpha}}\qquad \alpha = 1,2\dots,N
# $$
# where $\braket{\Delta_{\alpha}|\Delta_{\beta}}=W_{\alpha}\delta_{\alpha\beta}$, with $W_{\alpha}>0$. This is just orthogonal property of the vector set $\{\ket{\Delta_{\alpha}}\}_{\alpha}$. 
# Another property requires the $\{\ket{\Delta_{\alpha}}\}_{\alpha}$ to be complete in the subspace $\mathcal{S}$, that means any vector in $\mathcal{S}$ can be represented exactly using $\{\ket{\Delta_{\alpha}}\}_{\alpha}$
#
# As the weight factor $W_{\alpha}$ may be not necessarily equal to unity, Then we can define:
# $$
# \ket{F_{\alpha}}=\frac{1}{\sqrt{N_{\alpha}}}\ket{\Delta_{\alpha}},\qquad \braket{F_{\alpha}|F_{\beta}}=\delta_{\alpha \beta}
# $$
#
# The basis function set for the DVR in $M$ can be defined:
# $$
# F_{\alpha}(x)=\braket{x|F_{\alpha}},\qquad \alpha=1,2,\dots N
# $$
# then any function lives inside the space $\mathcal{S}$ can be exactly expressed as:
# $$
# g(x)=\sum_{i=1}^N {c_i F_i(x)}
# $$
# $c_i$ are the expansion coefficients. So far there is nothing special about the DVR method, and we have not explain the purpose of the grid point set. 
#
# Recall the way we define the basis function, to be explicitly, let write down a basis function:
#
# \begin{align}\begin{split}\label{eq:dvr_basis_function}
# F_i(x)=\braket{x|F_i}
# &=\frac{1}{N_i}\braket{x|\Delta_i}\\
# &=\frac{1}{N_i}\braket{x|P|x_i}\\
# \end{split}\end{align}
#
# For the projector $\op{P}$, by its definition, its satisfies $\op{P}^{\dagger}=\op{P}=\op{P}^2$. the last line of the last equation can be written as:
# \begin{align}\begin{split}
# F_i(x)=
# &=\frac{1}{N_i}\braket{x|P|x_i}\\
# &=\frac{1}{N_i}\braket{x|P^2|x_i}\\
# \end{split}\end{align}
# Evaluate the basis function at all grid points:
# $$
# F_i(x_j)=\frac{1}{N_i}\braket{x_j|P^2|x_i}=\frac{1}{N_i}\braket{\Delta_i|\Delta_j}=\frac{1}{N_i}\delta_{ij}
# $$
# This is an very interesting observation, it says that any basis function is non-zero only at its one grid point(the point where it is defined on, see Eq.\ref{eq:dvr_basis_function}), it is zero at all other grid points. For $x$ other than the grid points, it is non-zero in general.
#
#
# One of the difference between the spectrum representation and the DVR is the way to determine the function expansion coefficients, let a function $f(x)$ be expanded in a basis(can be spectrum basis or DVR basis), and the spectrum basis function set is $\{\phi_i(x)\}_{i=1}^N$ for, the DVR basis function set is $\{F_j(x)\}_{j=1}^M$. Then $f(x)$ can be expressed by the form:
#
# $$
# f(x)=\sum_{i=1}^N{c_i \phi_i(x)}=\sum_{j=1}^M{C_j F_j(x)}
# $$


