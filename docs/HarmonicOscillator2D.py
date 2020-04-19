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

# # Harmonic Oscillator in Two Dimensions

# Two dimensional harmonic oscillator is a solvable and very instructive system. First, it's simple enough for a graduate student to play to gain the essential properties of a multiple dimensional system, such as angular momentum and degeneracy of a quantum problem. Secondly, since this system can be solved analytically in both Cartesian coordinate as well as cylindrical coordinate, it severed as a good example to connect quantum properties in these two different coordinate system, such as how the angular momentum in cylindrical case is relate to the mathematical form of  wave-functions in the Cartesian coordinate. Third, similar to the cylindrical coordinate, a DVR basis can be used to expand the phase space of the Hilbert space of the problem,  numerical results in Cartesian and Cylindrical coordinates can be used as a benchmark for the case when a Bessel DVR base is used.

# ##  Cartesian Coordinates
# Let the angular frequencies of the harmonic oscillator in two dimensions be $\omega_x$ and $\omega_y$, then the Hamiltonian of the system has the Schrodinger equation($\hbar$ is set to 1):
#
# $$
# -\left(\frac{\partial^{2} \psi}{\partial x^{2}}+\frac{\partial^{2} \psi}{\partial y^{2}}\right)+\left(\omega_x^2x^{2}+\omega_y^{2} y^{2}\right) \psi=2 E \psi
# $$

#
# Since there is no coupling between the x and y component of the system, we can safely assume the wavefunction $\psi$ is separable, ie: $\psi(x,y)=X(x)Y(x)$, substitute into the Schrodinger equation yields:
# $$
# \left(-\frac{1}{X} \frac{d^{2} X}{d x^{2}}+\omega_x^{2}x^{2}\right)+\left(-\frac{1}{Y} \frac{d^{2} Y}{d y^{2}}+\omega_y^{2} y^{2}\right)=2 E
# $$
# The overall energy$E=E_x + E_y$, so we can rewrite the above equation into two equations:
#
# \begin{align}
# -\frac{d^{2} X}{d x^{2}}+\omega_x x^{2} X&=2 E_{x} X \\ 
# -\frac{d^{2} Y}{d y^{2}}+\omega_y^{2} y^{2} Y&=2 E_{y} Y
# \end{align}
#
# It's not hard to find the above two equations can be recognized as separate 1D harmonic oscillator equation in x and y direction, both of them have the eigen spectrum(by a factor of $\hbar \omega_x$ and$\hbar\omega_y$ respectively:
#
# $$
# E_x = \frac{1}{2}, \frac{3}{2}, \dots, \frac{2n+1}{2}, \qquad n=0, 1, 2, \dots \\
# E_y = \frac{1}{2}, \frac{3}{2}, \dots, \frac{2n+1}{2}, \qquad n=0, 1, 2, \dots
# $$
# The total energy of the 2D system is the sum of these two independent 1D system, if  $\omega_x=\omega_y=1$, the eigenvalue of $E$( with a factor of $\hbar$) corresponds to degeneracy of $E$, i.e. if $E=N$, then there are $N$ different wavefunction of the 2D system will yield same expected energy of $N$, to be more specific, the spectrum of the system should be:
#
# $$
# E=1, 2, 2, 3, 3, 3, 4, 4, 4, 4,\dots, 
# $$

# ## Angular momentum
# Recall that for 1D harmonic oscillator, the three lowest energy state wavefunctions are($C_s$ are some normalization constants):
# \begin{align}
# u_{0}(x)&=\left(\frac{m \omega}{\pi \hbar}\right)^{\frac{1}{4}} e^{-m \omega x^{2} / 2 \hbar}\qquad \text{ground state}\\
# u_{1}(x)&=\left(\frac{m \omega}{\pi \hbar}\right)^{\frac{1}{4}} \sqrt{\frac{2 m \omega}{\hbar}} x e^{-m \omega x^{2} / 2 \hbar}\qquad \text{first excited state}\\
# u_{2}(x)&=C\left(1-2 \frac{m \omega x^{2}}{\hbar}\right) e^{-m \omega x^{2} / 2 \hbar}\qquad \text{second excited state}
# \end{align}
#
# For $n>3$:
# $$
# u_{n}(x)=\sum_{k=0}^{\infty} a_{k} y^{k} e^{-y^{2} / 2}
# $$
# where
#
# \begin{align}
# a_{k+2}&=\frac{2(k-n)}{(k+1)(k+2)} a_{k}\\
# y&=\sqrt{\frac{m\omega}{\hbar}x}
# \end{align}
#
# We discuss the angular momentum from the perspective of the spectrum, for example, for $E=1$, the only possible combination is $E_x=E_y=1/2$,  since the two components have same energy in this case and also in ground state, they should be no "oscillation" in neither x or y direction(as $E=(\frac{1}{2}+n)\hbar\omega$, and n=0 for this case, the ground state is Gaussian without spatial factor in front of it), then the overall monition of the system should have no angular momentum.
# However, for $E=2$ case, we need the first excited state in 1D.
# if $n_x=1, n_y=0$, or $n_x=0, n_y=1$, their corresponding wave functions are:
#
# \begin{align}
# \psi_{1}(x, y)=C_0 x e^{-m \omega (x^{2}+y^2) / 2 \hbar}\\
# \psi_{2}(x, y)=C_0 y e^{-m \omega (x^{2}+y^2) / 2 \hbar}
# \end{align}
#
# The linear combination of these two wave function is also eigen function of the harmonic oscillator Hamiltonian.We can construct two wavefunction as follows:
#
# \begin{align}
# \psi_{1}(x, y)=C_1(x+iy)e^{-m \omega (x^{2}+y^2) / 2 \hbar}\\
# \psi_{2}(x, y)=C_1(x-iy)e^{-m \omega (x^{2}+y^2) / 2 \hbar}
# \end{align}
#
# These two equations be expressed in polar coordinate as:
#
# \begin{align}
# \psi_1(r, \theta)=C_1 r e^{i\theta} e^{-m\omega r^2/2\hbar}\\
# \psi_2(r, \theta)=C_1 r e^{-i\theta} e^{-m\omega r^2/2\hbar}
# \end{align}
#
# Now it's clear from these two wave-functions, we can tell these are two modes with angular momentum quantum number equal to one.
#
# For the $E=3$ excited states, we need to second excited state of the 1D wavefunction as the possible combination of $x$, $y$ components are: $(n_x=0, n_y=2)$, $(n_x=1, n_y=1)$ and $(n_x=2, n_y=2)$
#
# Then, the overall wavefunction for these cases are:
#
# \begin{align}
# \psi_{1}(x, y)&=C_1(1-2 \frac{m \omega x^{2}}{\hbar}) e^{-m \omega (x^{2}+y^2) / 2 \hbar}\\
# \psi_{2}(x, y)&=C_2 xy e^{-m \omega (x^{2}+y^2) / 2 \hbar}\\
# \psi_{3}(x, y)&=C_1(1-2 \frac{m \omega y^{2}}{\hbar}) e^{-m \omega (x^{2}+y^2) / 2 \hbar}
# \end{align}
# Similar argument as $E=2$ case, we can construct three norm-orthogonal wavefunctions(different angular momentum modes) by linear combination of these degenerate state functions:
#
# for $l=0$, we find:
# \begin{align}
# \psi_{0}(x, y)&=\left[C1+C_1(x^2+y^2)\right] e^{-m \omega (x^{2}+y^2)/2\hbar}
# \end{align}
# or in polar coordinate:
# \begin{align}
# \psi_{0}(r,\theta)&=\left[C1+C_1 r^2\right] e^{-m \omega r^2/2\hbar}
# \end{align}
# for $l=1$, there is no way we can construct a wavefunction like in the $E=2$ case that yield the desired angular momentum since all the coefficients are quadratic.
#
# for $l=2$
#
# \begin{align}
# \psi_{\pm}(x, y)&=C_3(x\pm iy)^2 e^{-m \omega (x^{2}+y^2) / 2 \hbar}
# \end{align}
# or in polar coordinate:
# \begin{align}
# \psi_{\pm}(r,\theta)&=C_3r^2e^{i2\theta} e^{-m \omega r^2 / 2 \hbar}
# \end{align}
#
# It's can be seen $\psi_0(r, \theta)$ has no angular momentum, while $\psi_{\pm}(r,\theta)$ are with angular momentum quantum number equal to $2$.
#
# Based on the above discussion, it might be found that: For $E=N$, if $N$ is odd, its degenerate states have single $l=0$ state, all other states are double degeneracy with the values of $l$ share the same parity(even). If $N$ is even, all its states are double degeneracy with values $l$ are odd.

# ## Cylindrical Coordinates
# To understand angular momentum better, it's convenient to present the Schrodinger equation in  2D polar coordinates, or Cylindrical coordinate using relations:
# $$
# r^2=x^2+y^2\\
# tan(\theta)=\frac{y}{x}
# $$
# Set $\omega_x=\omega_y=1$ for the rest of the discussion. Then the Schrodinger equation in polar coordinates:
# $$
# -\frac{\partial^{2} \psi}{\partial r^{2}}-\frac{1}{r} \frac{\partial \psi}{\partial r}-\frac{1}{r^{2}} \frac{\partial^{2} \psi}{\partial \theta^{2}}+r^{2} \psi=2 E \psi
# $$

# # 1D Harmonic Oscillator

from mmf_hfb.potentials import HarmonicOscillator
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate

h = HarmonicOscillator()

L = 10
N = 128
xs = np.linspace(0, L, N) - L/2
def fb(n, x):
    n = n+1
    k_n = n*np.pi/L
    if n%2 == 1:
        return (1/L)**0.5*np.cos(k_n*x)
    return (1/L)**0.5*np.sin(k_n*x)



for n in range(5):
    plt.plot(xs, h.get_wf(n=n, x=xs))


def overlap(m=0, n=0):
    def f(x):
        return h.get_wf(n=m, x=x).conj()*fb(n=n, x=x)
    return (sp.integrate.quad(f, -L/2, L/2)[0])**2


overlap(3, 0)

for m in range(5):
    str = f"$\psi_{m}(x)$"
    for n in range(5):
        str = f"{str}&{overlap(m=m,n=n):4.3}"
    print(str + "\\\\")
    print("\\hline")

plt.plot(xs, h.get_wf(n=0, x=xs))
plt.plot(xs, fb(n=0, x=xs))

plt.plot()
for n in range(6):
    plt.plot(xs, fb(n=n, x=xs))


