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
from nbimports import *            # Convenience functions like clear_output
# -

# # HFB Equations in 1D (BdG)
#
# Here we present some examples of solving the BdG equations (Hartree-Fock_Bogoliubov or HFB) for fermions in 1D.
#
# The 1D system is not a very good model for physics (mean-field theory works better in higher dimensions) but contains all of the ingredients of the 3D theory in a form that can be quickly simulated.
#
# <!-- END_TEASER -->

# # HFB Theory (BCS Superconductivity)

# ## <font color='green'>The Variational Mean-field Method</font>

# Here is how I tend to view the mean-field approximation.  It is not the only way of thinking about things, but satisfies my notion of being well defined and general.  It is based on the following theorem from [Feynman:1998]:
#
# **Theorem:** The thermodynamic potential $F$ of a given system described by the Hamiltonian $\op{H}$ is bounded:
#
# $$
#   F \leq F_0 + \braket{\op{H} - \op{H}_0}_0
# $$
#
# where $F_0$ is the thermodynamic potential of the system described by the Hamiltonian $\op{H}_0$ and the average $\braket{}_0$ is performed with respect to the thermal ensemble of $\op{H}_0$:
#
# $$
#   \braket{\op{A}}_0 = \frac{\Tr\bigl[\op{A}e^{-\beta\op{H}_0}\bigr]}{\Tr e^{-\beta\op{H}_0}}
# $$
#
# What this means is that we can choose any Hamiltonian $\op{H}_0$ for which we can exactly solve the problem, and use it to obtain a variational upper bound on the thermodynamic potential $F$.  A general strategy is thus to introduce some solvable Hamiltonian $\op{H}_0$ that depends on some parameters, then choose these parameters so as to minimize the right-hand size of the previous equation.
#
# [Feynman:1998]: http://search.perseusbooksgroup.com/book/paperback/statistical-mechanics/9780201360769 'Richard P. Feynman, "Statistical Mechanics: A Set of Lectures", (1998)'

# ## Alternative Formulation

# Another solution to this problem has a nice formulation in terms of an energy density functional $E[\mat{R}]$ as a function of the full density matrix $\mat{R}$ which we may express in terms of the minimization problem:
#
# $$
#   \min_\mat{R} \Bigl\{
#     E(\mat{R}) + T \Tr \bigl[\mat{R}\ln \mat{R}  + (\mat{1} - \mat{R})\ln(\mat{1}-\mat{R})\bigr]
#     \Bigr\}, \qquad
#     \Tr \mat{R} = 1, \qquad
#     \mat{R} = \mat{R}^\dagger.
# $$
#
# The solution is
#
# $$
#   \mat{R} = f_\beta(\mat{H}) = \frac{1}{1+e^{\beta \mat{H}}}, \qquad
#   \beta = \frac{1}{k_B T}, \qquad
#   \mat{H} = \frac{\delta E[\mat{R}]}{\delta \mat{R}^T}.
# $$
#
# ## Regularization
#
# This provides a very simple formulation, but poses an implementation issue when attempting to regularize the theory.

# # <font color='orange'>BCS Theory</font>

# We start with standard BCS theory, which is the HFB approximation to the following family of Hamiltonions:
#
# $$
#   \op{H} = \sum_{k} \left(
#     E^{a}_{k}\op{a}_{k}^\dagger\op{a}_{k}
#     +
#     E^{b}_{k}\op{b}_{k}^\dagger\op{b}_{k}
#   \right)
#   +
#   \int\d{x}\d{y}\; V(x-y)\op{n}_{a}(x)\op{n}_{b}(y).
# $$
#
# This describes two species of particle ($a$, and $b$) interacting with a potential $V(x-y)$ (in 1D we have $V(x-y) = g\delta(x-y)$ in 1D).  We first applying the Feynman variational principle, considering a trial Hamiltonian of the form:
#
# $$
#   \op{H}_0 = \sum_{k>0} 
#   \begin{pmatrix}
#     \op{a}_{k}^\dagger &
#     \op{b}_{-k}\\    
#   \end{pmatrix}
#   \cdot
#   \begin{pmatrix}
#     E^{a}_{k} + \Sigma^a_k & \Delta_k\\
#     \Delta^*_{k} & -E^{b}_{-k} - \Sigma^b_{-k}   
#   \end{pmatrix}
#   \cdot
#   \begin{pmatrix}
#     \op{a}_{k}\\
#     \op{b}_{-k}^\dagger
#   \end{pmatrix}\\
#   =
#   \sum_{k>0}\Bigl(
#   (E^{a}_{k} + \Sigma^a_k)\op{a}_{k}^\dagger\op{a}_{k} +
#   (E^{b}_{k} + \Sigma^b_k)\op{b}_{k}^\dagger\op{b}_{k} + 
#   \Delta_k\op{a}_{k}^\dagger\op{b}_{-k}^\dagger +
#   \Delta^*_{k}\op{b}_{-k}\op{a}_{k}
#   - E^{b}_{k} - \Sigma^b_k\Bigr)
# $$
#
# Here we will consider the self-energies $\Sigma$ and pairing gap $\Delta$ as variational parameters.
#
# Notice that in the ensemble of this quadratic Hamiltonian the expectation of the interaction looks like this, after Wick contracting:
#
# $$
#   g\braket{\op{n}_a\op{n}_b} = 
#   \overbrace{
#     g\braket{\op{a}^\dagger\op{a}}\braket{\op{b}^\dagger\op{b}}
#   }^{\text{Hartree}}
#   - \underbrace{
#   g\braket{\op{a}^\dagger\op{b}}\braket{\op{b}^\dagger\op{a}}
#   }_{\text{Fock}}
#   + \overbrace{
#     g\braket{\op{a}^\dagger\op{b}^\dagger}\underbrace{\braket{\op{b}\op{a}}}_{\nu}
#   }^{\text{Pairing}}.
# $$
#
# The opposite sign on the Fock term generally means that if there are pairing interactions which condense (for attractive interactions $g < 0$), then the Fock channel will be repulsive and will not condense.  In the following, we thus neglect the Fock channel to simplify the discussion.
#
# These signs give us the following identifications after minimizing:
#
# $$
#   \nu = \braket{\op{b}\op{a}}, \qquad
#   n_a = \braket{\op{a}^\dagger\op{a}}, \qquad
#   n_b = \braket{\op{b}^\dagger\op{b}}, \qquad
#   \Delta = g\nu, \qquad
#   \Sigma_{a,b} = gn_{b,a}.
# $$

# ## <font color='green'>Diagonalization</font>

# Consider diagonalizing the quadratic Hamiltonian
#
# $$
#   \op{H}_0 = \op{\Psi}^\dagger \cdot \mat{M} \cdot \op{\Psi}
#            = \op{C}^\dagger\cdot\diag{E}\cdot\op{C}, \qquad
#   \mat{U}^\dagger\cdot\mat{M}\cdot\mat{U} = \diag(E), \qquad
#   \op{\Psi} = \mat{U} \cdot \op{C}.
# $$
#
# This Hamiltonian is diagonal in terms of the quasi-particle operators $\op{c}_n$ and so the vacuum state will occupy these (independent) levels with probability $f_\beta(E_n)$:
#
# $$
#   \braket{\op{c}_m^\dagger\op{c}_n} = \delta_{mn}f_\beta(E_n).
# $$
#
# We can arrange this in matrix form:
#
# $$
#   \braket{\op{C}\op{C}^\dagger}
#   = \mat{1} - \braket{\op{C}^\dagger\op{C}}
#   = \mat{1} - f_\beta\bigl(\diag(E)\bigr),\\
#   \braket{\op{\Psi}\op{\Psi}^\dagger} 
#     = \mat{U}\cdot\braket{\op{C}\op{C}^\dagger}\cdot\mat{U}^\dagger
#     = \mat{1} - f_\beta(\mat{M})
#     = \mat{1} - \mat{R}.
# $$
#
# We can arrange this in matrix form by noting that $f_\beta(E) + f_\beta(-E) = 1$:
#
# $$
#   \braket{\op{C}\op{C}^\dagger}
#   = \mat{1} - \braket{\op{C}^\dagger\op{C}}
#   = \mat{1} - f_\beta\bigl(\diag(E)\bigr) = f_\beta\bigl(-\diag(E)\bigr),\\
#   \braket{\op{\Psi}\op{\Psi}^\dagger} 
#     = \mat{U}\cdot\braket{\op{C}\op{C}^\dagger}\cdot\mat{U}^\dagger
#     = f_\beta(-\mat{M}) = \mat{1} - f_\beta(\mat{M})
#     = \mat{1} - \mat{R}.
# $$
#
# For standard BCS theory, we have for example
#
# $$
#   \op{\Psi} = \begin{pmatrix}
#     \op{a}\\
#     \op{b}^\dagger
#   \end{pmatrix}, \qquad
#   \braket{\op{\Psi}\op{\Psi}^\dagger} = 
#   \mat{1} - \mat{R} = 
#   \begin{pmatrix}
#     \braket{\op{a}\op{a}^\dagger} & \braket{\op{a}\op{b}}\\
#     \braket{\op{b}^\dagger\op{a}^\dagger} &
#     \braket{\op{b}^\dagger\op{b}}
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#     \mat{1}-\mat{n}_a & -\nu\\
#     -\nu^\dagger & \mat{n}_{b}^*
#   \end{pmatrix},\\
#   \mat{R} = 
#   \begin{pmatrix}
#     \mat{n}_a & \nu\\
#     \nu^\dagger & \mat{1}-\mat{n}_{b}^*
#   \end{pmatrix}.  
# $$

# Note: although working with the density matrix is convenient, it is a bit tricky to work with when using a cutoff.  In particular, one must use the following procedure when trying to extract the sub-matrices with a cutoff $f(E) = f_\beta(E)f_c(\abs{E})$ where $f_c(\abs{E}) = 0$ for $\abs{E}>E_c$:
#
# * $\mat{n}_a = [f(\mat{M})]_{00}$,
# * $\mat{n}_b = [f(-\mat{M})]_{11}$,
# * $\mat{\nu} = \frac{[f(\mat{M})]_{01} - [f(-\mat{M})]_{10}^\dagger}{2}$.
#
# These subtleties result from the cutoff breaking the relationship $f_\beta(E) + f_\beta(-E) = 1$.

# ### On the sign of $\nu$

# There is some confusion in the literature about the sign of $g$ and $\nu$, etc.  Here we take the following convention:
#
# * Attractive interactions: $g < 0$ (required for pairing):
#
#   $$
#     \mathcal{E} = \cdots + g\nu^\dagger\nu + \cdots.
#   $$
#   
#   Agrees with (73) from book chapter.
#   
# * If $\Delta >0$, then $\nu < 0$.
# * $\Delta = g \nu$.  This is in error in many places: for example(74) and (75) of the book chapter have an error here.  The functional should have:
#
#   $$
#     \kappa = \frac{\hbar^2}{2m}\tau_+ + \Delta^\dagger \nu.
#   $$
#   
# * In terms of the $u$ and $v$ factors:
#   $$
#     \mat{U} = \begin{pmatrix}
#       u_0(r) & u_1(r) & \cdots \\
#       v_0(r) & v_1(r) & \cdots
#     \end{pmatrix},\\
#     \nu(r) = \sum_{n}u_n(r)v^*_n(r)f_\beta(E_n) = \sum_{n}u_n(r)v^*_n(r)\frac{f_\beta(E_n) - f_\beta(-E_n)}{2}
#   $$
#   
#   This agrees with (80) of the book chapter, but disagrees with (72).
# * The integrals give:
#
#   $$
#     \nu = -\int\frac{\d^{d}{k}}{(2\pi)^d} 
#     \frac{\Delta}
#          {2\sqrt{\epsilon_+^2 + \abs{\Delta}^2}}\bigl(f_\beta(\omega_-) - f_\beta(\omega_+)\bigr)
#   $$

import numpy as np
from scipy.linalg import expm, inv
delta = 1+2j
M = np.array([[0, delta], 
              [delta.conjugate(), -0]])
E, UV = np.linalg.eigh(M)
T = 0.1
R = inv(np.eye(2) + expm(M/T))
u, v = UV[0, :], UV[1, :]
def f(E): return 1./(1+np.exp(E/T))
nu = sum(u*v.conj()*(f(E) - f(-E))/2)
assert np.allclose(R[0,1], nu)
assert np.allclose(R[1,0], nu.conj())
g = nu/delta
assert g < 0


# ### HFB

# A quick note about the HFB equations in nuclear theory.  Here we have simplified our discussion considering only the Hartree and Bogoliubov terms.  If one wants to generalize this to allow for all possible pairing, then one should write the following:
#
# $$
#   \op{\Psi} = \begin{pmatrix}
#     \op{a}\\
#     \op{b}\\    
#     \op{a}^\dagger\\
#     \op{b}^\dagger
#   \end{pmatrix}, \\
#   \begin{aligned}
#   \braket{\op{\Psi}\op{\Psi}^\dagger} = 
#   \mat{1} - \mat{\mathcal{R}} &= 
#   \begin{pmatrix}
#     \braket{\op{a}\op{a}^\dagger} & 
#     \braket{\op{a}\op{b}^\dagger} &
#     \braket{\op{a}\op{a}} & 
#     \braket{\op{a}\op{b}}
#     \\
#     \braket{\op{b}\op{a}^\dagger} & 
#     \braket{\op{b}\op{b}^\dagger} &
#     \braket{\op{b}\op{a}} & 
#     \braket{\op{b}\op{b}}
#     \\
#     \braket{\op{a}^\dagger\op{a}^\dagger} & 
#     \braket{\op{a}^\dagger\op{b}^\dagger} &
#     \braket{\op{a}^\dagger\op{a}} & 
#     \braket{\op{a}^\dagger\op{b}}
#     \\
#     \braket{\op{b}^\dagger\op{a}^\dagger} & 
#     \braket{\op{b}^\dagger\op{b}^\dagger} &
#     \braket{\op{b}^\dagger\op{a}} & 
#     \braket{\op{b}^\dagger\op{b}}
#   \end{pmatrix}\\
#   &=\begin{pmatrix}
#     1-\mat{n}_a & \braket{\op{a}\op{b}^\dagger} & & -\mat{\nu}\\
#     \braket{\op{b}\op{a}^\dagger} & 1-\mat{n}_b & \mat{\nu}^T & \\
#     & \mat{\nu}^* & \mat{n}_a^* & \braket{\op{a}^\dagger\op{b}} \\
#     -\mat{\nu}^\dagger & & \braket{\op{b}^\dagger\op{a}} & \mat{n}_b^*
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#     \mat{1}-\mat{\rho} & -\mat{\kappa}\\
#     \mat{\kappa}^* & \mat{\rho}^*
#   \end{pmatrix},
#   \end{aligned}\\
#   \mat{\mathcal{R}} =
#   \begin{pmatrix}
#     \mat{\rho} & \mat{\kappa}\\
#     -\mat{\kappa}^* & \mat{1}-\mat{\rho}^*
#   \end{pmatrix},\qquad
#   \mat{\rho} = 
#   \begin{pmatrix}
#     \mat{n}_a & -\braket{\op{a}\op{b}^\dagger}\\
#     -\braket{\op{b}\op{a}^\dagger} & \mat{n}_b
#   \end{pmatrix}, \qquad
#   \mat{\kappa} = 
#   \begin{pmatrix}
#     & \mat{\nu}\\
#     -\mat{\nu}^T &
#   \end{pmatrix}.
# $$
#
# The conventions for the matrix definitions have some transposes:
#
# $$
#   \mat{n}_a^T = \mat{n}_a^* =\braket{\op{a}^*\op{a}^T}, \qquad
#   [\mat{n}_a]_{mn} = \braket{\op{a}^\dagger_{n}\op{a}_m},\\
#   \mat{\nu}^T = \braket{\op{b}\op{a}^T}, \qquad
#   [\mat{\nu}]_{mn} = \braket{\op{b}_{n}\op{a}_m}.
# $$

# ## <font color='green'>Homogeneous Matter</font>

# For homogeneous matter, momentum $p = \hbar k$ is a good quantum number, and so we can label all eigenstates of the system with the wavenumber $k$.  To describe two component superfluids, we introduce the quadratic Hamiltonian
#
# $$
#   \op{H}_0(k) = \op{\Psi}^\dagger_{k} \cdot \mat{H}_{k} \cdot \op{\Psi}_{k}
# $$
#
# where $\op{\Psi}_k = (\op{a}_{k}, \op{b}_{-k}^\dagger)$ and 
#
# $$
#   \mat{H}_{k} = \begin{pmatrix}
#     A & \Delta \\
#     \Delta^\dagger & -B
#   \end{pmatrix}
# $$
#
# with $A$ and $B$ are the dispersion relationships for the two particles:
#
# $$
#   A = \frac{\hbar^2k^2}{2m_A} - \mu_A.
# $$
#
# This quadratic Hamiltonian can be "solved" by diagonalizing the matrix with a unitary matrix $\mat{U}$:
#
# $$
#   \mat{U}^\dagger\mat{H}\mat{U} = \diag(E_-, E_+).
# $$
#
# Once this is done, we end up with a diagonal Hamiltonian
#
# $$
#   \op{H}_0 = \op{C}^\dagger \begin{pmatrix}
#     E_-\\
#     & E_+
#   \end{pmatrix}
#   \op{C}
#   = E_- \op{c}_-^\dagger\op{c}_- + E_+ \op{c}_+^\dagger\op{c}_+
# $$
#
# where $\op{c}_{\pm}$ are the creation operators for fermionic "quasiparticles", which are linear combinations of the original particles $\op{a}$ and $\op{b}^\dagger$.  Since the Hamiltonian is now diagonal, the thermal ensemble can be simply presented in terms of the Fermi distribution and inverse temperature $\beta = 1/k_BT$:
#
# $$
#   f_{\beta}(E) = \frac{1}{1+e^{\beta E}}.
# $$
#
# Note: The energy $E$ here will contain the chemical potential $E-\mu$.

# ## <font color='green'>Inhomogeneous Systems</font>

# For inhomogeneous systems, the same approach works, but one cannot use momentum as a good quantum number.  Thus, the Hamiltonian $\mat{H}$ will simply need to be diagonalized into a complete set of states:
#
# $$
#   \mat{H}_0\cdot \vect{U}_{\pm n} = \vect{U}_{\pm n}E_{\pm n}, \qquad
#   \vect{U}_{\pm n} = 
#   \begin{pmatrix}
#     u_{\pm n} \\
#     v^*_{\pm n}
#   \end{pmatrix}.
# $$
#
# The columns of the matrix $\mat{U}$ are these eigenvectors.

# # Homogeneous Matter

# ## <font color='orange'> Exact Solution (Gaudin)</font>

# The homogeneous equations for a two-component Fermi gas with attractive short-range interactions $-v_0 \delta(x-y)$ (note that $v_0 = -g$ here) are known exactly from the Gaudin equations [Gaudin:1967] which give the numerical solution [Casas:1991]:
#
# $$
#   F(x) = 2 - \frac{K}{\pi}\int_{-1}^{1} dy\; \frac{F(y)}{1+K^2(x-y)^2}\\
#   \frac{1}{\lambda}= \frac{K}{\pi}\int_{-1}^{1}dy\;F(y)\\
#   \frac{E_0(N)/N}{\abs{E_0(2)/2}} = -1 + \frac{4}{\pi}K^3\lambda\int_{-1}^{1}dy\; y^2F(y),
# $$
#
# where $E_0(2) = -mv_0^2/4\hbar^2$ is the two-body binding energy and $\lambda = mv_0/\hbar^2/\rho$ is the dimensionless coupling constant. This solution is only valid in homogeneous matter, but should give some idea of how well mean-field techniques work.  A direct comparison with the BCS solution [Quick:1993] shows that there are significant disagreements in the strong-coupling regime $0 < \lambda ^{-1} < 1$.
#
# [Gaudin:1967]: http://dx.doi.org/10.1016/0375-9601(67)90193-4 'M. Gaudin, "Un Systeme a Une Dimension de Fermions en Interaction", Phys. Lett. A24(1), 55-56 (1967)'
# [Casas:1991]: http://dx.doi.org/10.1103/PhysRevA.44.4915 'M. Casas, C. Esebbag, A. Extremera, J. M. Getino, M. de Llano, A. Plastino, and H. Rubio, "Cooper pairing in a soluble one-dimensional many-fermion model", Phys. Rev. A 44(8), 4915--4922 (1991)'
# [Quick:1993]: http://dx.doi.org/10.1103/PhysRevB.47.11512 'R. M. Quick, C. Esebbag, and M. de Llano, "BCS theory tested in an exactly solvable fermion fluid", Phys. Rev. B 47, 11512--11514 (1993)'
#
# To compare results note that [Quick:1993] and [Casas:1991] use the dimensionless $\epsilon = E(N)/\abs{E_0(N)}$ where $E_0(N)$ is the energy of $N$ particles in the zero-density limit.  I.e. where one has $N/2$ dimers, each with energy $E_0(2)$.  Thus $E_0(N) = E_0(2) N/2$.  The interaction strength is expressed in terms of $\lambda^{-1} \propto n$.

# +
# %pylab inline --no-import-all

def gaudin(K, N=64, tol=1e-6):
    """Return `E, lam` for exact Gaudin solution in box"""
    x = np.linspace(-1, 1, N)
    
    F0 = 0*x
    F1 = 2 - F0

    n = 0
    while abs(F0-F1).max() > tol:
        F0 = F1
        F1 = 2.0 - K/np.pi * np.trapz(
            F0[None, :]/(1+K**2*(x[:,None] - x[None,:])**2), x, axis=1)
        n += 1
    lam = np.pi / K / np.trapz(F1, x)
    e = -1 + 4./np.pi*K**3*lam * np.trapz(x**2*F1, x)
    return e, lam


# -

gaudin(1.0)

# Test the series expansion
e, lam = gaudin(0.001)
(e - (-1.0 + np.pi**2/12/(lam)**2 + np.pi**2/24/(lam)**3))

# ## <font color='orange'>The Thomas-Fermi Approximation (TF) or Local Density Approximation (LDA) at T=0</font>

# The solution for homogeneous states within HFB theory is called the Local Density Approximation (LDA).  For non-uniform systems, one treats each region of space as if it were locally homogeneous, using this solution with a spatially dependent chemical potential.  Here we present the equations for the LDA for a homogeneous gas of equal numbers of two fermionic species interacing with a delta-function interactions $V(x-y) = -v_0\delta(x-y)$.  The gap equations have the form (see for example [Quick:1993]):
#
# $$
#   \Delta = v_0 \nu = \frac{v_0}{2}\int\frac{\d{k}}{2\pi}\;\frac{\Delta}{\sqrt{\epsilon_+^2 + \Delta^2}},\quad
#   n_+ = \frac{N_a + N_b}{L} = \int\frac{\d{k}}{2\pi}
#     \left[1 - \frac{\epsilon_+}{\sqrt{\epsilon_+^2 + \abs{\Delta}^2}}\right],\quad
#   \epsilon_+ = \frac{\hbar^2k^2}{2m} \overbrace{- \frac{1}{2}n_+v_0}^{\Sigma} - \mu = \frac{\hbar^2k^2}{2m} - \mu_{\text{eff}},\\
#   \frac{E}{L} = 
#   \int\frac{\d{k}}{2\pi} \frac{\hbar^2k^2}{2m}
#     \left[1 - \frac{\epsilon_+}{\sqrt{\epsilon_+^2 + \abs{\Delta}^2}}\right]
#   - v_0(n_an_b + \nu^\dagger\nu)
#   = 
#   \int\frac{\d{k}}{2\pi} \frac{\hbar^2k^2}{2m}
#     \left[1 - \frac{\epsilon_+}{\sqrt{\epsilon_+^2 + \abs{\Delta}^2}}\right]
#   - \frac{v_0 n_+^2}{4}
#   - \frac{\abs{\Delta}^2}{v_0}.
# $$
#
# where $\mu = \mu_0 - V_\text{ext}$ is the effective local chemical potential.  These are easily solved by choosing $\Delta$ and $\mu_\mathrm{eff}$, then integrating to determine $v_0$, $n_+$, and $\mu$.
#
# [Quick:1993]: http://dx.doi.org/10.1103/PhysRevB.47.11512 'R. M. Quick, C. Esebbag, and M. de Llano, "BCS theory tested in an exactly solvable fermion fluid", Phys. Rev. B 47, 11512--11514 (1993)'

# <font color='red'>** To perform these integrals numerically, we need to deal with potential singularities.  As $\Delta \rightarrow 0$, there is a potential singularity when $\epsilon_+(k) = 0$ i.e. when $k = k_F$.  This is easily dealt with by breaking the integrand up into regions $(0, k_F) \cup (k_F, \infty)$.  One might occasionally have problems with the energy integral at large $k$.** </font>

# ## <font color='green'>LDA at finite T</font>

# We now generalize these results to finite temperature and allow for asymmetry.  Here we will obtain occupation numbers described by the Fermi distribution function
#
# $$
#   f(E) \equiv f_\beta(E) = \frac{1}{1 + e^{\beta E}} = \frac{1-\tanh(\beta E/2)}{2}, 
#   \qquad \beta = \frac{1}{k_B T}, \qquad
#   f(E) + f(-E) = 1,
# $$
#
# The BdG equations follow from the single-particle Hamiltonian (assuming homogeneous states)
#
# $$
#   \mat{M} = \begin{pmatrix}
#     \epsilon_k^{\uparrow} & \Delta\\
#     \Delta^\dagger & -\epsilon_{-k}^{\downarrow}
#   \end{pmatrix}
#    = \mat{U}\cdot
#    \begin{pmatrix}
#      \omega_+ & \\
#      & \omega_-
#    \end{pmatrix}
#    \cdot
#    \mat{U}^{\dagger},\qquad
#    \mat{U} = \begin{pmatrix}
#      u_+ & u_-\\
#      v_+^* & v_-^*
#    \end{pmatrix}.
# $$
#
# where
#
# $$
#   \epsilon_k^{\uparrow} = \frac{k^2}{2m}  - \mu_\uparrow, \qquad
#   \epsilon_{-k}^{\downarrow} = \frac{k^2}{2m} - \mu_\downarrow.
# $$
#
# This has eigenvectors and eigenvalues:
#
# $$
#   \omega_{\pm} = \epsilon^{-}_k \pm E_k, \qquad
#   E_k = \sqrt{(\epsilon^{+}_k)^2 + \Delta^2}, \qquad
#   \epsilon^{\pm}_{k} = \frac{\epsilon^\uparrow_k \pm \epsilon^\downarrow_{-k}}{2},\\
#   \epsilon^{\uparrow}_k = K^{\uparrow}_{k} - v_0n^{\downarrow} - \mu^{\uparrow} 
#                       = K^{\uparrow}_{k} - \mu^{\uparrow}_{\text{eff}},\\
#   \epsilon^{\downarrow}_{-k} = K^{\downarrow}_{-k} - v_0n^{\uparrow} - \mu^{\downarrow} 
#                       = K^{\downarrow}_{-k} - \mu^{\downarrow}_{\text{eff}},\\
#   u_{\pm} = e^{\I\phi}\cos\theta_\pm, \qquad v_{\pm} = \sin\theta_\pm, \qquad
#   \Delta = e^{\I\phi}\abs{\Delta}, \qquad
#   \tan\theta_\pm = \frac{\pm E - E_+}{\abs{\Delta}},\\
#   \abs{u^{\pm}_k}^2 = \frac{1\pm\epsilon^+_k/E_k}{2}, \qquad
#  \abs{v^{\pm}_k}^2 = \frac{1\mp\epsilon^+_k/E_k}{2},
# $$

# This is testing the previous equations
import numpy as np
np.random.seed(1)
N = 100
a = np.random.random(N) - 0.5
b = np.random.random(N) - 0.5
d = np.random.random(N) + 1j*np.random.random(N) - 0.5 - 0.5j
e_m, e_p = (a-b)/2, (a+b)/2
E = np.sqrt(e_p**2 + abs(d)**2)
es = np.array([e_m-E, e_m+E])
thetas = np.arctan2([-E-e_p, E-e_p], abs(d))
us = np.exp(1j*np.angle(d))*np.cos(thetas)
vs = np.sin(thetas)
assert np.allclose(a*us[0] + d*vs[0], es[0]*us[0])
assert np.allclose(d.conj()*us[0] - b*vs[0], es[0]*vs[0])

# Here is what the dispersions $\omega_{\pm}(k)$ look like:

# +
# %pylab inline --no-import-all
from ipywidgets import interact
def f(E, T):
    """Fermi distribution function"""
    T = max(T, 1e-32)
    return 1./(1+np.exp(E/T))


@interact(delta=(0, 1, 0.1), 
          mu_eF=(0, 2, 0.1),
          dmu=(-0.4, 0.4, 0.01),
          T=(0, 0.1, 0.01),
          dq=(0, 1, 0.1)
         )
def go(delta=0.1, mu_eF=1.0, dmu=0.0, dq=0, T=0.02):
    plt.figure(figsize=(12, 8))

    k = np.linspace(0, 1.4, 1000)
    hbar = m = kF = 1.0
    eF = (hbar*kF)**2/2/m
    mu = mu_eF*eF
    #dmu = dmu_delta*delta
    mu_a, mu_b = mu + dmu, mu - dmu
    e_a, e_b = (hbar*k+dq)**2/2/m - mu_a, (hbar*k-dq)**2/2/m - mu_b
    e_p, e_m = (e_a + e_b)/2, (e_a - e_b)/2
    E = np.sqrt(e_p**2+abs(delta)**2)
    w_p, w_m = e_m + E, e_m - E
    
    # Occupation numbers
    f_p = 1 - e_p/E*(f(w_m, T) - f(w_p, T))
    f_m = f(w_p, T) - f(-w_m, T)
    f_a, f_b = (f_p+f_m)/2, (f_p-f_m)/2

    plt.subplot(211);plt.grid()
    plt.plot(k/kF, f_a, label='a')
    plt.plot(k/kF, f_b, '--', label='b',);plt.legend()
    plt.ylabel('n')
    plt.subplot(212);plt.grid()
    plt.plot(k/kF, w_p/eF, k/kF, w_m/eF)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('$k/k_F$')
    plt.ylabel(r'$\omega_{\pm}/\epsilon_F$')
    plt.axhline(0, c='y')


# -

@interact(delta=(0, 1, 0.1), 
          mu_eF=(0, 2, 0.1),
          dmu=(-0.4, 0.4, 0.01),
          T=(0, 0.1, 0.01))
def fn(delta=0.1, mu_eF=1.0, dmu=0.0, T=0.02):
    plt.figure(figsize=(12, 8))

    k = np.linspace(0, 1.4, 1000)
    hbar = m = kF = 1.0
    eF = (hbar*kF)**2/2/m
    mu = mu_eF*eF
    #dmu = dmu_delta*delta
    mu_a, mu_b = mu + dmu, mu - dmu
    e_a, e_b = (hbar*k)**2/2/m - mu_a, (hbar*k)**2/2/m - mu_b
    e_p, e_m = (e_a + e_b)/2, (e_a - e_b)/2
    E = np.sqrt(e_p**2+abs(delta)**2)
    w_p, w_m = e_m + E, e_m - E
    
    # Occupation numbers
    #f_p = 1 - e_p/E*(f(w_m, T) - f(w_p, T))
    #f_m = f(w_p, T) - f(-w_m, T)
    #f_a, f_b = (f_p+f_m)/2, (f_p-f_m)/2
    u2, v2 = 0.5*(1+e_p/E), 0.5*(1-e_p/E)
    f_a = u2*f(w_p, T)+v2*f(w_m, T)
    f_b = u2*f(-w_m, T)+ v2*f(-w_p, T)
    plt.subplot(211);plt.grid()
    plt.plot(k/kF, f_a, label='a')
    plt.plot(k/kF, f_b, '--', label='b',);plt.legend()
    plt.ylabel('n')
    plt.subplot(212);plt.grid()
    plt.plot(k/kF, w_p/eF, k/kF, w_m/eF)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('$k/k_F$')
    plt.ylabel(r'$\omega_{\pm}/\epsilon_F$')
    plt.axhline(0, c='y')


x = sympy.var('x', positive=True)
(abs(us[0])**2).subs(k,1/x).series(x, n=8 ).subs(x, 1/k)

(abs(us[0])**2).series

# **if we may assume** 
# $
# \epsilon_k^{\uparrow} =\epsilon_{-k}^{\downarrow}
# $, **then** $\omega_- = -\omega_+$
#
#  **Let $n_+$ is the total particle number, while $n_-$ is the number difference**
#  
# \begin{align}
#   n_+ 
#   &= n(\epsilon_+)+n(\epsilon_-)\\
#   &= \int\frac{\d{k}}{2\pi}\left(
#     1 - \frac{\epsilon^+_k}{2E_k}
#     \left[\tanh(\beta\omega_+/2) - \tanh(\beta\omega_-/2)\right]
#   \right)\\
#   &= \int\frac{\d{k}}{2\pi}\left(
#     1 - \frac{\epsilon^+_k}{E_k}
#     \bigl(f(\omega_-) - f(\omega_+)\bigr)\right),\\
#   n_- 
#   &= n(\epsilon_+)-n(\epsilon_-)\\
#   &= \int\frac{\d{k}}{2\pi}\left(
#       - \frac{1}{2}\left[
#     \tanh(\beta\omega_+/2) + \tanh(\beta\omega_-/2)
#   \right]\right)\\
#   &= \int\frac{\d{k}}{2\pi}
#   \bigl(f(\omega_+) - f(-\omega_-)\bigr),\\
#     \Delta 
#   &= \frac{v_0}{2}\int \frac{\d{k}}{2\pi}\frac{\Delta}{E_k}\frac{\left[\tanh(\beta\omega_+/2) - \tanh(\beta\omega_-/2)\right]}{2},\\
#   &= \frac{v_0}{2}\int \frac{\d{k}}{2\pi}\frac{\Delta}{E_k}\bigl(f(\omega_-)-f(\omega_+)\bigr),\\
#   \frac{1}{v_0} 
#   &= \frac{1}{2}\int \frac{\d{k}}{2\pi}\frac{1}{E_k}\bigl(f(\omega_-)-f(\omega_+)\bigr).
# \end{align}

# To match our review paper...
#
# $$
#   n_+ = 
#   1 - \frac{\epsilon^+_k}{E_k}\bigl(f(\omega_-) - f(\omega_+)\bigr)\\
#   = 
#   1 - \frac{\epsilon^+_k}{E_k}(1 - 2\bar{f}(E_k))\\
#   = 
#   1 - \frac{\epsilon^+_k}{E_k} 
#   + 2\frac{\epsilon^+_k}{E_k}\bar{f}(E_k))
# $$
# **I changed the first $\omega_-$ to $\omega_+$ for $\bar{f}(E_k)$:**
#
# $$
#   \bar{f}(E_k) = \frac{f(E_k + \epsilon_+) + f(E_k - \epsilon_-)}{2}\\
#   1 - 2\bar{f}(E_k)
#   = f(\omega_-) - f(\omega_+)
# $$

# ## <font color='green'>3D case</font>

# In 3D case, all integration should be done in 3 dimensions. $d^3k$ is the volumn. If the $k_z$ in the $z$ direction is different from other tow dimensions($k_\perp$ ), we pick the cylindrical coordinate: $d^3k$ = $2{\pi}k_{\perp}d{k_z}dk_\perp $(where the integration over $d\theta$ is carried out and equal to $2\pi$), 
# \begin{align}
#   n_+ 
#   &= n(\epsilon_+)+n(\epsilon_-)\\
#   &= \int\frac{\d^3{k}}{8{\pi}^3}\left(
#     1 - \frac{\epsilon^+_k}{2E_k}
#     \left[\tanh(\beta\omega_+/2) - \tanh(\beta\omega_-/2)\right]
#   \right)\\
#   &= \int\frac{k_{\perp}d{k_z}dk_{\perp}}{4{\pi}^2}\left(
#     1 - \frac{\epsilon^+_k}{E_k}
#     \bigl(f(\omega_-) - f(\omega_+)\bigr)\right),\\
#   n_- 
#   &= n(\epsilon_+)-n(\epsilon_-)\\
#   &= \int\frac{k_{\perp}d{k_z}dk_{\perp}}{4{\pi}^2}\left(
#       - \frac{1}{2}\left[
#     \tanh(\beta\omega_+/2) + \tanh(\beta\omega_-/2)
#   \right]\right)\\
#   &= \int\frac{k_{\perp}d{k_z}dk_{\perp}}{4{\pi}^2}
#   \bigl(f(\omega_+) - f(-\omega_-)\bigr),\\
#   \Delta 
#   &= -\frac{g}{2}\int \frac{\d^3{k}}{8\pi^3}\frac{\Delta}{E_k}\frac{\left[\tanh(\beta\omega_+/2) - \tanh(\beta\omega_-/2)\right]}{2},\\
#   &= -\frac{g}{2}\int \frac{k_{\perp}d{k_z}dk_{\perp}}{4{\pi}^2}\frac{\Delta}{E_k}\bigl(f(\omega_-)-f(\omega_+)\bigr),\\
#   -\frac{1}{g} 
#   &=\frac{m}{4\pi a}-\sum_{k}{{\frac{1}{2\epsilon^+_k}}}\\
#   &=\sum_{k}{\left[\frac{1}{2\epsilon^+_k}-\frac{1-2\bar{f}(E_k)}{2E_k}\right]}-\sum_{k}{{\frac{1}{2\epsilon^+_k}}} \qquad \text{ONLY VALID IF $\Delta \neq 0$!}\\
#   &=\sum_{k}{\left[-\frac{1-2\bar{f}(E_k)}{2E_k}\right]}\\
#   &=-\int\frac{k_{\perp}d{k_z}dk_{\perp}}{4{\pi}^2}{\left[\frac{1-f(E_k + \epsilon_+) + f(E_k - \epsilon_-)}{2E_k}\right]}\\
# \end{align}

# ## <font color='orange'>Regularization for 3D</font>
# As discussed above, the converges of the gap equation is quite poor in 1D.  In 3D the situation is even worse and the gap equation diverges.  To deal with this, some sort of regularization condition is needed.  There are two common stratgies:
#
# 1. Fixed scattering length.  Here the approach is to replace the coupling constant $v_0$ with some physical observable such as the two-body scattering length.  The idea is to calculate the two-body scattering length $a_s$ using some convenient potential (with an appropriate regulator such as an energy cutoff $E_c$ or a lattice momentum cutoff $k_c$).  Then, adjust the coupling constant $v_0(E_c)$ as a function of this cutoff to hold the physical observable $a_s$ fixed.
#
# To relate with Braaten and Hammer (2006) [Physics Reports 428 (2006) 259 – 390], we identify $g_2 = -v_0$.  (Equations are numbered as in the paper.)  Next, calculate the s-wave scattering $a(g_2, \Lambda)$ as a function of the interaction strength and cutoff:
#
# \begin{gather}
#   \frac{1}{a} - \frac{2\Lambda}{\pi} = \frac{8\pi}{g_2} \tag{305}.
# \end{gather}
#
# Next, rearrange the gap equation in terms of $8\pi/g_2$:
#
# $$
#   \frac{8\pi}{g_2} = \frac{1}{a} - \frac{2\Lambda}{\pi} = -\frac{8\pi}{2}\int\frac{\d{k}^3}{(2\pi)^3}\;\frac{1}{\sqrt{\epsilon_+^2 + \Delta^2}}.
# $$
#
# Now substitute for your expression of $\Lambda$.  The divergences should cancel leaving a convergent equation that is valid in the limit of $\Lambda \rightarrow \infty$.  (Since we have not worked this out yet, we work backwards to cancel the divergence.)
#
# $$
#   \frac{1}{a} = \frac{2\Lambda}{\pi} - \frac{8\pi}{2}\int_{k < k_c}\frac{\d{k}^3}{(2\pi)^3}\;\frac{1}{\sqrt{\epsilon_+^2 + \Delta^2}}
#   = \frac{8\pi}{2}\int_{k < k_c}\frac{\d{k}^3}{(2\pi)^3}\;\left(
#   \frac{1}{E_+} - \frac{1}{\sqrt{\epsilon_+^2 + \Delta^2}}
#   \right)\\
#   E_+ = \frac{\hbar^2k^2}{2m}, \qquad
#   \Lambda = 2\pi^2\int_{k< k_c}\frac{\d{k}^3}{(2\pi)^3}\;\frac{1}{E_+}
#           = 2\pi^2\int_{k< k_c}\frac{\d{k}^3}{(2\pi)^3}\;\frac{2m}{\hbar^2k^2}
#           = \frac{2m}{\hbar^2}k_c
# $$
#
# 2. The second improvement due to Aurel Bulgac is to note that you can do the integrals with $1/\epsilon_+ = 1/(E_+ - \mu + \I 0^+)$ instead of $1/E_+$, which improves the order of convergence further.  Basically, this redefines the cutoff in terms of $k_c$ which satisfies:
#
#    $$
#      E_c = \frac{\hbar^2k_c^2}{2m} - \mu
#    $$
#
#    instead of $E_c = \hbar^2k_c^2/2m$ which is done above, effectively changing the meaning of $\Lambda$.
#
# 3. Aurel's approach works with density dependent $\mu(x)$ etc.
#
#

# #### More details
#
# From the work of [A Bulgac, MMN Forbes, P Magierski](https://arxiv.org/abs/1008.3933) pp 41, and with a cutoff $k_c$, the relation betweem the scattering length $a$ and the potential strength can be written:
# \begin{align}
# \frac{m}{4\pi\hbar^2 a}
# &=\frac{1}{g} + \frac{1}{2}\int_{0\le k \le k_c}\frac{d^3k}{(2\pi)^3}\frac{1}{\frac{\hbar^2 k^2}{2m} +i0^+}\\
# &=\frac{1}{g}+\frac{m}{2\hbar^2\pi^2}k_c \tag{83}
# \end{align}
#
# For a given scattering length $a$, we got an effective $g_e$,so:
# $$
# \frac{1}{g_e}= \frac{m}{4\pi\hbar^2a}-\frac{mk_c}{2\hbar^2\pi^2}
# $$
#
# To match the result from equation 305, the above can be mutiplied by a factor for both side to get:
# $$
# \frac{8\pi}{g_e}=\frac{2m}{\hbar^2 a} - \frac{4mk_c}{\hbar^2\pi}
# $$
# By setting $m=\hbar=1$,it becomes:
# $$
# \frac{4\pi}{g_e}=\frac{1}{a} - \frac{2k_c}{\pi}
# $$
# Here we got a factor of 2 for $\frac{1}{a}$ in contrast to the result from  Braaten and Hammer (2006), when compared the one used in the paper we reviewed, they also use the one has a factor of 2, so we will stick to this relation instead of  Braaten and Hammer (2006).

# ##  <font color='green'>Validation of convergence</font>

# We expect the following:
#
# $$
#   \frac{\Delta}{\mu} = 1.16220056, \qquad
#   \frac{\Delta}{\epsilon_F} = 0.68640205, \qquad
#   \frac{\mu}{\epsilon_F} = 0.5906055,
# $$
#
# where
#
# $$
#   n_+ = \frac{k_F^3}{3\pi^2}, \qquad
#   \epsilon_F = \frac{\hbar^2k_F^2}{2m}
# $$
#
# With the full solution
# Within the single-particle dispersion:
#
# $$
#   \mu = \frac{\hbar^2k_0^2}{2m}.
# $$
#
# We will choose our $\mu$ such that $k_0 = 1$, i.e. $\mu = \hbar^2k_0^2/2m = 1/2$.

# %pylab inline --no-import-all
from mmf_hfb import homogeneous;reload(homogeneous)
from mmf_hfb.homogeneous import *
h3 = Homogeneous3D(T=0.0)
k0 = 1.0
mu = k0**2/2
eF = mu/0.5906055
kF = np.sqrt(2*eF)
n_p = kF**3/3/np.pi**2
mus_eff = (mu,)*2
delta = 1.16220056*mus_eff[0]
k_c = 10.0
Lambda = h3.get_inverse_scattering_length(mus_eff=mus_eff, delta=delta, k_c=k_c)/4/np.pi
Lambda


# +
def f(k):
    ep = k**2/2 - mus_eff[0]
    E = np.sqrt(ep**2 + abs(delta)**2)
    return 1 - ep/E

def integrand(k):
    return k**2/2/np.pi**2*f(k)

import scipy as sp
ks = np.linspace(0,10,100)
print(sp.integrate.quad(integrand, 0, np.inf)[0])
plt.plot(ks, f(ks))
# -

h32 = Homogeneous3D(T=0)
ks = np.linspace(-30,30,100)
v0, ns, mus = h32.get_BCS_v_n_e_in_cylindrical(mus_eff=mus_eff, delta=delta, k_c=np.inf)
ns, mus, mus_eff,delta

v0, ns, mus = h32.get_BCS_v_n_e_in_spherical(mus_eff=mus_eff, delta=delta, k_c=np.inf)
print(sum(ns))
plt.plot(ks, [h32.f(_k) for _k in ks])

# ### Unitary Case
# At unitary case, we have $a=0$, then from the relation:
# $$
# \frac{4\pi}{g_e}=\frac{1}{a} - \frac{2k_c}{\pi}
# $$
# we can calculate the gap $\Delta$ and the effective interaction strength $g_e$:
# $$
# g_e = -\frac{2\pi^2}{k_c}
# $$
# Then:
# $$
# v_0 = -g_e = \frac{2\pi^2}{k_c}
# $$
#
#

# ## <font color='orange'>Inhomogeneous States</font>

# We now present the solution for inhomogeneous states.
#
# $$
#   \epsilon_k^{\uparrow} = \frac{k_\perp^2}{2m} + 2t[1-\cos(k_z d)] - \mu_\uparrow, \qquad
#   \epsilon_k^{\downarrow} = \frac{k^2}{2m} - \mu_\downarrow.
# $$
#
# The dispersion relationship here is a bit of a misnomer.  There low-energy states are just the single Bloch band $\abs{k_z} \leq \pi/d$.  The other bands are much higher separated by a large gap.
#
# The BdG equations follow from the single-particle Hamiltonian (assuming homogeneous states)
#
# $$
#   \begin{pmatrix}
#     \epsilon_k^{\uparrow} & \Delta\\
#     \Delta & -\epsilon_k^{\downarrow}
#   \end{pmatrix}
# $$
#
# which has eigenvectors and eigenvalues
#
# $$
#   \omega_{\pm} = \epsilon^{-}_k \pm E_k, \qquad
#   E_k = \sqrt{(\epsilon^{+}_k)^2 + \Delta^2}, \\
#   \epsilon^{\pm}_{k} = \frac{\epsilon^\uparrow_k \pm \epsilon^\downarrow_k}{2}\\
#   \abs{u^{\pm}_k}^2 = \frac{1\pm\epsilon^+_k/E_k}{2}, \qquad
#   \abs{v^{\pm}_k}^2 = \frac{1\mp\epsilon^+_k/E_k}{2},\\
#   n_+ = 1 - \frac{\epsilon^+_k}{2E_k}\left[\tanh(\beta\omega_+/2) - \tanh(\beta\omega_-/2)\right], \qquad
#   n_- = - \frac{1}{2}\left[\tanh(\beta\omega_+/2) + \tanh(\beta\omega_-/2)\right]\\
#   \Delta = -g\int \frac{\d^{3}\vect{k}}{(2\pi)^3}\frac{\Delta}{2E_k}\frac{\left[\tanh(\beta\omega_+/2) - \tanh(\beta\omega_-/2)\right]}{2}
# $$

# $$
#   \Delta = v_0 \nu = \frac{v_0}{2}\int\frac{\d{k}}{2\pi}\;\frac{\Delta}{\sqrt{\epsilon_+^2 + \Delta^2}},\quad
#   n_+ = \frac{N_a + N_b}{L} = \int\frac{\d{k}}{2\pi}
#     \left[1 - \frac{\epsilon_+}{\sqrt{\epsilon_+^2 + \abs{\Delta}^2}}\right],\quad
#   \epsilon_+ = \frac{\hbar^2k^2}{2m} - \frac{1}{2}n_+v_0 - \mu = \frac{\hbar^2k^2}{2m} - \mu_{\text{eff}},\\
#   \frac{E}{L} = 
#   \int\frac{\d{k}}{2\pi} \frac{\hbar^2k^2}{2m}
#     \left[1 - \frac{\epsilon_+}{\sqrt{\epsilon_+^2 + \abs{\Delta}^2}}\right]
#   - v_0(n_an_b + \nu^\dagger\nu)
#   = 
#   \int\frac{\d{k}}{2\pi} \frac{\hbar^2k^2}{2m}
#     \left[1 - \frac{\epsilon_+}{\sqrt{\epsilon_+^2 + \abs{\Delta}^2}}\right]
#   - \frac{v_0 n_+^2}{4}
#   - \frac{\abs{\Delta}^2}{v_0}.
# $$
#

# ## <font color='orange'> Error Analysis</font>

# We start with the base BCS class which solves the problem in a 1D periodic universe.  This class will form the base for subsequent work, but is limited by the issue discussed above with discrete $k_n$.  As we shall see, there are two forms of errors: UV errors resulting from a limited $k_\max = \pi N/L$ and IR errors from the discrete $\d{k} = \pi/L$.  To estimate the UV errors, we consider the asymptotic form of the integrals:
#
# $$
#   \delta_{UV}\Delta = \frac{v_0}{2}\overbrace{2}^{\pm k}\int_{k_\max}^{\infty}
#     \frac{\d{k}}{2\pi}\;\frac{\Delta}{\sqrt{\epsilon_+^2 + \Delta^2}} 
#   \approx v_0\int_{k_\max}^{\infty} \frac{\d{k}}{2\pi}\;\frac{2m\Delta}{\hbar^2k^2}
#   = \frac{v_0m\Delta}{\pi\hbar^2k_\max} + \frac{2v_0 m^2\mu_\mathrm{eff}}{3\pi\hbar^4k_\max^3},\\
#   \delta_{UV}n_+ = 2\int_{k_\max}^{\infty}\frac{\d{k}}{2\pi}
#     \left[1 - \frac{\epsilon_+}{\sqrt{\epsilon_+^2 + \abs{\Delta}^2}}\right]
#   \approx \int_{k_\max}^{\infty}\frac{\d{k}}{2\pi}
#     \frac{4m^2\abs{\Delta}^2}{\hbar^4k^4}
#   = \frac{2m^2\abs{\Delta}^2}{3\pi\hbar^4k_{\max}^3} 
#     + \frac{8m^3\mu_{\mathrm{eff}}\abs{\Delta}^2}{5\pi \hbar^6 k_\max^5}
# $$
#
# The error in $\Delta$ is largest, so we can set the lattice spacing to achieve the desired accuracy:
#
# $$
#   \frac{L}{N} \lesssim \frac{\pi^2\hbar^2}{v_0 m}\frac{\delta_{UV}\Delta}{\Delta}.
# $$

# Estimating the IR errors is more difficult: they arise from the variations of the integrand over the range $\d{k}$:

# $$
#   \frac{1}{\d{k}}\int_{-\d{k}/2}^{\d{k}/2}\d{k_b}\left\{
#     \frac{\d{k}}{2\pi}\sum_{n}f(k_n + k_b)
#   \right\} 
#   \approx
#   \frac{1}{\d{k}}\int_{-\d{k}/2}^{\d{k}/2}\d{k_b}\left\{
#     \frac{\d{k}}{2\pi}\sum_{n}\left[f(k_n) + k_bf'(k_n) + \frac{k_b^2}{2}f''(k_n)\right]
#   \right\}\\
#   =
#     \frac{\d{k}}{2\pi}\sum_{n}
#     \left\{
#       f(k_n)
#       +
#       \frac{\d{k}^2}{24}f''(k_n)
#   \right\}.
# $$

# We thus expect the error to scale like 
#
# $$
#   \delta_{IR} \sim \frac{\d{k}^2}{24} = \frac{\pi^2}{3L^2}
# $$
#
# but the coefficient is difficult to calculate.

# ### <font color='green'> Twist-Averaged Boundary Conditions(TBC)</font>

# For many-body wave function in periodic boundary conditions, one may assume the phase of the wavefunction returns to the same wavlue when a particle goes around the boundaries and returns to its original position. [Lin:2001] points out that such assumption may lead to a slow-down of converge for deloclized fermion systems, due to the shell efficts in the filling of single particle states. So to alleviate the shell effect, we allow the overall many-body wave funtions to pick up a phase when particles in the system wrap around the periodic boundaries:
# $$
# \Psi(r_1+L\hat{x},r_2,...)=e^{i\theta_x}\Psi(r_1,r_2...)
# $$
# Generally, the $\theta$ is restirced in the range:
# $$
# -\pi<\theta_x\le\pi
# $$
#
# Then the twist average of any oberserable is defined:
# $$
# \braket{\hat{A}}=(2\pi)^{-d} \int_{-\pi}^{\pi} d\theta\braket{\psi(R,\theta)|\hat{A}|\psi(R,\theta)}
# $$
#
# Numerically, we will only sample some values of $\theta$ and avaerage over the results, such method may be well enough. One can also randomly displace the origin of the grid for a number of time during compuation..

import mmfutils
mmfutils.__version__

# ## Computing $t$

# +
# %pylab inline --no-import-all
from scipy.optimize import leastsq
L = 0.1
kF = 20.0/L
EF = kF**2/2
mu = 9.5*EF
t = 0.1*EF
N = 2**8
r_tol = 1e-4
m = hbar = 1
v_0 = N*np.pi**2/L/m*r_tol

print(mu, t)

l = Lattice(T=0.0, N=N, L=L, v0=v_0, power=1, V0=-EF/40)
twists = np.linspace(-np.pi, np.pi, 40)
ks_b = twists/L
Es0 = [np.linalg.eigvalsh(l.get_H(mus=(1.0, 0.0), delta=1.0, twist=_t)[:N,:N])[0]
       for _t in twists]
Es = [np.linalg.eigvalsh(l.get_H(mus=(1.0, 0.0), delta=1.0, twist=_t)[:N,:N])[1:3]
      for _t in twists]

def f((t, E0)):
    return 2*t*(1-np.cos(ks_b*L)) + E0 - Es0

(t, E0), err = leastsq(f, ((max(Es0)-min(Es0))/2, min(Es0)))

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(ks_b, Es0, '+')
plt.plot(ks_b, f((t, E0)) + Es0)

plt.subplot(122)
plt.plot(ks_b, Es0, '+')
plt.plot(ks_b, f((t, E0)) + Es0)
plt.plot(ks_b, Es)
print(t)
# -

import homogeneous;reload(homogeneous)
h = homogeneous.Homogeneous()
delta = 1.0
mu_eff = 1.0
m = 1.0
mus_eff = [mu_eff]*2
v_0, (na, nb), (mua, mub) = h.get_BCS_v_n_e(delta=delta, mus_eff=mus_eff)
v_0_, n_, mu_, e_0_ = homogeneous.get_BCS_v_n_e(delta=delta, mu_eff=mu_eff)
assert np.allclose([v_0, na+nb, (mua+mub)/2], [v_0_, n_, mu_])


# +
class HomogeneousLattice(homogeneous.Homogeneous):
    t = 0.1
    L = 0.46

    def get_es(self, k, mus_eff):
        return (2*self.t*(1-np.cos(k*self.L)) - mus_eff[0],
                k**2/2.0/m - mus_eff[1])

h = HomogeneousLattice()
v_0, (na, nb), (mua, mub) = h.get_BCS_v_n_e(delta=delta, mus_eff=mus_eff)
# -

# For the paper I am reviewing, the choose parameters as follows:
#
# $$
#   k_FL = 50, \qquad
#   t/E_F = 0.1, \qquad
#   \mu/E_F = 9.5.
# $$

L = 20.0
kF = 50.0/L
EF = kF**2/2
mu = 9.5*EF
t = 0.1*EF
t, mu

# +

l = Lattice(T=0.000001, N=128, L=10.0, cells=1, v0=2.5)
#qT = (1.0, 1.0) + (np.ones(l.N),)*3
N_twist = 1
with NoInterrupt() as interrupted:
    while not interrupted:
        qT = l.iterate_full(qT, plot=False, N_twist=N_twist, 
                            na_avg=0.5, nb_avg=0.5, abs_tol=1e-2)
# -

from mmfutils.contexts import NoInterrupt
l = Lattice(T=0, N=128, L=10.0, mu_a=2.0, cells=1, v0=2.5)
#q1 = np.array((np.ones(l.N),)*3)
with NoInterrupt() as interrupted:
    while not interrupted:
        q1 = l.iterate(q1, plot=False)

from mmfutils.contexts import NoInterrupt
l = Lattice(T=0, N=128, L=10.0, mu_a=2.0, cells=4, v0=2.5, )
#q4 = np.array((np.ones(l.N),)*3)
with NoInterrupt() as interrupted:
    while not interrupted:
        q4 = l.iterate(q4, plot=False)

from mmfutils.contexts import NoInterrupt
l = Lattice(T=0, N=128, L=10.0, mu_a=2.0, cells=1, v0=2.5)
qt4 = np.array((np.ones(l.N),)*3)
N_twist = 4
with NoInterrupt() as interrupted:
    while not interrupted:
        qt4 = l.iterate(qt4, plot=False, N_twist=N_twist)

from mmfutils.contexts import NoInterrupt
l = Lattice(T=0, N=128, L=10.0, mu_a=2.0, cells=1, v0=2.5)
#qt12 = np.array((np.ones(l.N),)*3)
N_twist = 12
with NoInterrupt() as interrupted:
    while not interrupted:
        qt12 = l.iterate(qt12, plot=False, N_twist=N_twist)

from mmfutils.contexts import NoInterrupt
l = Lattice(T=0, N=128, L=10.0, mu_a=2.0, cells=1, v0=2.5)
#qt24 = np.array((np.ones(l.N),)*3)
N_twist = 24
with NoInterrupt() as interrupted:
    while not interrupted:
        qt24 = l.iterate(qt24, plot=False, N_twist=N_twist)

from mmfutils.contexts import NoInterrupt
l = Lattice(T=0, N=128, L=10.0, mu_a=2.0, cells=1, v0=2.5)
#q = np.array((np.ones(l.N),)*3)
#q = qt24
N_twist = np.inf
with NoInterrupt() as interrupted:
    while not interrupted:
        q = l.iterate(q, plot=False, N_twist=N_twist, abs_tol=1e-8)

# +
cells = 10
L_cell = 10.0
N = 2*64*cells
L = L_cell*cells

r0 = 2.0
p = np.ceil(L**2/2/r0**2/cells**2/np.pi**2/2)*2
V0 = -0.5
dx = L/N
x = np.arange(N)*dx - L/2
k = 2*np.pi * np.fft.fftfreq(N, dx)
k_bloch = 2*np.pi * np.fft.fftfreq(cells, L_cell)
V = V0 *(((1+np.cos(2*np.pi*cells*x/L))/2)**p)

Q = np.exp(1j*k[:,None]*x[None,:])
K = Q.T.conj().dot(k[:,None]**2/2 * Q)/N
H = K + np.diag(V)
d, psi = np.linalg.eigh(H)
plt.plot(x, psi[:,0])
plt.twinx()
plt.plot(x, V, 'r:')
plt.figure()

plt.plot(d[:cells])

# +
from scipy.optimize import leastsq

def f(q):
    t, E0 = q
    return (d[:cells] - E0 - sorted(2*t*(1-np.cos(k_bloch*L_cell))))

(t, E0) = leastsq(f, x0=(0.0000001, -1))[0]
plt.plot(f((t, E0)))
print(t, E0)
# -

k_bloch.max(), np.pi/L_cell

cells = 20
print(p)
#V = V0*sum(np.exp(-(x-n*L/cells)**2/2/r0**2) for n in range(-cells, cells))


M = np.array([[ep[0], -delta], [-delta, -ep[0]]])
np.linalg.eigh(M)[1], np.sqrt((1+ep[0]/E[0])/2)

# +
delta = 10.0
mu_eff = 1.0
b = bcs.BCS(T=0, N=128, L=10.0)
H = b.get_H(mus=(mu_eff, mu_eff), delta=delta)
k = np.fft.fftshift(b.k)
ep = (b.hbar*k)**2/2/b.m - mu_eff
em = 0
E = np.sqrt(ep**2 + delta**2)
wp = em + E
wm = em - E
assert np.allclose(np.linalg.eigvalsh(H),
                   sorted(np.concatenate([wp, wm])))

n = 1 - ep/E
plt.plot(k, n)
np.trapz(n, k/2/np.pi)
# -

k2[:N]

# $$
#   \frac{\d{k}}{2\pi} \equiv \frac{1}{L}, \qquad
#   \int \frac{\d{k}}{2\pi} \equiv \frac{1}{L}\sum_{k}
# $$

d, UV = np.linalg.eigh(H)
self = b
u = UV[:N, :N]
v = UV[N:, :N].conj()
k2 = (np.sqrt(d**2 - delta**2) + mu_eff)*2*b.m/b.hbar**2
plt.plot(np.sqrt(k2[:N]), (abs(u)**2).sum(axis=0), '+');
plt.plot(k, n/2);

n/2, (1-np.diag(R)[:N])*b.L

plt.plot(np.sqrt(k2[:N])[1:-1:2], (abs(v[:,1:-1].reshape(N, N//2-1, 2))**2).sum(axis=-1).sum(axis=0)/N)
