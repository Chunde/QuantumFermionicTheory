# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %pylab inline --no-import-all
from nbimports import *   
# %pylab inline

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

# The eigenvectors for $\epsilon_{\pm}$
# \begin{align}
# \ket{\omega_+} &= \left(-\frac{1}{2E_k},\frac{\Delta}{2E_k(E_k -\epsilon_+ )}\right)\\
# \ket{\omega_-} &= \left(\frac{1}{2E_k},\frac{\Delta}{2E_k(E_k +\epsilon_+ )}\right)
# \end{align}

# # Particle densities

# \begin{align}
# N_a 
# &=\sum_k\braket{\hat{C}_k^\dagger\hat{C}_k}\\
# &=\sum_k\braket{(U_k\hat{\gamma}_+^\dagger + V_k\hat{\gamma}_-)(U_k\hat{\gamma}_+ + V_k\hat{\gamma}_-^\dagger)}\\
# &=\sum_k\braket{U_k^2\hat{\gamma}_+^\dagger\hat{\gamma}_+ + V_k^2\hat{\gamma}_-\hat{\gamma}_-^\dagger+U_kV_k\hat{\gamma}_-\hat{\gamma}_+ + U_kV_k\hat{\gamma}_-^\dagger\hat{\gamma}_+^\dagger}\\
# &=\sum_k\braket{U_k^2\hat{\gamma}_+^\dagger\hat{\gamma}_+ + V_k^2\hat{\gamma}_-\hat{\gamma}_-^\dagger}\\
# &=\sum_k\braket{U_k^2f(\omega_+) + V_k^2f(-\omega_-)}\\
# N_b 
# &=\sum_k\braket{U_k^2\hat{\gamma}_-^\dagger\hat{\gamma}_- + V_k^2\hat{\gamma}_+\hat{\gamma}_+^\dagger}\\
# &=\sum_k\braket{U_k^2f(\omega_-) + V_k^2f(-\omega_+)}\\
# \end{align}

# * With these two equations, accupany for both speices can be computed:
# \begin{align}
# f_+ &= 1-\frac{\epsilon_+}{2E}\left[f(\omega_+)-f(-\omega_-)\right]\\
# f_- &= f(\omega_+)-f(\omega_-)\\
# \end{align}

# # Hartree-Fock-Bogoliubov Method
# In the mean-field approximation, a superconductor is described by a free Hamiltonian, i.e., quadratic in the electron creation and annihilation operators. Note that although the number of fermions is not conserved, the parity is.

# ## Thouless' Theorem
# Thouless' theorem states: 
# Any general product wave function $\ket{\Phi_I}$ which is not orthogonal to the quasi-particle vacuum $\ket{\Psi_0}$ can be expressed as:
#
# $$
# \ket{\Psi_I}=\Omega e^{\sum_{k<k'}{Z_{kk'}\beta_k^+\beta_{k'}^+}}\ket{\Psi_0}
# $$
# where
# $$
# \Omega=\braket{\Psi_0|\Psi_I}
# $$

# ## Two-dody interaction:
# $$
# \hat{H}=\sum_{ij}t_{ij}c_i^\dagger c_j+\frac{1}{4}\sum_{ijkl}\bar{\nu}_{ijkl}c_i^{\dagger}c_j^{\dagger}c_lc_k
# $$

# ## Variational principle:
# $$
# \delta E[\Psi]=0;\qquad E[\Psi]=\frac{\bra{\Psi}\hat{H}\ket{\Psi}}{\braket{\Psi|\Psi}};\qquad E[\Psi]>E_0
# $$
#
# Trail Wave Functions:
# $$
# \beta_k^\dagger=\sum_{i}(u_{ik}c_i^\dagger + v_{ik} c_i)\\
# \ket{\Psi}=\prod_{k}\beta_k\ket{-};\qquad \beta_k\ket{\Psi}=0
# $$

# # S-Wave Scattering

# $$
# -\frac{\hbar^2}{2m}u''(r)=Eu(r)
# $$
# solutions:
# $$
# u(r)=Ae^{ikr}+Be^{-ikr}
# $$
# For given condition: $u(r_0)=0$
# \begin{align}
# u(r_0)
# &=Ae^{ikr_0}+Be^{-ikr_0} \qquad\text{A,B are complex numbers}\\
# &=(A+B)cos(kr_0)+i(A-B)sin(kr_0)=0
# \end{align}

# The condition to make $u(r_0)=0$ can be achived:
# $$
# u(r_0)=sin(kx+\delta_0) \qquad \text{where}\qquad \delta_0 = kr_0
# $$

# # Commutation of field operator and mometum operator
# In plane-wave basis, the momentum operator is given by:
# $$
# \mathbf{P} \equiv \sum_{\alpha} \int d^{3} x \hat{\psi}_{\alpha}^{\dagger}(\mathbf{x})(-i \hbar \nabla) \hat{\psi}_{\alpha}(\mathbf{x})=\sum_{\mathbf{k} \lambda} \hbar \mathbf{k} c_{\mathbf{k} \lambda}^{\dagger} c_{\mathbf{k} \lambda}
# $$
# A field operaotr $\hat{\psi}(x)$ can be expanded in the plane wave basis as:

# $$
# \hat{\psi_{\alpha}}(x)=\sum_{\alpha}\braket{k|x}c_{k\alpha}=\sum_{\alpha}e^{ikx}c_{c\alpha}
# $$

# Then the commutator of $\hat{P}$ with the field operator can be evaluated as follow:
# $$
# \begin{align}
# \left[\hat{\psi}_{\alpha}(\mathbf{x}), \mathbf{P}\right]
# &=\sum_{\alpha k,\beta, k'}\hbar k e^{kx}\left[c_{k\alpha}c_{k'\beta}c_{k'\beta} - c_{k'\beta}c_{k'\beta}c_{k\alpha}\right]\\
# &=\sum_{\alpha k}e^{ikx}\hbar k \left[c_{k\alpha}c_{k\alpha}c_{k\alpha} - c_{k\beta}c_{k\alpha}c_{k\alpha}\right]\qquad \text{[Apply commuation relation]}\\
# &=\sum_{\alpha k}e^{ikx}\hbar k c_{\alpha k}\\
# &=-i\hbar \nabla\hat{\psi}(x)
# \end{align}
# $$


