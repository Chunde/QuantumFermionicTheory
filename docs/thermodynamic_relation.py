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

# # Thermodynamic Relations

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all

# \begin{align}
# k_a &= k + q + dq, \qquad k_b = k+q - dq ,\qquad
# \epsilon_a = \frac{\hbar^2}{2m}k_a^2 - \mu_a, \qquad \epsilon_b = \frac{\hbar^2}{2m}k_b^2 - \mu_b,\\
# E&=\sqrt{\epsilon_+^2+\abs{\Delta}^2},\qquad \omega_+= \epsilon_-+E, \qquad \omega_- = \epsilon_- - E\\
# \epsilon_+&= \frac{\hbar^2}{4m}(k_a^2+k_b^2) - \mu_+= \frac{\hbar^2}{2m}\left[(k+q)^2 + dq^2\right] - \mu_+\\
# \epsilon_-&= \frac{\hbar^2}{4m}(k_a^2-k_b^2) - \mu_-=\frac{\hbar^2}{m}(k +q)dq - \mu_-\tag{1}\\
# \end{align}

# # Analytical Results for Free Fermi Gas
# * Check numerical results against analytical derivaions,[ECE 407 Farhan Rana]
# [ECE 407 Farhan Rana]: https://courses.cit.cornell.edu/mse5470/handout3.pdf 'Farhan Rana "Free Electron Gas in 2D and 1D"'

# ### Energy Density for 1D free Fermi Gas($\Delta=0$)
# $$
# \mathcal{E}=\frac{\sqrt{2m}}{\pi\hbar}\frac{E_F^{3/2}}{3}\qquad n=\frac{\sqrt{2mE_F}}{\pi \hbar}\tag{2}
# $$
# Since there are two components, then the overall energy density
# \begin{align}
# \mathcal{E}
# &=\frac{\sqrt{2m}}{\pi\hbar}\frac{\mu_a^{3/2}+\mu_b^{3/2}}{3}=\frac{\sqrt{2m}}{\pi\hbar}\frac{(\mu+d\mu)^{3/2}+(\mu-d\mu)^{3/2}}{3}
# \end{align}

# ### Energy Density for 2D free Fermi Gas($\Delta=0$)
# For single component
# $$
# \mathcal{E}=\frac{m}{4\pi\hbar^2}E_F^2,\qquad n=\frac{m}{2\pi\hbar^2}E_F, \qquad \mathcal{E}=\frac{1}{2}nE_F\tag{3}
# $$
#
# Since there are two components, then the overall energy density
# \begin{align}
# \mathcal{E}
# &=\frac{m}{4\pi\hbar^2}(\mu_a^2+\mu_b^2)=\frac{m}{4\pi\hbar^2}\left((\mu+d\mu)^2+(\mu-d\mu)^2\right)\\
# \end{align}

# ### Energy Density for 3D free Fermi Gas($\Delta=0$)
# For single component
# $$
# \mathcal{E}=\frac{\hbar^2}{10m\pi^2}k_F^5=\frac{1}{10\pi^2}\frac{(2m)^{3/2}}{\hbar^3}E_F^{5/2}, \qquad n=\frac{k_F^3}{3\pi^2}=\frac{1}{3\pi^2}\frac{(2m)^{3/2}}{\hbar^3}E_F^{3/2}\tag{4}
# $$
# Since there are two components, then the overall energy density
# \begin{align}
# \mathcal{E}
# &=\frac{1}{10\pi^2}\frac{(2m)^{3/2}}{\hbar^3}(\mu_a^{5/2}+\mu_b^{5/2})=\frac{1}{10\pi^2}\frac{(2m)^{3/2}}{\hbar^3}\left((\mu+d\mu)^{5/2}+(\mu+d\mu)^{5/2}\right)\\
# \end{align}

# \begin{align}
# \frac{d E}{d n}
# &=\frac{d E(n_a, n_b)}{d n}\\
# &=\frac{\partial E(n_a, n_b)}{\partial n_a}\frac{\partial n_a}{\partial n}+\frac{\partial E(n_a, n_b)}{\partial n_b}\frac{\partial n_b}{\partial n}\\
# &=\frac{1}{2}\left[\frac{\partial E(n_a, n_b)}{\partial n_a}+\frac{\partial E(n_a, n_b)}{\partial n_b}\right]\\
# &=\frac{\mu_a + \mu_b}{2}=\mu\tag{6}
# \end{align}
# * Since $n=n_+=n_a + n_b$, $n_-=n_a - n_b$, so $n_a = (n_++n_-)/2$, $n_b = (n_+-n_-)/2$, then $\frac{\partial n_a}{\partial n_+}=1/2$

# # Conditions for FF State
# * Thermodymamcally, for given $\mu$,$d\mu$, the pressure should be maximized with respect to $q$ and $dq$
# * $dq>q$, and dq can be many times of q, which means 2q is the speed difference
# * $dq=\frac{n_+}{n_-}$, smaller density difference means bigger dq, or small q will give better chance to find a FF State



