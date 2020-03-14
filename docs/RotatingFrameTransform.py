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

# # BdG in Ratating Frame Transform

# * In theoretical calculation, some quantities may gain space-dependence phase that would make calculation hard or more expensive. To transform a rotating phase away one can transform the Hamiltonian to a rotating frame.In theoretical calculation, some quantities may gain space-dependence phase that would make calculation hard or more expensive. To transform a rotating phase away one can transform the Hamiltonian to a rotating frame. A typical application of the rotating frame is in the GPE simulation, where the time dependent driving potential from laser beam to couple different pseudo-spin states can be absorbed into kinetics terms by transforming to the rotating frame from a lab frame. In BCS theories, similar application can be found in literature too.
#

# * Mathmatical identity
# \begin{align}
# \nabla^2\left[U(x)e^{iqx}\right]
# &=\nabla\left[\nabla U(x)e^{iqx}+U(x)iqe^{iqx}\right]\\
# &=\nabla^2U(x)e^{iqx}+2iq\nabla U(x)e^{iqx}-q^2U(x)e^{iqx}\\
# &=(\nabla+iq)^2U(x)e^{iqx}
# \end{align}

# * Let $2q=q_a + q_b$
# \begin{align}
# \begin{pmatrix}
# -\nabla^2-\mu_a & \Delta e^{2iqx}\\
# \Delta^*  e^{-2iqx} & \nabla^2 + \mu_b\\
# \end{pmatrix}
# \begin{pmatrix}
# U(x)e^{iq_a x}\\
# V^*(x)e^{-iq_bx}
# \end{pmatrix}
# &=\begin{pmatrix}
# (-\nabla^2-\mu_a)U(x)e^{iq_a x}+ \Delta e^{2iqx}V^*(x)e^{-iq_bx}\\
# \Delta^*  e^{-2iqx}U(x)e^{iq_a x} + (\nabla^2 + \mu_b)V^*(x)e^{-iq_bx}\\
# \end{pmatrix}\\
# &=\begin{pmatrix}\left[-(\nabla+iq_a)^2-\mu_a\right]U(x)e^{iq_a x}+ \Delta V^*(x)e^{iq_ax}\\
# \Delta^*  U(x)e^{-iq_b x} + \left[(\nabla-iq_b)^2 + \mu_b)\right]V^*(x)e^{-iq_bx}\\
# \end{pmatrix}\\
# &=\begin{pmatrix}\left[-(\nabla+iq_a)^2-\mu_a\right]U(x)e^{iq_a x}+ \Delta V*(x)e^{iq_ax}\\
# \Delta^* U(x)e^{-iq_b x} + \left[(\nabla-iq_b)^2 + \mu_b)\right]V^*(x)e^{-iq_bx}\\
# \end{pmatrix}
# =\begin{pmatrix}
# E & 0\\
# 0&-E
# \end{pmatrix}\begin{pmatrix}
# U(x)e^{iq_a x}\\
# V(x)^*e^{-iq_bx}
# \end{pmatrix}
# \end{align}
# * By canceling out the phase terms:
#

# \begin{align}
# \begin{pmatrix}\left[(i\nabla-q_a)^2-\mu_a\right] &\Delta\\
# \Delta^* & -\left[(i\nabla + q_b)^2 - \mu_b)\right]\\
# \end{pmatrix}
# \begin{pmatrix}
# U(x)\\
# V^*(x)\\
# \end{pmatrix}=\begin{pmatrix}
# E & 0\\
# 0&-E
# \end{pmatrix}
# \begin{pmatrix}
# U(x)\\
# V^*(x)
# \end{pmatrix}
# \end{align}

# * Let $\delta q = q_a - q_b$, then:
# $$
# q_a = q + \delta q\\
# q_b = q - \delta q
# $$
# * So:

# \begin{align}
# \begin{pmatrix}\left[(i\nabla-q - \delta q)^2-\mu_a\right] &\Delta\\
# \Delta^* & -\left[(i\nabla + q - \delta q)^2 - \mu_b)\right]\\
# \end{pmatrix}
# \begin{pmatrix}
# U(x)\\
# V^*(x)\\
# \end{pmatrix}=\begin{pmatrix}
# E & 0\\
# 0&-E
# \end{pmatrix}
# \begin{pmatrix}
# U(x)\\
# V^*(x)
# \end{pmatrix}
# \end{align}

# ## Notation Difference with the code
#
# That means in our code, $\delta q$ is the one enters the pairing field, while we need to max pressure over $q$. In a vertox , we require $\delta q\propto \frac{1}{2r}$

#
