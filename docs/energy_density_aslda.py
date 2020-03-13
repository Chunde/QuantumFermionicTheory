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

# ## Potential terms
# The potential terms is not totally different from the simple external potentials, we need to incorporate lots of tems into the modified potentials, as can be seen from their expressions:
#
# $$
# V_a=\frac{\partial\alpha_-(n_a,n_b)}{\partial n_a}\frac{\hbar^2\tau_-}{2m} + \frac{\partial \alpha_+(n_a,n_b)}{\partial n_a}\left(\frac{\hbar^2\tau_+}{2m} - \frac{\Delta^\dagger\upsilon}{\alpha_+(n_a,n_b)}\right)-\frac{\partial\tilde{C}(n_a,n_b)}{\partial n_a}\frac{\Delta^\dagger \Delta}{\alpha_+} + \frac{\hbar^2}{m} \frac{\partial D(n_a,n_b)}{\partial n_a}+U_a(r)
# $$
# where
#
# \begin{align}
# n_a(r)&=\sum_{\abs{E_n}<E_c} \abs{u_n(r)}^2 f_\beta(E_n)\\
# n_b(r)&=\sum_{\abs{E_n}<E_c} \abs{v_n(r)}^2 f_\beta(-E_n)\\
# \tau_a(r)&=\sum_{\abs{E_n}<E_c} \abs{\nabla u_n(r)}^2 f_\beta(E_n)\\
# \tau_b(r)&=\sum_{\abs{E_n}<E_c} \abs{\nabla v_n(r)}^2 f_\beta(-E_n)\\
# p(r)&=\frac{n_a(r)-n_b(r)}{n_a(r)+n_b(r)}\\
# \alpha(p)&=1.094+0.156p \left(1-\frac{2p^2}{3}+\frac{p^4}{5}\right)-0.532p^2\left(1-p^2+\frac{p^4}{3} \right)\\
# \alpha_a(n_a,n_b)&=\alpha(p)\\
# \alpha_b(n_a,n_b)&=\alpha(-p)\\
# \alpha_{\pm}(n_a,n_b)&=\frac{1}{2}\left[\alpha_a(n_a,n_b)\pm\alpha_b(n_a,n_b)\right]\\
# \tau_{\pm}&=\tau_a \pm \tau_b\\
# G(p)&=0.357+0.642p^2\\
# \tilde{C}(n_a,n_b)&=\frac{\alpha_+(p)(n_a+n_b)^{1/3}}{\gamma(p)}\\
# D(n_a,n_b)&=\frac{\left(6\pi^2(n_a+n_b)\right)^{5/3}}{20\pi^2}\left[G(p)-\alpha(p)\left(\frac{1+p}{2}\right)^{5/3}-\alpha(-p)\left(\frac{1-p}{2}\right)^{5/3}\right]\\
# D(n_a,n_b)&=\frac{\left(6\pi^2(n_a+n_b)\right)^{5/3}}{20\pi^2}2^{-2/3}\beta(p)\\
# \end{align}

# ## Energy density
# Here I will compare all the possible forms of energy density show up in papers(all from ASLDA by Dr.Forbes and his cooperators) I read. All in all, I need to make sure the one given in the ASLDA review paper is the one that rules all others.
# $$
# \varepsilon_{ASLDA}=\frac{\hbar^2}{m}\left(\alpha_a(n_a,n_b)\frac{\tau_a}{2}+\alpha_b(n_a,n_b)\frac{tau_b}{2}+D(n_a,n_b)\right)+g_{eff}\nu^\dagger\nu
# $$
# For SLDA phase:
# $$
# \varepsilon_{SLDA}=\frac{\hbar^2}{m}\left(\alpha_a(n_a,n_b)\frac{\tau_a}{2}+\alpha_b(n_a,n_b)\frac{tau_b}{2}+D(n_a,n_b)\right)+g_{eff}\nu^\dagger\nu
# $$
# For normal Phase:
# $$
# \varepsilon_{Normal}=\frac{\hbar^2}{m}\frac{\left(6\pi^2(n_a+n_b)\right)^{5/3}}{20\pi^2}G(p)
# $$

# For normal phase, when $\Delta=0$, $\nu=0$, then:
# \begin{align}
# \varepsilon_{ASLDA}&=\frac{\hbar^2}{m}\left(\alpha_a(n_a,n_b)\frac{\tau_a}{2}+\alpha_b(n_a,n_b)\frac{\tau_b}{2}+D(n_a,n_b)\right)\\
# &=\frac{\hbar^2}{m}\left(\alpha_a(n_a,n_b)\frac{\tau_a}{2}+\alpha_b(n_a,n_b)\frac{\tau_b}{2}\right) + \frac{\hbar^2}{m}D(n_a,n_b)\\
# &=\frac{\hbar^2}{m}\left(\alpha_a(n_a,n_b)\frac{\tau_a}{2}+\alpha_b(n_a,n_b)\frac{\tau_b}{2}\right)-\frac{\left(6\pi^2\hbar^2(n_a+n_b)\right)^{5/3}}{20m\pi^2}\left[\alpha(p)\left(\frac{1+p}{2}\right)^{5/3}+\alpha(-p)\left(\frac{1-p}{2}\right)^{5/3}\right]+\varepsilon_{Normal}
# \end{align}
#


