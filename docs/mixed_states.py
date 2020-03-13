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

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import *  

# # Mixed states
# $$
#   \mu = \pdiff{\mathcal{E}(n)}{n}, \qquad
#   n = \pdiff{P(\mu)}{\mu}, \qquad
#   P = \mu n - \mathcal{E}, \qquad
#   \mathcal{E} = \mu n - P,\\
#   \mathcal{E}_1 = n^2, \qquad
#   \mathcal{E}_2 = 1 + (n-1)^2\\
#   \mu_1 = 2n, \qquad \mu_2 = 2(n-1)\\
#   n_1 = \mu/2, \qquad n_2 = 1+\mu/2\\
#   P_1 = \frac{\mu^2}{4}, \qquad
#   P_2 = \mu - 1 + \frac{\mu^2}{4}
# $$
#
# First order transition at $\mu_c = 1$, $P=\mu_c^2/4$, with $n \in (\mu/2, 1 + \mu/2)$ and
# $\mathcal{E} = \mu_c n - \mu_c^2/4$.

# +
n = np.linspace(0,2,100)
mu_c = 1
E1 = n**2
E2 = 1+(n-1)**2
n_mix = np.linspace(mu_c/2, 1+mu_c/2, 100)
E_mix = mu_c*n_mix - mu_c**2/4

plt.subplot(121)
plt.plot(n, E1, n, E2)
plt.plot(n_mix, E_mix, '-k')
plt.xlabel(r'$n$'); plt.ylabel(r'$\mathcal{E}$');

mu = np.linspace(0, 2, 100)
P1 = mu**2/4
P2 = mu**2/4 + mu - 1
ax = plt.subplot(122); ax.yaxis.tick_right()
plt.plot(mu, P1, mu, P2)
plt.plot(mu, np.maximum(P1, P2), 'k')
plt.xlabel(r'$\mu$'); plt.ylabel('$P$');
# -


