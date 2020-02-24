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

# # Subspace Cooling

# Here we organize the sp wavefunctions into groups as if they were located on different compute nodes.

# +
import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from importlib import reload
import mmf_hfb.subspace_cooling;reload(mmf_hfb.subspace_cooling)
b = mmf_hfb.subspace_cooling.SubspaceCooling(N=64, L=24)

Nodes = 1
wf_per_node = 10
N = Nodes*wf_per_node
n0 = 10               # Start filling plane-waves from here.
w = 1.0

b.V_ext = b.m*(w*b.x)**2/2
E0 = b.hbar*w*sum(np.arange(N) + 0.5)

ks = b.kx[np.argsort(abs(b.kx))]
Psis = []
n = n0
for node in range(Nodes):
    # Assumes 1D
    psis = np.exp(1j*ks[n:n+wf_per_node][:,None]*b.x[None,...])/np.sqrt(b.Lxyz)
    n += wf_per_node
    Psis.append(psis)
    
assert np.allclose(b.get_N(Psis), Nodes*wf_per_node)
# -

# Here is how we compute the dot products.
norm = psis.conj().dot(psis.T)
HPsis = b.apply_H(Psis)
E, N = b.get_E_N(Psis)
E

ts, ys = b.solve(Psis, T=1.5, rtol=1e-8, atol=1e-8)
Es, Ns = np.transpose([b.get_E_N(psis) for psis in ys])
ns = [b.get_density(psis) for psis in ys]
Es, E0

plt.semilogy(ts, abs(Es-E0))

plt.semilogy(ts, abs(Es-E0))

np.diag(ys[-1].dot(ys[-1].conj().T))




