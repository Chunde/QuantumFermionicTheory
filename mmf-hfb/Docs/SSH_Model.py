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

# + {"id": "_762UXGJ2yRO", "colab_type": "code", "outputId": "9a316bc9-2617-4d3b-bbcb-c7b52a5d48c1", "colab": {"base_uri": "https://localhost:8080/", "height": 34}}
# %pylab inline --no-import-all

# + {"id": "AtypauH1Auxw", "colab_type": "text", "cell_type": "markdown"}
# # SSH Model
#
# Here we consider the SSH model as presented and discussed in [[Léséleuc:2019]](https://science.sciencemag.org/content/365/6455/775) which represents a finite 1D lattice of fermions with the following Hamiltonian (from their supplement):
#
# \begin{gather}
#   H_F = - J_1\sum_{i=1}^{L}c_{2i-1}^\dagger c_{2i} 
#         - J_2\sum_{i=1}^{L-1}c_{2i}^\dagger c_{2i+1}.\tag{S3}
# \end{gather}

# + {"id": "K_h6pxB5WQzM", "colab_type": "text", "cell_type": "markdown"}
# ## Brute Force Diagonalization
# * np.kron is the tensor product operation

# + {"id": "9v_3EuFM21uF", "colab_type": "code", "outputId": "67f4c617-6642-4725-862c-6578547fd9f3", "colab": {"base_uri": "https://localhost:8080/", "height": 54}}
import numpy as np
def get_c(N, n):
    c = np.array([[0, 0], 
                    [1, 0]])
    one = np.eye(2)
    factors = [one]*N
    factors[n] = c
    res = 1
    for f in factors:
        res = np.kron(res, f)
    return res
    
def get_cs(N):
    """Return a list of the operators c_i.
    
    Arguments
    ---------
    N : int
        N = 2L = number of lattice sites.
    """    
    cs = []
    for n in range(N):
        res = get_c(N=N, n=n)
        cs.append(res)
    return cs

def get_H(L=4, J1=1, J2=2, get_cs=get_cs):
    """Return the Hamiltonian for a chain of N=2L sites.
    """
    N = 2*L
    cs = get_cs(N)
    H1 = 0
    H2 = 0
    for i in range(1, L+1):
        # -1 for 0-indexing in python
        H1 += cs[2*i-1-1].T.dot(cs[2*i-1])
    for i in range(1, L):
        H2 += cs[2*i-1].T.dot(cs[2*i+1-1])
    H = -(J1*H1 + J2*H2)
    return H + H.T

# %time H = get_H(L=5)
# %time E, psi = np.linalg.eigh(H)


# -

c = get_c(N=10, n=1)
ct = c.T
op = np.matmul(ct, c)


# + {"id": "IikbBXMnH1Ja", "colab_type": "text", "cell_type": "markdown"}
# Here we plot the energes of the lowest excited states.  In the topological phase, the energy of the ground state has a four-fold degeneracy, so there are three degenerate "excited" states in the plot since we subtract off the energy of the ground state.
# -

def get_Ns(N, state_id=0, J1=1, J2=2):
    """
    return occupancy number for given manybody state id
    """
    H = get_H(L=N//2, J1=J1, J2=J2)
    E, psi = np.linalg.eigh(H)
    Ns = []
    for n in range(N):
        c = get_c(N=N, n=n)
        ct = c.T
        op_n = ct.dot(c)
        n_ = op_n.dot(psi[:,state_id])
        Ns.append(n_.dot(psi[:,state_id]))
    return Ns


state_id = 1
Jprime=0.2
J=1.0
Ns=get_Ns(N=10, state_id=state_id, J2=Jprime, J1=J)
plt.subplot(121)
plt.bar(list(range(len(Ns))),Ns)
plt.subplot(122)
Ns=get_Ns(N=10, state_id=state_id, J2=J, J1=Jprime)
plt.bar(list(range(len(Ns))),Ns)

# + {"id": "H4MeDsvXENMD", "colab_type": "code", "outputId": "88ec8436-93b6-4491-9192-70f500493c1b", "colab": {"base_uri": "https://localhost:8080/", "height": 349}}
Nstates = 10
Jprime = 0.2
J = 1.0

# Trivial state
H = get_H(L=4, J1=J, J2=Jprime)
E = np.linalg.eigvalsh(H)
plt.plot(range(Nstates), (E[0:Nstates]-E[0])/J, '+', label='Trivial')

# Topological state:
H = get_H(L=4, J1=Jprime, J2=J)
E, psi = np.linalg.eigh(H)
plt.plot(range(Nstates), (E[0:Nstates]-E[0])/J, 'o', label='Topological')

plt.legend()
plt.xlabel("State")
plt.ylabel(r"$(E-E_0)/J$")
plt.title("Excited State Energies")

# + {"id": "TY-RcwRcJU3x", "colab_type": "text", "cell_type": "markdown"}
# ##### Exercise: Site Occupations
#
# It would be nice to plot the occupancies.  Compute the expectation value of the occupation of each site by forming $n_i = c_i^\dagger c_i$ and plot these for the lowest states to see if these agree with the Fig. 1 in the paper.

# + {"id": "y7xK69t2JL6M", "colab_type": "code", "outputId": "a932759b-cde3-46c9-b4e5-1ebf4d4fbb16", "colab": {"base_uri": "https://localhost:8080/", "height": 286}}
plt.plot(abs(psi[2, :]))


# + {"id": "pbZigkLBEUAw", "colab_type": "code", "outputId": "af34fef5-89d5-4ab9-cab4-b16fb7fe90db", "colab": {"base_uri": "https://localhost:8080/", "height": 323}}
Jprime = 0.1
J = 2.0
H = get_H(L=5, J1=J, J2=Jprime)
E = np.linalg.eigvalsh(H)
#[plt.axhline(_E) for _E in E];
plt.plot(E, '+')
plt.ylim(-J-Jprime, J+Jprime)

# + {"id": "UqTa4DnG3O4D", "colab_type": "code", "outputId": "22124be6-a755-47cd-a580-30fa831cf85c", "colab": {"base_uri": "https://localhost:8080/", "height": 323}}
Jprime = 0.1
J = 2.0
H = get_H(L=5, J1=Jprime, J2=J)
E = np.linalg.eigvalsh(H)
[plt.axhline(_E) for _E in E];
plt.ylim(-J-Jprime, J+Jprime)

# + {"id": "bJ8s4CRSCwB8", "colab_type": "text", "cell_type": "markdown"}
# ### Profiling

# + {"id": "mNsT4Qab_nsg", "colab_type": "text", "cell_type": "markdown"}
# The maximum lattice size is about 6 or 7 if using full matrices ($L=7$ will take about an hour, $L=8$ will take about a day).

# + {"id": "vKALyaSR-HX5", "colab_type": "code", "outputId": "b742156c-7c5d-4bcd-875c-a4f369a12d83", "colab": {"base_uri": "https://localhost:8080/", "height": 88}}
import time
reps = 3
ts = []
Ls = range(1, 6)
for L in Ls:
    tic = time.time()
    for n in range(reps):
        H = get_H(L=L)
        E = np.linalg.eigvalsh(H)
    ts.append((time.time()-tic)/reps)

def get_T(L, Ls=Ls, ts=ts):
    """Return the expected execution time in s."""
    return np.exp(np.polyval(np.polyfit(Ls[3:], np.log(ts[3:]), deg=1), L))

print(f"T_6~{get_T(L=6):.0f}s, T_7~{get_T(L=7)/60:.0f}min, T_8~{get_T(L=8)/60/60:.0f}h")

# + {"id": "JDvU6hrD3Q-L", "colab_type": "text", "cell_type": "markdown"}
# ## Sparse Implementation

# + {"id": "P_7andqOWXlL", "colab_type": "code", "colab": {}}
import functools
import scipy.sparse.linalg
sp = scipy

def get_cs_sparse(N):
    """Return a list of the operators c_i.
    
    Arguments
    ---------
    N : int
        N = 2L = number of lattice sites.
    """
    c = sp.sparse.csr_matrix([[0, 0], [1, 0]])
    one = sp.sparse.eye(2)
    cs = []
    for n in range(N):
        factors = [one]*N
        factors[n] = c
        res = 1
        for f in factors:
            res = sp.sparse.kron(res, f, format='csr')
        cs.append(res)
    return cs


# + {"id": "hHeOCFyRXcCp", "colab_type": "code", "outputId": "2c8f2503-95b5-43c7-da36-eba7472c3688", "colab": {"base_uri": "https://localhost:8080/", "height": 334}}
import time
reps = 3
ts = []
Ls = range(1, 10)
for L in Ls:
    tic = time.time()
    for n in range(reps):
        H = get_H(L=L, get_cs=get_cs_sparse)
        E, psi = sp.sparse.linalg.eigsh(H, k=min(H.shape[0]-1, 10))
    ts.append((time.time()-tic)/reps)

def get_T(L, Ls=Ls, ts=ts):
    """Return the expected execution time in s."""
    return np.exp(np.polyval(np.polyfit(Ls[-3:], np.log(ts[-3:]), deg=1), L))

_Ls = np.arange(1, 10)
plt.semilogy(Ls, ts);plt.xlabel('L');plt.ylabel('t [s]')
plt.semilogy(_Ls, [get_T(_L) for _L in _Ls])
for L0 in [13, 14, 15]:
    print(f"T_{L0}~{get_T(L=L0):.0f}s")

# + {"id": "y7mXpcReX9zm", "colab_type": "code", "outputId": "3cd6cdf8-2fb5-4d7c-f8f0-a5b436930063", "colab": {"base_uri": "https://localhost:8080/", "height": 119}}
# %time H = get_H(9, get_cs=get_cs_sparse)
H = H.tocsc()
# %time E, psi = sp.sparse.linalg.eigsh(H, k=10)
E[:5]

# + {"id": "aFfZh_-jYErG", "colab_type": "code", "outputId": "039123ab-8d5b-45d9-a598-2b614a36fa86", "colab": {"base_uri": "https://localhost:8080/", "height": 329}}
# %%time 
L = 8
Nstates = 10
Jprime = 0.2
J = 1.0

# Trivial state
H = get_H(L=L, J1=J, J2=Jprime, get_cs=get_cs_sparse)
E, psi = sp.sparse.linalg.eigsh(H, k=2*Nstates)
plt.plot(range(Nstates), (E[0:Nstates]-E[0])/J, '+', label='Trivial')

# Topological state:
H = get_H(L=L, J1=Jprime, J2=J,get_cs=get_cs_sparse)
E, psi = sp.sparse.linalg.eigsh(H, k=2*Nstates)
plt.plot(range(Nstates), (E[0:Nstates]-E[0])/J, 'o', label='Topological')

plt.legend()
plt.xlabel("State")
plt.ylabel(r"$(E-E_0)/J$")
plt.title("Excited State Energies")

# + {"id": "6BNQChwZa9Gy", "colab_type": "code", "outputId": "a64ab16f-f7bf-4ed4-d5a7-56b8787a53fc", "colab": {"base_uri": "https://localhost:8080/", "height": 323}}
H0 = get_H(L=5, J1=Jprime, J2=J,get_cs=get_cs)
H = get_H(L=5, J1=Jprime, J2=J,get_cs=get_cs_sparse)
E0, psi0 = np.linalg.eigh(H0)
E, psi = sp.sparse.linalg.eigsh(H, k=2*Nstates)
plt.plot(range(Nstates), E0[:Nstates], '+', label='dense')
plt.plot(range(Nstates), E[:Nstates], 'x', label='sparse')
plt.legend()


# + {"id": "4bXjVg84g5bY", "colab_type": "text", "cell_type": "markdown"}
# # GPU

# + {"id": "BNXNtoQgk-fz", "colab_type": "text", "cell_type": "markdown"}
# Be sure to select the GPU accelerator before using the following code.  This version works, and is fast, but the GPU memory is limited, so we cannot do more than $L=5$.

# + {"id": "dQy9I68wk4nA", "colab_type": "code", "colab": {}}
import cupy as cp


# + {"id": "dADZiYT-k1qw", "colab_type": "code", "colab": {}}
def get_cs_GPU(N):
    """Return a list of the operators c_i.

    Arguments
    ---------
    N : int
     N = 2L = number of lattice sites.
    """
    c = cp.array([[0, 0], 
                    [1, 0]])
    one = cp.eye(2)
    cs = []
    for n in range(N):
        factors = [one]*N
        factors[n] = c
        res = 1
        for m in range(N):
            # Implement the tensor product by using indices
            # and broadcasting.
            inds = [None, ]*2*N
            inds[m] = inds[N+m] = slice(None)
            res = res * factors[m][inds]
            cs.append(res.reshape(2**N, 2**N))
    return cs

def get_H_GPU(L=4, J1=1, J2=2, get_cs=get_cs_GPU):
    """Return the Hamiltonian for a chain of N=2L sites.
    """
    N = 2*L
    cs = get_cs(N)
    H = 0
    for i in range(1, L+1):
        # -1 for 0-indexing in python
        H += - J1*cs[2*i-1-1].T.dot(cs[2*i-1])
    for i in range(1, L):
        H += - J2*cs[2*i-1].T.dot(cs[2*i+1-1])
    return H + H.T

# + {"id": "nNSgEDF_lIwO", "colab_type": "code", "outputId": "6af7ba44-2dcd-4343-bf3a-aa2b0bea837b", "colab": {"base_uri": "https://localhost:8080/", "height": 85}}
# %time H = get_H_GPU(L=6)
# %time E, psi = cp.linalg.eigh(H)

# + {"id": "dzfdF9oJlKiG", "colab_type": "code", "colab": {}}

