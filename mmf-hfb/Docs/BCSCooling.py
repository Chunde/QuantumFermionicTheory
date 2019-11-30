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
import matplotlib.pyplot as plt
# %pylab inline --no-import-all
from nbimports import *
import numpy as np

# + {"id": "ptb73kVS8ceS", "colab_type": "text", "cell_type": "markdown"}
# # BCS Cooling Class Test
# * To properly display equations, we define some command to make life easier, this commands are invisible
# $$
#   \newcommand{\I}{\mathrm{i}}
#   \newcommand{\d}{\mathrm{d}}
#   \newcommand{\vect}[1]{\vec{#1}}
#   \newcommand{\op}[1]{\hat{#1}}
#   \newcommand{\abs}[1]{\lvert#1\rvert}
#   \newcommand{\pdiff}[2]{\frac{\partial #1}{\partial #2}}
#   \newcommand{\ket}[1]{\lvert#1\rangle}
#   \newcommand{\bra}[1]{\langle#1\rvert}
#   \newcommand{\braket}[1]{\langle#1\rangle}
#   \DeclareMathOperator{\Tr}{Tr}
# $$
# * A class implement local friction that supports GP type cooling(single wave function) and BCS type cooling(multiple wavefunctions). When applied to BCS orbits, it will maintain the particle number and orthogonality of these orbits

# +
import mmf_hfb.BCSCooling as bcsc; reload(bcsc)
from mmf_hfb.BCSCooling import BCSCooling
from mmf_hfb.SolverABM import ABMEvolverAdapter
from mmf_hfb.Cooling import Cooling
from mmf_hfb.Potentials import HarmonicOscillator
from IPython.core.debugger import set_trace
from IPython.display import display, clear_output
import numpy as np
import scipy as sp

def Normalize(psi):
    return psi/psi.dot(psi.conj())**0.5

def Prob(psi):
    return np.abs(psi)**2
   
def check_uv_ir_error(psi, plot=False):
    """check if a lattice configuration(N, L, dx) is good"""
    psi_k = np.fft.fft(psi)
    psi_log = np.log10(abs(psi))
    psi_log_k = np.log10(abs(psi_k))
    if plot:
        l, =plt.plot(psi_log_k)
        plt.plot(psi_log,'--', c=l.get_c())
        print(np.min(psi_log), np.min(psi_log_k))
    # assert the log10 value to be close to machine precision
    #assert np.min(psi_log)<-15
    assert np.min(psi_log_k) < -15


# -

# ## Test Derivative Cooling
# * As the derivative cooling potential Vd is not diagonilzed in either postion space nor momenutum space, it can't be used in split-operator method. 
# * It's found that discarding the highest momentum $k_{max}$ can cool down the energy using $V_{d}$ to some energy and may stall there.
# ### Analytical Check
# Start with plance wave $\psi_n(x)= e^{i n kx}$, and free particle Hamiltonian $H$ with eigen energy $E_n$ for nth wave function $\psi_n(x)$, then
# $$
# H\psi=i\hbar \frac{\partial \psi(x, t)}{\partial t} = E_n \psi=\frac{n^2k^2}{2}\psi(x), \qquad \dot{\psi(x)}=\frac{\partial \psi(x)}{\partial t} = \frac{E_n\psi(x)}{i\hbar}\\
# \psi^a(x)=\nabla^a \psi(x) = \frac{d^a \psi(x)}{dx^a} =  (i n k )^a e^{i n k x} = (ink)^a \psi(x)\\
# \dot{\psi}^a(x)=(i n k )^a e^{i n k x} = (ink)^a\frac{E_n \psi(x)}{i\hbar}\\
# $$
# The defination of $V_{11}$ is given by:
#
# $$
#   \hbar V_{ab}(x) 
#   = (\I\hbar\braket{x|\op{D}_b[\op{R},\op{H}]\op{D}_a^\dagger|x})^*
#   = \I\hbar\braket{x|\op{D}_a[\op{R},\op{H}]\op{D}_b^\dagger|x}\\
#   = \braket{x|\op{D}_a|\psi}\braket{x|\op{D}_b|\dot{\psi}}^*
#   + \braket{x|\op{D}_a|\dot{\psi}}\braket{x|\op{D}_b|\psi}^*.
# $$
#
# If $\op{D}_{a,b}(x)$ are just derivative operators$\nabla^a$, then $\braket{x|\op{D}_{a}|\psi} = \psi^{(a)}(x)$, then we have
#
# $$
# \begin{align}
#  \hbar V_{ab}(x) 
#   &= \psi^{(a)}(x)\overline{\dot{\psi}^{(b)}(x)}+ \dot{\psi}^{(a)}(x)\overline{\psi^{(b)}(x)}\\
#   &=\frac{(ink)^a(-ink)^b E_n\psi(x)\psi^*(x)}{-i\hbar}+\frac{(ink)^a(-ink)^b E_n\psi(x)\psi^*(x)}{i\hbar}\\
#   &=\frac{(nk)^{a+b}E_n\psi\psi^*}{ih}\bigl[(i)^a(-i)^b-i^a(-i)^b\bigr]=0
# \end{align}
# $$
#
# It's found that $V_{ab}=0$, since any wave function can be expanded as plane wave, that means for any wave function, as long as the Hamiltonian is free-partile type, $V_{ab}=0$

# * The following code check the UV and IR depedence of the derivative potential. 

psi_err_real = [-0.00218921, -0.00226835, -0.00282437, -0.00360555, -0.00501059,  -0.00685029, -0.00951956, -0.0127734 , -0.01684204, -0.02124532,  -0.02588368, -0.02995909, -0.03310807, -0.03448526, -0.03393179,  -0.03113281, -0.02663775, -0.02086866, -0.01482283, -0.0089401 ,  -0.00373432,  0.00104928,  0.00547406,  0.0098595 ,  0.01384601,   0.01710405,  0.01897185,  0.01938578,  0.01841146,  0.01652955,   0.01381004,  0.01019765,  0.00539174, -0.00051897, -0.00721161,  -0.01431977, -0.02175712, -0.02969925, -0.03800296, -0.04647457,  -0.05466709, -0.06283147, -0.07077758, -0.07868313, -0.08590425,  -0.09288341, -0.09926526, -0.10549211, -0.11085466, -0.11593967,  -0.12040231, -0.12456927, -0.12803744, -0.13119387, -0.13395077,  -0.13628036, -0.13820641, -0.1399491 , -0.14125757, -0.142399  ,  -0.14322768, -0.14393405, -0.14432862, -0.14463055, -0.14470525,  -0.14463055, -0.14432862, -0.14393405, -0.14322768, -0.142399  ,  -0.14125757, -0.1399491 , -0.13820641, -0.13628036, -0.13395077,  -0.13119387, -0.12803744, -0.12456927, -0.12040231, -0.11593967,  -0.11085466, -0.10549211, -0.09926526, -0.09288341, -0.08590425,  -0.07868313, -0.07077758, -0.06283147, -0.05466709, -0.04647457,  -0.03800296, -0.02969925, -0.02175712, -0.01431977, -0.00721161,  -0.00051897,  0.00539174,  0.01019765,  0.01381004,  0.01652955,   0.01841146,  0.01938578,  0.01897185,  0.01710405,  0.01384601,   0.0098595 ,  0.00547406,  0.00104928, -0.00373432, -0.0089401 ,  -0.01482283, -0.02086866, -0.02663775, -0.03113281, -0.03393179,  -0.03448526, -0.03310807, -0.02995909, -0.02588368, -0.02124532,  -0.01684204, -0.0127734 , -0.00951956, -0.00685029, -0.00501059,  -0.00360555, -0.00282437, -0.00226835]

psi_err = [-0.00218921+4.50738870e-05j, -0.00226835+3.27814682e-04j, -0.00282437+5.97283274e-04j, -0.00360555+1.39230139e-03j,    -0.00501059+2.07960895e-03j, -0.00685029+3.09336714e-03j, -0.00951956+3.68280893e-03j, -0.0127734 +4.12272628e-03j,    -0.01684204+3.54005357e-03j, -0.02124532+2.11877210e-03j,    -0.02588368-9.62888633e-04j, -0.02995909-5.33343049e-03j,    -0.03310807-1.14164114e-02j, -0.03448526-1.83242920e-02j,    -0.03393179-2.59022377e-02j, -0.03113281-3.28407813e-02j,    -0.02663775-3.88468295e-02j, -0.02086866-4.29065158e-02j,    -0.01482283-4.53694929e-02j, -0.0089401 -4.60280349e-02j,    -0.00373432-4.58378021e-02j,  0.00104928-4.47449411e-02j,     0.00547406-4.33346633e-02j,  0.0098595 -4.09341245e-02j,     0.01384601-3.76976409e-02j,  0.01710405-3.30584247e-02j,     0.01897185-2.77803042e-02j,  0.01938578-2.19596106e-02j,     0.01841146-1.65698780e-02j,  0.01652955-1.12854342e-02j,     0.01381004-6.45290020e-03j,  0.01019765-1.48229183e-03j,     0.00539174+2.91830327e-03j, -0.00051897+6.83454930e-03j,    -0.00721161+9.48325482e-03j, -0.01431977+1.14661610e-02j,    -0.02175712+1.24614862e-02j, -0.02969925+1.29921508e-02j,    -0.03800296+1.22802124e-02j, -0.04647457+1.07622808e-02j,    -0.05466709+8.09754723e-03j, -0.06283147+5.04299703e-03j,    -0.07077758+1.04844071e-03j, -0.07868313-3.50850290e-03j,    -0.08590425-8.93007232e-03j, -0.09288341-1.45492125e-02j,    -0.09926526-2.06476776e-02j, -0.10549211-2.70484057e-02j,    -0.11085466-3.37529097e-02j, -0.11593967-4.04560507e-02j,    -0.12040231-4.71200358e-02j, -0.12456927-5.38792982e-02j,    -0.12803744-6.03759482e-02j, -0.13119387-6.66371139e-02j,    -0.13395077-7.25906880e-02j, -0.13628036-7.82334154e-02j,    -0.13820641-8.33861957e-02j, -0.1399491 -8.80135257e-02j,    -0.14125757-9.22296061e-02j, -0.142399  -9.57805140e-02j,    -0.14322768-9.87048205e-02j, -0.14393405-1.01118355e-01j,    -0.14432862-1.02752808e-01j, -0.14463055-1.03759408e-01j,    -0.14470525-1.04155158e-01j, -0.14463055-1.03759408e-01j,    -0.14432862-1.02752808e-01j, -0.14393405-1.01118355e-01j,    -0.14322768-9.87048205e-02j, -0.142399  -9.57805140e-02j,    -0.14125757-9.22296061e-02j, -0.1399491 -8.80135257e-02j,    -0.13820641-8.33861957e-02j, -0.13628036-7.82334154e-02j,    -0.13395077-7.25906880e-02j, -0.13119387-6.66371139e-02j,    -0.12803744-6.03759482e-02j, -0.12456927-5.38792982e-02j,    -0.12040231-4.71200358e-02j, -0.11593967-4.04560507e-02j,    -0.11085466-3.37529097e-02j, -0.10549211-2.70484057e-02j,    -0.09926526-2.06476776e-02j, -0.09288341-1.45492125e-02j,    -0.08590425-8.93007232e-03j, -0.07868313-3.50850290e-03j,    -0.07077758+1.04844071e-03j, -0.06283147+5.04299703e-03j,    -0.05466709+8.09754724e-03j, -0.04647457+1.07622808e-02j,    -0.03800296+1.22802124e-02j, -0.02969925+1.29921508e-02j,    -0.02175712+1.24614862e-02j, -0.01431977+1.14661610e-02j,    -0.00721161+9.48325482e-03j, -0.00051897+6.83454930e-03j,     0.00539174+2.91830327e-03j,  0.01019765-1.48229183e-03j,     0.01381004-6.45290021e-03j,  0.01652955-1.12854342e-02j,     0.01841146-1.65698780e-02j,  0.01938578-2.19596107e-02j,     0.01897185-2.77803042e-02j,  0.01710405-3.30584247e-02j,     0.01384601-3.76976409e-02j,  0.0098595 -4.09341245e-02j,     0.00547406-4.33346633e-02j,  0.00104928-4.47449411e-02j,    -0.00373432-4.58378021e-02j, -0.0089401 -4.60280349e-02j,    -0.01482283-4.53694929e-02j, -0.02086866-4.29065158e-02j,    -0.02663775-3.88468295e-02j, -0.03113281-3.28407813e-02j,    -0.03393179-2.59022377e-02j, -0.03448526-1.83242920e-02j,    -0.03310807-1.14164114e-02j, -0.02995909-5.33343049e-03j,    -0.02588368-9.62888632e-04j, -0.02124532+2.11877210e-03j,    -0.01684204+3.54005357e-03j, -0.0127734 +4.12272628e-03j,    -0.00951956+3.68280893e-03j, -0.00685029+3.09336714e-03j,    -0.00501059+2.07960895e-03j, -0.00360555+1.39230139e-03j,    -0.00282437+5.97283275e-04j, -0.00226835+3.27814682e-04j]

plt.figure(figsize(10,5))
args = dict(beta_K=1, beta_V=1, beta_D=1, beta_Y=1, divs=(1, 1))
s = BCSCooling(N=128, dx=0.1, g=-1,**args)
s.erase_max_ks()
x = s.xyz[0]
s.V = x**2/2
psi0 = psi_err #np.exp(-x**2/2.0)*np.exp(1j*x)
plt.subplot(121)
plt.plot(x, psi0)
plt.subplot(122)
Vc = s.get_Dyadic(s.apply_H([psi0]))
plt.plot(x, Vc)


def Check_UV_IR():
    dx = 0.1
    plt.figure(figsize=(15,5))
    args = dict(beta_K=10000, beta_V=1, beta_D=1, beta_Y=1, divs=(1, 1))
    for Nx in [128, 256, 512]:
        offset = np.log(Nx)*0.1 # add a small offset in y direction
        uv = BCSCooling(N=256, dx=dx*256/Nx,**args)
        ir  = BCSCooling(N=Nx, dx=dx, beta_0=1,**args)
        for s, i in zip([uv, ir],[2, 3]):           
            s.g = -1
            x = s.xyz[0]
            s.V = x**2/2
            psi0 = np.exp(-x**2/2.0)*np.exp(1j*x)
            plt.subplot(1,3,1)
            plt.plot(x, psi0 + offset)
            plt.subplot(1,3,i)
            psis_k = s.get_psis_k([psi0])
            Vc = s.get_Vd(psis=[psi0], psis_k=psis_k)
            l, = plt.plot(x, Vc + offset)  # add some offset in y direction to separate plots
    plt.subplot(131)
    plt.xlim(-10, 10)
    plt.subplot(132)
    plt.xlim(-4, 4)
    plt.subplot(133)
    plt.xlim(-4,4)
Check_UV_IR()

import time
def test_der_cooling(plot=True, plot_dE=True, T=0.5, log=False, **args):   
    b = BCSCooling(**args)
    h0 = HarmonicOscillator(w=1)
    h = HarmonicOscillator()
    da, db=b.divs    
    k0 = 2*np.pi/b.L
    x = b.xyz[0]
    V = x**2/2
    b.V = V
    H0 = b._get_H(mu_eff=0, V=0)
    H1 = b._get_H(mu_eff=0, V=V)
    U0, E0 = b.get_U_E(H0, transpose=True)
    U1, E1 = b.get_U_E(H1, transpose=True)
    psi0 = b.Normalize(h.get_wf(x))
    psi0 = b.Normalize(U1[0])
    psi = b.Normalize(h0.get_wf(x, n=2))
    psi = b.Normalize(U0[1])
    start_time = time.time()
    ts, psis, nfev = b.solve([psi], T=T, rtol=1e-5, atol=1e-6,solver=None, method='BDF')
    wall_time = time.time() - start_time
    if plot:
        #b.erase_max_ks()
        if plot_dE:
            N_plot=3
        else:
            N_plot=2
        plt.figure(figsize=(5*N_plot, 5))
        plt.subplot(1, N_plot, 1)
        E0, _ = b.get_E_Ns([psi0])
        Es = [b.get_E_Ns(_psi)[0] for _psi in psis]
        dE_dt= [-1*b.get_dE_dt(_psi) for _psi in psis]
        plt.plot(x, Prob(psis[0][0]), "+", label='init')
        plt.plot(x, Prob(psis[-1][0]), '--',label="final")
        plt.plot(x, Prob(psi0), label='Ground')
        plt.xlabel("x")
        plt.ylabel(r"$\psi(x)$")
        plt.legend()
        plt.subplot(1,N_plot,2)
        if log:
            plt.semilogy(ts[:-1], (Es[:-1] - E0)/abs(E0), label="E")
        else:
            plt.plot(ts[:-1], (Es[:-1] - E0)/abs(E0), label="E")
        if (Es[-1] - E0)/abs(E0) < 2.5:
            plt.axhline(1, linestyle='dashed')
        plt.xlabel("Time")
        plt.ylabel("(E-E0)/E0")
        if plot_dE:
            plt.subplot(1, N_plot, 3)
            plt.plot(ts[:-1], dE_dt[:-1], label='-dE/dt')
            plt.axhline(0, linestyle='dashed')
            plt.xlabel("Time")
            plt.ylabel("-dE/dt")
        plt.legend()
        plt.show()
    print(f"Wall Time={wall_time}, nfev={nfev}")
    return (wall_time, nfev)


args = dict(N=128, dx=.1, divs=(1, 1), beta_Y=0, beta_S=5, T=0.05, check_dE=False)
psi = test_der_cooling(plot_dE=True, **args)

args = dict(N=128, dx=.1, divs=(1, 1), beta_0=-1j, beta_S=0, T=.05, log=False,check_dE=False)
psi = test_der_cooling(plot_dE=True, **args)

args = dict(N=128, dx=0.2, divs=(1, 1), beta_K=0, beta_V=5, beta_D=0, T=3, check_dE=True)
res = test_der_cooling(plot_dE=True, **args)

args = dict(N=128, dx=0.2, divs=(1, 1), beta_K=100, beta_V=0, beta_D=0, T=1, check_dE=True)
res = test_der_cooling(plot_dE=True, **args)

args = dict(N=128, dx=0.1, divs=(1, 1), beta_K=0, beta_V=0, beta_D=5, T=3, check_dE=False)
psi = test_der_cooling(plot_dE=True, **args)

# ## Dyadic Cooling
# To minimize the communication costs, we consider approximating $\op{H}_c$ by a set of dyads:
#
# $$
#   \op{H}_c = \sum_{n}\ket{a_n}f_n\bra{b_n} + \text{h.c.}, \qquad
#   \I\hbar\dot{E} = \sum_{i}\left(
#     f_i\sum_{n}\left(
#       \braket{\psi_n|\op{H}|a_i}\braket{b_i|\psi_n}
#       - 
#       \braket{\psi_n|a_i}\braket{b_i|\op{H}|\psi_n}
#     \right)
#     + \text{h.c.}
#   \right), \\
#   f_i = \frac{\I}{\hbar}
#   \sum_{n}\left(
#     \braket{a_i|\op{H}|\psi_n}\braket{\psi_n|b_i}
#     - 
#     \braket{a_i|\psi_n}\braket{\psi_n|\op{H}|b_i}
#   \right).
# $$
#
# A simplification occurs if $\ket{a_n} = \ket{b_n}$:
#
# $$
#   \op{H}_c = \sum_{n}\ket{a_n}f_n\bra{a_n}, \qquad
#   f_i = \frac{\I}{\hbar}
#   \sum_{n}\left(
#     \braket{a_i|\op{H}|\psi_n}\braket{\psi_n|a_i}
#     - 
#     \braket{a_i|\psi_n}\braket{\psi_n|\op{H}|a_i}
#   \right).
# $$
#
# Choosing $\ket{a_n} = \ket{x}$ leads to our local cooling potential $\op{V}_c$ while choosing $\ket{a_n} = \ket{k}$ leads to $\op{K}_c$.

# + {"id": "lLIWw-ya8ceW", "colab_type": "text", "cell_type": "markdown"}
# ## Free Fermions and Fermions in a Harmonic Trap

# + {"id": "EyLTqCwj8ceX", "colab_type": "code", "colab": {}}
Nx = 128
L = 23.0
dx = L/Nx
bcs = BCSCooling(N=Nx, L=None, dx=dx, beta_0=1j, beta_K=0, beta_V=0)
np.random.seed(1)
psi_ = np.exp(-bcs.xyz[0]**2/2)/np.pi**0.25

# + {"id": "APSW0DGNiuV0", "colab_type": "code", "colab": {}}
H0 = bcs._get_H(mu_eff=0, V=0)  # free particle
x = bcs.xyz[0]
V = x**2/2
H1 = bcs._get_H(mu_eff=0, V=V)  # harmonic trap
U0, Es0 = bcs.get_U_E(H0, transpose=True)
U1, Es1 = bcs.get_U_E(H1, transpose=True)

# + {"id": "gmdvwhivQ6eN", "colab_type": "text", "cell_type": "markdown"}
# # Prerequisite Test

# + {"id": "B3ifArgkA90M", "colab_type": "text", "cell_type": "markdown"}
# ## Check relation of $V_c(x)$, $K_c(k)$ with $H_c$
# * By definition, $V_c$ should be equal to the diagonal terms of $H_c$ in position space while $K_c$ in momentum space

# + {"id": "TztPB7ZFipxu", "colab_type": "code", "outputId": "08e80d3e-a3e6-4403-cd6c-389dcccd7c0a", "colab": {"base_uri": "https://localhost:8080/", "height": 34}}
np.random.seed(2)
psi = [np.random.random(np.prod(bcs.Nxyz)) - 0.5]
bcs.V=0
Vc = bcs.get_Vc(psi)
Kc = bcs.get_Kc(psi)
Hc = bcs.get_Hc(psi)
Hc_k = np.fft.ifft(np.fft.fft(Hc, axis=0), axis=1)
np.allclose(np.diag(Hc_k).real - Kc, 0), np.allclose(np.diag(Hc) - Vc, 0)

# + {"id": "Nkt4KOoaRQ0C", "colab_type": "text", "cell_type": "markdown"}
# ## Demostrate the $V_c$ and $K_c$ are Independent of Box Size
# * with fixed $dx$
#
# $$
# \hat{R}=\sum_n \ket{\psi_n}\bra{\psi_n}\qquad
# \hat{V}_c(x)=\int dx V_c(x) \ket{x}\bra{x} \qquad\\
# N=\braket{\psi|\psi}=\int dx\psi(x)^*\psi(x)\qquad
# V_c(x) =\braket{x|H_c|x}
# $$

# + {"id": "ysb1C9Hu8ces", "colab_type": "text", "cell_type": "markdown"}
# # Evolve in Imaginary Time

# + {"id": "D2BW3sz38cet", "colab_type": "code", "colab": {}}
dx = 0.1
def ImaginaryCooling():
    plt.rcParams["figure.figsize"] = (15, 4)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for Nx in [64, 128, 256]:
        labels = ["IR", "UV"]
        args = dict(N=Nx, beta_0=-1j, beta_K=0, beta_V=0)
        ir = BCSCooling(dx=dx, **args)  # dx fixed, L changes, IR
        uv = BCSCooling(dx=dx*64.0/Nx, **args)  # dx changes, L fixed: UV
        for i, s in enumerate([ir, uv]):
            s.g = 0# -1
            x = s.xyz[0]
            r2 = x**2
            V = x**2/2
            s.V = V
            u0 = np.exp(-x**2/2)/np.pi**4
            u0 = u0/u0.dot(u0.conj())**0.5
            u1=(np.sqrt(2)*x*np.exp(-x**2/2))/np.pi**4
            u1 = u1/u1.dot(u1.conj())**0.5

            psi_0 = Normalize(V*0 + 1+0*1j) # np.exp(-r2/2.0)*np.exp(1j*s.xyz[0])
            ts, psis,_ = s.solve([psi_0], T=10, rtol=1e-5, atol=1e-6, method='BDF')
            psi0 = psis[0][-1]
            E0, N0 = s.get_E_Ns([psi0])
            Es = [s.get_E_Ns([_psi + 1j*0])[0] for _psi in psis[0]]
            line, = ax1.semilogy(ts[0][:-2], (Es[:-2] - E0)/abs(E0), label=labels[i] + f":Nx={Nx}")
            plt.sca(ax2)
            l, = plt.plot(x, psi0)  # ground state
            plt.plot(x, psi_0, '--', c=l.get_c(), label=labels[i] + f":Nx={Nx}")  # initial state
            plt.plot(x, u0, '+', c=l.get_c())  # desired ground state
            E, N = s.get_E_Ns([V])
            # plt.title(f"E={E:.4f}, N={N:.4f}")
    plt.sca(ax1)
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('abs((E-E0)/E0)')
    plt.sca(ax2)
    plt.legend()
    plt.xlim(-5, 5)
    clear_output()
    plt.show()


# + {"id": "-0u8hZIMBjN2", "colab_type": "code", "outputId": "aad3f16e-6edb-41c6-8343-3869e67a5517", "colab": {"base_uri": "https://localhost:8080/", "height": 283}}
ImaginaryCooling()

# + {"id": "p5nZgiVpBr6w", "colab_type": "text", "cell_type": "markdown"}
# # Evolve in Real Time(Locally)
# * Unlike the imaginary time situation, where all wavefunction or orbits are used to renormlized the results, which can be expensive. Here wave functions are evolved in real time only using the local wavefunctions to cool down the energy.

# + {"id": "-Wxtf4KV8cew", "colab_type": "text", "cell_type": "markdown"}
# ## Split-operator method

# + {"id": "C5kkQAZcja8V", "colab_type": "text", "cell_type": "markdown"}
# * Assume all orbits are mutually orthogonal. For any given two obits , the state can be put as $\ket{\psi}=\ket{\psi_1 ⊗\psi_2}$. To compute the probability of a particle showing up in each of the ground state orbit, ie. $\ket{\phi_0}$ and $\ket{\phi_1}$:
#
# $$
#   P_i = (\abs{\braket{\phi_i|\psi_1}}^2+\abs{\braket{\phi_i|\psi_1}}^2)\qquad \text{i=0, 1}
# $$

# + {"id": "EIyZmO5HCt5x", "colab_type": "text", "cell_type": "markdown"}
# # Cool Down the Energy
# In this section, all kinds of configuration will be presented. The initial state(s) is (are) picked from the free fermions in a box.
#
#
# -

Cooling(N_state=6, Nx=128, N_data=25, start_state=4,  N_step=100, beta_V=5, beta_K=0, beta_D=0, plot_K=False);

Cooling(N_state=3, Nx=256, N_data=25, start_state=2,  N_step=100, beta_V=1, beta_K=1, beta_D=0);

N_data = 20
N_step = 100
Cooling(N_state=4, Nx=128, N_data=10, 
        init_state_ids=list(range(2, 4)), V0=1,
        N_step=N_step*10, beta_V=1, beta_K=1, divs=(1, 1), beta_D=0, plot_k=False);

# + {"id": "Eo0kxBxAVMhZ", "colab_type": "text", "cell_type": "markdown"}
# ## The simplest single wave function.
# * In the follow demo, we will show the efficiency of  the Cooling algorithm in different condition. Start with the simplest case where the inital state is a uniform wavefunction, then we turn on the hamonic potential, and monitor how the wave function evolve and the true ground state of the harmonic system is pupulated as the cooling proceeds. In the plot, the left panel plot the true ground state probability distribution $\psi^\dagger\psi$ in '+', and the evolving wavefunction probability distribution in solid line. 

# + {"id": "BXaJWUplV13u", "colab_type": "code", "colab": {}}
rets = Cooling(N_state=1, Nx=64, init_state_ids=(3,), N_data=25, N_step=100, beta_0=-1j, beta_V=0, beta_K=0, beta_D=0., divs=(1, 1), plot_k=False)

# + {"id": "Tr365cInZDqJ", "colab_type": "text", "cell_type": "markdown"}
# ### Double States
# However, in multiple state situation, if we state of state 1 and 3, it may cool down to the ground states

# + {"id": "ceTa7P2bZQax", "colab_type": "code", "colab": {}}
x, rets = Cooling(N_state=2, Nx=128, Lx=23, init_state_ids=(1,3), N_data=20, N_step=1000, beta_V=1, beta_K=0, beta_D=0., divs=(1, 1))

# + {"id": "P14489lt3y5X", "colab_type": "text", "cell_type": "markdown"}
# ### Triple States
# * if set Nx=128, the environment of Google collaboratory will yield different result than than I run locally. Where it not converge properly, but will give desired result on my local environment.

# + {"id": "ZBaymdxh3zaN", "colab_type": "code", "colab": {}}
Cooling(N_state=3, Nx=256, N_data=25, start_state=2, N_step=1000, beta_V=1, beta_K=1, beta_D=0);

# + {"id": "QETrGFXTGhcb", "colab_type": "text", "cell_type": "markdown"}
# # Experiment with another wavefunction
# * All the above trails used the 1D harmonic wavefunction, in which case, the $V_c$ and $K_c$ both works well to cool the energy($K_c$ performs better). However, in some case, $K_c$ may fail to cool the energy. The follow example we use GP wavefunction with interaction strength $g=1$, and no external potential.

# + {"id": "TzOboV3sDYN5", "colab_type": "code", "colab": {}}
args = dict(N=4, g=1)
egs = [BCSCooling(beta_0=-1j, beta_V=0.0, beta_K=0.0, **args),
       BCSCooling(beta_0=0.0, beta_V=0.0, beta_K=1.0, **args),
       BCSCooling(beta_0=1.0, beta_V=0.0, beta_K=1.0, **args),      
       BCSCooling(beta_0=0.0, beta_V=1.0, beta_K=0.0, **args),
       BCSCooling(beta_0=1.0, beta_V=1.0, beta_K=0.0, **args),
       BCSCooling(beta_0=0.0, beta_V=1.0, beta_K=1.0, **args),
       BCSCooling(beta_0=1.0, beta_V=1.0, beta_K=1.0, **args)]
labels = ['Imaginary Time',
          'K', 'H+K',
          'V', 'H+V',
          'V+K', 'H+V+K']
eg = egs[0]
psi0 = 2*(np.random.random(eg.Nxyz[0]) + 1j*np.random.random(eg.Nxyz[0]) - 0.5 - 0.5j)
V = np.array(psi0)*0
x=egs[0].xyz[0]
#psi0 = 0*x + 1.5 + 1.5*np.exp(-x**2/2)
psi_ground = 0*psi0 + np.sqrt((abs(psi0)**2).mean())
E0, N0 = eg.get_E_Ns([psi_ground], V=V)
Es = [[] for _n in range(len(egs))]
psis = [psi0.copy() for _n in range(len(egs))]
t_max = 3.0
Nstep = 4
Ndata = int(np.round(t_max/eg.dt/Nstep))
ts = np.arange(Ndata)*Nstep*eg.dt
for _n in range(Ndata):
    for n, eg in enumerate(egs):
        ps = [psis[n]]
        ps = eg.step(psis=ps, n=Nstep, V=V)
        psis[n] = ps[0]
        E, N = eg.get_E_Ns(psis=ps, V=V) 
        Es[n].append(E/E0 - 1.0)
Es = np.asarray(Es)

# + {"id": "rUPY34nVHSux", "colab_type": "code", "colab": {}}
plt.semilogy(ts, Es[1], c='C0', ls=':', label=labels[1])
plt.semilogy(ts, Es[2], c='C0', ls='-', label=labels[2])
plt.semilogy(ts, Es[3], c='C1', ls=':', label=labels[3])
plt.semilogy(ts, Es[4], c='C1', ls='-', label=labels[4])
plt.semilogy(ts, Es[5], c='C2', ls=':', label=labels[5])
plt.semilogy(ts, Es[6], c='C2', ls='-', label=labels[6])
plt.semilogy(ts, Es[0], c='k', ls='-', label=labels[0], scaley=False)
plt.xlabel("t")
plt.ylabel("E-E0")
plt.legend()

# + {"id": "LedHtcO3PO-1", "colab_type": "text", "cell_type": "markdown"}
# # With Pairing Field
# * to-do: update the code to support BCS with pairing field.
# -

b = BCSCooling(N=64, dx=0.1, beta_0=1, beta_V=1, delta=1, mus=(2, 2))
x = b.xyz[0]
V0 = x**2/3
V1 = x**2/2
H0 = b.get_H(mus_eff=b.mus, delta=b.delta, Vs=(V0, V0))
H1 = b.get_H(mus_eff=b.mus, delta=b.delta, Vs=(V1, V1))
U0, Es0 = b.get_U_E(H0, transpose=True)
U1, Es1 = b.get_U_E(H1, transpose=True)
N_state = 1
psi0 = U1[64]
psi = U0[64]
plt.plot(psi0)
E0, N0 = b.get_E_Ns(psis=[U1[64]], V=V1)
psis = [psi]
for i in range(20):
    psis = b.step(psis=psis, n=10, V=V1)
    plt.plot(psis[0],'--')
    plt.plot(psi0,'-')
    #print(psis[0].real)
    E, N = b.get_E_Ns(psis=psis, V=V1)
    plt.title(f"E0={E0.real},E={E.real}")
    plt.show()
    clear_output(wait=True)


