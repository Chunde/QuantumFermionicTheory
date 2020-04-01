# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
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

# + [markdown] {"id": "ptb73kVS8ceS", "colab_type": "text"}
# # Quantum Friction Demonstration
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
# Here we demonstrate and compare a couple of forms of quantum friction.  We do this with free fermions in an external Harmonic oscillator potential.  We demonstrate the following:
#
# * Both $\op{V}_c$ and $\op{K}_c$ need evolution with the Hamiltonian or else they stall.  ($\op{H}$ is needed to generate the currents that will be cooled.)
# * cooling with the kinetic term $\op{K}_K$ does improve cooling, but only in conjunction with $\op{V}_c$.
# * Derivative cooling is still undergoing test.
# * In answer to Aurel's question: the Fermi surface is properly filled, even if the initial state does not have the same symmetry as the ground state.  (This does not work for a single state...)
# * We check the basis and box size by looking at semilog plots of the various states in both $x$ and $k$ space to make sure they decay by a factor of $\epsilon \sim 10^{-16}$.
# * We evolve the states using split-operator method with a fixed timestep chosen so that ???

# + {"id": "uV6ryeSMJsdb", "colab_type": "code", "colab": {}}
# run this cell first to set up the environment
import matplotlib.pyplot as plt
# %matplotlib inline
from IPython.display import display, clear_output
from mmf_hfb.bcs import BCS
from mmf_hfb.bcs_cooling import BCSCooling
from mmf_hfb.cooling import cooling, check_uv_ir_error
import numpy as np
import scipy as sp

def Prob(psi):
    return np.abs(psi)**2

def Normalize(psi):
    return psi/psi.dot(psi.conj())**0.5

N_data = 20
N_step = 100

# + [markdown] {"id": "zgRcXDoEJfi3", "colab_type": "text"}
# # Formation of a Fermi Surface
#
# ## Start with 20 initial plane-wave states
#
# ### cooling Function parameters:
#
# * N_State: integer, specify number of gound states to be observed(check occupancy)
# * Nx: integer,  number of lattice points
# * N_data: integer,  number of images used for animation
# * init_state_ids: list of index of plane-wave functions used as initial states
# * N_step: integer,  specify time step for each image update
# * beta_V: float, a factor acts on Vc
# * beta_K: float, a factor acts on Kc
# * beta_D: float, a factor acts on Vd(derivitive cooling potential, still undergoing test)
# * plot_n: bool,  toggle image for occupancy vs time
# * plot_k: bool, toggle image for n(k) vs k
# * V0: 0 or 1, toggle the harmonic potential, V0=0 will turn off the potential

# + {"id": "JNYtqSURH67W", "colab_type": "code", "outputId": "3243f8b2-5c8b-4720-9ebf-3fabc1d70a22", "colab": {"base_uri": "https://localhost:8080/", "height": 392}}
cooling(N_state=20, Nx=128, N_data=25, V0=1,
        init_state_ids=list(range(5,25)),
        N_step=N_step*10, beta_V=1, beta_K=1, divs=(1,1), beta_D=0, plot_n=True, plot_k=True);

# + {"id": "OZF4gADgKYks", "colab_type": "code", "outputId": "c3d79647-6663-4baa-9c81-afcd844ffc72", "colab": {"base_uri": "https://localhost:8080/", "height": 392}}
cooling(N_state=6, Nx=128, N_data=25,
        init_state_ids=list(range(4,8)),
        N_step=N_step, beta_V=1, beta_K=1, divs=(1,1), beta_D=0, plot_n=True, plot_k=True);

# + {"id": "vdPU-DSiO54k", "colab_type": "code", "outputId": "8da89236-f4ad-44ad-e7c1-37d2bcdbcf4b", "colab": {"base_uri": "https://localhost:8080/", "height": 392}}
cooling(N_state=5, Nx=128, N_data=20, 
        init_state_ids=list(range(3,6)),
        N_step=N_step, beta_V=1, beta_K=1, divs=(1,1), beta_D=0, plot_n=True, plot_k=True);

# + {"id": "9gXHZKlLI7Ar", "colab_type": "code", "outputId": "d8074c7e-759e-4f52-eba8-bcf3f1848b57", "colab": {"base_uri": "https://localhost:8080/", "height": 405}}
cooling(N_state=3, Nx=128, N_data=20, 
        init_state_ids=(1, 3, 5),
        N_step=N_step, beta_V=2, beta_K=2, divs=(1,1), beta_D=0, plot_n=True, plot_k=False);

# + [markdown] {"id": "gfSJNhrj7vcr", "colab_type": "text"}
# $$
#   n(x) = \sum_{n} \braket{x|\psi_n}\braket{\psi_n|x}\qquad
#   n_k = \sum_{n} \braket{k|\psi_n}\braket{\psi_n|k}, \qquad
#   n_n = \sum_{i} \braket{n|\psi_i}\braket{\psi_i|n},  
# $$
#
# $$
#   \sum_n\int\d{x} e^{-\I k x}n(x) \braket{x|\psi_n}\braket{\psi_n|x}
# $$

# + [markdown] {"id": "lLIWw-ya8ceW", "colab_type": "text"}
# # Free Fermions and Fermions in a Harmonic Trap

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
U0, Es0 = bcs.get_psis_es(H0, transpose=True)
U1, Es1 = bcs.get_psis_es(H1, transpose=True)

# + [markdown] {"id": "gmdvwhivQ6eN", "colab_type": "text"}
# # Prerequisite Test

# + [markdown] {"id": "B3ifArgkA90M", "colab_type": "text"}
# ## Check relation of $V_c(x)$, $K_c(k)$ with $H_c$
# * By defination, $V_c$ should be equal to the diagonal terms of $H_c$ in position space while $K_c$ in momentum space

# + {"id": "TztPB7ZFipxu", "colab_type": "code", "outputId": "89504aa9-2913-42e9-ddc3-ad7ce8a2769f", "colab": {"base_uri": "https://localhost:8080/", "height": 34}}
np.random.seed(2)
psi = [np.random.random(np.prod(bcs.Nxyz)) - 0.5]
Vc = bcs.get_Vc(psi, V=0)
Kc = bcs.get_Kc(psi, V=0)
Hc = bcs.get_Hc(psi, V=0)
Hc_k = np.fft.ifft(np.fft.fft(Hc, axis=0), axis=1)
np.allclose(np.diag(Hc_k).real - Kc, 0), np.allclose(np.diag(Hc) - Vc, 0)

# + [markdown] {"id": "cBYbgtdFBF72", "colab_type": "text"}
# ## Check Derivatives
# * As derivatitves will be used, we need to make sure the numerical method works by comparing its results to analytical ones

# + {"id": "cz7QCOsFBHI-", "colab_type": "code", "outputId": "4a2b63e3-a4b7-4eb6-dcb3-a12dd77fef3e", "colab": {"base_uri": "https://localhost:8080/", "height": 429}}
y = np.cos(x)**2
plt.subplot(211)
plt.plot(x, y)
dy = bcs.Del(y, n=1)
plt.plot(x, dy)
plt.plot(x, -np.sin(2*x), '+')
plt.subplot(212)
dy = bcs.Del(y, n=2)
plt.plot(x, dy)
plt.plot(x, -2*np.cos(2*x), '+')

# + [markdown] {"id": "Nkt4KOoaRQ0C", "colab_type": "text"}
# ## Demostrate the $V_c$ and $K_c$ are Independent of Box Size
# * with fixed $dx$
#
# $$
# \hat{R}=\sum_n \ket{\psi_n}\bra{\psi_n}\qquad
# \hat{V}_c(x)=\int dx V_c(x) \ket{x}\bra{x} \qquad\\
# N=\braket{\psi|\psi}=\int dx\psi(x)^*\psi(x)\qquad
# V_c(x) =\braket{x|H_c|x}
# $$

# + {"id": "uW47ksLDRUia", "colab_type": "code", "outputId": "db5557bc-6686-46a4-b907-d5e33700690f", "colab": {"base_uri": "https://localhost:8080/", "height": 591}}
dx = 0.1

def Check_Vc():
    #plt.rcParams["figure.figsize"] = (16,8)
    plt.figure(figsize=(16,8))
    Nx0 = 128
    for n in [1]: 
        Nx = n*Nx0
        args=dict(beta_0=-1j, beta_K=1, beta_V=1)
        offset = np.log(Nx)*0.1  # add a small offset so the plots would be visible for all Nx 
        s_ir = BCSCooling(N=Nx, dx=dx, **args)
        s_uv = BCSCooling(N=Nx, dx=dx*n, **args)
        for s in [s_ir, s_uv]:
            s.g=-1
            x = s.xyz[0]
            V_ext = x**2/2
            psi0 = np.exp(-x**2/2.0)*np.exp(1j*x)
            plt.subplot(121)
            plt.plot(x, Prob(psi0) + offset)
            plt.subplot(122)
            Vc = s.get_Vc(s.apply_H([psi0], V=V_ext), V=V_ext) 
            l, = plt.plot(x, Vc + offset)  # add some offset in y direction to seperate plots
    plt.subplot(121)
    plt.xlim(-10, 10)
    plt.subplot(122)
    plt.xlim(-10,10)
    
Check_Vc()


# + [markdown] {"id": "s5LBtTtXSt_a", "colab_type": "text"}
# ## Check machine precision
# * Also do this for exiting states in both postion and momentum space

# + {"id": "BDFIOcIYSqcT", "colab_type": "code", "outputId": "1ed6c686-4e59-4cc8-e83d-ef6459b32475", "colab": {"base_uri": "https://localhost:8080/", "height": 395}}
bcs = BCSCooling(N=Nx, L=None, dx=dx, beta_0=1, beta_V=1, beta_K=0, smooth=True) 
H = bcs._get_H(mu_eff=0, V=V)  # harmonic trap
Us, Es = bcs.get_psis_es(H, transpose=True)
check_uv_ir_error(Us[0], plot=True)


# + [markdown] {"id": "ysb1C9Hu8ces", "colab_type": "text"}
# # Evolve in Imaginary Time

# + {"id": "D2BW3sz38cet", "colab_type": "code", "colab": {}}
def ImaginaryCooling():
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for Nx in [64, 128, 256]:
        labels = ["IR", "UV"]
        args = dict(N=Nx, beta_0=-1j, beta_K=0, beta_V=0)
        ir = BCSCooling(dx=dx, **args) # dx fixed, L changes, IR
        uv = BCSCooling(dx=dx*64.0/Nx, **args) # dx changes, L fixed: UV
        for i, s in enumerate([ir, uv]):
            ir.g = 0# -1
            x = s.xyz[0]
            r2 = x**2
            V = x**2/2
            u0 = np.exp(-x**2/2)/np.pi**4
            u0 = u0/u0.dot(u0.conj())**0.5
            u1=(np.sqrt(2)*x*np.exp(-x**2/2))/np.pi**4
            u1 = u1/u1.dot(u1.conj())**0.5

            psi_0 = Normalize(V*0 + 1) # np.exp(-r2/2.0)*np.exp(1j*s.xyz[0])
            ts, psis, _ = s.solve([psi_0], T=10, rtol=1e-5, atol=1e-6, V=V, method='BDF')
            psi0 = psis[0][-1]
            E0, N0 = s.get_E_Ns([psi0], V=V)
            Es = [s.get_E_Ns([_psi], V=V)[0] for _psi in psis[0]]
            line, = ax1.semilogy(ts[0][:-2], (Es[:-2] - E0)/abs(E0), label=labels[i] + f":Nx={Nx}")
            plt.sca(ax2)
            l, = plt.plot(x, psi0)  # ground state
            plt.plot(x, psi_0, '--', c=l.get_c(), label=labels[i] + f":Nx={Nx}")  # initial state
            plt.plot(x, u0, '+', c=l.get_c())  # desired ground state
            E, N = s.get_E_Ns([V], V=V)
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


# + {"id": "-0u8hZIMBjN2", "colab_type": "code", "outputId": "355cd01f-8894-4fb2-8dda-582f5d5450f4", "colab": {"base_uri": "https://localhost:8080/", "height": 392}}
ImaginaryCooling()

# + [markdown] {"id": "p5nZgiVpBr6w", "colab_type": "text"}
# # Evolve in Real Time(Locally)
# * Unlike the imagary time situation, where all wavefunction or orbits are used to renormlized the results, which can be expensive. Here wave functions are evolved in real time only using the local wavefunctions to cool down the enery.
#
# * Assume all orbits are mutually orthogonal. For any given two obits , the state can be put as $\ket{\psi}=\ket{\psi_1 ⊗\psi_2}$. To compute the probability of a patcile showing up in each of the ground state oribt, ie. $\ket{\phi_0}$ and $\ket{\phi_1}$:
#
# $$
#   P_i = (\abs{\braket{\phi_i|\psi_1}}^2+\abs{\braket{\phi_i|\psi_1}}^2)\qquad \text{i=0,1}
# $$
#
# * In this section, all kinds of configuration will be presented. The initial state(s) is (are) picked from the free fermions in a box. 
#

# + [markdown] {"id": "Eo0kxBxAVMhZ", "colab_type": "text"}
# ## single wave
# * In the follow demo, we will show the efficiency of the cooling algorithm in different condition. Start with the simplest case where the inital state is a uniform wavefunction, then we turn on the hamonic potential, and monitor how the wave function evolve and the true ground state of the harmonic system is pupulated as the cooling proceeds. In the plot, the left panel plot the true ground state probability distribution $\psi^\dagger\psi$ in '+', and the evolving wavefunction probability distribution in solid line. 

# + [markdown] {"id": "3g1oa3n8WRqx", "colab_type": "text"}
# ### Start with and Even Single State
# If we pick the initial state with even nodes(state id is even), then such state have some overlap with the ground state in a harmonic trap. It's expected to cooling down to the ground state as above case.

# + {"id": "vLvdhzU4WYFS", "colab_type": "code", "outputId": "a06373d9-f8df-4c59-fae3-7237025fe264", "colab": {"base_uri": "https://localhost:8080/", "height": 392}}
rets = cooling(N_state=1, Nx=64, init_state_ids=(2,), N_data=25, N_step=100, beta_V=1, beta_K=1, beta_D=0., divs=(1,1))

# + [markdown] {"id": "c7vQCjWHVsaW", "colab_type": "text"}
# ### Start with an odd single state
# * if the initial state has no overlap with the true ground state, in single state case, we will see the cooling does not works.

# + {"id": "BXaJWUplV13u", "colab_type": "code", "outputId": "484ada2f-5cc4-433e-c691-01e811d074fa", "colab": {"base_uri": "https://localhost:8080/", "height": 392}}
rets = cooling(N_state=1, Nx=64, init_state_ids=(3,), N_data=20, N_step=100, beta_V=1, beta_K=1, beta_D=0., divs=(1,1), use_sp=True)

# + [markdown] {"id": "he1QRomv6Ip8", "colab_type": "text"}
# ## Triple-States
# * if set Nx=128, the environment of Google colaberator will yield different result than than I run locally. Where it not converge properly, but will give desired result on my local environment.

# + {"id": "ZBaymdxh3zaN", "colab_type": "code", "outputId": "a187a942-69da-41c8-a7bb-ae9a4664621d", "colab": {"base_uri": "https://localhost:8080/", "height": 392}}
cooling(N_state=3, Nx=256, N_data=25, 
        init_state_ids=(0,2,4),
        N_step=N_step*10, beta_V=1, beta_K=1, divs=(1,1), beta_D=0);

# + {"id": "1ZqR94_A2P8h", "colab_type": "code", "outputId": "838f1f43-8e36-49c2-d2cf-345f3b04c999", "colab": {"base_uri": "https://localhost:8080/", "height": 392}}
cooling(N_state=6, Nx=128, N_data=25, 
        init_state_ids=(2,3,6,7),
        N_step=N_step*10, beta_V=2, beta_K=2, divs=(1,1), beta_D=0, plot_k=True);

# + [markdown] {"id": "kS_QPkqhM39e", "colab_type": "text"}
# ## Many-States
# * Here we demostrate initally highly exicted states can be cooled down to the fermi surface.

# + {"id": "kg6LrrzLdPNa", "colab_type": "code", "outputId": "e2457e12-fcc1-40b2-fa3f-3456558ec69a", "colab": {"base_uri": "https://localhost:8080/", "height": 392}}
cooling(N_state=5, Nx=128, N_data=25, 
        init_state_ids=list(range(2,5)), V0=1,
        N_step=N_step*10, beta_V=1, beta_K=1, divs=(1,1),  beta_D=0, plot_n=True, plot_k=True);

# + {"id": "LogUah-CNIu3", "colab_type": "code", "outputId": "ce0e047e-e7e0-4738-b505-d46d8e720572", "colab": {"base_uri": "https://localhost:8080/", "height": 392}}
cooling(N_state=20, Nx=128, N_data=15, 
        init_state_ids=list(range(5,25)),
        N_step=N_step*10, beta_V=5, beta_K=10, divs=(1,1), beta_D=0, plot_n=True, plot_k=True);

# + [markdown] {"id": "DtyhhJd5vM94", "colab_type": "text"}
# ## cooling With Derivatives
#

# + [markdown] {"id": "lyFRMrlPdgVO", "colab_type": "text"}
# Consider the following cooling Hamiltonian.  (The motivation here is that the operators $\op{D}$ are derivatives, so this Hamiltonian is quasi-local.)
#
# $$
#   \op{H}_c = \int \d{x}\; \op{D}_a^\dagger\ket{x}V_{ab}(x)\bra{x}\op{D}_b + \text{h.c.},\\
#   \hbar \dot{E} = -\I\left(
#     \int\d{x}\;V_{ab}(x)
#     \braket{x|\op{D}_b[\op{R},\op{H}]\op{D}_a^\dagger|x}
# + \text {"incorrectly_encoded_metadata": "{h.c.}"}
#   \right).
# $$
#
# We can ensure cooling if we take:
#
# $$
#   \hbar V_{ab}(x) 
#   = (\I\hbar\braket{x|\op{D}_b[\op{R},\op{H}]\op{D}_a^\dagger|x})^*
#   = \I\hbar\braket{x|\op{D}_a[\op{R},\op{H}]\op{D}_b^\dagger|x}\\
#   = \braket{x|\op{D}_a|\psi}\braket{x|\op{D}_b|\dot{\psi}}^*
# + \braket {"incorrectly_encoded_metadata": "{x|\\op{D}_a|\\dot{\\psi}}\\braket{x|\\op{D}_b|\\psi}^*."}
# $$
#
# If $\op{D}_{a,b}(x)$ are just derivative operators $\braket{x|\op{D}_{a}|\psi} = \psi^{(a)}(x)$, then we have
#
# $$
#   \hbar V_{ab}(x) 
#   = \psi^{(a)}(x)\overline{\dot{\psi}^{(b)}(x)}
# + \dot {"incorrectly_encoded_metadata": "{\\psi}^{(a)}(x)\\overline{\\psi^{(b)}(x)},"}
# $$
#
# where
#
# $$
#   \dot{\psi}(x) = -\I\hbar\braket{x|\op{H}|\psi}
# $$
#
# is the time-derivative with respect to the original Hamiltonian.  Note that these "potentials" are no longer diagonal in either momentum or position space, so they should be implemented in the usual fashion with an integrator like ABM.

# + [markdown] {"id": "8xx81MBqDWqL", "colab_type": "text"}
# * <font color='red'>if turn on the derivative terms(beta_D !=1), with both sides with the first order derivative of the wavefunction(divs=(1, 1)), it will screw up the cooling. beta_D=1 may be too big.
# </font>

# + {"id": "O0gXrx5UL8Nb", "colab_type": "code", "outputId": "01812047-37ae-400d-fab1-3e37d68b1f71", "colab": {"base_uri": "https://localhost:8080/", "height": 392}}
cooling(N_data=N_data, N_step=N_step, beta_0=1, beta_V=1, beta_K=0, beta_D=1, divs=(1, 1));

# + [markdown] {"id": "QETrGFXTGhcb", "colab_type": "text"}
# # Epxeriment with another wavefunction
# * All the above trails used the 1D harmonic wavefunction, in which case, the $V_c$ and $K_c$ both works well to cool the energy($K_c$ performs better). However, in some case, $K_c$ may fail to cool the energy. The follow example we use GP wavefunction with interaction strength $g=1$, and no external potential.

# + {"id": "TzOboV3sDYN5", "colab_type": "code", "colab": {}}
args = dict(N=32, g=1)
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

# + {"id": "rUPY34nVHSux", "colab_type": "code", "outputId": "8468c33f-b8c3-476d-d615-a97d872b8437", "colab": {"base_uri": "https://localhost:8080/", "height": 443}}
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

# + [markdown] {"id": "LedHtcO3PO-1", "colab_type": "text"}
# # With Pairing Field
# * to-do: update the code to support BCS with pairing field.

# + {"id": "2H01mNmwqzZl", "colab_type": "code", "outputId": "d9f50b93-8083-425a-c31b-2caf842cfd5b", "colab": {"base_uri": "https://localhost:8080/", "height": 34}}
# %%file cooling.py



# + [markdown] {"id": "rhoysAzbqs3Q", "colab_type": "text"}
# ## BCSCooling Class

# + {"id": "HUoihT6vokj6", "colab_type": "code", "colab": {}}

