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

# + [markdown] id="ptb73kVS8ceS" colab_type="text"
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
# * Cooling with the kinetic term $\op{K}_K$ does improve cooling, but only in conjunction with $\op{V}_c$.
# * Derivative cooling is still undergoing test.
# * In answer to Aurel's question: the Fermi surface is properly filled, even if the initial state does not have the same symmetry as the ground state.  (This does not work for a single state...)
# * We check the basis and box size by looking at semilog plots of the various states in both $x$ and $k$ space to make sure they decay by a factor of $\epsilon \sim 10^{-16}$.
# * We evolve the states using split-operator method with a fixed timestep chosen so that ???

# + id="uV6ryeSMJsdb" colab_type="code" colab={}
# run this cell first to set up the environment
import matplotlib.pyplot as plt
# %matplotlib inline
from IPython.display import display, clear_output
from BCS import BCS
from BCSCooling import BCSCooling
from Cooling import Cooling, check_uv_ir_error
import numpy as np
import scipy as sp

def Prob(psi):
    return np.abs(psi)**2

def Normalize(psi):
    return psi/psi.dot(psi.conj())**0.5

N_data = 20
N_step = 100

# + [markdown] id="zgRcXDoEJfi3" colab_type="text"
# # Formation of a Fermi Surface
#
# ##Start with 20 initial plane-wave states
#
# ### Cooling Function parameters:
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

# +
# matplotlib.rcParams.update({'font.size': 18})
# -

cooling(N_state=4, Nx=256, N_data=25, start_state=2,  N_step=500, beta_V=1, beta_K=1, beta_D=0);

# + id="JNYtqSURH67W" colab_type="code" outputId="d83e503f-d893-4258-e925-e739224d90ef" colab={"base_uri": "https://localhost:8080/", "height": 388}
Cooling(N_state=20, Nx=128, N_data=25, V0=1,
        init_state_ids=list(range(5,25)),
        N_step=N_step*10, beta_V=1, beta_K=1, divs=(1,1), beta_D=0, plot_n=True, plot_k=True);

# + id="OZF4gADgKYks" colab_type="code" outputId="31aa425e-1cb6-4158-e0ae-84fc676a0e38" colab={"base_uri": "https://localhost:8080/", "height": 388}
Cooling(N_state=6, Nx=128, N_data=50,
        init_state_ids=list(range(4,8)),
        N_step=N_step, beta_V=1, beta_K=1, divs=(1,1), beta_D=0, plot_n=True, plot_k=True);

# + id="vdPU-DSiO54k" colab_type="code" outputId="0427c979-c3f2-43bb-d9ea-9d23ae9c28e1" colab={"base_uri": "https://localhost:8080/", "height": 388}
Cooling(N_state=5, Nx=128, N_data=50, 
        init_state_ids=list(range(3,6)),
        N_step=N_step, beta_V=1, beta_K=1, divs=(1,1), beta_D=0, plot_n=True, plot_k=True);

# + id="9gXHZKlLI7Ar" colab_type="code" outputId="d8074c7e-759e-4f52-eba8-bcf3f1848b57" colab={"base_uri": "https://localhost:8080/", "height": 405}
Cooling(N_state=3, Nx=128, N_data=20, 
        init_state_ids=(1, 3, 5),
        N_step=N_step, beta_V=2, beta_K=2, divs=(1,1), beta_D=0, plot_n=True, plot_k=False);

# + [markdown] id="gfSJNhrj7vcr" colab_type="text"
# $$
#   n(x) = \sum_{n} \braket{x|\psi_n}\braket{\psi_n|x}\qquad
#   n_k = \sum_{n} \braket{k|\psi_n}\braket{\psi_n|k}, \qquad
#   n_n = \sum_{i} \braket{n|\psi_i}\braket{\psi_i|n},  
# $$
#
# $$
#   \sum_n\int\d{x} e^{-\I k x}n(x) \braket{x|\psi_n}\braket{\psi_n|x}
# $$

# + [markdown] id="lLIWw-ya8ceW" colab_type="text"
# # Free Fermions and Fermions in a Harmonic Trap

# + id="EyLTqCwj8ceX" colab_type="code" colab={}
Nx = 128
L = 23.0
dx = L/Nx
bcs = BCSCooling(N=Nx, L=None, dx=dx, beta_0=1j, beta_K=0, beta_V=0)
np.random.seed(1)
psi_ = np.exp(-bcs.xyz[0]**2/2)/np.pi**0.25
H0 = bcs._get_H(mu_eff=0, V=0)  # free particle
x = bcs.xyz[0]
V = x**2/2
H1 = bcs._get_H(mu_eff=0, V=V)  # harmonic trap
U0, Es0 = bcs.get_U_E(H0, transpose=True)
U1, Es1 = bcs.get_U_E(H1, transpose=True)
np.random.seed(2)
psi = [np.random.random(np.prod(bcs.Nxyz)) - 0.5]
Vc = bcs.get_Vc(psi, V=0)
Kc = bcs.get_Kc(psi, V=0)
Hc = bcs.get_Hc(psi, V=0)
Hc_k = np.fft.ifft(np.fft.fft(Hc, axis=0), axis=1)
np.allclose(np.diag(Hc_k).real - Kc, 0), np.allclose(np.diag(Hc) - Vc, 0)

# + [markdown] id="Nkt4KOoaRQ0C" colab_type="text"
# # Demostrate the $V_c$ and $K_c$ are Independent of Box Size
# * with fixed $dx$
#
# $$
# \hat{R}=\sum_n \ket{\psi_n}\bra{\psi_n}\qquad
# \hat{V}_c(x)=\int dx V_c(x) \ket{x}\bra{x} \qquad\\
# N=\braket{\psi|\psi}=\int dx\psi(x)^*\psi(x)\qquad
# V_c(x) =\braket{x|H_c|x}
# $$

# + id="uW47ksLDRUia" colab_type="code" outputId="db5557bc-6686-46a4-b907-d5e33700690f" colab={"base_uri": "https://localhost:8080/", "height": 591}
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


# + id="D2BW3sz38cet" colab_type="code" colab={}
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
            ts, psis = s.solve([psi_0], T=10, rtol=1e-5, atol=1e-6, V=V, method='BDF')
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


# + id="-0u8hZIMBjN2" colab_type="code" outputId="355cd01f-8894-4fb2-8dda-582f5d5450f4" colab={"base_uri": "https://localhost:8080/", "height": 392}
ImaginaryCooling()

# + [markdown] id="p5nZgiVpBr6w" colab_type="text"
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

# + [markdown] id="Eo0kxBxAVMhZ" colab_type="text"
# ## single wave
# * In the follow demo, we will show the efficiency of the Cooling algorithm in different condition. Start with the simplest case where the inital state is a uniform wavefunction, then we turn on the hamonic potential, and monitor how the wave function evolve and the true ground state of the harmonic system is pupulated as the cooling proceeds. In the plot, the left panel plot the true ground state probability distribution $\psi^\dagger\psi$ in '+', and the evolving wavefunction probability distribution in solid line. 

# + [markdown] id="3g1oa3n8WRqx" colab_type="text"
# ### Start with and Even Single State
# If we pick the initial state with even nodes(state id is even), then such state have some overlap with the ground state in a harmonic trap. It's expected to cooling down to the ground state as above case.

# + id="vLvdhzU4WYFS" colab_type="code" outputId="a06373d9-f8df-4c59-fae3-7237025fe264" colab={"base_uri": "https://localhost:8080/", "height": 392}
rets = Cooling(N_state=1, Nx=64, init_state_ids=(2,), N_data=25, N_step=100, beta_V=1, beta_K=1, beta_D=0., divs=(1,1))

# + [markdown] id="c7vQCjWHVsaW" colab_type="text"
# ### Start with an odd single state
# * if the initial state has no overlap with the true ground state, in single state case, we will see the cooling does not works.

# + id="BXaJWUplV13u" colab_type="code" outputId="484ada2f-5cc4-433e-c691-01e811d074fa" colab={"base_uri": "https://localhost:8080/", "height": 392}
rets = Cooling(N_state=1, Nx=64, init_state_ids=(3,), N_data=20, N_step=100, beta_V=1, beta_K=1, beta_D=0., divs=(1,1), use_sp=True)

# + [markdown] id="he1QRomv6Ip8" colab_type="text"
# ## Triple-States
# * if set Nx=128, the environment of Google colaberator will yield different result than than I run locally. Where it not converge properly, but will give desired result on my local environment.

# + id="ZBaymdxh3zaN" colab_type="code" outputId="a187a942-69da-41c8-a7bb-ae9a4664621d" colab={"base_uri": "https://localhost:8080/", "height": 392}
Cooling(N_state=3, Nx=256, N_data=25, 
        init_state_ids=(0,2,4),
        N_step=N_step*10, beta_V=1, beta_K=1, divs=(1,1), beta_D=0);

# + id="1ZqR94_A2P8h" colab_type="code" outputId="838f1f43-8e36-49c2-d2cf-345f3b04c999" colab={"base_uri": "https://localhost:8080/", "height": 392}
Cooling(N_state=6, Nx=128, N_data=25, 
        init_state_ids=(2,3,6,7),
        N_step=N_step*10, beta_V=2, beta_K=2, divs=(1,1), beta_D=0, plot_k=True);

# + [markdown] id="kS_QPkqhM39e" colab_type="text"
# ## Many-States
# * Here we demostrate initally highly exicted states can be cooled down to the fermi surface.

# + id="kg6LrrzLdPNa" colab_type="code" outputId="e2457e12-fcc1-40b2-fa3f-3456558ec69a" colab={"base_uri": "https://localhost:8080/", "height": 392}
Cooling(N_state=5, Nx=128, N_data=25, 
        init_state_ids=list(range(2,5)), V0=1,
        N_step=N_step*10, beta_V=1, beta_K=1, divs=(1,1),  beta_D=0, plot_n=True, plot_k=True);

# + id="LogUah-CNIu3" colab_type="code" outputId="ce0e047e-e7e0-4738-b505-d46d8e720572" colab={"base_uri": "https://localhost:8080/", "height": 392}
Cooling(N_state=20, Nx=128, N_data=15, 
        init_state_ids=list(range(5,25)),
        N_step=N_step*10, beta_V=5, beta_K=10, divs=(1,1), beta_D=0, plot_n=True, plot_k=True);


# + [markdown] id="DtyhhJd5vM94" colab_type="text"
# ## Cooling With Derivatives
#

# + [markdown] id="lyFRMrlPdgVO" colab_type="text"
# Consider the following cooling Hamiltonian.  (The motivation here is that the operators $\op{D}$ are derivatives, so this Hamiltonian is quasi-local.)
#
# $$
#   \op{H}_c = \int \d{x}\; \op{D}_a^\dagger\ket{x}V_{ab}(x)\bra{x}\op{D}_b + \text{h.c.},\\
#   \hbar \dot{E} = -\I\left(
#     \int\d{x}\;V_{ab}(x)
#     \braket{x|\op{D}_b[\op{R},\op{H}]\op{D}_a^\dagger|x}
#     + \text{h.c.}
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
#   + \braket{x|\op{D}_a|\dot{\psi}}\braket{x|\op{D}_b|\psi}^*.
# $$
#
# If $\op{D}_{a,b}(x)$ are just derivative operators $\braket{x|\op{D}_{a}|\psi} = \psi^{(a)}(x)$, then we have
#
# $$
#   \hbar V_{ab}(x) 
#   = \psi^{(a)}(x)\overline{\dot{\psi}^{(b)}(x)}
#   + \dot{\psi}^{(a)}(x)\overline{\psi^{(b)}(x)},
# $$
#
# where
#
# $$
#   \dot{\psi}(x) = -\I\hbar\braket{x|\op{H}|\psi}
# $$
#
# is the time-derivative with respect to the original Hamiltonian.  Note that these "potentials" are no longer diagonal in either momentum or position space, so they should be implemented in the usual fashion with an integrator like ABM.

# + [markdown] id="8xx81MBqDWqL" colab_type="text"
# * In the derivative cooling, the highest monentum is discarded.

# + id="O0gXrx5UL8Nb" colab_type="code" colab={}
def  derivative_cooling(psi = None, evolve=True, plot_dE=True, T=0.5, **args):   
    b = BCSCooling(**args)
    da, db=b.divs    
    k0 = 2*np.pi/b.L
    x = b.xyz[0]
    V = x**2/2
    H0 = b._get_H(mu_eff=0, V=0)
    H1 = b._get_H(mu_eff=0, V=V)
    U0, E0 = b.get_U_E(H0, transpose=True)
    U1, E1 = b.get_U_E(H1, transpose=True)
    psi_1 = Normalize(np.cos(k0*x))
    if psi is None:
        psi = U0[0]  # np.exp(1j*n*(k0*x))
    psi_a = b.Del(psi, n=da)
    Hpsi = np.array(b.apply_H([psi], V=V))[0]/(1j)
    plt.figure(figsize=(18, 6))
    N = 2  
    Hpsi_a = b.Del(Hpsi, n=da)
    if da == db:
        psi_b = psi_a
        Hpsi_b = Hpsi_a
    else:
        psi_b = b.Del(psi, n=db)
        Hpsi_b = b.Del(Hpsi, n=db)
    Vc =  psi_a*Hpsi_b.conj() + Hpsi_a*psi_b.conj()
    if evolve:
        b.erase_max_ks()
        plt.subplot(1,N,1)
        ts, psiss = b.solve([psi], T=T, rtol=1e-5, atol=1e-6, V=V, method='BDF')
        psi0 = U1[0]
        E0, _ = b.get_E_Ns([psi0], V=V)
        Es = [b.get_E_Ns([_psi], V=V)[0] for _psi in psiss[0]]
        dE_dt= [-1*b.get_dE_dt([_psi], V=V) for _psi in psiss[0]]
        plt.plot(x, Prob(psiss[0][0]), "+", label='init')
        plt.plot(x, Prob(psiss[0][-1]), '--',label="final")
        plt.plot(x, Prob(U1[0]), label='Ground')
        plt.legend()
        plt.subplot(1,N,2)
        plt.plot(ts[0][:-2], (Es[:-2] - E0)/abs(E0), label="E")
        if plot_dE:
            plt.plot(ts[0][:-2], dE_dt[:-2], label='-dE/dt')
            plt.axhline(0, linestyle='dashed')
        plt.legend()
        plt.show()
    return psiss[0][-1]


# + id="0pWSFKL5Nn2j" colab_type="code" outputId="9c38aa3f-df26-4df3-b0c4-217bc70a536b" colab={"base_uri": "https://localhost:8080/", "height": 408}
# %%time 
args = dict(N=128, dx=0.1, divs=(1, 1), beta_K=0, beta_V=0, T=150, beta_D=0.02, check_dE=False)
psi =  derivative_cooling(plot_dE=False, **args)

# + [markdown] id="QETrGFXTGhcb" colab_type="text"
# # Epxeriment with another wavefunction
# * All the above trails used the 1D harmonic wavefunction, in which case, the $V_c$ and $K_c$ both works well to cool the energy($K_c$ performs better). However, in some case, $K_c$ may fail to cool the energy. The follow example we use GP wavefunction with interaction strength $g=1$, and no external potential.

# + id="TzOboV3sDYN5" colab_type="code" colab={}
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

# + id="rUPY34nVHSux" colab_type="code" outputId="8468c33f-b8c3-476d-d615-a97d872b8437" colab={"base_uri": "https://localhost:8080/", "height": 443}
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

# + [markdown] id="LedHtcO3PO-1" colab_type="text"
# # With Pairing Field
# * to-do: update the code to support BCS with pairing field.

# + [markdown] id="nKSBb4KzpBYD" colab_type="text"
# # Initialization: Run These First!

# + [markdown] id="DhAwSz5Nqvyg" colab_type="text"
# ## Cooling Methods

# + id="2H01mNmwqzZl" colab_type="code" outputId="86b4b5bf-4c99-4fde-b46a-9e1bf0e75dd6" colab={"base_uri": "https://localhost:8080/", "height": 34}
# %%file Cooling.py

from BCSCooling import BCSCooling
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
eps = 7./3 - 4./3 -1  # machine precision
def check_uv_ir_error(psi, plot=False):
    """check if a lattice configuration(N, L, dx) is good"""
    psi_k = np.fft.fft(psi)
    psi_log = np.log10(abs(psi)+eps)
    psi_log_k = np.log10(abs(psi_k)+eps)
    if plot:
        l, =plt.plot(psi_log_k)
        plt.plot(psi_log,'--', c=l.get_c())
        print(np.min(psi_log), np.min(psi_log_k))
    # assert the log10 value to be close to machine precision
    #assert np.min(psi_log)<-15
    assert np.min(psi_log_k) < -15
    
def get_occupancy(psis0, psis):
    """return occupancy"""
    def p(psis0):
        ps =[np.abs(psis0.conj().dot(psi))**2 for psi in psis]
        return sum(ps)
    return [p(psi0) for psi0 in psis0]

def plot_occupancy_n(ts,nss):
    """plot occupancy as cooling preceeds"""
    if len(nss) == 0:
        return
    num_plt = len(nss[0])
    datas = []
    for i in range(num_plt):
        datas.append([])
    for ns in nss:
        for i in range(num_plt):
            v = ns[i]
            datas[i].append(v)
    for i, data in enumerate(datas):
        plt.plot(data, label=f'{i}')
    plt.axhline(1, linestyle='dashed')
    plt.xlabel("Time(s)")
    plt.ylabel("Occupancy")
    plt.legend()

def plot_occupancy_k(b, psis):
    n_k = 0
    dx = b.dxyz[0]
    ks = b.kxyz[0]
    for psi in psis:
        n_k += abs(np.fft.fft(psi))**2
    n_k = np.fft.fftshift(n_k)
    ks = np.fft.fftshift(ks)   
    plt.plot(ks, n_k)
    plt.xlabel("k")
    plt.ylabel("n_k")


def plot_psis(b, psis0, psis, E, E0):
    x = b.xyz[0]
    cs = []
    for psi in psis:
        ax, = plt.plot(x, abs(psi)**2)
        cs.append(ax.get_c())
        for i, psi in enumerate(psis0):
            if i < len(cs):
                plt.plot(x, abs(psi)**2,'+', c=cs[i])
            else:
                plt.plot(x, abs(psi)**2,'+')
        plt.title(
            f"E0={E0:5.4},E={E:5.4}, $" + r"\beta_0$" +f"={b.beta_0}, "
            +r"$\beta_V$"+f"={b.beta_V}, "+r" $\beta_K$" +f"={b.beta_K}"
            +r" $\beta_D$" +f"={b.beta_D}")
        
def PlayCooling(
        b, psis0, psis, V, N_data=10, N_step=100,
        plot=True, plot_n=True, plot_k=True, **kw):
    
    E0, N0 = b.get_E_Ns(psis0[:len(psis)], V=V)
    Es, Ns = [], []
    plt.rcParams["figure.figsize"] = (18,6)
    plot_n_k = plot_n and plot_k
    for _n in range(N_data):
        Ps = get_occupancy(psis0, psis)
        Ns.append(Ps)
        
        psis = b.step(psis, V=V, n=N_step)
        E, N = b.get_E_Ns(psis, V=V)
        Es.append(abs(E - E0))
        ts = b.dt*N_step*np.array(list(range(len(Es))))
        if plot:
            plt.subplot(131)
            if plot_n_k:
                plot_occupancy_n(ts=ts, nss=Ns)
            else:
                plot_psis(b=b, psis0=psis0, psis=psis, E0=E0, E=E)
            plt.subplot(132)
            if plot_k:
                plot_occupancy_k(b=b,psis=psis)
            else:
                plot_occupancy_n(ts=ts, nss=Ns)
            
            plt.subplot(133)
            plt.plot(ts, Es)
            plt.xlabel("time(s)")
            plt.ylabel("Enery Diff")
            plt.axhline(0, linestyle='dashed')
            plt.show()
            clear_output(wait=True)     
    return psis, Es, Ns


def Cooling(Nx=128, Lx=23, init_state_ids=None, V0=1, beta_0=1, N_state=1, plot_k=True, **args):
    L = Lx
    dx = L/Nx
    b = BCSCooling(N=Nx, L=None, dx=dx, **args)
    x = b.xyz[0]
    V = V0*x**2/2
    H0 = b._get_H(mu_eff=0, V=0)  # free particle
    H1 = b._get_H(mu_eff=0, V=V)  # harmonic trap
    U0, Es0 = b.get_U_E(H0, transpose=True)
    U1, Es1 = b.get_U_E(H1, transpose=True)
    if init_state_ids is None:
        psis = U0[:N_state] # change the start states here if needed.
    else:
        assert len(init_state_ids) <= N_state
        psis=[U0[id] for id in init_state_ids]
    psis0 = U1[:N_state] # the ground states for the harmonic potential
    for i in range(len(psis0)):
        check_uv_ir_error(psis0[i], plot=False)
    for i in range(len(psis)):
        check_uv_ir_error(psis[i], plot=False)
    return [x, PlayCooling(b=b,psis0=psis0, psis=psis, V=V, plot_k=plot_k, **args)]


# + [markdown] id="rhoysAzbqs3Q" colab_type="text"
# ## BCSCooling Class

# + id="6Ji5pra9HYh7" colab_type="code" outputId="1ffda666-8c74-46ff-b426-6d9539fd5d42" colab={"base_uri": "https://localhost:8080/", "height": 34}
# %%file BCSCooling.py
from scipy.integrate import solve_ivp
import numpy as np
import numpy.linalg
from BCS import BCS
import numpy as np
import numpy.linalg

def assert_orth(psis):
    y1, y2 = psis
    inner_prod = y1.dot(y2.conj())
    ret = np.allclose(inner_prod, 0, rtol=1e-16)
    assert ret


def Normalize(psi):
    return psi/psi.dot(psi.conj())**0.5


class HamiltonianC(object):
    pass


class BCSCooling(BCS):
    """
    1d Local Quantum Friction class
    """

    def __init__(
            self, N=256, L=None, dx=0.1, delta=0, mus=(0, 0),
            beta_0=1.0, beta_V=1.0, beta_K=1.0, beta_D=1.0,
            dt_Emax=1.0, g=0, divs=None, check_dE=True, **args):
        """
        Arguments
        ---------
        beta_0 : float
           Portion of the original Hamiltonian H to include.
        beta_V : float
           Portion of the position cooling potential V_c.
        beta_K : float
           Portion of the momentum cooling potential K_c.
        beta_D: float
            Portion of the position cooling potential V_c with derivative
        """
        if L is None:
            L = N*dx
        BCS.__init__(self, Nxyz=(N,), Lxyz=(L,))
        self.L=L
        self.N=N
        self.dx=dx
        self.delta = delta
        self.mus = mus
        self.beta_0 = beta_0
        self.beta_V = beta_V
        self.beta_K = beta_K
        self.beta_D = beta_D
        self.g = g
        self.divs = divs
        self.check_dE = check_dE
        self._K2 = (self.hbar*np.array(self.kxyz[0]))**2/2/self.m
        self.dt = dt_Emax*self.hbar/self._K2.max()
        self.E_max = self._K2.max()
    
    def _get_uv(self, psi):
        uv = psi.reshape(2, len(psi)//2)
        return uv

    def get_V_eff(self, psis, V):
        """
            return effective potential for
            given external potential V and
            states
        """
        if self.g==0:
            return V
        ns = self.get_ns(psis)
        if self.delta !=0:  # n_a + n_b
            ns = ns[:len(ns)//2] + ns[len(ns)//2:]
        return sum(self.g*ns) + V
    
    def _apply_H(self, psi, V):
        if self.delta == 0:
            psi_k = self.fft(psi)
            Hpsi = self.ifft(self._K2*psi_k) + V*psi
            return Hpsi
        H = self.get_H(mus_eff=self.mus, delta=self.delta, Vs=(V, V))
        return H.dot(psi)  # apply H on psi

    def apply_H(self, psis, V):
        """compute dy/dt=H\psi"""
        V_eff = self.get_V_eff(psis, V=V)
        Hpsis = []
        for psi in psis:
            Hpsi = self._apply_H(psi, V=V_eff)
            Hpsis.append(Hpsi)
        return Hpsis

    def _apply_K(self, psi, V):
        if self.delta == 0:
            psi_k = self.fft(psi)
            Kpsi = self.ifft(self._K2*psi_k)
            return Kpsi
        u, v = self._get_uv(psi)
        uv = (u, -v)
        Kpsi = []
        for psi in uv:
            psi_k = self.fft(psi)
            Kpsi.extend(self.ifft(self._K2*psi_k))
        return Kpsi

    def apply_K(self, psis, V):
        """compute dy/dt with kinetic part only"""
        Hpsis = []
        V_eff = self.get_V_eff(psis, V=V)
        for psi in psis:
            Kpsi = self._apply_K(psi=psi, V=V_eff)
            Hpsis.append(Kpsi)
        return Hpsis

    def _apply_V(self, psi, V):
        if self.delta == 0:
            return V*psi
        return np.array([V - self.mus[0], -V + self.mus[1]]).ravel()*psi

    def apply_V(self, psis, V):
        """compute dy/dt with effective potential only"""
        V_eff = self.get_V_eff(psis, V=V)
        Hpsis = []
        for psi in psis:
            Vpsi = self._apply_V(psi, V=V_eff)
            Hpsis.append(Vpsi)
        return Hpsis

    def _div(self, psi, n=1):
        if self.delta == 0:
            for _ in range(n):
                psi = self._Del(alpha=(np.array([psi]).T,))[:, 0, ...][0].T[0]
            return psi

        u, v = np.array(psi).reshape(2, len(psi)//2)
        for _ in range(n):
            u = self._Del(alpha=(np.array([u]).T,))[:, 0, ...][0].T[0]
            v = self._Del(alpha=(np.array([v]).T,))[:, 0, ...][0].T[0]
        return u.extend(v)

    def Del(self, psi, n=1):
        """
        return the nth order derivative for psi
        support 1D function, should be generalized later
        """
        if n <=0:
            return psi
        psi = self._div(psi, n=n)
        return psi

    def get_N(self, psis):
        """return total particle number"""
        N = 0
        for psi in psis:
            N = N + psi.dot(psi.conj())*self.dV
        return N

    def _normalize_potential(self, Vc):
        """
        normalize a given cooling potential so that its max value
        is not larger than the maximum energy of the system.
        """
        return Vc
        E0 = 0.01*self.E_max
        V_max = np.max(abs(Vc))
        Vc = Vc/V_max*E0
        return Vc

    def _get_Vs(self, psis, V, divs=None):
        """
        return Vc or Vd
        -------------------
        Normalization
            Vc should not depend on particle number as
            it applies on single particle orbit(divided by N)
            it should not depend on lattice setting, dx or L
            its maximum value should be smaller than the energy
            cutoff self.E_max in order to be compatible with
            time step. How to rescale the cooling potential is
            not clear yet.
        """
        
        N = self.get_N(psis)
        Vc = 0
        if divs is None:
            # can also apply_H, but result is unchanged.
            # but with pairing, apply_H should be used
            Hpsis = self.apply_H(psis, V=V)
            for i, psi in enumerate(psis):
                Vc = Vc + 2*(psi.conj()*Hpsis[i]).imag
        else:  # Departure from locality
            da, db = self.divs
            # compute d^n \psi / d^n x
            psis_a = [self.Del(psi, n=da) for psi in psis]
            # d[d^n \psi / d^n x] / dt
            Hpsis = np.array(self.apply_H(psis, V=V))/(1j*self.hbar)
            Hpsis_a = [self.Del(psi, n=da) for psi in Hpsis]

            if da == db:
                psis_b = psis_a
                Hpsis_b = Hpsis_a
            else:
                psis_b = [self.Del(psi, n=db) for psi in psis]
                Hpsis_b = [self.Del(psi, n=db) for psi in Hpsis]
            for i in range(len(psis)):
                Vc = Vc + (
                    (psis_a[i]*Hpsis_b[i].conj()
                        +Hpsis_a[i]*psis_b[i].conj()))
        return Vc/N

    def get_Vc(self, psis, V):
        """return Vc potential"""
        Vc = 0*np.array(psis[0])
        if self.beta_V != 0:
            Vc = Vc + self._get_Vs(psis, V)
        return Vc

    def get_Vd(self, psis, V):
        """return the derivative cooling potential"""
        Vd = 0*np.array(psis[0])
        if self.beta_D !=0 and self.divs is not None:
            Vd = Vd + self._get_Vs(psis, V, self.divs)
        return Vd

    def _get_Kc(self, Hpsi, psi, V, N):
        """
        Kc is the diagonal of the H in k space, so
        even in the case with pairing, it is good to
        use psi as a single wavefunction without
        dividing it in to u, v components.
        """
        psi_k = self.fft(psi)*self.dV
        Vpsi_k = self.fft(Hpsi)*self.dV
        return 2*(psi_k.conj()*Vpsi_k).imag/N*self.dV/np.prod(self.Lxyz)

    def get_Kc(self, psis, V):
        N = self.get_N(psis)
        Kc = 0*np.array(psis[0])
        if self.beta_K == 0:
            return Kc
        Hpsis = self.apply_H(psis, V=V)  # use apply_H instead of apply_V for pairing
        for i, psi in enumerate(psis):
            Kc = Kc + self._get_Kc(Hpsi=Hpsis[i], psi=psi, V=V, N=N)
        return Kc

    def get_Hc(self, psis, V):
        """Return the full cooling Hamiltonian in position space."""
        size = np.prod(self.Nxyz)
        Hc = 0
        Hpsis = self.apply_H(psis, V=V)
        for _, (psi, Hpsi) in enumerate(zip(psis, Hpsis)):
            Hc_ = (1j*psi.reshape(size)[:, None]*Hpsi.conj().reshape(size)[None, :])
            Hc_ += Hc_.conj().T
            Hc = Hc + Hc_
        N = self.get_N(psis)
        return Hc/N

    def apply_Vd(self, psis, V):
        """
            apply Vd such as (V11) to the wave-functions
            NOTE: This may not be unitary
        """
        Vmn = self.beta_D*self.get_Vd(psis=psis, V=V)
        da, db = self.divs
        V11_psis = [-self.Del(Vmn*self.Del(psi=psi, n=da), n=db) for psi in psis]
        return V11_psis

    def _apply_expK(self, psi, V, Kc, factor=1):
        if self.delta == 0:
            psi_k = self.fft(psi)
            return self.ifft(np.exp(-1j*self.dt*factor*(self.beta_0*self._K2 + Kc))*psi_k)
        kuv = [self.fft(psi) for psi in self._get_uv(psi)]
        kc_uv = self._get_uv(Kc)
        signs = [1, -1]  # used to change the sign of k2
        expK = [self.ifft(
            np.exp(-1j*self.dt*factor*(
                self.beta_0*self._K2*sign + Kc))*psi_k) for (
                    sign, psi_k, Kc) in zip(signs, kuv, kc_uv)]
        return np.array(expK).ravel()

    def apply_expK(self, psis, V, factor=1):
        Kc = self.beta_K*self.get_Kc(psis=psis, V=V)
        for i, psi in enumerate(psis):
            psis[i] = self._apply_expK(psi, V=V, Kc=Kc, factor=factor)
        return psis

    def _apply_expV(self, psi, V, Vc, factor):
        if self.delta == 0:
            return np.exp(-1j*self.dt*factor*(self.beta_0*V +self.beta_V*Vc))*psi
        Vc_uv = self._get_uv(Vc)
        uv = self._get_uv(psi)
        Vs = (V - self.mus[0], -V + self.mus[1])
        expV = [np.exp(
            -1j*self.dt*factor*(
                self.beta_0*V_ +self.beta_V*Vc))*psi for (V_, psi, Vc) in zip(Vs, uv, Vc_uv)]
        return np.array(expV).ravel()

    def apply_expV(self, psis, V, factor=1):
        Vc = self.get_Vc(psis=psis, V=V)
        V_eff = self.get_V_eff(psis, V=V)
        for i, psi in enumerate(psis):
            psis[i] = self._apply_expV(psi=psi, V=V_eff, Vc=Vc, factor=factor)
        return psis
    
    def apply_Hc(self, psis, V):
        """
        Apply the cooling Hamiltonian.
        or, compute dy/dt w.r.t to Hc
        """
        Hc_psis = []
        H_psis = self.apply_H(psis=psis, V=V) if self.beta_0 != 0 else 0
        Vc = self.get_Vc(psis=psis, V=V) if self.beta_V !=0 else 0
        Kc = self.get_Kc(psis, V=V) if self.beta_K !=0 else 0
        Vd_psis = self.apply_Vd(psis=psis, V=V) if self.beta_D !=0 else (0,)*len(psis)
        for i, psi in enumerate(psis):
            Vc_psi = Vc*psi
            Kc_psi = self.ifft(Kc*self.fft(psi))
            Hc_psi = (
                self.beta_0*H_psis[i] + self.beta_V*Vc_psi
                +self.beta_K*Kc_psi + self.beta_D*Vd_psis[i])
            Hc_psis.append(Hc_psi)
        return Hc_psis

    def get_dE_dt(self, psis, V):
        """compute dE/dt"""
        H_psis = self.apply_H(psis, V=V)
        Hc_psis = self.apply_Hc(psis=psis, V=V)
        dE_dt = sum(
            [H_psi.conj().dot(Hc_psi)- Hc_psi.conj().dot(H_psi)
                for (H_psi, Hc_psi) in zip(H_psis, Hc_psis)])/(1j)
        return dE_dt

    def step(self, psis, V, n=1):
        """
        Evolve the state psi by applying n steps of the
        Split-Operator method.
        """
        psis = self.apply_expK(psis=psis, V=V, factor=0.5)
        for _ in range(n):
            psis = self.apply_expV(psis=psis, V=V)
            psis = self.apply_expK(psis=psis, V=V)
        psis = self.apply_expK(psis=psis, V=V, factor=-0.5)
        return psis

    def compute_dy_dt(self, t, psi, subtract_mu=True):
        """Return dy/dt for ODE integration."""
        if self.check_dE:
            dE_dt = self.get_dE_dt(psis=[psi], V=self.V)
            if abs(dE_dt) > 1e-16:
                assert dE_dt<= 0
        Hpsi = self.apply_Hc([psi], V=self.V)[0]
        if subtract_mu:
            Hpsi -= psi.conj().dot(Hpsi)/psi.dot(psi.conj())*psi
        return Hpsi/(1j*self.hbar)
    
    def get_U_E(self, H, transpose=False):
        """return Us and Vs and energy"""
        Es, U = numpy.linalg.eigh(H)
        if transpose:
            return (U.T, Es)
        return (U, Es)

    def solve(self, psis, T, V, dy_dt=None, **kw):
        self.V = V  # external potential
        self.psis = psis  # all single particle states
        ts, ys = [], []
        if dy_dt is None:
            dy_dt = self.compute_dy_dt
        for psi0 in psis:  # can be parallelized
            res = solve_ivp(fun=dy_dt, t_span=(0, T), y0=psi0, **kw)
            if not res.success:
                raise Exception(res.message)
            ts.append(res.t)
            ys.append(res.y.T)
        return(ts, ys)

    def get_ns(self, psis, shrink=False):
        """compute densities"""
        # if self.delta == 0:
        #     return sum(abs(psis)**2)
        # psis_ = psis.reshape(psis.shape[:1] + (2, psis.shape[1]//2))
        # Us, Vs = psis_[:, 0, ...], psis_[:, 1, ...]
        # return (sum(abs(Us)**2), sum(abs(Vs)**2))
        return sum(np.abs(psis)**2)

    def get_E_Ns(self, psis, V):
        E = 0
        N = 0
        ns = self.get_ns(psis)
        N = sum(ns)*self.dV
        if self.delta == 0:
            for psi in psis:
                K = psi.conj().dot(self.ifft(self._K2*self.fft(psi)))
                E = E + K.real*self.dV
            V_eff = (self.get_V_eff(psis, V=0)/2 + V)*ns
            E = E + V_eff.sum()*self.dV
        else:
            H = self.get_H(mus_eff=self.mus, delta=self.delta, Vs=(V, V))
            for psi in psis:
                E = E + psi.conj().dot(H.dot(psi))
        return E, N



# + [markdown] id="KeqyxIM9qp4q" colab_type="text"
# ## BCS Class

# + id="jHuskBS3_1uX" colab_type="code" outputId="1b14317a-2e2a-4348-8922-73d796aaad11" colab={"base_uri": "https://localhost:8080/", "height": 34}
# %%file BCS.py
"""BCS Equations in 1D, 2D, and 3D.

This module provides a class BCS for solving the BCS (BdG functional)
a two-species Fermi gas with short-range interactions.
"""
from collections import namedtuple
import itertools
import numpy
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class BCS(object):
    """Simple implementation of the BCS equations in a periodic box.

    We use all states in the box, regularizing the theory with a fixed
    coupling constant g_c which will depend on the box parameters.
    """
    hbar = 1.0
    m = 1.0

    def __init__(self, Nxyz=None, Lxyz=None, dx=None, T=0, E_c=None):
        """Specify any two of `Nxyz`, `Lxyz`, or `dx`.

        Arguments
        ---------
        Nxyz : (int, int, ...)
           Number of lattice points.  The length specifies the dimension.
        Lxyz : (float, float, ...)
           Length of the periodic box.
           Can also be understood as the largest wavelength of
           possible waves host in the box. Then the minimum
           wave-vector k0 = 2*pi/lambda = 2*pi/L, and all
           possible ks should be integer times of k0.
        dx : float
           Lattice spacing.
        T : float
           Temperature.
        """
        if dx is not None:
            if Lxyz is None:
                Lxyz = numpy.multiply(Nxyz, dx)
            elif Nxyz is None:
                Nxyz = numpy.ceil(numpy.divide(Lxyz, dx)).astype(int)
        dxyz = numpy.divide(Lxyz, Nxyz)

        self.xyz = np.meshgrid(*[np.arange(_N) * _d - _L / 2
                                 for _N, _L, _d in zip(Nxyz, Lxyz, dxyz)],
                               indexing='ij')
        self.kxyz = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(_N, _d)
                                  for _N, _d in zip(Nxyz, dxyz)],
                                indexing='ij')

        self.dxyz = dxyz
        self.Nxyz = Nxyz
        self.Lxyz = Lxyz
        self.E_c = E_c
        self.T = T

    @property
    def dim(self):
        return len(self.Nxyz)

    @property
    def dV(self):
        return numpy.prod(self.dxyz)

    def erase_max_ks(self):
        """set the max abs(ks) to zero as they may cause problems"""
        self.max_ks = []
        for i in range(self.dim):
            self.max_ks.append(self.kxyz[i][self.Nxyz[i]//2])
            if self.Nxyz[i] % 2 == 0:
                self.kxyz[i][self.Nxyz[i]//2]=0

    def restore_max_ks(self):
        """restore the original max ks"""
        for i in range(self.dim):
            if self.Nxyz[i] % 2 == 0 and self.kxyz[i][self.Nxyz[i]//2] == 0:
                self.kxyz[i][self.Nxyz[i]//2] = self.max_ks[i]
                
    def _get_K(self, k_p=0, twists=0, **kw):
        """Return the kinetic energy matrix."""
        ks_bloch = numpy.divide(twists, self.Lxyz)
        ks = [_k + _kb for _k, _kb in zip(self.kxyz, ks_bloch)]

        # Here we use a simple trick of applying the FFT to an
        # identify matrix.  This ensures that we appropriately
        # calculate the matrix structure without having to worry about
        # indices and phase factors.  The following transformation
        # should be correct however:
        #
        # U = np.exp(-1j*k[:, None]*self.x[None, :])/self.Nx
        #
        mat_shape = (numpy.prod(self.Nxyz),) * 2
        tensor_shape = tuple(self.Nxyz) * 2

        K = np.eye(mat_shape[0]).reshape(tensor_shape)
        bcast = (slice(None),)*self.dim + (None,)*self.dim
        K = (
            self.hbar**2/2/self.m*self.ifft(
                sum(_k**2 for _k in ks)[bcast]*self.fft(K))).reshape(mat_shape)
        if not np.allclose(k_p, 0, rtol=1e-16):
            k_p = np.diag(np.ones_like(sum(self.xyz).ravel())*k_p)
            K = K + k_p
        return K

    def _get_Del(self, twists=0, **kw):
        """
            Return the first order derivative matrix
        """
        ks_bloch = numpy.divide(twists, self.Lxyz)
        ks = [_k + _kb for _k, _kb in zip(self.kxyz, ks_bloch)]
        mat_shape = (numpy.prod(self.Nxyz),)*2
        tensor_shape = tuple(self.Nxyz)*2

        K = np.eye(mat_shape[0]).reshape(tensor_shape)
        bcast = (slice(None),) * self.dim + (None,)*self.dim
        K = (
            self.hbar**2/2.0/self.m*self.ifft(
                1j*sum(_k for _k in ks)[bcast]*self.fft(K))).reshape(mat_shape)
        return K

    def _Del(self, alpha, twists=0):
        """
        Apply the Del, or Nabla operation on a function alpha
        -------
        Note:
            Here we compute the first derivatives and pack them so that
            the first component is the derivative in x, y, z, etc.
        """
        ks_bloch = numpy.divide(twists, self.Lxyz)
        ks = [_k + _kb for _k, _kb in zip(self.kxyz, ks_bloch)]
        axes = range(1, self.dim + 1)
        alpha_t = self.fft(alpha, axes=axes)
        return np.stack(
            [self.ifft(1j*_k[None, ..., None]*alpha_t, axes=axes) for _k in ks])

    def get_Ks(self, twists, **args):
        K = self._get_K(twists=twists, **args)
        return (K, K)

    def fft(self, y, axes=None):
        if axes is None:
            axes = range(self.dim)
        return np.fft.fftn(y, axes=axes)

    def ifft(self, y, axes=None):
        if axes is None:
            axes = range(self.dim)
        return np.fft.ifftn(y, axes=axes)

    def get_v_ext(self, **kw):
        """Return the external potential."""
        return (0, 0)


    def _get_H(self, mu_eff, twists=0, V=0, **kw):
        K = self._get_K(twists=twists, **kw)
        mu_eff = np.zeros_like(sum(self.xyz)) + mu_eff
        return K - np.diag((mu_eff - V).ravel())

   


# + id="_nyORkFAKtJk" colab_type="code" outputId="9b5a6da9-b593-4ffe-d8b9-fa1e969949e9" colab={"base_uri": "https://localhost:8080/", "height": 34}
# !ls

# + id="HUoihT6vokj6" colab_type="code" colab={}

