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
from nbimports import  *
from IPython.core.debugger import set_trace
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# + {"id": "ptb73kVS8ceS", "colab_type": "text", "cell_type": "markdown"}
# # BCS Cooling Class
#
# * A class implement local friction that supports GP type cooling(single wave function) and BCS type cooling(multiple wavefunctions). When applied to BCS orbits, it will maintian the particle number and orthogonality of these orbits

# + {"id": "a8GAbW-pn2cI", "colab_type": "text", "cell_type": "markdown"}
# ## Define some commands
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

# + {"id": "kharf_G6odN6", "colab_type": "code", "colab": {}}

from mmf_hfb.BCSCooling import BCSCooling

def Normalize(psi):
    return psi/psi.dot(psi.conj())**0.5

def Prob(psi):
    return np.abs(psi)**2


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

# + {"id": "B3ifArgkA90M", "colab_type": "text", "cell_type": "markdown"}
# ## Check relation of $V_c(x)$, $K_c(k)$ with $H_c$
# * By defination, $V_c$ should be equal to the diagonal terms of $H_c$ in position space while $K_c$ in momentum space

# + {"id": "TztPB7ZFipxu", "colab_type": "code", "outputId": "57844bef-c978-406d-82ed-a3d476b44ec3", "colab": {"base_uri": "https://localhost:8080/", "height": 34}}
np.random.seed(2)
psi = [np.random.random(np.prod(bcs.Nxyz)) - 0.5]
Vc = bcs.get_Vc(psi, V=0)
Kc = bcs.get_Kc(psi, V=0)
Hc = bcs.get_Hc(psi, V=0)
Hc_k = np.fft.ifft(np.fft.fft(Hc, axis=0), axis=1)
np.allclose(np.diag(Hc_k).real - Kc, 0), np.allclose(np.diag(Hc) - Vc, 0)

# + {"id": "cBYbgtdFBF72", "colab_type": "text", "cell_type": "markdown"}
# ## Check Derivatives
# * As derivatitves will be used, we need to make sure the numerical method works by comparing its results to analytical ones

# + {"id": "cz7QCOsFBHI-", "colab_type": "code", "outputId": "ba8422d3-03d4-4e93-99b0-3d65368bc044", "colab": {"base_uri": "https://localhost:8080/", "height": 320}}
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


# + {"id": "ysb1C9Hu8ces", "colab_type": "text", "cell_type": "markdown"}
# ## Evolve in Imaginary Time

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
    plt.sca(ax1)
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('abs((E-E0)/E0)')
    plt.sca(ax2)
    plt.legend()
    plt.xlim(-5, 5)
    clear_output()
    plt.show()


# + {"id": "-0u8hZIMBjN2", "colab_type": "code", "outputId": "48948a71-4737-4695-ad7f-318ce6a5b616", "colab": {"base_uri": "https://localhost:8080/", "height": 283}}
ImaginaryCooling()


# + {"id": "p5nZgiVpBr6w", "colab_type": "text", "cell_type": "markdown"}
# ## Demostrate the $V_c$ and $K_c$ are Independent of Box Size
# * with fixed $dx$
#
# $$
# \hat{R}=\sum_n \ket{\psi_n}\bra{\psi_n}\qquad
# \hat{V}_c(x)=\int dx V_c(x) \ket{x}\bra{x} \qquad\\
# N=\braket{\psi|\psi}=\int dx\psi(x)^*\psi(x)\qquad
# V_c(x) =\braket{x|H_c|x}
# $$

# + {"id": "bXGfXLh_CB_m", "colab_type": "code", "colab": {}}
def Check_Vc():
    for Nx in [64, 128, 256]:
        offset = np.log(Nx)*0.1  # add a small offset so the plots would be visible for all Nx 
        s = BCSCooling(N=Nx, dx=dx,  beta_0=-1j, beta_K=1, beta_V=1)
        s.g = -1
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
    plt.xlabel("x"); plt.ylabel(f"$V_c$");
Check_Vc()

# + {"id": "-Wxtf4KV8cew", "colab_type": "text", "cell_type": "markdown"}
# ## Split-operator method

# + {"id": "nPRsT-oL8ce2", "colab_type": "code", "colab": {}}
N_data = 20
N_step = 100


# + {"id": "C5kkQAZcja8V", "colab_type": "text", "cell_type": "markdown"}
# * Assume all orbits are mutually orthogonal. For any given two obits , the state can be put as $\ket{\psi}=\ket{\psi_1 ⊗\psi_2}$. To compute the probability of a patcile showing up in each of the ground state oribt, ie. $\ket{\phi_0}$ and $\ket{\phi_1}$:
#
# $$
#   P_i = (\abs{\braket{\phi_i|\psi_1}}^2+\abs{\braket{\phi_i|\psi_1}}^2)\qquad \text{i=0,1}
# $$

# + {"id": "F_qwEBVdjaq-", "colab_type": "text", "cell_type": "markdown"}
#

# + {"id": "stIKJpqBgl8Z", "colab_type": "code", "colab": {}}
def get_occupancy(psis0, psis):
    """return occupancy"""
    def p(psi0):
        ps =[np.abs(psi0.conj().dot(psi))**2 for psi in psis]
        return sum(ps)
    return [p(psi0) for psi0 in psis0]

def plot_occupancy(nss):
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
    plt.legend()


# -

def check_uv_ir_error(psi, plot=False):
    """check if a lattice configuration(N, L, dx) is good"""
    psi_k = np.fft.fft(psi)
    psi_log = np.log10(abs(psi))
    psi_log_k = np.log10(abs(psi_k))
    if plot:
        plt.plot(psi_log)
        plt.plot(psi_log_k)
    assert np.min(psi_log)<-15
    assert np.min(psi_log_k) < -15


# + {"id": "8d7MfMQH8ce6", "colab_type": "code", "colab": {}}
def PlayCooling(b, psis0, psis, V, N_data=10, N_step=100, plot=True, **kw):
    x = b.xyz[0]
    E0, N0 = b.get_E_Ns(psis0, V=V)
    Es, cs, Ns = [], [], []
    for _n in range(N_data):
        psis = b.step(psis, V=V, n=N_step)
        E, N = b.get_E_Ns(psis, V=V)
        Es.append(abs(E - E0)/E0)
        if plot:
            plt.subplot(121)
            for psi in psis:
                ax, = plt.plot(x, abs(psi)**2)
                cs.append(ax.get_c())
            for i, psi in enumerate(psis0):
                plt.plot(x, abs(psi)**2,'+', c=cs[i])
            plt.title(
                f"E0={E0:5.4},E={E:5.4}, $" + r"\beta_0$" +f"={b.beta_0}, "
                +r"$\beta_V$"+f"={b.beta_V}, "+r" $\beta_K$" +f"={b.beta_K}"
                +r" $\beta_D$" +f"={b.beta_D}")
            plt.subplot(122)
            plot_occupancy(Ns)
            plt.show()
            clear_output(wait=True)
        Ps = get_occupancy(psis0, psis)
        Ns.append(Ps)
    return psis, Es, Ns


# + {"id": "NXzWTiBwCNVS", "colab_type": "code", "colab": {}}
def Cooling(Nx=128, init_state_ids=None, beta_0=1, N_state=1, **args):
    L = 23.0
    dx = L/Nx
    b = BCSCooling(N=Nx, L=None, dx=dx, **args)
    x = b.xyz[0]
    V = x**2/2
    H0 = b._get_H(mu_eff=0, V=0)  # free particle
    H1 = b._get_H(mu_eff=0, V=V)  # harmonic trap
    U0, Es0 = b.get_U_E(H0, transpose=True)
    U1, Es1 = b.get_U_E(H1, transpose=True)
    if init_state_ids is None:
        psis = U0[:N_state] # change the start states here if needed.
    else:
        assert len(init_state_ids) == N_state
        psis=[U0[id] for id in init_state_ids]
    psis0 = U1[:N_state] # the ground states for the harmonic potential
    return PlayCooling(b=b,psis0=psis0, psis=psis, V=V, **args)


# + {"id": "pU_3rmBeCjJI", "colab_type": "text", "cell_type": "markdown"}
# ## Check machine precision
# * Also do this for exiting states in both postion and momentum space

# + {"id": "EpAFrPKCCf45", "colab_type": "code", "outputId": "88eed221-c857-42a0-d2fc-237f330d43c7", "colab": {"base_uri": "https://localhost:8080/", "height": 286}}
bcs = BCSCooling(N=128, L=40, dx=None, beta_0=1, beta_V=1, beta_K=0, smooth=True)
x = bcs.xyz[0]
V = x**2/2
H = bcs._get_H(mu_eff=0, V=V)  # harmonic trap
Us, Es = bcs.get_U_E(H, transpose=True)
#plt.plot(x, np.log10(abs(Us[0])))
check_uv_ir_error(Us[0])

# + {"id": "EIyZmO5HCt5x", "colab_type": "text", "cell_type": "markdown"}
# ## Cool down the energy

# + {"id": "H0EtjerlWq-U", "colab_type": "code", "colab": {"base_uri": "https://localhost:8080/", "height": 282}, "outputId": "5b65bcb2-d55a-4186-b298-914fe847f8dd"}
N_data = 20
N_step = 100
rets = Cooling(N_state=2, Nx=128, init_state_ids=(1,3), N_data=50, N_step=200, beta_V=1, beta_K=1, beta_D=0., divs=(1,1))

# + {"id": "qesmvV_pC-M-", "colab_type": "text", "cell_type": "markdown"}
# ### Evolve with Original Harmitonian

# + {"id": "6DH4J9me8cfB", "colab_type": "code", "colab": {}}
Cooling(N_data=N_data, N_step=N_step, beta_V=0, beta_K=0, beta_D=0, plot=False);

# + {"id": "__QlP1023TOV", "colab_type": "text", "cell_type": "markdown"}
# ### With $V_c$ only

# + {"id": "Gtt1t3XH3R0r", "colab_type": "code", "outputId": "47aecb40-9348-4d8a-a3b3-1156db095a41", "colab": {"base_uri": "https://localhost:8080/", "height": 282}}
Cooling(N_data=N_data, N_step=N_step, beta_V=2, beta_K=0, beta_D=0);

# + {"id": "crFVHZeADJef", "colab_type": "text", "cell_type": "markdown"}
# ### With $K_c$ only

# + {"id": "9jtz-B_OC46R", "colab_type": "code", "outputId": "422844bc-3176-4bfd-c146-581559adc41e", "colab": {"base_uri": "https://localhost:8080/", "height": 282}}
Cooling(N_data=N_data, N_step=N_step, beta_V=0, beta_K=2, beta_D=0);

# + {"id": "J7AxO9tZDP43", "colab_type": "text", "cell_type": "markdown"}
# ### With $V_c$ and $K_c$

# + {"id": "9yFVXjhIDLYa", "colab_type": "code", "colab": {}}
Cooling(N_data=N_data, N_step=N_step, beta_V=1, beta_K=2, beta_D=0);

# + {"id": "P14489lt3y5X", "colab_type": "text", "cell_type": "markdown"}
# ### Double States
# * if set Nx=128, the environment of Google colaberator will yield different result than than I run locally. Where it not converge properly, but will give desired result on my local environment.

# + {"id": "ZBaymdxh3zaN", "colab_type": "code", "outputId": "2e90fd9a-2512-44e6-8a48-9c62d48c6616", "colab": {"base_uri": "https://localhost:8080/", "height": 282}}
Cooling(N_state=3, Nx=256, N_data=100,start_state=2, N_step=N_step*10, beta_V=1, beta_K=1, beta_D=0);

# + {"id": "lyFRMrlPdgVO", "colab_type": "text", "cell_type": "markdown"}
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

# + {"id": "8xx81MBqDWqL", "colab_type": "text", "cell_type": "markdown"}
# ### With Derivatives
# * <font color='red'>if turn on the derivative terms(beta_D !=1), with both sides with the first order derivative of the wavefunction(divs=(1, 1)), it will screw up the cooling. beta_D=1 may be too big.
# </font>

# + {"id": "O0gXrx5UL8Nb", "colab_type": "code", "outputId": "a3075d95-2bbb-4d00-ec61-d6b7492e68ff", "colab": {"base_uri": "https://localhost:8080/", "height": 282}}
Cooling(N_data=N_data, N_step=N_step, beta_0=1, beta_V=2, beta_K=0, beta_D=1, divs=(1, 1))

# + {"id": "nKevmpDPMDBX", "colab_type": "text", "cell_type": "markdown"}
# ### Turn on a little derivative term
# * if set beta_D to small value 0.2, the cooling precedure will be more effienct compared with $V_c$ only.

# + {"id": "2ej15LjHNLXu", "colab_type": "text", "cell_type": "markdown"}
#

# + {"id": "pfCjQEUCNMUi", "colab_type": "code", "outputId": "ada93709-59e8-4b93-843e-80cda6d64f6b", "colab": {"base_uri": "https://localhost:8080/", "height": 282}}
Cooling(N_data=N_data, N_step=N_step, beta_0=1, beta_V=2, beta_K=0, beta_D=0, divs=(1, 1))

# + {"id": "I18eY3dgDRTq", "colab_type": "code", "outputId": "0d853c38-f4de-4900-9815-0f479abcec45", "colab": {"base_uri": "https://localhost:8080/", "height": 282}}
Cooling(N_data=N_data, N_step=N_step, beta_0=1, beta_V=2, beta_K=0, beta_D=0.2, divs=(1, 1))

# + {"id": "BpsL9KrUMeB0", "colab_type": "text", "cell_type": "markdown"}
# ### Check how the beta_D effects the cooling

# + {"id": "46e5O5q8INNH", "colab_type": "code", "colab": {}}
beta_Ds = np.linspace(0, 0.23, 10)
Es = [Cooling(N_data=N_data, N_step=500, beta_0=1, beta_V=2, beta_K=0, beta_D=beta_D, divs=(1, 1), plot=False) for beta_D in beta_Ds]  

# + {"id": "V1krCSOkJuBJ", "colab_type": "code", "colab": {"base_uri": "https://localhost:8080/", "height": 301}, "outputId": "8651c737-4aff-44b1-8114-b53feeba1665"}
plt.plot(beta_Ds, Es)
plt.xlabel("beta_D")
plt.ylabel("(E-E0)/E0")

# + {"id": "QETrGFXTGhcb", "colab_type": "text", "cell_type": "markdown"}
# ### Epxeriment with another wavefunction
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

# + {"id": "rUPY34nVHSux", "colab_type": "code", "outputId": "5c92fd72-3cef-41c3-95b6-c379d5a44a49", "colab": {"base_uri": "https://localhost:8080/", "height": 334}}
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

# + {"id": "6Ji5pra9HYh7", "colab_type": "code", "colab": {}}

