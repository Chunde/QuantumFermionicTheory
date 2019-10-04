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
# %pylab inline --no-import-all
from nbimports import *
import matplotlib.pyplot as plt
import numpy as np

# ## Some Helper Functions

# +
from mmf_hfb.BCSCooling import BCSCooling
from mmf_hfb.CoolingEg import CoolingEg

def H_exp(H, psi):
    return H.dot(psi).dot(psi.conj()).real

def Normalize(psi):
    return psi/psi.dot(psi.conj())**0.5

def Prob(psi):
    return np.abs(psi)**2


# -

# ## Analytical vs Numerical

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

# ## Check relation of $V_c(x)$, $K_c(k)$ with $H_c$
# * By defination, $V_c$ should be equal to the diagonal terms of $H_c$ in position space while $K_c$ in momentum space

np.random.seed(2)
psi = [np.random.random(np.prod(bcs.Nxyz)) - 0.5]
Vc = bcs.get_Vc(psi, V=0)
Kc = bcs.get_Kc(psi, V=0)
Hc = bcs.get_Hc(psi, V=0)
Hc_k = np.fft.ifft(np.fft.fft(Hc, axis=0), axis=1)
np.allclose(np.diag(Hc_k).real - Kc, 0), np.allclose(np.diag(Hc) - Vc, 0)

# ## Check Derivatives
# * As derivatitves will be used, we need to make sure the numerical method works by comparing its results to analytical ones

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


# ## Evolve with Imaginary Time

def ImaginaryCooling():
#    plt.figure(figsize(16, 8))
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


ImaginaryCooling()


# # Cooling Hamiltonian

# Start with Schrodinger Equation:
#
# $$
#   \I\hbar \ket{\dot{\psi}} \equiv \I\hbar \pdiff{\ket{\psi}}{t}
#   = \op{H}\ket{\psi}.
# $$
#
# Assume the original Hamiltonian does not depend on time explicitly, then the change of energy can be computed as:
# $$
# \dot{E}=\dot{\braket{H}}=\dot{\bra{\psi}}H\ket{\psi} + \bra{\psi}H\dot{\ket{\psi}}
# $$
# Here is we evolve the wavefunction with original hamiltonian, the energy will be conserved, so we need to find a different hamiltonian $H_c$ so that
# $$
# H_c\ket{\psi}=i\hbar\dot{\ket{\psi}}\qquad \dot{\ket{\psi}}=-\frac{i}{\hbar} H_c\ket{\psi}
# $$
# Then
# $$
# \dot{E}=\dot{\bra{\psi}}H\ket{\psi} + \bra{\psi}H\dot{\ket{\psi}}=\frac{i}{\hbar} \braket{\psi |H_c H|\psi} -\frac{i}{\hbar}  \braket{\psi |H_c H|\psi}=\frac{i}{\hbar}\braket{\psi [H_c, H]\psi}=\frac{\braket{\psi [H, H_c]\psi}}{i\hbar}
# $$
#
# If we can choose $\op{H}_c$ to ensure that the last term is negative-definite, then we have a cooling procedure.  The last term can be more usefully expressed in terms of the normalized density operator $\op{R} = \ket{\psi}\bra{\psi}/\braket{\psi|\psi}$ and using the cyclic property of the trace:
#
# $$
#   \frac{\braket{\psi|[\op{H},\op{H}_c]|\psi}}{\braket{\psi|\psi}} 
#   = \Tr\left(\op{R}[\op{H},\op{H}_c]\right)
#   = \Tr\left(\op{H}_c[\op{R},\op{H}]\right),\qquad
#   \hbar\dot{E} = -\braket{\psi|\psi}\Tr\left(\I[\op{R},\op{H}]\op{H}_c\right).
# $$
#
# This gives the optimal choice:
#
# $$
#   \op{H}_c = \left(\I[\op{R},\op{H}]\right)^{\dagger}
#            = \I[\op{R},\op{H}], \qquad
#   \hbar\dot{E} = -\braket{\psi|\psi}\Tr(\op{H}_c^\dagger\op{H}_c),\tag{1}
# $$
#

# ## Units

# First, $\braket{\psi|\psi}$ is dimensionless because is the particle number, the matrix $R=\sum_{\psi}\ket{\psi}\bra{\psi}$ should has not unit in all bases to satisifed $R^n=R$ for pure state. In $\ket{\psi}$ basis:
#
# $$
# R_{mn}=\braket{\psi_m|R|\psi_n}=\sum_l \braket{\psi_m|\psi_l}\braket{\psi_l|\psi_n}=\sum_l \delta_{lm}\delta_{ln}=\delta_{mn}\qquad \text{(dimensionless)}
# $$
#
# Then from the relation in (1), we can see $H_c$ and $H$ should have same unit.
#
# Since $H\ket{\psi}=E\ket{\psi}$, the $H$ must have unit of energy, its matrix element in bais of $\psi$ can be derived as:
# $$
# H_{mn}=\braket{\psi_m|H|\psi_n}=\int dx dx'\braket{\psi_m|x}\braket{x|H|x'}\braket{x'|\psi_n}=\int dx dx'\psi^*_m(x)\psi(x')\braket{x|H|x'}
# $$
# It's well known that the wave function $\psi(x)$ has unit of $\frac{1}{\sqrt{V}}$($V$ is the spatial volumn), and $\psi^*_m(x)\psi_n(x')$ has unit of $\frac{1}{V}$, $dx$ and $dx'$ are with unit $V$, then the units for the matrix element of Hamiltonian in basis $\psi$ and $x$ are different and connected by:
# $$
# [\braket{\psi|H|\psi}=\frac{[\braket{x|H|x}]}{V}
# $$
# That means the matrix elements of an operator would be different in different bases.

# The Hamiltonian matrix in a basis applied on a vector in that the space spaned by that basis will yield a new vector,

# Start with relations:
#
# $$
# R = \sum_n \ket{\psi_n}\bra{\psi_n}\qquad H_c = i[R, H]\\
# \dot{N}=\frac{d}{dt}\braket{\psi|\psi}=\braket{\dot{\psi}|\psi}+\braket{\psi|\dot{\psi}}=0\\
# V_c(x)=i\braket{x|H_c|x}=i\braket{x|[R, H]|x}=-\hbar\dot{n(x)}
# $$
#

# ## Demostrate the $V_c$ and $K_c$ are Independent of Box Size
# * with fixed $dx$

def Check_Vc():
    for Nx in [64, 128, 256]:
        offset = np.log(Nx)*0
        s = BCSCooling(N=Nx, dx=dx*64/Nx, beta_0=-1j, beta_K=1, beta_V=1)
        s.g = -1
        x = s.xyz[0]
        V_ext = x**2/2
        psi0 = np.exp(-x**2/2.0)*np.exp(1j*x)
        plt.subplot(121)
        plt.plot(x, Prob(psi0) + offset)
        plt.subplot(122)
        Vc = s.get_Vc(s.apply_H([psi0], V=V_ext), V=V_ext)
        l, = plt.plot(x, Vc + offset)  # add some offset in y direction to separate plots
    plt.subplot(121)
    plt.xlim(-10, 10)
    plt.subplot(122)
    plt.xlim(-10, 10)
    plt.xlabel("x"); plt.ylabel(f"$V_c$")
    clear_output()


Check_Vc()

# ## Split-operator method

from IPython.display import display, clear_output
from mmf_hfb.CoolingEg import CoolingEg


def PlayCooling(b, psis0, psis, V, N_data=10, N_step=100, **kw):
    x = b.xyz[0]
    E0, N0 = b.get_E_Ns(psis0, V=V)
    Es, cs, steps = [], [], list(range(N_data))
    for _n in range(N_data):
        psis = b.step(psis, V=V, n=N_step)
        E, N = b.get_E_Ns(psis, V=V)
        Es.append(abs(E - E0)/E0)
        for psi in psis:
            ax, = plt.plot(x, abs(psi)**2)
            cs.append(ax.get_c())
        for i, psi in enumerate(psis0):
            plt.plot(x, abs(psi)**2,'+', c=cs[i])
        # derivative:   
        #for i, psi in enumerate(psis):
        #    dpsi = bcs.Del(psi, n=1)
        #   plt.plot(x, abs(dpsi)**2,'--', c=cs[i])
        plt.title(
            f"E0={E0:5.4},E={E:5.4}, $" + r"\beta_0$" +f"={b.beta_0}, "
            +r"$\beta_V$"+f"={b.beta_V}, "+r" $\beta_K$" +f"={b.beta_K}"
            +r" $\beta_D$" +f"={b.beta_D}")
        plt.show()
        clear_output(wait=True)
    return psis


bcs = BCSCooling(N=Nx, L=None, dx=dx, beta_0=1, beta_V=1, beta_K=0, smooth=True) 
H = bcs._get_H(mu_eff=0, V=V)  # harmonic trap
Us, Es = bcs.get_U_E(H, transpose=True)
#plt.plot(x, np.log10(abs(Us[0])))

def Cooling(Nx=64, beta_0=1, N_state=1, **args):
    L = 23.0
    dx = L/Nx
    b = BCSCooling(N=Nx, L=None, dx=dx, **args)
    x = b.xyz[0]
    V = x**2/2
    H0 = b._get_H(mu_eff=0, V=0)  # free particle
    H1 = b._get_H(mu_eff=0, V=V)  # harmonic trap
    U0, Es0 = b.get_U_E(H0, transpose=True)
    U1, Es1 = b.get_U_E(H1, transpose=True)
    psis0 = U1[:N_state]
    psis = U0[:N_state]
    psis=PlayCooling(b=b,psis0=psis0, psis=psis, V=V, **args)


N_data = 20
N_step = 100

# ### Evolve with Original Hamiltonian

Cooling(N_data=N_data, N_step=N_step, beta_V=0, beta_K=0, beta_D=0)

# ### With $V_c$ Only

Cooling(N_data=N_data, N_step=N_step, beta_V=2, beta_K=0, beta_D=0)

# ### With $K_c$ only

Cooling(N_data=N_data, N_step=N_step, beta_V=0, beta_K=2, beta_D=0)

# ### With $V_c$ and $K_c$

Cooling(N_data=N_data, N_step=N_step, beta_V=1, beta_K=2, beta_D=0)

# ### Double States

Cooling(N_state=2, Nx=64, N_data=10, N_step=N_step, beta_V=1, beta_K=1, beta_D=0)

# ### With Derivatives

Cooling(N_data=N_data, N_step=N_step, beta_V=1, beta_K=0, beta_D=1, divs=(1, 1))

# # Another wave function

# +
L = 23.0
Nx=4
dx = L/Nx

egs = [
    BCSCooling(N=Nx,g=1,L=None, dx=dx, beta_V=1, beta_K=0, beta_D=0),
    BCSCooling(N=Nx,g=1,L=None, dx=dx, beta_V=0, beta_K=1, beta_D=0)]
psi0 = 2*(np.random.random(egs[0].Nxyz[0]) + 1j*np.random.random(egs[0].Nxyz[0]) - 0.5 - 0.5j)
x=egs[0].xyz[0]
#psi0 = 0*x + 1.5 + 1.5*np.exp(-x**2/2)
psi_ground = 0*psi0 + np.sqrt((abs(psi0)**2).mean())
V = np.array(psi0)*0
E0, N0 = egs[0].get_E_Ns([psi_ground], V=V)
# -

psis = [psi0]
Ess = []
for eg in egs:
    Es = []
    cs = []
    for _n in range(10):
        psis = eg.step(psis, V=V, n=N_step)
        E, N = eg.get_E_Ns(psis, V=V)
        Es.append(abs(E - E0)/E0)
        plt.subplot(121)
        for psi in psis:
            ax, = plt.plot(x, abs(psi)**2)
            cs.append(ax.get_c())
        for i, psi in enumerate(psis):
            plt.plot(x, abs(psi_ground)**2,'+', c=cs[i])
        plt.subplot(122)
        plt.plot(Es)
        plt.show()
        clear_output(wait=True)
    Ess.append(Es)

for Es in Ess:
    plt.plot(Es)

# ### Epxeriment with another wavefunction
# * All the above trails used the 1D harmonic wavefunction, in which case, the $V_c$ and $K_c$ both works well to cool the energy($K_c$ performs better). However, in some case, $K_c$ may fail to cool the energy. The follow example we use GP wavefunction with interaction strength $g=1$, and no external potential.

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

_n = 0
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

args = dict(N=32)
egs = [CoolingEg(beta_0=-1j, beta_V=0.0, beta_K=0.0, **args),
       CoolingEg(beta_0=0.0, beta_V=0.0, beta_K=1.0, **args),
       CoolingEg(beta_0=1.0, beta_V=0.0, beta_K=1.0, **args),      
       CoolingEg(beta_0=0.0, beta_V=1.0, beta_K=0.0, **args),
       CoolingEg(beta_0=1.0, beta_V=1.0, beta_K=0.0, **args),
       CoolingEg(beta_0=0.0, beta_V=1.0, beta_K=1.0, **args),
       CoolingEg(beta_0=1.0, beta_V=1.0, beta_K=1.0, **args)]
labels = ['Imaginary Time',
          'K', 'H+K',
          'V', 'H+V',
          'V+K', 'H+V+K']
eg = egs[0]
E0, N0 = eg.get_E_N(psi_ground)
Es = [[] for _n in range(len(egs))]
psis = [psi0.copy() for _n in range(len(egs))]
t_max = 3.0
Nstep = 4
Ndata = int(np.round(t_max/eg.dt/Nstep))
ts = np.arange(Ndata)*Nstep*eg.dt
for _n in range(Ndata):
    for n, eg in enumerate(egs):
        psis[n] = eg.step(psis[n], Nstep)
        E, N = eg.get_E_N(psis[n]) 
        Es[n].append(E/E0 - 1.0)
Es = np.asarray(Es)

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


