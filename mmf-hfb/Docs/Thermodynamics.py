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

# + {"id": "40nYf7WoWEp7", "colab_type": "text", "cell_type": "markdown"}
# # Equation of State and Thermodynamics

# + {"id": "4grqmC9WWKOO", "colab_type": "text", "cell_type": "markdown"}
# Suppose we have energy density $\mathcal{E}(n)$.
# -

import mmf_setup;mmf_setup.nbinit()
# %pylab inline --no-import-all
from nbimports import *
from mmf_hfb.BCSCooling import BCSCooling
import matplotlib.pyplot as plt
import numpy as np
import numpy
import time
import sympy

# + {"id": "tjs-c6dv4rHG", "colab_type": "code", "colab": {}}
n_ = sympy.var('n', positive=True)
E_ = (n_**2-1)**2
mu_ = E_.diff(n_)
P_ = mu_*n_ - E_
E, mu, P = [sympy.lambdify([n_], f_)
            for f_ in [E_, mu_, P_]]

# + {"id": "6UM532NZ9ixU", "colab_type": "code", "colab": {}}
n_ = sympy.var('n', real=True)
E0_ = (n_**2-1)**2
E_ = sympy.Piecewise((E0_.subs(n_,0)-100*n_, n_<0), (E0_, True))
mu_ = E_.diff(n_)
P_ = mu_*n_ - E_
E, mu, P = [sympy.lambdify([n_], f_)
            for f_ in [E_, mu_, P_]]

# + {"id": "SxVmohcn5VD7", "colab_type": "code", "outputId": "8823595a-8103-4be2-856e-29079de7ef4a", "colab": {"base_uri": "https://localhost:8080/", "height": 355}}
plt.figure(figsize=(15, 6))
n = np.linspace(-0.01, 1.5, 100)
plt.figure(figsize=(10,5))
ax = plt.subplot(221);
ax.plot(n, E(n))
ax.set(xlabel='n', ylabel='E')
ax = plt.subplot(222)
ax.plot(mu(n), P(n))
ax.set(xlabel=r'$\mu$', ylabel='P', xlim=(-2,0), ylim=(-1.5,0))
ax = plt.subplot(223)
ax.plot(n, (E(n)-E(0))/n)
ax.set(xlabel=r'n', ylabel='(E-E_0)/n', ylim=(-2, 1))


# + {"id": "KwyfZRh9WOTN", "colab_type": "code", "outputId": "54590f3e-32e8-4c6b-fca3-ad1574ae49b4", "colab": {"base_uri": "https://localhost:8080/", "height": 350}}
# %pylab inline --no-import-all

def E(n, d=0):
    if d == 0:
        return (n**2-1)**2-1
    elif d == 1:
        return 4*n*(n**2-1)
    elif d == 2:
        return 12*n**2-4
    elif d == 3:
        return 24*n

def mu(n, d=0):
    return E(n, d=1+d)

def P(n, d=0):
    if d == 0:
        return mu(n)*n - E(n)
    elif d == 1:
        return mu(n, d=1)*n + mu(n) - E(n, d=1)

n = np.linspace(0,1.5,100)
plt.plot(n, E(n)/n)
plt.ylim(-1.5,1)


# + {"id": "F8ViDMXNkz3R", "colab_type": "code", "colab": {}, "cell_type": "markdown"}
# # 2D BCS Cooling
# -
def plot_occupancy_k(b, psis):
    n_k = 0
    ks = b.kxyz
    for psi in psis:
        n_k += abs(b.fft(psi))**2
    n_k = np.fft.fftshift(n_k)
    ks = np.fft.fftshift(ks)   
    plt.plot(ks, n_k)
    plt.xlabel("k")
    plt.ylabel("n_k")


class FissionCooling(BCSCooling):
    """Fission Potential??"""
    def get_Vint(self, psis):
        ns = self.get_ns(psis)
        return (ns**2 - 1)**2


def Cooling(plot=True, N_state=2, plot_dE=True, T=0.5, log=False, **args):  
    b = FissionCooling(**args)
    da, db=b.divs    
    k0 = 2*np.pi/b.L
    x, y = b.xyz
    V = x**2/2
    V = sum(_x**2 for _x in b.xyz)
    #b.V = V/2
    #b.g = -1
    x0 = 0.5
    H0 = b._get_H(mu_eff=0, V=0)
    H1 = b._get_H(mu_eff=0, V=V)
    U0, E0 = b.get_U_E(H0, transpose=True)
    U1, E1 = b.get_U_E(H1, transpose=True)
    psi0 = [b.Normalize(U1[i]) for i in range(N_state)]
    phase = ((x-x0) + 1j*y)*((x+x0) - 1j*y)
    psi =  [b.Normalize(U0[i]) for i in range(N_state)] # b.Normalize(1.0*np.exp(1j*np.angle(phase))) #
    start_time = time.time()
    ts, psis, nfev = b.solve([psi], T=T, rtol=1e-5, atol=1e-6,solver=None, method='BDF')
    wall_time = time.time() - start_time
    if plot:
        E0, _ = b.get_E_Ns(psis[-1])
        Es = [b.get_E_Ns(_psi)[0] for _psi in psis]
        dE_dt= [-1*b.get_dE_dt(_psi) for _psi in psis]
        for i in range(N_state):
            b.plot(psis[-1][i])
        plt.semilogy(ts, Es)
    print(f"Wall Time={wall_time}, nfev={nfev}")
    return (wall_time, nfev, b, psis)


args = dict(N=32, dx=0.25, dim=2, N_state=2, beta_0=1, beta_V=0.1, beta_S=0, T=1, log=False, check_dE=False)
wall_time, nfev, b, psis=BCS2D_Cooling(**args)

from mmfutils import plot as mmfplt            
def plot_occupancy_k(b, psis):
    n_k = 0
    kx, ky = b.kxyz
    kx = kx.T[0]
    ky = ky[0]
    x, y = b.xyz
    for psi in psis:
        n_k += abs(b.fft(psi))**2
    n_k = np.fft.fftshift(n_k)
    kx, ky = np.fft.fftshift(kx), np.fft.fftshift(ky)
    #print(ks)
    mmfplt.imcontourf(x, y, n_k)
    plt.colorbar()


plot_occupancy_k(b, psis[-1])

b = BCSCooling(N=64, dx=0.1,beta_K=1, delta=1, mus=(2, 2))
x = b.xyz[0]
V0 = x**2/3
V1 = x**2/2
H0 = b.get_H(mus_eff=b.mus, delta=b.delta, Vs=(V0, V0))
H1 = b.get_H(mus_eff=b.mus, delta=b.delta, Vs=(V1, V1))
U0, Es0 = b.get_U_E(H0, transpose=True)
U1, Es1 = b.get_U_E(H1, transpose=True)
psi0 = U1[0]
psi = U0[10]
plt.plot(psi0)
b.V = V1
E0, N0 = b.get_E_Ns(psis=[psi0])
psis = [psi]
Es = []
for i in range(100):
    plt.figure(figsize=(15, 6))

    psis = b.step(psis=psis, n=100)
    plt.subplot(121)
    plt.plot(abs(psis[0])**2,'--')
    plt.plot(abs(psi0)**2,'-')
    #print(psis[0].real)
    E, N = b.get_E_Ns(psis=psis)
    Es.append(E)
    plt.title(f"E0={E0.real},E={E.real}")
    plt.subplot(122)
    plt.semilogy(abs((np.array(Es)-E0)/E0))
    plt.show()
    
    clear_output(wait=True)


