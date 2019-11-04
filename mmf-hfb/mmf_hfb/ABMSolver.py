from mmf_hfb.SolverABM import ABMEvolverAdapter
from mmf_hfb.BCSCooling import BCSCooling
from mmf_hfb.Potentials import HarmonicOscillator
import matplotlib.pyplot as plt
import numpy as np


def Normalize(psi):
    return psi/psi.dot(psi.conj())**0.5


def Prob(psi):
    return np.abs(psi)**2


def test_der_cooling(evolve=True, plot_dE=True, T=0.5, **args):
    b = BCSCooling(**args)
    h0 = HarmonicOscillator(w=1)
    h = HarmonicOscillator()
    da, db=b.divs    
    # k0 = 2*np.pi/b.L
    x = b.xyz[0]
    V = x**2/2
    H0 = b._get_H(mu_eff=0, V=0)
    H1 = b._get_H(mu_eff=0, V=V)
    U0, E0 = b.get_U_E(H0, transpose=True)
    U1, E1 = b.get_U_E(H1, transpose=True)
    psi0 = h.get_wf(x)
    psi0 = U1[0]
    psi = h0.get_wf(x, n=2)
    psi = U0[1]
    if evolve:
        #b.erase_max_ks()
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        ts, psiss = b.solve([psi], T=T, solver=ABMEvolverAdapter, rtol=1e-5, atol=1e-6, V=V, method='BDF')
        E0, _ = b.get_E_Ns([psi0], V=V)
        Es = [b.get_E_Ns([_psi], V=V)[0] for _psi in psiss[0]]
        dE_dt= [-1*b.get_dE_dt([_psi], V=V) for _psi in psiss[0]]
        plt.plot(x, Prob(psiss[0][0]), "+", label='init')
        plt.plot(x, Prob(psiss[0][-1]), '--', label="final")
        plt.plot(x, Prob(psi0), label='Ground')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(ts[0][:-2], (Es[:-2] - E0)/abs(E0), label="E")
        if plot_dE:
            plt.plot(ts[0][:-2], dE_dt[:-2], label='-dE/dt')
            plt.axhline(0, linestyle='dashed')
        plt.legend()
        plt.show()
    return psiss[0][-1]

if __name__ == '__main__':
    args = dict(N=128, dx=0.1, divs=(1, 1), beta0=1, beta_K=0, beta_V=0, beta_D=3, beta_Y=1, T=3, check_dE=False)
    psi = test_der_cooling(plot_dE=False, **args)
