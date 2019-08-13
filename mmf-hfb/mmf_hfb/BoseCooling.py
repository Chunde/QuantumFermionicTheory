import matplotlib.pyplot as plt
import numpy as np


class CoolingBose(object):
    g = hbar = m = 1.0

    def __init__(self, beta_0=1.0, beta_V=1.0, beta_K=1.0, N=256, dx=0.1, 
                 dt_Emax=1.0):
        """
        Arguments
        ---------
        beta_0 : float
           Portion of the original Hamiltonian H to include.
        beta_V : float
           Portion of the position cooling potential V_c.
        beta_K : float
           Portion of the momentum cooling potential K_c.
        """
        self.N = N
        self.dx = dx
        self.L = dx*N
        self.x = np.arange(N)*dx - self.L/2
        self.k = 2*np.pi * np.fft.fftfreq(N, dx)
        self.beta_0 = beta_0
        self.beta_V = beta_V
        self.beta_K = beta_K
        self._K2 = (self.hbar*self.k)**2/2/self.m
        Emax = self._K2.max()
        self.dt = dt_Emax * self.hbar/Emax

    def step(self, psi, n=1):
        """Evolve the state psi by applying n steps of the
        Split-Operator method."""
        psi = self.apply_expK(psi, factor=0.5)
        for n in range(n):
            psi = self.apply_expV(psi)
            psi = self.apply_expK(psi)
        psi = self.apply_expK(psi, factor=-0.5)
        return psi

    def apply_expK(self, psi, factor=1):
        psi_k = np.fft.fft(psi)
        Kc = self.get_Kc(psi=psi)
        return np.fft.ifft(
            np.exp(-1j*self.dt*factor*(self.beta_0*self._K2 + Kc))
            * psi_k)
        
    def apply_expV(self, psi, factor=1):
        Vc = self.get_Vc(psi)
        n = abs(psi)**2
        V = self.g*n
        return np.exp(-1j*self.dt*factor*(self.beta_0*V + Vc)) * psi

    def apply_H(self, psi):
        n = abs(psi)**2
        psi_k = np.fft.fft(psi)
        Kpsi = self._K2*psi_k
        Vpsi = self.g*n*psi
        return Kpsi + Vpsi

    def get_Vc(self, psi):
        psi_k = np.fft.fft(psi)
        Kpsi = np.fft.ifft(self._K2*psi_k)
        Vc = 2*(psi.conj()*Kpsi).imag/(abs(psi)**2).max()
        return self.beta_V*Vc

    def get_Kc(self, psi):
        n = abs(psi)**2
        psi_k = np.fft.fft(psi)
        V = self.g*n
        Vpsi_k = np.fft.fft(V*psi)
        Kc = 2*(psi_k.conj()*Vpsi_k).imag/(abs(psi_k)**2).max()
        return self.beta_K*Kc

    def get_E_N(self, psi):
        """Return the energy and particle number `(E,N)`."""
        dx = self.dx
        n = abs(psi)**2
        K = abs(np.fft.ifft(self.hbar*self.k*np.fft.fft(psi)))**2/2/self.m
        E = ((self.g*n**2/2 + K).sum()*dx).real
        N = n.sum()*dx
        return E, N
    
    def plot(self, psi):
        plt.clf()
        plt.plot(self.x, abs(psi)**2)
        plt.ylim(0, 2)
        plt.twinx()
        Vc = self.get_Vc(psi)
        plt.plot(self.x, Vc, 'C1')
        plt.ylim(-2, 2)
        E, N = self.get_E_N(psi)
        plt.title(f"E={E:.4f}, N={N:.4f}")
        return plt.gcf()


if __name__ == "__main__":
    eg = eg_VK = CoolingBose()
    eg_K = CoolingBose(beta_0=1, beta_V=0.0)
    eg_V = CoolingBose(beta_0=1, beta_K=0.0)
    #psi = np.random.random(eg.N) + 1j*np.random.random(eg.N) - 0.5 - 0.5j
    psi0 = 0*eg.x + 1 + 1.0*np.exp(-eg.x**2/2)
    psi_ground = 0*psi0 + np.sqrt((abs(psi0)**2).mean())
    E0, N0 = eg.get_E_N(psi_ground)
    Es = [[], [], []]
    psis = [psi0, psi0, psi0]
    egs = [eg_VK, eg_K, eg_V]
    Ndata = 100
    Nstep = 100
    for _n in range(Ndata):
        for n, eg in enumerate(egs):
            psis[n] = eg.step(psis[n], Nstep)
            E, N = eg.get_E_N(psis[n])
            Es[n].append(E - E0)
    Es = np.asarray(Es)
    plt.semilogy(Es.T)
    plt.xlabel("Step")
    plt.ylabel("E-E0")
    plt.legend(['V+K', 'K', 'V'])
    plt.show()
