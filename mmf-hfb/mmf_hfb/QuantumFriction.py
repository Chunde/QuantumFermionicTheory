import matplotlib.pyplot as plt
from mmf_hfb.bcs import BCS
import numpy as np


class BCSCooling(BCS):
    """
    1d Local Quamtum Friction class
    """
    def get_v_ext(self, V0=None, **kw):
        """Return the external potential."""
        if V0 is None:
            V0 = self.V0
        V = V0*np.array(self.xyz)**2
        return (V, V)

    def __init__(self, N=64, L=0.46, beta_0=1.0, beta_V=1.0, beta_K=1.0, dt_Emax=1.0):
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
        self.V0 = 0
        self.Vs = None
        BCS.__init__(self, Nxyz=(N,), Lxyz=(L,))
        self.beta_0 = beta_0
        self.beta_V = beta_V
        self.beta_K = beta_K
        self._K2 = (self.hbar*np.array(self.kxyz))**2/2/self.m
        Emax = self._K2.max()
        self.dt = dt_Emax * self.hbar/Emax
        # initial U, V for the free fermions
        self.H0 = self.get_H(mus_eff=(0, 0), delta=0)
        self.UV0 = self.get_U_V(H=self.H0, transpose=True)
        
    def apply_expK(self, psi, V, factor=1):
        psi_k = np.fft.fft(psi)
        Kc = self.get_Kc(psi=psi, V=V)
        return np.fft.ifft(
            np.exp(-1j*self.dt*factor*(self.beta_0*self._K2 + Kc))
            * psi_k)
        
    def apply_expV(self, psi, V, factor=1):
        Vc = self.get_Vc(psi)
        return np.exp(-1j*self.dt*factor*(self.beta_0*V + Vc)) * psi

    def apply_H(self, psi, V):
        psi_k = np.fft.fft(psi)
        Kpsi = self._K2*psi_k
        Vpsi = V*psi
        return Kpsi + Vpsi

    def get_Vc(self, psi):
        psi2=(abs(psi)**2).max()
        if psi2 == 0:
            return psi
        psi_k = np.fft.fft(psi)
        Kpsi = np.fft.ifft(self._K2*psi_k)
        Vc = 2*(psi.conj()*Kpsi).imag/psi2
        return self.beta_V*Vc

    def get_Kc(self, psi, V):
        psi2=(abs(psi)**2).max()
        if psi2 == 0:
            return psi
        psi_k = np.fft.fft(psi)
        Vpsi_k = np.fft.fft(V*psi)
        Kc = 2*(psi_k.conj()*Vpsi_k).imag/psi2
        return self.beta_K*Kc

    def step(self, psi, V, n=1):
        """
        Evolve the state psi by applying n steps of the
        Split-Operator method.
        """
        psi = self.apply_expK(psi=psi, V=V, factor=0.5)
        for n in range(n):
            psi = self.apply_expV(psi=psi, V=V)
            psi = self.apply_expK(psi=psi, V=V)
        psi = self.apply_expK(psi=psi, V=V, factor=-0.5)
        return psi

    def evolve(self, n=1):
        self.V0 = 1
        Vs = self.get_v_ext()
        Us = self.UV0
        for U, V in zip(Us, Vs):
            for i in range(len(U)):
                U[i] = self.step(psi=U[i], V=V, n=n)

    def get_configuration(self, id):
        """return the states with given index: id"""
        self.V0 = 1
        Vs = self.get_v_ext()
        Us = self.UV0
        U_, V_ = self.get_U_V(self.get_H(mus_eff=(0, 0), delta=0), transpose=True)
        V = Vs[0]
        U = Us[0]
        psi = U[id]
        psi_ = U_[id]
        return (psi, psi_, V)
        # if True:
        #     U_, V_ = self.get_U_V(self.get_H(mus_eff=(0, 0), delta=0), transpose=True)
        #     V = Vs[0]
        #     U = Us[0]
        #     i = 300
        #     psi = U[i]
        #     psi_ = U_[i]
        #     return (psi, psi_, V)
        #     for _ in range(100):
        #         psi = self.step(psi=psi, V=V, n=100)
        #         print(np.max(psi + psi_), np.max(psi - psi_))
        # else:
        #     for U, V in zip(Us, Vs):
        #         for i in range(len(U)):
        #             U[i] = self.step(psi=U[i], V=V, n=n)

    def get_E_N(self, psi, V):
        """Return the energy and particle number `(E,N)`."""
        dx = self.dV
        n = abs(psi)**2
        K = abs(np.fft.ifft(self.hbar*self.kxyz[0]*np.fft.fft(psi)))**2/2/self.m
        E = ((V*n**2 + K).sum()*dx).real
        N = n.sum()*dx
        return E, N

if __name__ == "__main__":
    eg = eg_VK = BCSCooling(N=256)
    psi, psi0, V = eg.get_configuration(id=257)
    eg_K = BCSCooling(N=256, beta_0=1, beta_V=0.0)
    eg_V = BCSCooling(N=256, beta_0=1, beta_K=0.0)
    x = eg.xyz[0]
    # psi = np.random.random(eg.N) + 1j*np.random.random(eg.N) - 0.5 - 0.5j
    psi = 0*eg.xyz[0] + 1 + 1.0*np.exp(-eg.xyz[0]**2/2)
    #psi_ground = 0*psi0 + np.sqrt((abs(psi0)**2).mean())
    plt.plot(x, psi)
    plt.plot(x, psi0, '--')
    plt.show()
    E0, N0 = eg.get_E_N(psi0, V=V)
    Es = [[], [], []]
    psis = [psi, psi, psi]
    egs = [eg_VK, eg_K, eg_V]
    Ndata = 100
    Nstep = 100
    steps = list(range(100))
    for _n in range(Ndata):
        for n, eg in enumerate(egs):
            psis[n] = eg.step(psis[n], V=V, n=Nstep)
            E, N = eg.get_E_N(psis[n], V=V)
            Es[n].append(E - E0)
        for n, eg in enumerate(egs):
            plt.plot(x, psis[n][0])
        plt.show()
    
    for n, eg in enumerate(egs):
        plt.plot(steps, Es[n])
    plt.xlabel("Step")
    plt.ylabel("E-E0")
    plt.legend(['V+K', 'K', 'V'])
    plt.show()
