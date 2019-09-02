import matplotlib.pyplot as plt
from mmf_hfb.bcs import BCS
import numpy as np



def assert_orth(psis):
    y1, y2 = psis
    assert np.allclose(y1.dot(y2.conj()), 0, rtol=1e-16)


def Assert(a, b, rtol=1e-10):
    assert np.allclose(a, b, rtol=rtol)

    
class BCSCooling(BCS):
    """
    1d Local Quamtum Friction class
    """

    def __init__(self, N=256, L=32, beta_0=1.0, beta_V=1.0, beta_K=1.0, dt_Emax=1.0):
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
        BCS.__init__(self, Nxyz=(N,), Lxyz=(L,))
        self.beta_0 = beta_0
        self.beta_V = beta_V
        self.beta_K = beta_K
        self._K2 = (self.hbar*np.array(self.kxyz[0]))**2/2/self.m
        Emax = self._K2.max()
        self.dt = dt_Emax*self.hbar/Emax

    def apply_H(self, psi, V):
        """compute dy/dt"""
        psi_k = np.fft.fft(psi)
        Kpsi = self._K2*psi_k
        Vpsi = V*psi
        return Kpsi + Vpsi

    def get_Vc(self, psi, V):
        N = psi.dot(psi.conj())*self.dV
        Hpsi = self.apply_H(psi, V=V)
        Vc = 2*(psi.conj()*Hpsi).imag*self.dV/N
        return Vc

    def get_Kc(self, psi, V):
        N = psi.dot(psi.conj())*self.dV
        psi_k = np.fft.fft(psi)*self.dV
        Hpsi = self.apply_H(psi, V=V)
        Vpsi_k = np.fft.fft(Hpsi)*self.dV
        Kc = 2*(psi_k.conj()*Vpsi_k).imag/N /self.dV
        return Kc
 
    def apply_expK(self, psi, V, factor=1):
        len0 = psi.dot(psi.conj())
        psi_k = np.fft.fft(psi)
        Kc = self.beta_K*self.get_Kc(psi=psi, V=V)
        psi = np.fft.ifft(
            np.exp(-1j*self.dt*factor*(self.beta_0*self._K2 + Kc))*psi_k)
        len1 = psi.dot(psi.conj())
        Assert(len0, len1)
        return psi
        
    def apply_expV(self, psi, V, factor=1):
        len0 = psi.dot(psi.conj())
        Vc = self.beta_V*self.get_Vc(psi, V=V)
        psi = np.exp(-1j*self.dt*factor*(self.beta_0*V + Vc)) * psi
        len1 = psi.dot(psi.conj())
        Assert(len0, len1)
        return psi
    
    def apply_Hc(self, psi, V):
        """
        Apply the cooling Hamiltonian.
        or, compute dy/dt w.r.t to Hc
        """
        H_psi = self.apply_H(psi, V=V)
        Vc_psi = self.get_Vc(psi, V=V)*psi
        Kc_psi = self.ifft(self.get_Kc(psi, V=V)*self.fft(psi))
        return (self.beta_0 * H_psi + self.beta_V * Vc_psi + self.beta_K * Kc_psi)
    
# old implementation
#     def get_Vc(self, psi):
#         psi2=(abs(psi)**2).max()
#         if psi2 == 0:
#             return psi
#         psi_k = np.fft.fft(psi)
#         Kpsi = np.fft.ifft(self._K2*psi_k)
#         Vc = 2*(psi.conj()*Kpsi).imag/psi2
#         return self.beta_V*Vc

#     def get_Kc(self, psi, V):
#         psi2=(abs(psi)**2).max()
#         if psi2 == 0:
#             return psi
#         psi_k = np.fft.fft(psi)
#         Vpsi_k = np.fft.fft(V*psi)
#         Kc = 2*(psi_k.conj()*Vpsi_k).imag/psi2
#         return self.beta_K*Kc

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
       
    def compute_dy_dt(self, t, psi, V, subtract_mu=True):
        """Return dy/dt for ODE integration."""
        Hpsi = self.apply_Hc(psi, V=V)
        if subtract_mu:
            Hpsi -= self.dotc(psi, Hpsi)/self.dotc(psi, psi)*psi
        return Hpsi/(1j*self.hbar)
    
    def get_U_E(self, H, transpose=False):
        """return Us and Vs and energy"""
        Es, U = sp.linalg.eigh(H)
        if transpose:
            return (U.T, Es)
        return (U, Es)

    def evolve(self, Us, V, n=1):
        """Evolve all stats"""
        for i in range(len(Us)):
            Us[i] = self.step(psi=Us[i], V=V, n=n)
        return Us
    
    def get_E_Ns(self, psis, V):
        E = 0
        N = 0
        for psi in psis:
            E_, N_ = self.get_E_N(psi=psi, V=V)
            E = E + E_
            N = N + N_
        return E, N
    
    def get_E_N(self, psi, V):
        """Return the energy and particle number `(E,N)`."""
        dx = self.dV
        n = abs(psi)**2
        K = abs(np.fft.ifft(self.hbar*self.kxyz[0]*np.fft.fft(psi)))**2/2/self.m
        E = ((V + K).sum()*dx).real
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
