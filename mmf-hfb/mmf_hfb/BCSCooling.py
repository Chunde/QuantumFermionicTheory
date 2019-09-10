from scipy.integrate import solve_ivp
from mmf_hfb.bcs import BCS
import numpy as np


def Assert(a, b, rtol=1e-10):
    assert np.allclose(a, b, rtol=rtol)

    
def assert_orth(psis):
    y1, y2 = psis
    inner_prod = y1.dot(y2.conj())
    ret = np.allclose(inner_prod, 0, rtol=1e-16)
    assert ret


def H_exp(H, psi):
    return H.dot(psi).dot(psi.conj()).real


def Normalize(psi):
    return psi/psi.dot(psi.conj())**0.5


class BCSCooling(BCS):
    """
    1d Local Quamtum Friction class
    """

    def __init__(
            self, N=256, L=None, dx=0.1,
            beta_0=1.0, beta_V=1.0, beta_K=1.0,
            dt_Emax=1.0, g=0, divs=None):
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
        if L is None:
            L = N*dx
        BCS.__init__(self, Nxyz=(N,), Lxyz=(L,))
        self.beta_0 = beta_0
        self.beta_V = beta_V
        self.beta_K = beta_K
        self._K2 = (self.hbar*np.array(self.kxyz[0]))**2/2/self.m
        self.dt = dt_Emax*self.hbar/self._K2.max()
        self.g = g
        self.divs = divs

    def get_V(self, psis, V):
        return sum(self.g*np.abs(psis)**2) + V
    
    def apply_H(self, psis, V):
        """compute dy/dt"""
        V = self.get_V(psis, V=V)
        Hpsis = []
        for psi in psis:
            psi_k = np.fft.fft(psi)
            Kpsi = self.ifft(self._K2*psi_k)
            Vpsi = V*psi
            Hpsis.append(Kpsi + Vpsi)
        return Hpsis
    
    def Del(self, psi, n):
        """Now only support 1D function, should be genenilzed later"""
        if n <=0:
            return psi
        for _ in range(n):
            psi = self._Del(alpha=(np.array([psi]).T,))[:, 0, ...][0].T[0]
        return psi

    def get_N(self, psis):
        N = 0
        for psi in psis:
            N = N + psi.dot(psi.conj())*self.dV
        return N

    def get_Vc(self, psis, V):
        N = self.get_N(psis)
        Hpsis = self.apply_H(psis, V=V)
        Vc = 0
        if self.divs is None:
            Hpsis = self.apply_H(psis, V=V)
            for i, psi in enumerate(psis):
                Vc = Vc + 2*(psi.conj()*Hpsis[i]).imag*self.dV/N
        else:  # Departure from locality
            da, db = self.divs
            psis_a = [self.Del(psi, n=da) for psi in psis]
            Hpsis_a = self.apply_H(psis_a, V=V)
            if da == db:
                psis_b = psis_a
                Hpsis_b = Hpsis_a
            else:
                psis_b = [self.Del(psi, n=db) for psi in psis]
                Hpsis_b = self.apply_H(psis_b, V=V)
            for i in range(len(psis)):
                Vc = Vc + (
                    (Hpsis_a[i]*psis_b[i].conj()
                        -psis_a[i]*Hpsis_b[i].conj())).imag*self.dV/N
        return Vc

    def get_Kc(self, psis, V):
        N = self.get_N(psis)
        Kc = 0
        Hpsis = self.apply_H(psis, V=V)
        for i, psi in enumerate(psis):
            psi_k = np.fft.fft(psi)*self.dV
            Vpsi_k = np.fft.fft(Hpsis[i])*self.dV
            Kc = Kc + 2*(psi_k.conj()*Vpsi_k).imag/N /self.dV
        return Kc
 
    def apply_expK(self, psis, V, factor=1):
        Kc = self.beta_K*self.get_Kc(psis=psis, V=V)
        for i, psi in enumerate(psis):
            len0 = psi.dot(psi.conj())
            psi_k = np.fft.fft(psi)
            psi = np.fft.ifft(
                np.exp(-1j*self.dt*factor*(self.beta_0*self._K2 + Kc))*psi_k)
            len1 = psi.dot(psi.conj())
            Assert(len0, len1)
            psis[i] = psi
        return psis
        
    def apply_expV(self, psis, V, factor=1):
        Vc = self.beta_V*self.get_Vc(psis=psis, V=V)
        for i, psi in enumerate(psis):
            len0 = psi.dot(psi.conj())
            psi = np.exp(-1j*self.dt*factor*(self.beta_0*V + Vc))*psi
            len1 = psi.dot(psi.conj())
            Assert(len0, len1)
            psis[i]=psi
        return psis
    
    def apply_Hc(self, psi, V):
        """
        Apply the cooling Hamiltonian.
        or, compute dy/dt w.r.t to Hc
        """
        H_psi = self.apply_H(psis=[psi], V=V)[0] if self.beta_0 !=0 else 0
        Vc_psi = self.get_Vc(psis=[psi], V=V)[0]*psi if self.beta_V !=0 else 0
        Kc_psi = (
            self.ifft(self.get_Kc([psi], V=V)[0]*self.fft(psi))
            if self.beta_K !=0 else 0)
        return (self.beta_0*H_psi + self.beta_V*Vc_psi + self.beta_K*Kc_psi)
    
    def step(self, psis, V, n=1):
        """
        Evolve the state psi by applying n steps of the
        Split-Operator method.
        """
        psis = self.apply_expK(psis=psis, V=V, factor=0.5)
        for n in range(n):
            psis = self.apply_expV(psis=psis, V=V)
            psis = self.apply_expK(psis=psis, V=V)
        psis = self.apply_expK(psis=psis, V=V, factor=-0.5)
        return psis
       
    def compute_dy_dt(self, t, psi, subtract_mu=True):
        """Return dy/dt for ODE integration."""
        Hpsi = self.apply_Hc(psi, V=self.V)
        if subtract_mu:
            Hpsi -= psi.conj().dot(Hpsi)/psi.dot(psi.conj())*psi
        return Hpsi/(1j*self.hbar)
    
    def get_U_E(self, H, transpose=False):
        """return Us and Vs and energy"""
        Es, U = np.linalg.eigh(H)
        if transpose:
            return (U.T, Es)
        return (U, Es)

    def solve(self, psis, T, V, **kw):
        self.V = V  # external potential
        self.psis = psis  # all single particle states
        ts, ys = [], []
        for psi0 in psis:  # can be parallelized
            res = solve_ivp(fun=self.compute_dy_dt, t_span=(0, T), y0=psi0, **kw)
            if not res.success:
                raise Exception(res.message)
            ts.append(res.t)
            ys.append(res.y.T)
        return(ts, ys)

    def get_density(self, psis):
        """compute densities"""
        ns = np.abs(psis)**2
        return sum(ns)
   
    def get_E_Ns(self, psis, V):
        E = 0
        N = 0
        n0 = self.get_density(psis)
        N = sum(n0)*self.dV
        V = (self.g*n0**2/2 + V*n0).sum()
        for psi in psis:
            K = psi.conj().dot(self.ifft(self._K2*self.fft(psi)))
            E = E + K.real*self.dV
        E = E + V*self.dV
        return E, N
