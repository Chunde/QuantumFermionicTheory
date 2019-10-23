from scipy.integrate import solve_ivp
from mmf_hfb.bcs import BCS
import numpy as np
import numpy.linalg


def assert_orth(psis):
    y1, y2 = psis
    inner_prod = y1.dot(y2.conj())
    ret = np.allclose(inner_prod, 0, rtol=1e-16)
    assert ret


def Normalize(psi):
    return psi/psi.dot(psi.conj())**0.5


class BCSCooling(BCS):
    """
    1d Local Quantum Friction class
    """

    def __init__(
            self, N=256, L=None, dx=0.1, delta=0, mus=(0, 0),
            beta_0=1.0, beta_V=1.0, beta_K=1.0, beta_D=1.0,
            dt_Emax=1.0, g=0, divs=None, **args):
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
        V11_psis = [-1*self.Del(Vmn*self.Del(psi=psi)) for psi in psis]
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
    
    def apply_Hc(self, psi, V):
        """
        Apply the cooling Hamiltonian.
        or, compute dy/dt w.r.t to Hc
        """
        H_psi = self.apply_H(psis=[psi], V=V)[0] if self.beta_0 !=0 else 0
        Vc_psi = self.get_Vc(psis=[psi], V=V)*psi
        Kc_psi = (
            self.ifft(self.get_Kc([psi], V=V)*self.fft(psi))
            if self.beta_K !=0 else 0)
        Vd_psi = self.apply_Vd(psis=[psi], V=V)[0]  # apply the V11 on psi
        return (
            self.beta_0*H_psi + self.beta_V*Vc_psi
            + self.beta_K*Kc_psi + self.beta_D*Vd_psi)

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
        Hpsi = self.apply_Hc(psi, V=self.V)
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    b = BCSCooling(N=64, dx=0.1, beta_0=1, beta_V=1, delta=1, mus=(2, 2))
    x = b.xyz[0]
    V = x**2/2
    H0 = b.get_H(mus_eff=b.mus, delta=b.delta, Vs=(0, 0))
    H1 = b.get_H(mus_eff=b.mus, delta=b.delta, Vs=(V, V))
    U0, Es0 = b.get_U_E(H0, transpose=True)
    U1, Es1 = b.get_U_E(H1, transpose=True)
    N_state = 1
    psi0 = U1[64]
    psi = U0[64]
    plt.plot(psi0)
    E0, N0 = b.get_E_Ns(psis=[U1[64]], V=V)
    psis = [psi]
    for i in range(1):
        E, N = b.get_E_Ns(psis=psis, V=V)
        psis = b.step(psis=psis, n=10, V=V)
        plt.plot(psis[0], '--')
        plt.plot(psi0, '-')
        plt.title(f"E0={E0.real},E={E.real}")
        plt.show()
