"""The unitary cooling class that can cool more than one states
with unitary operations that will maintain the normorthogonal
among states.
Serveal cooling potentials are supported. But it turns out the
V_c and K_c potentials work much better than others in general
cases. It also support cooling with pairing and multiple dimensional
systems.
---------------------
Example
-------

if __name__ == "__main__":
    args = dict(beta_K=1, beta_V=1, beta_D=1, beta_Y=1, divs=(1, 1))
    s = BCSCooling(N=128, dx=0.1, g=-1, **args)
    s.erase_max_ks()
    x = s.xyz[0]
    s.V = x**2/2
    psi0 = np.exp(-x**2/2.0)*np.exp(1j*x)
    plt.subplot(121)
    plt.plot(x, psi0)
    plt.subplot(122)
    Vc = s.get_Dyadic(s.apply_H([psi0]))
    plt.plot(x, Vc)
    plt.show()

"""

import time
import numpy as np
import numpy.linalg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mmf_hfb.hfb import BCS
#  import warnings
#  warnings.filterwarnings("error")


class BCSCooling(BCS):
    """
    Local Quantum Friction class that supports GPE(single wavefunction)
    and BdG type system.
    -----------
    EXAMPLE:

    def ImaginaryCooling():
        args = dict(N=64, dx=0.1, beta_0=-1j)
        s = BCSCooling(**args)  # dx fixed, L changes, IR
        x = s.xyz[0]
        V = x**2/2
        s.V = V
        u0 = np.exp(-x**2/2)/np.pi**4
        u0 = u0/u0.dot(u0.conj())**0.5
        u1=(np.sqrt(2)*x*np.exp(-x**2/2))/np.pi**4
        u1 = u1/u1.dot(u1.conj())**0.5
        psi_0 = Normalize(V*0 + 1+0*1j)
        ts, psis, _ = s.solve(
            [psi_0], T=10, rtol=1e-5, atol=1e-6, method='BDF')
        psi_ground = psis[-1]
        E_ground = s.get_E_Ns([psi_ground])

    2D BCS EXAMPLE:

    def BCS2D_Cooling(N_state=2, T=0.5,**args):
        b = BCSCooling(**args)
        da, db=b.divs
        k0 = 2*np.pi/b.L
        x, y = b.xyz
        V = x**2/2
        V = sum(_x**2 for _x in b.xyz)
        b.V = V/2
        x0 = 0.5
        H0 = b._get_H(mu_eff=0, V=0)
        H1 = b._get_H(mu_eff=0, V=V)
        U0, E0 = b.get_psis_es(H0, transpose=True)
        U1, E1 = b.get_psis_es(H1, transpose=True)
        psi0 = [b.Normalize(U1[i]) for i in range(N_state)]
        phase = ((x-x0) + 1j*y)*((x+x0) - 1j*y)
        psi =  [b.Normalize(U0[i]) for i in range(N_state)]
        ts, psis, nfev = b.solve(
            [psi], T=T, rtol=1e-5, atol=1e-6, method='BDF')
        E, N = b.get_E_Ns(psis[-1])
        print(f"Final Energy={E}, nfev={nfev}")

    args = dict(
        N=32, dx=0.25, dim=2, N_state=2,
        beta_V=0.3, T=20, check_dE=False)
    BCS2D_Cooling(**args)
    ------------
    To too down to some level above ground energy, please check
    the test case in '/test/test_BCS_Cooling.py'
    """

    def __init__(
            self, N=256, L=None, dx=0.1, delta=0, mus=(0, 0), dim=1, V=0,
            beta_H=1, beta_0=1.0, beta_V=0, beta_K=0, beta_D=0, beta_Y=0,
            beta_S=0, g=0, dE_dt=1, divs=(1, 1), check_dE=False,
            time_out=None, E_stop=None, **args):
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
        beta_Y: float
            Portion of the Dyadic cooling potential V_Dyadic.
        E_stop: float
            the target energy, integration will stop if E_stop is reached
        """
        if L is None:
            L = N*dx
        BCS.__init__(self, Nxyz=(N,)*dim, Lxyz=(L,)*dim)
        self.L = L
        self.N = N
        self.dx = dx
        self.delta = delta
        self.mus = mus
        self.beta_H = beta_H
        self.beta_0 = beta_0
        self.beta_V = beta_V
        self.beta_K = beta_K
        self.beta_D = beta_D
        self.beta_Y = beta_Y
        self.beta_S = beta_S
        self.g = g
        self.V = V
        self.time_out = time_out
        self.start_time = 0
        self.divs = divs
        self.check_dE = check_dE
        self.dE_dt = dE_dt
        self._K2 = self.hbar**2/2/self.m*sum(_k**2 for _k in self.kxyz)
        self.dt = dE_dt*self.hbar/self._K2.max()
        self.E_max = self._K2.max()
        self.E_stop = E_stop

    def _get_uv(self, psi):
        """slice a uv into u, and v components"""
        shape = (2,) + self.Nxyz
        return psi.reshape(shape)

    def get_Vext(self):
        return self.V

    def get_Vint(self, psis):
        if self.g == 0:
            return 0
        ns = self.get_ns(psis)
        if self.delta != 0:  # n_a + n_b
            ns = ns[:len(ns)//2] + ns[len(ns)//2:]
        return self.g*ns

    def get_V(self, psis):
        """
            return effective potential for
            given external potential V and
            states
        """
        Vext = self.get_Vext()
        Vint = self.get_Vint(psis)
        return Vint + Vext

    def _apply_H(self, psi, psi_k, V):
        # if psi_k is None:
        #      psi_k = self.fft(psi)
        if self.delta == 0:
            Hpsi = self.ifft(self._K2*psi_k) + V*psi
            return Hpsi
        H = self.get_H(
            mus_eff=self.mus, delta=self.delta, Vs=(V, V))
        return H.dot(psi)  # apply H on psi

    def apply_H(self, psis, psis_k=None):
        """compute dy/dt=H psi"""
        V_eff = self.get_V(psis)
        Hpsis = []
        if psis_k is None:
            psis_k = self.get_psis_k(psis)
        for psi, psi_k in zip(psis, psis_k):
            Hpsi = self._apply_H(psi, psi_k=psi_k, V=V_eff)
            Hpsis.append(Hpsi)
        return Hpsis

    def _apply_K(self, psi, psi_k):
        if self.delta == 0:
            Kpsi = self.ifft(self._K2*psi_k)
            return Kpsi
        Kpsi = [self.ifft(
            self._K2*psi_k) for psi_k_ in self._get_uv(psi_k)]
        return Kpsi.reshape(psi.shape)

    def apply_K(self, psis, psis_k):
        """
        compute dy/dt with kinetic part only
        NOTE: Should be tested
        """
        Hpsis = []
        if psis_k is None:
            psis_k = self.get_psis_k(psis)
        for (i, psi) in enumerate(psis):
            Kpsi = self._apply_K(psi=psi, psi_k=psis_k[i])
            Hpsis.append(Kpsi)
        return Hpsis

    def _apply_V(self, psi, V):
        if self.delta == 0:
            return V*psi
        return np.array([V - self.mus[0], -V + self.mus[1]]).ravel()*psi

    def apply_V(self, psis):
        """compute dy/dt with effective potential only"""
        V_eff = self.get_V(psis)
        Hpsis = []
        for psi in psis:
            Vpsi = self._apply_V(psi, V=V_eff)
            Hpsis.append(Vpsi)
        return Hpsis

    def _div(self, psi, n=1):
        assert self.dim == 1
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
        if n <= 0:
            return psi
        psi = self._div(psi, n=n)
        return psi

    def get_N(self, psis):
        """return total particle number"""
        N = 0
        for psi in psis:
            N = N + psi.dot(psi.conj())*self.dV
        return N

    def get_psis_k(self, psis):
        """return FFT of all wave functions"""
        if self.delta == 0:
            return [self.fft(psi) for psi in psis]
        else:  # can be optimized in consice way
            return [
                self.fft(
                    self._get_uv(psi),
                    axes=self.axes(bdg=True)).reshape(
                        psi.shape) for psi in psis]

    def _get_Vc(self, psis, psis_k, divs=None):
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
        Vc = 0
        if divs is None:
            # can also apply_H, result is unchanged.
            # but with pairing, apply_H should be used
            Hpsis = self.apply_H(psis, psis_k=psis_k)
            for i, psi in enumerate(psis):
                Vc = Vc + 2*(psi.conj()*Hpsis[i]).imag
        else:  # Departure from locality
            da, db = self.divs
            assert da <= 1
            assert db <= 1  # now only da=db=1 is tested
            if da != db:
                raise ValueError("Only support V11(da=db=1)")
            # compute $d^n \psi / d^n x$
            psis_a = [self.Del(psi, n=da) for psi in psis]
            # $d[d^n \psi / d^n x] / dt$
            Hpsis = np.array(self.apply_H(psis, psis_k=psis_k))/(1j*self.hbar)
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
                        + Hpsis_a[i]*psis_b[i].conj()))
        return Vc/sum(self.get_ns(psis))/self.dV  # divided by total density

    def get_Vc(self, psis, psis_k=None):
        """return Vc potential"""
        if psis_k is None:
            psis_k = self.get_psis_k(psis)
        Vc = 0*np.array(psis[0])
        if self.beta_V != 0:
            Vc = Vc + self._get_Vc(psis=psis, psis_k=psis_k)
        return Vc

    def get_Vd(self, psis, psis_k=None):
        """return the derivative cooling potential"""
        Vd = 0*np.array(psis[0])
        if psis_k is None:
            psis_k = self.get_psis_k(psis)
        if self.beta_D != 0 and self.divs is not None:
            Vd = Vd + self._get_Vc(
                psis=psis, psis_k=psis_k, divs=self.divs)
        return Vd

    def get_Dyadic(self, psis, psis_k=None):
        """
        mixed Vc and Kc
        $$
        H_C=f(\bra{K}\ket{x}+\bra{x}\ket{k})
        $$
        """
        V_dy = 0
        if psis_k is None:
            psis_k = self.get_psis_k(psis)
        Hpsis = self.apply_H(psis=psis, psis_k=psis_k)
        Hpsis_k = self.get_psis_k(Hpsis)
        for i in range(len(psis_k)):
            V_dy = (
                V_dy + (
                    Hpsis_k[i].conj()*psis[i]
                    - psis[i].conj()*Hpsis_k[i]
                    + Hpsis[i].conj()*psis_k[i]
                    - psis_k[i].conj()*Hpsis[i]))
        return -1j*V_dy/self.hbar*self.dV  # minus sign?

    def get_Sc(self, psis, psis_k=None):
        """
        NOTE: experiemental code trying to cool
            to ground state as soon as possible
        """
        Sc = 0
        if psis_k is None:
            psis_k = self.get_psis_k(psis)
        Hpsis = self.apply_H(psis=psis, psis_k=psis_k)
        Hpsis_k = self.get_psis_k(Hpsis)
        for i in range(len(psis_k)):
            Sc = Sc + (
                Hpsis[i]*psis_k[i].conj()
                - psis[i]*Hpsis_k[i].conj())
        return -1j*Sc/self.hbar*self.dV

    def apply_Sc(self, psis, psis_k):
        """
        Note: experiemental code trying to cool
            to ground state as soon as possible
        """
        if self.beta_S == 0:
            return (0,)*len(psis)
        Sc = self.beta_S*self.get_Sc(psis=psis, psis_k=psis_k)
        Sc_psis = [self.ifft(Sc*psi_k) for psi_k in psis_k]  # Sharp cooling
        # Vdy_psis = [Vdy*psi_k for psi_k in psis_k]
        return Sc_psis

    def _get_Kc(self, Hpsi, psi, psi_k, N):
        """
        Kc is the diagonal of the H in k space, so
        even in the case with pairing, it is good to
        use psi as a single wavefunction without
        dividing it in to u, v components.
        """
        psi_k = psi_k*self.dV  # this line is confusing[check]
        Vpsi_k = self.fft(Hpsi)*self.dV/np.prod(self.Lxyz)
        Kc = 2*(psi_k.conj()*Vpsi_k).imag/N*self.dV
        return Kc

    def get_Kc(self, psis, psis_k=None):
        Kc = 0*np.array(psis[0])
        if self.beta_K == 0:
            return Kc
        N = self.get_N(psis)
        if psis_k is None:
            psis_k = self.get_psis_k(psis)
        # use apply_H instead of apply_V
        Hpsis = self.apply_H(psis, psis_k=psis_k)
        for i, psi in enumerate(psis):
            Kc = Kc + self._get_Kc(
                Hpsi=Hpsis[i], psi=psi, psi_k=psis_k[i], N=N)
        return Kc

    def get_Hc(self, psis, psis_k=None):
        """Return the full cooling Hamiltonian in position space."""
        size = np.prod(self.Nxyz)
        Hc = 0
        if psis_k is None:
            psis_k = self.get_psis_k(psis)
        Hpsis = self.apply_H(psis, psis_k=psis_k)
        for _, (psi, Hpsi) in enumerate(zip(psis, Hpsis)):
            Hc_ = (
                1j*psi.reshape(
                    size)[:, None]*Hpsi.conj().reshape(
                        size)[None, :])
            Hc_ += Hc_.conj().T
            Hc = Hc + Hc_
        N = self.get_N(psis)
        return Hc/N

    def apply_Dyadic(self, psis, psis_k):
        if self.beta_Y == 0:
            return (0,)*len(psis)
        Vdy = self.beta_Y*self.get_Dyadic(psis=psis, psis_k=psis_k)
        Vdy_psis = [
            self.ifft(Vdy*psi) + Vdy*psi_k
            for (psi, psi_k) in zip(psis, psis_k)]
        return Vdy_psis

    def apply_Vd(self, psis, psis_k):
        """
            apply Vd such as (V11) to the wave-functions
            NOTE: This may not be unitary
        """
        if self.beta_D == 0:
            return (0,)*len(psis)
        Vmn = self.beta_D*self.get_Vd(psis=psis, psis_k=psis_k)
        da, db = self.divs
        if db == 1:
            V11_psis = [-self.Del(
                Vmn*self.Del(psi=psi, n=da), n=db) for psi in psis]
        elif db == 0:
            V11_psis = [Vmn*self.Del(psi=psi, n=da) for psi in psis]
        else:
            raise ValueError(
                "Derivative order should be no larger than 1")
        return np.array(V11_psis)*self.dV**(2*sum(self.divs))

    def _apply_expK(self, psi, psi_k, Kc, factor=1):
        factor_ = -1j*self.dt*factor*self.beta_H
        if self.delta == 0:
            psi_new = self.ifft(
                np.exp(factor_*(
                    self.beta_0*self._K2 + Kc))*psi_k)
            psi_new *= np.sqrt(
                (abs(psi)**2).sum()/(abs(psi_new)**2).sum())
            return psi_new
        kc_uv = self._get_uv(Kc)
        kuv = psi_k.reshape(kc_uv.shape)
        # [bug]
        signs = [1, -1]  # used to change the sign of k2
        expK = [self.ifft(
            np.exp(factor_*(
                self.beta_0*self._K2*sign + kc))*psi_k) for (
                    sign, psi_k, kc) in zip(signs, kuv, kc_uv)]
        return np.array(expK).ravel()

    def apply_expK(self, psis, psis_k=None, factor=1):
        if psis_k is None:
            psis_k = self.get_psis_k(psis)
        Kc = self.beta_K*self.get_Kc(psis=psis, psis_k=psis_k)
        for i, psi in enumerate(psis):
            psis[i] = self._apply_expK(
                psi, psi_k=psis_k[i], Kc=Kc, factor=factor)
        return psis

    def _apply_expV(self, psi, V, Vc, factor):
        factor_ = -1j*self.dt*factor*self.beta_H
        if self.delta == 0:
            psi_new = np.exp(
                factor_*(self.beta_0*V + self.beta_V*Vc))*psi
            psi_new *= np.sqrt((abs(psi)**2).sum()/(abs(psi_new)**2).sum())
            return psi_new
        Vc_uv = self._get_uv(Vc)
        uv = self._get_uv(psi)
        Vs = (V - self.mus[0], -V + self.mus[1])
        expV = [np.exp(
            factor_*(
                self.beta_0*V_ + self.beta_V*Vc))*psi for (
                    V_, psi, Vc) in zip(Vs, uv, Vc_uv)]
        return np.array(expV).ravel()

    def apply_expV(self, psis, psis_k=None, factor=1):
        if psis_k is None:
            psis_k = self.get_psis_k(psis)
        Vc = self.get_Vc(psis=psis, psis_k=psis_k)
        V_eff = self.get_V(psis)
        for i, psi in enumerate(psis):
            psis[i] = self._apply_expV(
                psi=psi, V=V_eff, Vc=Vc, factor=factor)
        return psis

    def apply_Hc(self, psis, psis_k=None):
        """
        Apply the cooling Hamiltonian.
        or, compute dy/dt w.r.t to Hc
        """
        Hc_psis = []
        if psis_k is None:
            psis_k = self.get_psis_k(psis)
        H_psis = self.apply_H(
            psis=psis, psis_k=psis_k) if self.beta_0 != 0 else 0
        Vc = self.get_Vc(
            psis=psis, psis_k=psis_k) if self.beta_V != 0 else 0
        Kc = self.get_Kc(
            psis=psis, psis_k=psis_k) if self.beta_K != 0 else 0
        Vd_psis = self.apply_Vd(psis=psis, psis_k=psis_k)
        V_Dyadic_psis = self.apply_Dyadic(psis, psis_k=psis_k)
        Vs_psis = self.apply_Sc(psis, psis_k=psis_k)

        for i, psi in enumerate(psis):
            Vc_psi = Vc*psi
            Kc_psi = self.ifft(Kc*psis_k[i])
            Hc_psi = (
                self.beta_0*H_psis[i] + self.beta_V*Vc_psi
                + self.beta_K*Kc_psi + Vd_psis[i]
                + V_Dyadic_psis[i] + Vs_psis[i])
            Hc_psis.append(self.beta_H*Hc_psi)
        return Hc_psis

    def get_dE_dt(self, psis, psis_k=None):
        """compute dE/dt"""
        if psis_k is None:
            psis_k = self.get_psis_k(psis)
        H_psis = self.apply_H(psis, psis_k=psis_k)
        Hc_psis = self.apply_Hc(psis=psis, psis_k=psis_k)
        dE_dt = 2*sum(
            [self.dotc(H_psi, Hc_psi).imag
                for (H_psi, Hc_psi) in zip(H_psis, Hc_psis)])
        return dE_dt

    def step(self, psis, n=1):
        """
        Evolve the state psi by applying n steps of the
        Split-Operator method.
        """
        psis = self.apply_expK(psis=psis, factor=0.5)
        for _ in range(n):
            psis = self.apply_expV(psis=psis)
            psis = self.apply_expK(psis=psis)
        psis = self.apply_expK(psis=psis, factor=-0.5)
        return psis

    def check_time_out(self):
        """a function used to enforce timing"""
        if self.time_out is None:
            return
        wall_time = time.time() - self.start_time
        if wall_time > self.time_out:
            print(f"Solver time out[{wall_time}>{self.time_out}], skip...")
            raise ValueError("Time Out")

    def pack(self, psi):
        return np.ascontiguousarray(psi).view().ravel()

    def unpack(self, y):
        n = len(y)//np.prod(self.Nxyz)
        shape = (n, ) + self.Nxyz
        return np.ascontiguousarray(y).view().reshape(shape)

    def compute_dy_dt(self, t, psis, subtract_mu=True, **args):
        """Return dy/dt for ODE integration."""
        self.check_time_out()
        psis = self.unpack(y=psis)
        if self.check_dE:
            dE_dt = self.get_dE_dt(psis=psis)
            if abs(dE_dt) > 1e-10:
                assert dE_dt <= 0
        Hpsis = self.apply_Hc(psis)
        if subtract_mu:
            for i, psi in enumerate(psis):
                Hpsis[i] -= self.dotc(psi, Hpsis[i])/self.dotc(psi, psi)*psi
        return self.pack(np.array(Hpsis)/(1j*self.hbar))

    def get_psis_es(self, H, transpose=False):
        """return Us and Vs and energy"""
        Es, U = numpy.linalg.eigh(H)
        if transpose:
            return (U.T, Es)
        return (U, Es)

    def solve(self, psis, T, dy_dt=None, solver=None, **kw):
        self.psis = psis  # all single particle states
        self.start_time = time.time()
        if dy_dt is None:
            dy_dt = self.compute_dy_dt
        if solver is None:
            solver = solve_ivp
        else:
            kw.update(dt=self.dt)
        psis0 = self.pack(psis)

        def ivp_event(t, y):
            psis = self.unpack(y)
            E = self.get_E_Ns(psis)[0]
            return E - self.E_stop

        ivp_event.terminal = True
        event = None if self.E_stop is None else ivp_event

        res = solver(
            fun=dy_dt, t_span=(0, T), y0=psis0,
            events=event, **kw)

        if not res.success:
            raise Exception(res.message)
        ys = list(map(self.unpack, res.y.T))
        return(res.t, ys, res.nfev)

    def get_ns(self, psis):
        """compute densities"""
        return sum(np.abs(psis)**2)

    def get_E_Ns(self, psis):
        E = 0
        N = 0
        ns = self.get_ns(psis)
        N = ns.sum()*self.dV
        if self.delta == 0:
            for psi in psis:
                K = self.dotc(psi, self.ifft(self._K2*self.fft(psi)))
                E = E + K.real*self.dV
            V_eff = (self.get_Vint(psis)/2 + self.V)*ns
            E = E + V_eff.sum()*self.dV
        else:
            H = self.get_H(mus_eff=self.mus, delta=self.delta, Vs=(self.V, )*2)
            for psi in psis:
                E = E + psi.conj().dot(H.dot(psi))
        return E, N

    def plot(self, psi, **kw):
        if self.dim == 1:
            x = self.xyz[0].ravel()
            plt.plot(x, abs(psi)**2, **kw)
        elif self.dim == 2:
            from mmfutils import plot as mmfplt
            x, y = self.xyz
            mmfplt.imcontourf(x, y, self.get_ns([psi]))
            plt.colorbar()
        E, N = self.get_E_Ns([psi])
        plt.title(f"E={E:.4f}, N={N:.4f}")
        plt.show()
