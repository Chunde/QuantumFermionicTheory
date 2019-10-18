from scipy.integrate import solve_ivp
from mmf_hfb.bcs import BCS
import numpy as np
from scipy import signal as sg
import numpy.linalg


def assert_orth(psis):
    y1, y2 = psis
    inner_prod = y1.dot(y2.conj())
    ret = np.allclose(inner_prod, 0, rtol=1e-16)
    assert ret


def Normalize(psi):
    return psi/psi.dot(psi.conj())**0.5


class SubspaceCooling(BCS):
    """Set of tools for cooling in subspaces.

    Psis = [psis1, psis2, psis3,...]

    Each of these is a collection of sp wavefunctions assumed to be on
    different nodes.  Each psis should be an array of row vectors.  To
    construct from individual states:

    psis1 = np.asarray([psi1, psi2, psi3, ...])

    Sums over states should be done over the first index.
    """

    def __init__(
            self, N=256, L=None, dx=0.1,
            beta_0=1.0, beta_V=1.0, beta_K=1.0,
            beta_F=1.0, beta_D=1.0,
            dt_Emax=1.0, g=0, divs=None, smooth=False,**args):
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
        self.beta_0 = beta_0
        self.beta_V = beta_V
        self.beta_K = beta_K
        self.beta_D = beta_D
        self.beta_F = beta_F
        self.g = g
        self.divs = divs
        self.smooth = smooth
        self._K2 = (self.hbar*np.array(self.kxyz[0]))**2/2/self.m
        self.dt = dt_Emax*self.hbar/self._K2.max()

        kc = abs(self.kxyz[0]).max()
        Ec = abs(self._K2).max()

        self.V_ext = None

    @property
    def x(self):
        return self.xyz[0].ravel()
    
    @property
    def kx(self):
        return self.kxyz[0].ravel()

    def get_density(self, Psis):
        """compute densities"""
        return sum(sum(np.abs(psis)**2) for psis in Psis)

    def get_N(self, Psis):
        """return total particle number"""
        return sum(self.get_density(Psis=Psis)) * self.dV
    
    def get_V_eff(self, Psis):
        """Return the effective potential."""
        n = self.get_density(Psis=Psis)
        return self.g * n + self.V_ext

    def get_E_eff(self, Psis):
        """Return the effective energy density."""
        n = self.get_density(Psis=Psis)
        return self.g * n**2/2 + self.V_ext*n
    
    def apply_K(self, Psis):
        """compute dy/dt with kinetic part only"""
        return [
            np.array(
                [self.ifft(self._K2*self.fft(psi))
                 for psi in psis])
            for psis in Psis]
    
    def apply_V(self, Psis):
        """compute dy/dt with effective potential only"""
        V_eff = self.get_V_eff(Psis)
        return [V_eff[None, ...]*psis
                for psis in Psis]

    def apply_H(self, Psis):
        """compute dy/dt"""
        return [Kpsis + Vpsis
                for Kpsis, Vpsis in zip(
                        self.apply_K(Psis),
                        self.apply_V(Psis))]

    def get_E_N(self, Psis):
        KPsis = self.apply_K(Psis)
        tau = sum(
            sum(psi.conj().dot(Kpsi.T)
                for psi, Kpsi in zip(psis, Kpsis))
            for psis, Kpsis in zip(Psis, KPsis))
        E_eff = sum(self.get_E_eff(Psis))
        E = (tau + E_eff)*self.dV
        N = self.get_N(Psis)
        return E.real, N

    def get_Vc(self, Psis):
        HPsis = self.apply_H(Psis)
        n = self.get_density(Psis=Psis)
        Vc = sum(2*(psis.conj()*Hpsis.T).imag
                 for psis, Hpsis in zip(Psis, HPsis))/n
        return Vc
    
    def apply_subspace_Hc(self, Psis):
        HPsis = self.apply_H(Psis)
        HcPsis = []
        for psis, Hpsis in zip(Psis, HPsis):
            norm = np.diag(psis.conj().dot(psis.T))
            Hij = psis.conj().dot(Hpsis.T)
            HcPsis.append(
                -1j*(psis.T.dot((Hij - Hij.T)/norm[:, None])).T)
        return HcPsis

    def get_Hcs(self, Psis):
        HPsis = self.apply_H(Psis)
        Hcs = []
        for psis, Hpsis in zip(Psis, HPsis):
            norm = np.diag(psis.conj().dot(psis.T))[None, :]
            psis_ = psis.T/np.sqrt(norm)
            Hpsis_ = Hpsis.T/np.sqrt(norm)
            Hc = 1j*(psis_.dot(Hpsis_.conj().T)
                     - Hpsis_.dot(psis_.conj().T))
            Hcs.append(Hc)
        return Hcs

    def get_PHcs(self, Psis):
        HPsis = self.apply_H(Psis)
        Hcs = []
        for psis, Hpsis in zip(Psis, HPsis):
            norm = np.diag(psis.conj().dot(psis.T))[None, :]
            psis_ = psis.T/np.sqrt(norm)
            P = psis_.dot(psis_.conj().T)
            Hpsis_ = Hpsis.T/np.sqrt(norm)
            Hc = 1j*(psis_.dot(Hpsis_.conj().T)
                     - Hpsis_.dot(psis_.conj().T))
            Hcs.append(P.dot(Hc).dot(P))
        return Hcs
            
    def apply_subspace_Hc(self, Psis):
        HPsis = self.apply_H(Psis)
        Hcs = self.get_Hcs(Psis)
        HcPsis = [Hcs.dot(psis.T).T
                  for psis, Hcs in zip(Psis, Hcs)]
        return HcPsis
        
        HcPsis = []
        for psis, Hpsis in zip(Psis, HPsis):
            norm = np.diag(psis.conj().dot(psis.T))
            psis.T.dot(psis.conj().dot(Hpsis.T)/norm[:, None]).T
            
            psis.T.dot(psis.conj().dot(Hpsis.T)/norm[:, None]).T
            Hij = psis.conj().dot(Hpsis.T)
            HcPsis.append(
                -1j*(psis.T.dot((Hij - Hij.T)/norm[:, None])).T)
        return HcPsis
    
    def apply_full_Hc(self, psis, psis0=None):
        """Apply the full cooling potential."""
        if psis0 is None:
            psis0 = psis
        psis0 = np.asarray(psis0).T
        Hc = self.get_full_Hc(psis)
        return Hc.dot(psis0).T
    
    def apply_Hc(self, psi):
        """
        Apply the cooling Hamiltonian.
        or, compute dy/dt w.r.t to Hc
        """
        H_psi = self.apply_H(psis=[psi])[0] if self.beta_0 !=0 else 0
        Vc_psi = self.get_Vc(psis=[psi])[0]*psi
        Kc_psi = (
            self.ifft(self.get_Kc([psi])[0]*self.fft(psi))
            if self.beta_K !=0 else 0)
        Vd_psi = self.get_Vd(psis=[psi])[0]*psi  # apply the V11 on psi
        return (
            self.beta_0*H_psi + self.beta_V*Vc_psi
            + self.beta_K*Kc_psi + self.beta_D*Vd_psi)

    def apply_Hc(self, psis):
        """My version of cooling."""
        H_psis = np.asarray(self.apply_H(psis))
        Vc = self.get_Vc(psis)
        Vc_psis = np.asarray([Vc*psi for psi in psis])
        full_Hc_psis = np.asarray(self.apply_full_Hc(psis))
        Kc_psis = 0
        return (
            self.beta_0*H_psis
            + self.beta_V*Vc_psis
            + self.beta_K*Kc_psis
            + self.beta_F*full_Hc_psis)
    
    def compute_dy_dt(self, Psis, subtract_mu=False):
        """Return dy/dt for ODE integration."""
        #assert np.allclose(np.array(psis).conj().dot(psis.T),
        #                   np.eye(len(psis)))
        
        HPsis = self.apply_subspace_Hc(Psis)
        if subtract_mu:
            HPsis_ = []
            for psis, Hpsis in zip(Psis, HPsis):
                Hpsis_ = []
                for psi, Hpsi in zip(psis, Hpsis):
                    mu = psi.conj().dot(Hpsi.T)/psi.conj().dot(psi.T)
                    Hpsis_.append(Hpsi - mu*psi)
                HPsis_.append(np.asarray(Hpsis_))
            HPsis = HPsis_
        
        dy_dt = [Hpsis/(1j*self.hbar) for Hpsis in HPsis]
        #assert np.allclose(np.array(psis).conj().dot(dy_dt.T), 0)
        return dy_dt

    def get_U_E(self, H, transpose=False):
        """return Us and Vs and energy"""
        Es, U = numpy.linalg.eigh(H)
        if transpose:
            return (U.T, Es)
        return (U, Es)

    def solve(self, psis, T, Nt=100, **kw):
        shape = np.shape(psis)
        y0 = np.ravel(psis)

        def dy_dt(t, y):
            psis = y.reshape(shape)
            return np.ravel(self.compute_dy_dt(psis))

        res = solve_ivp(fun=dy_dt, t_span=(0, T), y0=y0,
                        t_eval=np.linspace(0, T, Nt),
                        **kw)
        if not res.success:
            raise Exception(res.message)
        ts = res.t
        ys = res.y.T.reshape((len(res.t),) + shape)
        return (ts, ys)

'''
    def apply_Vd(self, psis):
        """
            apply Vd such as (V11) to the wavefunctions
            NOTE: This may not be unitary
        """
        Vmn = self.beta_D*self.get_Vd(psis=psis)
        V11_psis = [-1*self.Del(Vmn*self.Del(psi=psi)) for psi in psis]
        return V11_psis

    def apply_expK(self, psis, factor=1):
        Kc = self.beta_K*self.get_Kc(psis=psis)
        for i, psi in enumerate(psis):
            psi_k = np.fft.fft(psi)
            psi_new = np.fft.ifft(
                np.exp(-1j*self.dt*factor*(self.beta_0*self._K2 + Kc))*psi_k)
            # Not sure if the next line is necessary
            psi_new *= np.sqrt((abs(psi)**2).sum()/(abs(psi_new)**2).sum())
            psis[i] = psi_new
        return psis
        
    def apply_expV(self, psis, factor=1):
        Vc = self.get_Vc(psis=psis)
        V_eff = self.get_V_eff(psis)
        for i, psi in enumerate(psis):
            psi_new = np.exp(-1j*self.dt*factor*(self.beta_0*V_eff +self.beta_V*Vc))*psi
            # Not sure if the next line is necessary
            psi_new *= np.sqrt((abs(psi)**2).sum()/(abs(psi_new)**2).sum())
            psis[i]=psi_new
        return psis

    def evolve_V(self, psis, n=1):
        """
        evolve the states using Vmn(such as V11)
        """
        if self.beta_D == 0 or self.divs is None:
            return psis
        T = self.dt*n
        psiss = self.solve(psis=psis, T=T, dy_dt=self.compute_dy_dt_v11)[1]
        psis_new = [psiss[i][-1] for i in range(len(psis))]
        for wf1, wf2 in zip(psis, psis_new):
            len0 = wf1.conj().dot(wf1)
            len1 = wf2.conj().dot(wf2)
            assert np.allclose(len0, len1)
        return psis_new

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

    def compute_dy_dt_v11(self, t, psi, subtract_mu=True):
        """Return dy/dt for ODE integration."""
        Hpsi = self.apply_Vd([psi], V=self.V)[0]
        return Hpsi/(1j*self.hbar)

    def get_Kc(self, Psis):
        Kc = 0
        Hpsis = self.apply_HPpsis)  # [check]apply_V or apply_H
        for i, psi in enumerate(psis):
            psi_k = np.fft.fft(psi)*self.dV
            Vpsi_k = np.fft.fft(Hpsis[i])*self.dV
            Kc = Kc + 2*(psi_k.conj()*Vpsi_k).imag/N*self.dV/np.prod(self.Lxyz)
        return Kc

    
    def Del(self, psi, n=1):
        """Now only support 1D function, should be genenilzed later"""
        if n <=0:
            return psi
        for _ in range(n):
            psi = self._Del(alpha=(np.array([psi]).T,))[:, 0, ...][0].T[0]
            if self.smooth:
                psi = sg.savgol_filter(psi, 5, 2, mode='nearest')
        return psi


    def get_Vs(self, psis, divs=None):
        """return Vc or Vd"""
        N = self.get_N(psis)
        Vc = 0
        if divs is None:
            Hpsis = self.apply_H(psis)
            for i, psi in enumerate(psis):
                Vc = Vc + 2*(psi.conj()*Hpsis[i]).imag/N  # *self.dV
        else:  # Departure from locality
            da, db = self.divs
            psis_a = [self.Del(psi, n=da) for psi in psis]  # r'\frac{d^n\psi}{dx^n}'
            # r'\frac{\frac{d[d^n\psi}{dx^n}]}{dt} = \frac{\frac{d^n[d\psi}{dt}]}{dx^n}'
            Hpsis_a = self.apply_H(psis_a)
            if da == db:
                psis_b = psis_a
                Hpsis_b = Hpsis_a
            else:
                psis_b = [self.Del(psi, n=db) for psi in psis]
                Hpsis_b = self.apply_H(psis_b)
            for i in range(len(psis)):
                Vc = Vc + (
                    (Hpsis_a[i]*psis_b[i].conj()
                        +psis_a[i]*Hpsis_b[i].conj()))/N  # no image
        return Vc

'''
