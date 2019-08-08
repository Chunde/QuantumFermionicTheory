import numpy as np
from enum import Enum
from mmf_hfb.utils import block

def nan0(data):
    return np.nan_to_num(data, 0)

class DVRBasisType(Enum):
    """Types of DVR basis"""
    POLYNOMIAL = 0
    BESSEL = 1
    SINC = 2
    AIRY = 3


class BesselDVR(object):
    """SLDA with Bessel DVR"""
    def __init__(self, N_c=100, R_c=9.0, T=0.025, **args):
        self.N_c = N_c
        self.R_c = R_c
        self.T = T
        self.E_c = 0.95*(N_c + 1.5)
        self.dE_c = self.E_c/50
        self.k_c = np.sqrt(2*N_c + 3) + 3
        self.xi = 0.42
        self.eta=0.504
        self.alpha=1.14
        self.beta = -0.55269
        self.gamma = -1/0.090585
        self.eps = 2.2204e-16
    
    def f(self, E, E_c=None):
        if E_c is None:
            E_c = self.E_c
        if E > E_c:
            return 0
        if E < -E_c:
            return 1
        return 1.0/(1 + np.exp(E/(self.T + self.eps)))
  
    def get_zeros(self):
        N_max = 500
        nn = np.array(list(range(0, N_max, 1))) + 1
        Z0 = nn*np.pi
        Z1 = Z0 + 0.5*np.pi
        for _ in range(20):
            Z1 = nn*np.pi + np.arctan(Z1)
        
        z_c = self.R_c*self.k_c
        i0, i1 = 0, 0
        for i in range(len(Z0)):
            if Z0[i] > z_c:
                i0 = i
                break
        for i in range(len(Z1)):
            if Z1[i] > z_c:
                i1 = i
                break
        
        return (Z0[:i0], Z1[:i1])

    def get_Cs(self, zs=None):
        if zs is None:
            zs = self.get_zeros()
        z0, z1 = zs
        C0 = np.sqrt(np.pi/self.k_c)
        C1 = np.sqrt(np.pi/self.k_c/np.sin(z1)**2)
        return (C0, C1)

    def get_Us(self, zs=None):
        """return the coordinate convert matrix"""
        if zs is None:
            zs =self.get_zeros()
        z0, z1 = zs
        z0, z1 = np.array(z0), np.array(z1)
        a = np.cos(z0)/np.sqrt(z0)  # dim=49
        b = np.sin(z1)/np.sqrt(z1)  # dim=48
        # U10 from dim 49->48 with shape(48, 49)
        U10 = 2*np.sqrt(z1[:, None]*z0[None, :])/(z1[:, None]**2 - z0[None, :]**2)*b[:, None]/a[None, :]
        a = np.sin(z1)/np.sqrt(z1)   # dim=48
        b = -np.cos(z0)/np.sqrt(z0)  # dim=49
        # U01 from dim 48->49 with shape(49, 48)
        U01 = 2*np.sqrt(z0[:, None]*z1[None, :])/(z0[:, None]**2 - z1[None, :]**2)*b[:, None]/a[None, :]
        return (U10, U01)

    def _get_K(self, nu, zeros):
        """return kinetic matrix for a given angular momentum $\nu$"""
        zi = np.array(list(range(len(zeros)))) + 1
        xx, yy = np.meshgrid(zi, zi, sparse=False, indexing='ij')
        zx, zy = np.meshgrid(zeros, zeros, sparse=False, indexing='ij')
        K_diag = (1+2*(nu**2 - 1)/zeros**2)/3.0
        K_off = 8*(-1)**(abs(xx - yy))*zx*zy/(zx**2 - zy**2)**2+self.eps
        np.fill_diagonal(K_off, K_diag)
        T = self.k_c**2*K_off/2.0*self.alpha
        return T

    def get_Ks(self, zs=None):
        """return kinetic matrix for different angular momentums"""
        if zs is None:
            zs = self.get_zeros()
        z0, z1 = zs
        K0 = self._get_K(nu=0.5, zeros=z0)
        K1 = self._get_K(nu=1.5, zeros=z1)
        return (K0, K1)
    
    def get_Lambda(self, k0, kc, n):
        """compute the lambda for dim=3"""
        Lc = (kc - k0/2.0*np.log((kc + k0)/(kc - k0)))/(2*np.pi**2*self.alpha)
        # Lambda = kc*(1.0 - k0/kc/2.0*np.log((kc + k0)/(kc - k0)))/(2*np.pi**2*self.alpha)
        return nan0(Lc)
    
    def _get_k0_kc(self, mu, r2, V):
        k0 = np.sqrt(2*(mu - r2/2.0 - V)/self.alpha + 0*1j)
        kc = np.sqrt(2*(self.E_c + mu - r2/2 - V)/self.alpha + 0*1j)
        return (k0, kc)

    def get_g_eff(self, mu, r2, V, n, k0=None, kc=None):
        """compute the effective g"""
        if k0 is None or kc is None:
            k0, kc = self._get_k0_kc(mu=mu, r2=r2, V=V)
        Lc = self.get_Lambda(k0=k0, kc=kc, n=n)
        g_eff = 1.0/(n**(1/3.0)/self.gamma - Lc + self.eps)
        return g_eff

    def get_H(self, delta, mus, V, zs=None, Ts=None, angular_momentum=0):
        """return the Hamiltonian"""
        if zs is None:
            zs = self.get_zeros()
        if Ts is None:
            Ts = self.get_Ks(zs=zs)

        # zero = np.zeros_like(sum(self.xyz))
        Delta = np.diag(delta)
        mu_a, mu_b = mus
        L = angular_momentum
        L0 = L % 2
        LL = self.alpha*(L*(L + 1) - L0*(L0 + 1))/2.0
        r2 = (zs[L0]/self.k_c)**2
        V_ = LL /r2
        H_a = Ts[L0] + np.diag(V_ + V + r2/2 - mu_a)
        H_b = Ts[L0] + np.diag(V_ + V + r2/2 - mu_b)
        H = block(H_a, Delta, Delta.conj(), -H_b)
        return H


class AurelBesselDVR(BesselDVR):

    def __init__(self, N_c=100, R_c=9.0, T=0.025, **args):
        """
            intend to reproduce Aurel's matlab code
        """
        BesselDVR.__init__(self, N_c=N_c, R_c=R_c, T=T, **args)
        self.amix = 1.0
        self.bmix = 1.0
        self.cmix = 1.0
        self.EN_C = 40
        self.deps = 1.0e-15

    def beta_bar(self):
        """compute beta bar equation 20, PRA 76, 040502(R)(2007)"""
        return self.beta - self.eta**2*(3*np.pi**2)**(2.0/3)/self.gamma/6.0

    def get_last_correction(self, D, n, mu=None, r2=None, V=None, k0=None, kc=None):
        if k0 is None or kc is None:
            assert V is not None
            k0, kc = self._get_k0_kc(mu=mu, r2=r2, V=V)
        last_corr = 1 - nan0(D**2/(6*np.pi**2*self.alpha**2)*np.log((kc + k0)/(kc - k0))/(k0*n + self.eps))
        return last_corr

    def _get_den(self, H, L, V, D, mus, zs, Cs):
        eigen, phi = np.linalg.eigh(H)
        phi = phi.T  # to have same sine as given by matlab
        al = (2*L + 1)/4.0/np.pi
        
        Us = self.get_Us(zs=zs)
        mms = (len(zs[0]), len(zs[1]))
        mu_a, mu_b = mus

        def _den_kappa(fe, fc, u, v):
            rho_a = (1 - fe)*fc*al*v**2
            rho_b = fe*fc*al*u**2
            kappa = (1 - 2*fe)*fc*al*v*u
            return np.array([rho_a, rho_b, kappa])

        den_a, den_b, e = 0, 0, 0
        for i in range(len(eigen)):
            E = eigen[i]
            fe = self.f(E, E_c=self.EN_C)
            En_c = (abs(E) - self.E_c)/self.dE_c
            if En_c > self.EN_C:
                fc = 0
            elif En_c < -self.EN_C:
                fc = 1
            else:
                fc = 1/(1.0 + np.exp(En_c))
            
            if fc > 0:
                li = L % 2
                offset = mms[li]
                U = Us[li]
                if li == 0:
                    u0 = phi[i][:offset]/Cs[li]
                    v0 = phi[i][offset:]/Cs[li]
                    u1 = U.dot(u0)
                    v1 = U.dot(v0)
                else:
                    u1 = phi[i][:offset]/Cs[li]
                    v1 = phi[i][offset:]/Cs[li]
                    u0 = U.dot(u1)
                    v0 = U.dot(v1)
                den_a = den_a + _den_kappa(fe=fe, fc=fc, u=u0, v=v0)
                den_b = den_b + _den_kappa(fe=fe, fc=fc, u=u1, v=v1)
                ev = v0**2*(mu_a - eigen[i] - V) + v0*u0*D
                eu = u0**2*(mu_b + eigen[i] - V) - v0*u0*D
                e = e + 4*np.pi*al*sum(((1 - fe)*ev + fe*eu)*Cs[0]**2)*fc
        return np.array([den_a, den_b, e])

    def compute(self, N=2, **args):
        """compute the density for particle N"""
        N_a = np.ceil(N/2)
        N_b = np.floor(N/2)
        mu = (3.0*N)**(1/3.0)*np.sqrt(self.xi)
        mu_a = mu_b = mu
        # E0 = (3.0*N)**(4/3.0)/4/np.sqrt(self.xi)

        zs = self.get_zeros()
        z0, z1 = zs
        mm0, mm1= len(z0), len(z1)
        bbar = self.beta_bar()
        r0 = z0/self.k_c
        r02 = r0**2
        r12 = (z1/self.k_c)**2

        def _get_DV(r2):
            ir = ((2*mu - r2)>0).astype("uint8")
            rho_a = ((2*mu - r2)/(self.alpha*(1 + bbar)))**1.5/(6*np.pi**2)*ir
            rho_b = rho_a
            rho = rho_a + rho_b
            D = nan0(self.eta*(3*np.pi**2*rho)**(2/3.0)/2)
            V = nan0(bbar*(3*np.pi**2*rho)**(2/3.0)/2)
            return (D, V)

        def _update_DV(D, V, r2, mu, n, kappa):
            k0, kc = self._get_k0_kc(mu=mu, r2=r2, V=V)
            g_eff = self.get_g_eff(mu=mu, r2=r2, V=V, n=n, k0=k0, kc=kc)
            last_corr = self.get_last_correction(D=D, n=n, V=V, k0=k0, kc=kc)
            D = -g_eff*kappa
            V = self.beta*(3*np.pi**2*n)**(2/3.0)/2.0 - D**2/self.gamma/(3*(n + self.eps)**(2/3.0))
            V = V / last_corr.real
            return (D.real, V.real)

        D0, V0 = _get_DV(r2=r02)
        D1, V1 = _get_DV(r2=r12)
        Cs = self.get_Cs(zs=zs)
        C0, C1=Cs
        C02 = C0**2
        
        x0 = np.hstack([V0, D0, V1, D1, mu_a, mu_b])
        x1, dx, G0, G1, dG, = x0, x0, x0, x0, x0
        K0 = self.cmix*np.diag(np.ones_like(x0))
        K1 = K0
        iter = 0

        while(True):  # start iteration
            iter = iter + 1
            ret = 0
            Ds = (D0, D1)
            Vs = (V0, V1)
            for L in range(self.N_c):
                """L is the angular momentum quantum number"""
                H = self.get_H(
                    delta=Ds[L % 2], mus=(mu_a, mu_b),
                    V=Vs[L % 2], zs=(z0, z1), angular_momentum=L)
                ret = ret + self._get_den(H=H, L=L, V=V0, D=D0, mus=(mu_a, mu_b), zs=zs, Cs=Cs)
            # print(ret[2])
            den_a, den_b, e = ret
            na0, nb0, kappa0 = den_a/r02
            na1, nb1, kappa1 = den_b/r12
            n0, n1 = na0 + nb0, na1 + nb1
            kappa0, kappa1 = kappa0/2.0, kappa1/2.0
            # Update parameters at angular momentum = 0 site
            D0, V0 = _update_DV(D=D0, V=V0, r2=r02, mu=mu, n=n0, kappa=kappa0)
            # Update parameters at angular momentum = 1 site
            D1, V1 = _update_DV(D=D1, V=V1, r2=r12, mu=mu, n=n1, kappa=kappa1)
            e = e + 0.3*self.beta*(3*np.pi**2)**(2/3.0)*4*np.pi*sum(r02*n0**(5/3.0)*C0**2) - 4*np.pi*sum(r02*D0*kappa0*C02)
            N0_a = 4*np.pi*sum(r0**2*C0**2*na0)
            N0_b = 4*np.pi*sum(r0**2*C0**2*nb0)
            R2_0 = 4*np.pi*sum(r0**4*C0**2*n0)
            mu_a_ = mu_a + mu_a*(N_a/N0_a - 1)
            mu_b_ = mu_b + mu_b*(N_b/N0_b - 1)
            x = np.hstack([V0, D0, V1, D1, mu_a_, mu_b_])
            if iter == 1:
                G0 = x0 - x
                x1 = x0 - self.amix*G0
                dx = x1 - x0
                x0 = x1
            else:
                G1 = x0 - x
                dG = G1 - G0
                ket = dx - K0.dot(dG)
                ket = ket*(abs(ket)>self.deps*(abs(G0) + abs(G1))).astype("uint8")
                bra = dx.dot(K0)
                inorm = self.bmix/bra.dot(dG)
                K1 = K0 + ket[:, None]*bra[None, :]*inorm
                x1 = x0 - K1.dot(G1)
                K0 = K1
                dx = x1 - x0
                x0 = x1
                G0 = G1
            
            V0, D0, V1, D1, mu_a, mu_b = x0[:mm0], x0[mm0:2*mm0], x0[2*mm0: 2*mm0 + mm1], x0[2*mm0 + mm1:2*mm0 + 2*mm1], x0[-2], x0[-1]
            mu, dmu = (mu_a + mu_b)/2, (mu_a - mu_b)/2
            convergence = np.max(np.abs([G0, dx]))
            print(f"convergence={convergence}")
            if convergence < 1.0e-9:
                print(e)
                break



if __name__ == "__main__":
    dvr = AurelBesselDVR()
    #dvr.get_Ks()
    #dvr.get_Us()
    dvr.compute()


