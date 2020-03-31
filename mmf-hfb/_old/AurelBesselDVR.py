import numpy as np
from mmf_hfb.utils import block
import warnings
warnings.filterwarnings("ignore")


def nan0(data):
    """convert nan to zero"""
    return np.nan_to_num(data, 0)


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
        if E < -1*E_c:
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
        U10 = 2*np.sqrt(
            z1[:, None]*z0[None, :])/(z1[:, None]**2 - z0[None, :]**2)*b[:, None]/a[None, :]
        a = np.sin(z1)/np.sqrt(z1)   # dim=48
        b = -np.cos(z0)/np.sqrt(z0)  # dim=49
        # U01 from dim 48->49 with shape(49, 48)
        U01 = 2*np.sqrt(
            z0[:, None]*z1[None, :])/(z0[:, None]**2 - z1[None, :]**2)*b[:, None]/a[None, :]
        return (U10, U01)

    def _get_K(self, nu, zs):
        """return kinetic matrix for a given angular momentum $\nu$"""
        zi = np.array(list(range(len(zs)))) + 1
        xx, yy = np.meshgrid(zi, zi, sparse=False, indexing='ij')
        zx, zy = np.meshgrid(zs, zs, sparse=False, indexing='ij')
        K_diag = (1+2*(nu**2 - 1)/zs**2)/3.0
        K_off = 8*(-1)**(abs(xx - yy))*zx*zy/(zx**2 - zy**2)**2+self.eps
        np.fill_diagonal(K_off, K_diag)
        T = self.k_c**2*K_off/2.0*self.alpha
        return T

    def get_Ks(self, zs=None):
        """return kinetic matrix for different angular momentums"""
        if zs is None:
            zs = self.get_zeros()
        z0, z1 = zs
        K0 = self._get_K(nu=0.5, zs=z0)
        K1 = self._get_K(nu=1.5, zs=z1)
        return (K0, K1)
    
    def get_Lambda(self, k0, kc, n):
        """compute the lambda for dim=3"""
        Lc = (kc - k0/2.0*np.log((kc + k0)/(kc - k0)))/(2*np.pi**2*self.alpha)
        return nan0(Lc)
    
    def _get_k0_kc(self, mu, r2, V):
        k0 = np.sqrt(2*(mu - r2/2.0 - V)/self.alpha + 0*1j)
        kc = np.sqrt(2*(self.E_c + mu - r2/2 - V)/self.alpha + 0*1j)
        return (k0, kc)

    def getget_effective_g(self, mu, r2, V, n, k0=None, kc=None):
        """compute the effective g"""
        if k0 is None or kc is None:
            k0, kc = self._get_k0_kc(mu=mu, r2=r2, V=V)
        Lc = self.get_Lambda(k0=k0, kc=kc, n=n)
        g_eff = 1.0/(n**(1/3.0)/self.gamma - Lc + self.eps)
        return g_eff

    def get_H(self, delta, mus, V_mean, zs=None, Ts=None, l=0):
        """return the Hamiltonian"""
        if zs is None:
            zs = self.get_zeros()
        if Ts is None:
            Ts = self.get_Ks(zs=zs)

        # zero = np.zeros_like(sum(self.xyz))
        Delta = np.diag(delta)
        mu_a, mu_b = mus

        # the correction term for centrifugal potential if l !=\nu
        # But seem it should be l^2 - l0^2, so the follow code may
        # only be accurate when l = l0 ?
        l0 = l % 2
        ll = self.alpha*(l*(l + 1) - l0*(l0 + 1))/2.0
        r2 = (zs[l0]/self.k_c)**2
        V_corr = ll/r2
        V_harm = r2/2
        H_a = Ts[l0] + np.diag(V_corr + V_mean + V_harm - mu_b)
        H_b = Ts[l0] + np.diag(V_corr + V_mean + V_harm - mu_a)
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
        """
        Some unknown correction, need to be clarified
        """
        if k0 is None or kc is None:
            assert V is not None
            k0, kc = self._get_k0_kc(mu=mu, r2=r2, V=V)
        last_corr = 1 - nan0(D**2/(6*np.pi**2*self.alpha**2)*np.log(
            (kc + k0)/(kc - k0))/(k0*n + self.eps))
        return last_corr

    def _get_den(self, H, l, V, D, mus, zs, Cs):
        """
        return density for  particle a and b
        also return the energy density
        """
        eigen, phi = np.linalg.eigh(H)
        phi = phi.T  # to have same sine as given by matlab
        al = (2*l + 1)/4.0/np.pi
        
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
                li = l % 2
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
        """
        compute the density for particle N
        """
        N_a = np.ceil(N/2)
        N_b = np.floor(N/2)
        # the (3N)^(1/3) is defined as Fermi energy of the noninteracting gas.
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
            na = ((2*mu - r2)/(self.alpha*(1 + bbar)))**1.5/(6*np.pi**2)*ir
            nb = na
            n = na + nb
            D = nan0(self.eta*(3*np.pi**2*n)**(2/3.0)/2)
            V = nan0(bbar*(3*np.pi**2*n)**(2/3.0)/2)
            return (D, V)

        def _update_DV(D, V_mean, r2, mu, n, kappa):
            k0, kc = self._get_k0_kc(mu=mu, r2=r2, V=V_mean)
            g_eff = self.getget_effective_g(mu=mu, r2=r2, V=V_mean, n=n, k0=k0, kc=kc)
            last_corr = self.get_last_correction(D=D, n=n, V=V_mean, k0=k0, kc=kc)
            D = -g_eff*kappa
            V_mean = (
                self.beta*(3*np.pi**2*n)**(2/3.0)/2.0
                - D**2/self.gamma/(3*(n + self.eps)**(2/3.0)))
            V_mean = V_mean / last_corr.real
            return (D.real, V_mean.real)

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
            for l in range(self.N_c):
                """l is the angular momentum quantum number"""
                H = self.get_H(
                    delta=Ds[l % 2], mus=(mu_a, mu_b),
                    V_mean=Vs[l % 2], zs=(z0, z1), l=l)
                ret = ret + self._get_den(
                    H=H, l=l, V=V0, D=D0, mus=(mu_a, mu_b), zs=zs, Cs=Cs)
            # print(ret[2])
            den_a, den_b, e = ret
            na0, nb0, kappa0 = den_a/r02
            na1, nb1, kappa1 = den_b/r12
            n0, n1 = na0 + nb0, na1 + nb1
            kappa0, kappa1 = kappa0/2.0, kappa1/2.0
            # Update parameters at angular momentum = 0 site
            D0, V0 = _update_DV(D=D0, V_mean=V0, r2=r02, mu=mu, n=n0, kappa=kappa0)
            # Update parameters at angular momentum = 1 site
            D1, V1 = _update_DV(D=D1, V_mean=V1, r2=r12, mu=mu, n=n1, kappa=kappa1)
            e = e + 0.3*self.beta*(3*np.pi**2)**(2/3.0)*4*np.pi*sum(
                r02*n0**(5/3.0)*C0**2) - 4*np.pi*sum(r02*D0*kappa0*C02)
            N0_a = 4*np.pi*sum(r0**2*C0**2*na0)
            N0_b = 4*np.pi*sum(r0**2*C0**2*nb0)
            R2_0 = 4*np.pi*sum(r0**4*C0**2*n0)  # what's this?
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
            
            V0, D0, V1, D1, mu_a, mu_b = (
                x0[:mm0], x0[mm0:2*mm0], x0[2*mm0: 2*mm0 + mm1],
                x0[2*mm0 + mm1:2*mm0 + 2*mm1], x0[-2], x0[-1])
            mu = (mu_a + mu_b)/2
            convergence = np.max(np.abs([G0, dx]))
            # print(f"convergence={convergence}")
            if convergence < 1.0e-9:
                print(e)
                break
        return (e, R2_0)


def compute_particle(N):
    dvr = AurelBesselDVR()
    return dvr.compute(N=N)


def AurelPlot():
    import matplotlib.pyplot as plt
    from mmf_hfb.parallel_helper import PoolHelper
    En = [1.37, ]
    R2 = [1.37, ]
    Nn = [1]

    Ns = list(range(2, 31))
    rets = PoolHelper.run(compute_particle, Ns)
    Nn.extend(Ns)
    for ret in rets:
        En.append(ret[0])
        R2.append(ret[1])
    np0 = np.array(list(range(1, 23, 1)))
    en0 = np.array(
        [
            1.5, 2.01, 4.28, 5.1, 7.6, 8.7, 11.3, 12.6, 15.6, 17.2, 19.9, 21.5,
            25.2, 26.6, 30.0, 31.9, 35.4, 37.4, 41.1, 43.2, 46.9, 49.3])
    den0 = np.array(
        [
            0, 0.02, 0.04, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.2,
            0.4, 0.1, 0.3, 0.2, 0.3, 0.3, 0.4, 0.2, 0.1])

    np1 = list(range(1, 31, 1))  
    en1 = [
        1.5, 2.002, 4.281, 5.051, 7.610, 8.639, 11.362, 12.573, 15.691, 16.806,
        20.102, 21.278, 24.787, 25.923, 29.593, 30.876, 34.634, 35.971, 39.820,
        41.302, 45.474, 46.889, 51.010, 52.624, 56.846, 58.545, 63.238, 64.388,
        69.126, 70.927]
    den1 = [
        0.0, 0.0, 0.004, 0.009, 0.01, 0.03, 0.02, 0.03, 0.05, 0.04, 0.07, 0.05,
        0.09, 0.05, 0.1, 0.06, 0.12, 0.07, 0.15, 0.08, 0.15, 0.09, 0.18, 0.20,
        0.22, 0.18, 0.22, 0.31, 0.31, 0.3]
    plt.plot(Nn, En, label="SLDA")
    plt.plot(np0, en0, label="GFMC")
    plt.plot(np1, en1, label="FN-DMC")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    AurelPlot()
