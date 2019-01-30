"""BCS Equations in 2D

This module provides a class BCS2D for solving the BCS (BdG) equations in 2D for a
two-species Fermi gas with short-range interaction.
"""
from __future__ import division

import itertools
import numpy as np
import scipy.integrate

from mmfutils.math.integrate import mquad

class ASLDA(object):
    hbar = 1.0
    m = 1.0
    w = 1.0                     # Trapping potential
    
    def __init__(self, Nx=32, Lx=0.46, dx=None, T=0,E_c=np.inf):

        dy = dx
        if dx is None:
            dx = np.divide(Lx, Nx)
        elif Lx is None:
            Lx = np.prod(Nx, dx)
        elif Nx is None:
            Nx = np.ceil(Lx / dx).astype(int)
        self.x = (np.arange(Nx) * dx - Lx / 2)
        self.kx = 2*np.pi * np.fft.fftfreq(Nx, dx)
        self.dx = dx
        self.Nx = Nx
        self.Lx = Lx
        self.T = T
        self.E_c = E_c
        self.gamma = -11.11
        self.g_eff = 1.0
        self.alpha0 = 0 # the constant to turn on or off the alpha tems
        # External potential
        self.v_ext = self.get_v_ext()
        self._D2 = self.get_Laplacian()
        self._D1 = self.get_Del()
    def fft(self, y):
            return np.fft.fft(y)
            
    def ifft(self, y):
            return np.fft.ifft(y)
    def _get_p(self, ns):
        """return p for a given ns"""
        "[Numerical Test Status:Pass]"
        if ns == None:
            return 0
        n_a, n_b = ns
        n_p,n_m = n_a + n_b,n_a - n_b
        p = n_m / n_p 
        return p

    def _alpha(self, p=0):
        """return alpha"""
        "[Numerical Test Status:Pass]"
        p2,p3,p4 = p**2,p**3,p**4
        alpha = 1.0094 + 0.532 * p2 * (1 - p2 + p4 / 3.0) +  0.156 * p * (1 - 2.0 * p2 / 3.0 + p4 / 5.0)
        return alpha

    def _G(self,p):
        "[Numerical Test Status:Pass]"
        return 0.357 + 0.642 * p **2

    def _dG_dp(self,p):
        "[Numerical Test Status:Pass]"
        return 1.284 * p

    def _dalpha_dp(self,p):
        """return dalpha / dp"""
        "[Numerical Test Status:Pass]"
        #return -1.064*p**5 + 0.156*p**4 + 2.128*p**3 + 0.312*p**2 - 1.064*p + 0.156 # from mathmatica, wrong
        #return (133*p*(p**4/3 - p**2 + 1))/125 - (39*p*((4*p)/3 - (4*p**3)/5))/250 - (133*p**2*(2*p - (4*p**3)/3))/250 - (13*p**2)/125 + (39*p**4)/1250 + 39/250 # from matlab,right
        return ((266*p + 39)*(p**2 - 1)**2)/250 # from matlab, simplified version, pass the test

    def _gamma(self,p):
        return -11.11

    def _alpha_a(self,n_a,n_b):
        return self._alpha(self._get_p((n_a,n_b)))
     
    def _alpha_b(self,n_a,n_b):
        return self._alpha(-self._get_p((n_a,n_b)))

    def _alpha_p(self,p):
        return 0.5 * (self._alpha(p) + self._alpha(-p))

    def _alpha_m(self,p):
        return 0.5 * (self._alpha(p) - self._alpha(-p))

    def _dalpha_p_dp(self,p):
        """return dalpha_p / dp"""
        "[Matlab verified]"
        "[Numerical Test Status:Pass]"
        return 1.064*p*(p**2 - 1.0)**2

    def _dalpha_m_dp(self,p):
        """return dalpha_m / dp"""
        "[Matlab verified]"
        "[Numerical Test Status:Pass]"
        return 0.156*(p**2 - 1.0)**2

    def _C(self,n_a,n_b):
        """return C tilde"""
        "[Numerical Test Status:Pass]"
        p = self._get_p((n_a,n_b))
        return self._alpha_p(p) * (n_a + n_b)**(1.0/3) / self.gamma

    def _D(self,n_a,n_b):
        "[Numerical Test Status:Pass]"
        N1 = (6 * np.pi**2 *(n_a + n_b))**(5.0/3)/20/np.pi**2
        p = self._get_p((n_a,n_b))
        N2 = self._G(p) - self._alpha(p) * ((1+p)/2.0)**(5.0/3) - self._alpha(-p) * ((1-p)/2.0)**(5/3)
        return N1 * N2

    def _Dp(self,p):
        "[Numerical Test Status:Pass]"
        return self._G(p) - self._alpha(p) * ((1+p)/2.0)**(5.0/3) - self._alpha(-p) * ((1-p)/2.0)**(5/3)

    def _dD_dp(self,p):
        """return the derivative 'dD(p)/dp' """
        "[Numerical Test Status:Pass]"
        p_p, p_m = (1+p)**(2.0/3), (1-p)**(2.0/3)
        pp = p_p + p_m
        pm = p_p - p_m
        p2 = p**2
        p3 = p2*p
        p4 = p3*p
        p5 = p4*p
        p6 = p5*p
        #dD_p = 1.284*p + 0.57904*p_m + 0.51615*p**2*p_m +0.82315*p**3*p_m - 0.90042*p**4*p_m - 0.40065*p**5*p_m + 0.42823*p**6*p_m - 0.46617*p*p_p - 0.57904*p_p - 0.51615*p**2*p_p + 0.82315*p**3*p_p + 0.90042*p**4*p_p- 0.40065*p**5*p_p - 0.42823*p**6*p_p - 0.46617*p*p_m
        #dD_p = 1.284*p + (0.57904 - 0.46617*p+ 0.51615*p2 +0.82315*p3 - 0.90042*p4 - 0.40065*p5 + 0.42823*p6)*p_m + ( - 0.57904- 0.46617*p - 0.51615*p2 + 0.82315*p3 + 0.90042*p4- 0.40065*p5 - 0.42823*p6)*p_p
        #dD_p = 1.284*p - 0.57904 * pm - 0.46617 *p * pp - 0.51615*p2 * pm +0.82315*p3 * pp + 0.90042*p4 * pm - 0.40065*p5 * pp -0.42823*p6*pm
        dD_p = 1.284*p + (- 0.57904  - 0.51615*p2 + 0.90042*p4 -0.42823*p6)*pm +(0.82315*p3- 0.46617 *p- 0.40065*p5) * pp
        return dD_p

    def _dD_dn(self,ns):
        """Return the derivative `dD(n_a,n_b)/d n_a and d n_b` """
        "[Numerical Test Status:Pass]"
        na, nb = ns
        n = na + nb
        ### matlab gives long formulas used to check numerical code.
        # the follow 3 lines is from matlab
        #nb32 = (nb/(na + nb))**(2/3)
        #na32 = (na/(na + nb))**(2/3)
        #dD_n_a = 7.5963331205759943604501182783546*n**(2/3)*((0.642*(na - 1.0*nb)**2)/n**2 - (0.000066666666666666666666666666666667*(na/n)**(5/3)*(19049.0*na**6 + 114294.0*na**5*nb + 285735.0*na**4*nb**2 + 185780.0*na**3*nb**3 + 248295.0*na**2*nb**4 + 99318.0*na*nb**5 + 16553.0*nb**6))/n**6 - (0.000066666666666666666666666666666667*(nb/n)**(5/3)*(16553.0*na**6 + 99318.0*na**5*nb + 248295.0*na**4*nb**2 + 185780.0*na**3*nb**3 + 285735.0*na**2*nb**4 + 114294.0*na*nb**5 + 19049.0*nb**6))/n**6 + 0.357) - (0.00050642220803839962403000788522364*(92448.0*na*nb**6 - 23112.0*na**6*nb + 23112.0*nb**7 + 16553.0*nb**7*na32 - 19049.0*nb**7*nb32 + 115560.0*na**2*nb**5 - 115560.0*na**4*nb**3 - 92448.0*na**5*nb**2 + 99318.0*na*nb**6*na32 + 19049.0*na**6*nb*na32 - 114294.0*na*nb**6*nb32 - 16553.0*na**6*nb*nb32 + 248295.0*na**2*nb**5*na32 - 75724.0*na**3*nb**4*na32 + 637095.0*na**4*nb**3*na32 + 114294.0*na**5*nb**2*na32 - 637095.0*na**2*nb**5*nb32 + 75724.0*na**3*nb**4*nb32 - 248295.0*na**4*nb**3*nb32 - 99318.0*na**5*nb**2*nb32))/n**(19/3)
        #dD_n_b = (0.00050642220803839962403000788522364*(23112.0*na*nb**6 - 92448.0*na**6*nb - 23112.0*na**7 + 19049.0*na**7*na32 - 16553.0*na**7*nb32 + 92448.0*na**2*nb**5 + 115560.0*na**3*nb**4 - 115560.0*na**5*nb**2 + 16553.0*na*nb**6*na32 + 114294.0*na**6*nb*na32 - 19049.0*na*nb**6*nb32 - 99318.0*na**6*nb*nb32 + 99318.0*na**2*nb**5*na32 + 248295.0*na**3*nb**4*na32 - 75724.0*na**4*nb**3*na32 + 637095.0*na**5*nb**2*na32 - 114294.0*na**2*nb**5*nb32 - 637095.0*na**3*nb**4*nb32 + 75724.0*na**4*nb**3*nb32 - 248295.0*na**5*nb**2*nb32))/n**(19/3) + 7.5963331205759943604501182783546*n**(2/3)*((0.642*(na - 1.0*nb)**2)/n**2 - (0.000066666666666666666666666666666667*(na/n)**(5/3)*(19049.0*na**6 + 114294.0*na**5*nb + 285735.0*na**4*nb**2 + 185780.0*na**3*nb**3 + 248295.0*na**2*nb**4 + 99318.0*na*nb**5 + 16553.0*nb**6))/n**6 - (0.000066666666666666666666666666666667*(nb/n)**(5/3)*(16553.0*na**6 + 99318.0*na**5*nb + 248295.0*na**4*nb**2 + 185780.0*na**3*nb**3 + 285735.0*na**2*nb**4 + 114294.0*na*nb**5 + 19049.0*nb**6))/n**6 + 0.357)

        p = self._get_p(ns)
        p2 = p**2
        dp_n_a,dp_n_b = self._dp_dn(ns)
        dD_p = self._dD_dp(p=p)
        N0 = (6 * np.pi**2) ** (5.0/3) / 20 / np.pi**2
        N1 = self._D(na,nb) / 0.6 / n
        N2 = N0 * n**(5/3) * dD_p
        dD_n_a = N1 + N2 * dp_n_a
        dD_n_b = N1 + N2 * dp_n_b
        return (dD_n_a,dD_n_b)

    def _dC_dn(self,ns):
        """return dC / dn"""
        "[Numerical Test Status:Pass]"
        na, nb = ns
        n = na + nb
        p = self._get_p(ns)
        p2 = p**2
        dp_n_a,dp_n_b = self._dp_dn(ns)
        dC_dn_a = self._alpha_p(p) * n **(-2/3)/3 + n**(1/3)*self._dalpha_p_dp(p) * dp_n_a
        dC_dn_b = self._alpha_p(p) * n **(-2/3)/3 + n**(1/3)*self._dalpha_p_dp(p) * dp_n_b
        gamma = self.gamma # do not forget the gamma in the demonimor
        return (dC_dn_a/gamma,dC_dn_b/gamma)

    def _dp_dn(self,ns):
        na,nb = ns
        n = na + nb
        n2 = n*n
        dp_n_a, dp_n_b= 2*nb/n2,-2*na/n2
        return (dp_n_a, dp_n_b)

    def get_alphas(self,ns = None):
        p = self._get_p(ns)
        p2 = p**2
        p4 = p2**2
        # too make code faster, equation (98) is divived into even part and odd part
        alpha_even,alpha_odd = 1.0094 + 0.532 * p2 * (1 - p2 + p4 / 3.0), 0.156 * p * (1 - 2.0 * p2 / 3.0 + p4 / 5.0)
        alpha_a,alpha_b =  alpha_odd + alpha_even, -alpha_odd + alpha_even #fixed an error here
        alpha_p = alpha_even # it's defined as (alpha_a + alpha_b) /2 , which is just alpha_even
        return (alpha_a,alpha_b,alpha_p)

    def get_Del(self, twist=0):
        """return the second order derivative operator matrix"""
        k_bloch = twist/self.Lx
        k = self.kx + k_bloch
        N = self.Nx

        # Unitary matrix implementing the FFT including the phase
        # twist
        U = np.exp(-1j*k[:, None]*self.x[None, :])/np.sqrt(N)
        assert np.allclose(U.conj().T.dot(U), np.eye(N))
        
        nabla  = np.dot(U.conj().T, (1j*self.hbar * k)[:, None]/2/self.m * U)
        return nabla

    def get_Laplacian(self, twist=0):
        """return the second order derivative operator matrix"""
        k_bloch = twist/self.Lx
        k = self.kx + k_bloch
        N = self.Nx
        # Unitary matrix implementing the FFT including the phase
        # twist
        U = np.exp(-1j*k[:, None]*self.x[None, :])/np.sqrt(N)
        assert np.allclose(U.conj().T.dot(U), np.eye(N))
        # Kinetic energy matrix
        D2 = np.dot(U.conj().T, (self.hbar * k)[:, None]**2* U)

        #D_ = np.fft.ifft(-1j*k*np.fft.fft(np.eye(N), axis=1), axis=1)
        #D2_ = -D_.dot(D_.T.conj())
        #assert np.allclose(D2, D2_)

        return D2 # is there a minus sign??? Need to verify!!!!

    def get_Ks(self, kper=0,twist=0):
        """return the original kinetic density matrix for homogeneous system"""
        D2 = self.get_Laplacian(twist)
        K = D2  * self.hbar**2/2/self.m + kper
        return (K,K)

    def get_modified_K(self, D2,alpha):
        """"return a modified kinetic density  matrix"""
        "[Numerical Test Status:Pass]"
        A = np.diag(alpha)
        K = (D2.dot(A) - np.diag(self._D2.dot(alpha)) + A.dot(D2)) / 2
        #assert np.allclose(K, K.conj().T) # the assertion is not always good, K would be slighly off from being Haermintian
        return K

    def get_modified_Ks(self,alpha_a,alpha_b,twist=0):
        """return the modified kinetic density  matrix"""
        "[Numerical Test Status:Pass]"
        D2 = self.get_Laplacian(twist=twist)
        # K( A U') = [(A u')'= (A u)'' - A'' u + A u'']/2
        K_a =  self.get_modified_K(D2,alpha_a)
        K_b =  self.get_modified_K(D2,alpha_b)
        #assert np.allclose(K_b, K_b.conj().T)
        return (self.hbar**2/2/self.m * K_a,self.hbar**2/2/self.m * K_b)

    def get_modified_Vs(self,delta, ns=None, taus=None, kappa=0):
        """get the modified V functional terms"""
        #return self.v_ext
        if ns == None or taus == None:
            return self.v_ext
        U_a, U_b = self.v_ext
        tau_a, tau_b = taus
        tau_p, tau_m = tau_a + tau_b,tau_a - tau_b
        alpha_a, alpha_b, alpha_p = self.get_alphas(ns)
        p = self._get_p(ns)
        dp_n_a,dp_n_b = self._dp_dn(ns)
        dalpha_p = self._dalpha_p_dp(p)
        dalpha_m = self._dalpha_m_dp(p)
        dalpha_p_dn_a, dalpha_p_dn_b, dalpha_m_dn_a, dalpha_m_dn_b= dalpha_p * dp_n_a, dalpha_p * dp_n_b, dalpha_m * dp_n_a, dalpha_m * dp_n_b
        dC_dn_a,dC_dn_b = self._dC_dn(ns)
        dD_dn_a,dD_dn_b = self._dD_dn(ns=ns)
        C0 = self.hbar**2 /self.m 
        C1 = C0 / 2
        C2 = tau_p * C1 - delta.conj().T * kappa / alpha_p
        V_a = dalpha_m_dn_a * tau_m * C1 + dalpha_p_dn_a * C2 + dC_dn_a + C0 * dD_dn_a + U_a 
        V_b = dalpha_m_dn_b * tau_m * C1 + dalpha_p_dn_b * C2 + dC_dn_b + C0 * dD_dn_b + U_b
        return (V_a,V_b)

    def get_Lambda(self,k0,kc,dim = 3):
        """return the renomalization condition parameter Lambda"""
        if dim ==3:
            Lambda = self.m / self.hbar**2/2/np.pi**2 *(1.0 - k0/kc/2*np.log((kc+k0)/(kc-k0)))
        elif dim == 2:
            Lambda = self.m /self.hbar**2/4/np.pi *np.log((kc/k0)**2 - 1)
        elif dim == 1:
            Lambda = self.m/self.hbar**2/2/np.pi * np.log((kc-k0)/(kc+k0))/k0
        return Lambda

    def get_effective_g(self,ns,Vs,mus,alpha_p,dim = 3):
        """get the effective g"""
        V_a, V_b = Vs
        mu_p = (np.sum(mus) - V_a + V_b) / 2
        k0 = (2*self.m/self.hbar**2*mu_p/alpha_p)**0.5
        kc = (2*self.m/self.hbar**2 * (self.E_c + mu_p)/alpha_p)**0.5
        C = alpha_p * (np.sum(ns)**(1.0/3))/self.gamma
        Lambda = self.get_Lambda(k0=k0,kc=kc,dim=dim)
        g = alpha_p/(C - Lambda)
        return g

    def get_Ks_Vs(self, delta, mus=(0,0), ns=None, taus=None, kappa=0, ky=0, kz=0, twist=0): 
        """Return the kinetic energy and modifled potential matrics."""
        alphas = self.get_alphas(ns)
        k_per = self.hbar**2/2/self.m  *( kz**2 + ky**2)
        k_per = np.diag(np.ones(self.Nx)) * k_per
        if alphas == None or ns == None or taus == None:
            return (self.get_Ks(kper=k_per,twist=twist), self.get_modified_Vs(delta=delta,ns=ns,taus=taus,kappa=kappa))
        return (self.get_Ks(kper=k_per,twist=twist), self.get_modified_Vs(delta=delta,ns=ns,taus=taus,kappa=kappa,alphas=alphas))
        alpha_a, alpha_b, alpha_p = alphas
       
        K_a,K_b = self.get_modified_Ks(alpha_a=alpha_a,alpha_b=alpha_b,twist = twist)
        V_a, V_b = self.get_modified_Vs(delta=delta,ns=ns,taus=taus,kappa=kappa,alphas=alphas)
        self.g_eff = self.get_effective_g(ns = ns, Vs=(V_a,V_b), mus = mus,alpha_p = alpha_p)
        return ((K_a + k_per, K_b + k_per), (V_a, V_b))
            
    def get_v_ext(self):
        """Return the external potential."""
        #v_a = (-self.V0 * (1-((1+np.cos(2*np.pi * self.cells*self.x/self.L))/2)**self.power))
        #v_b = 0 * self.x
        #return v_a, v_b
        return (0, 0)

    def f(self, E, E_c=None):
        """Return the Fermi-Dirac distribution at E."""
        if E_c is None:
            E_c = self.E_c
        if self.T > 0:
            f = 1./(1+np.exp(E/self.T))
        else:
            f = (1 - np.sign(E))/2
        return f
        # the following line of code will remove some states when computing density, causing distortion
        #return np.where(abs(E)<E_c, f, 0)

    def get_H(self, mus, delta, ns=None,taus=None, kappa=0, ky=0, kz=0, twist=0):
        """Return the single-particle Hamiltonian with pairing. """
        zero = np.zeros_like(sum((self.Nx,self.Nx)))
        Delta = np.diag((delta.reshape(self.Nx) + zero)) 
        mu_a, mu_b = mus
        mu_a += zero
        mu_b += zero
        (K_a, K_b),(V_a,V_b) = self.get_Ks_Vs(delta = delta,mus=mus,kappa=kappa,taus=taus,ns=ns,ky=ky,kz=kz,twist=twist)
        #V_a, V_b = self.get_modified_Vs(delta=delta,ns=ns,taus=taus,kappa=kappa)
        #alpha_a, alpha_b, alpha_p = self.get_alphas(ns)
        #K_a,K_b = self.get_modified_Ks(alpha_a=alpha_a,alpha_b=alpha_b,twist = twist)
        Mu_a, Mu_b = np.diag((mu_a - V_a).ravel()), np.diag((mu_b - V_b).ravel())
        assert (Mu_a.shape[0] == self.x.shape[0])
        H = np.bmat([[K_a - Mu_a, -Delta], # may need remove the minus sign for Delta, need to check
                     [-Delta.conj(), -(K_b - Mu_b)]]) 
        assert np.allclose(H.real,H.conj().T.real)
        return np.asarray(H)

    def _get_modified_taus(self,taus,js):
        """return the modified taus with currents in it, not implement"""
        return taus

    def get_ns_taus_kappa(self, H, compute_current_flag=False): 
        """Return the n_a, n_b"""
        N = self.Nx
        Es, psi = np.linalg.eigh(H) 
        us, vs = psi.reshape(2, N, N*2)
        us,vs = us.T,vs.T
        j_a,j_b =None,None
        n_a, n_b = np.sum(np.abs(us[i])**2 * self.f(Es[i])  for i in range(len(us)))/self.dx, np.sum(np.abs(vs[i])**2 * self.f(-Es[i])  for i in range(len(vs)))/self.dx

        assert not np.allclose(n_a,0) and not np.allclose(n_b,0)
        nabla = self.get_Del()

        # From Dr. Forbes' implementaiton
        #tau_a = (6*np.pi**2*n_a)**(5/3)/10/np.pi**2
        #tau_b = (6*np.pi**2*n_b)**(5/3)/10/np.pi**2


        tau_a = np.sum(np.abs(nabla.dot(us[i]))**2 * self.f(Es[i]) for i in range(len(us)))/self.dx
        tau_b = np.sum(np.abs(nabla.dot(vs[i]))**2 * self.f(-Es[i]) for i in range(len(vs)))/self.dx
        kappa = 0.5 * np.sum(us[i]*vs[i].conj() *(self.f(Es[i]) - self.f(-Es[i])) for i in range(len(us)))/self.dx
        if compute_current_flag:
            j_a = 0.5 * sum( (us[i].conj()*nabla.dot(us[i])-us[i]*nabla.dot(us[i].conj())) * self.f(Es[i]) for i in range(len(us)))
            j_b = 0.5 * sum( (vs[i].conj()*nabla.dot(vs[i])-vs[i]*nabla.dot(vs[i].conj())) * self.f(Es[i]) for i in range(len(vs)))
        return ((n_a, n_b),self._get_modified_taus(taus=(tau_a,tau_b),js=(j_a,j_b)),kappa)

    def get_ns_taus_kappa_average_3d(self,mus,delta,ns=None,taus=None,kappa=None, kc=None, N_twist=8,abs_tol=1e-6):
        if kc is None:
            kc = 100 * np.sqrt(2 * self.m * self.E_c)/self.hbar
        twists = np.arange(0, N_twist)*2*np.pi/N_twist
        n_a,n_b,tau_a_,tau_b_,kappa_ =0,0,0,0,0

        zero = np.zeros_like(sum((self.Nx,self.Nx)))
        Delta = np.diag((delta.reshape(self.Nx) + zero)) 
        mu_a, mu_b = mus
        mu_a, mu_b= zero + mu_a, zero + mu_b
        V_a, V_b = self.get_modified_Vs(delta=delta, ns=ns, taus=taus, kappa=kappa)
        Mu_a, Mu_b = np.diag((mu_a - V_a).ravel()), np.diag((mu_b - V_b).ravel())
        alpha_a,alpha_b, alpha_p = self.get_alphas(ns)
        iMat = np.diag(np.ones(self.Nx))
        for twist in twists:
            def g(kz=0):
                def f(ky=0):
                    k_per = self.hbar**2/2/self.m  *ky**2
                    K_a,K_b = self.get_modified_Ks(alpha_a=alpha_a,alpha_b=alpha_b,twist = twist)
                    H = np.bmat([[K_a - Mu_a, -Delta], # may need remove the minus sign for Delta, need to check
                         [-Delta.conj(), -(K_b - Mu_b)]]) 
                    assert np.allclose(H.real,H.conj().T.real)
                    _ns,_taus,_kappa = self.get_ns_taus_kappa(np.asarray(H))
                    return np.concatenate((_ns[0].ravel(),_ns[1].ravel(),_taus[0].ravel(),_taus[1].ravel(),_kappa.ravel()))
                rets = mquad(f,-kc,kc,abs_tol=abs_tol)/2/kc
                return rets
            rets = mquad(g,-kc,kc,abs_tol=abs_tol)/2/kc
            rets = rets.reshape(5,self.Nx)
            n_a = n_a + rets[0]
            n_b = n_b + rets[1]
            tau_a_ = tau_a_ + rets[2]
            tau_b_ = tau_b_ + rets[3]
            kappa_ = kappa_ + rets[4]
        return ((n_a/N_twist,n_b/N_twist),(tau_a_/N_twist,tau_b_/N_twist),kappa_/N_twist)

    def get_ns_taus_kappa_average_2d(self,mus,delta,ns=None,taus=None,kappa=None, kc=None, N_twist=8,abs_tol=1e-6):
        if kc is None:
            kc = 100 * np.sqrt(2 * self.m * self.E_c)/self.hbar
        twists = np.arange(0, N_twist)*2*np.pi/N_twist
        n_a,n_b,tau_a_,tau_b_,kappa_ =0,0,0,0,0

        zero = np.zeros_like(sum((self.Nx,self.Nx)))
        Delta = np.diag((delta.reshape(self.Nx) + zero)) 
        mu_a, mu_b = mus
        mu_a, mu_b= zero + mu_a, zero + mu_b
        V_a, V_b = self.get_modified_Vs(delta=delta, ns=ns, taus=taus, kappa=kappa)
        Mu_a, Mu_b = np.diag((mu_a - V_a).ravel()), np.diag((mu_b - V_b).ravel())
        alpha_a,alpha_b, alpha_p = self.get_alphas(ns)
        iMat = np.diag(np.ones(self.Nx))
        for twist in twists:
            def f(ky=0):
                k_per = self.hbar**2/2/self.m  *ky**2
                K_a,K_b = self.get_modified_Ks(alpha_a=alpha_a,alpha_b=alpha_b,twist = twist)
                H = np.bmat([[K_a - Mu_a, -Delta], # may need remove the minus sign for Delta, need to check
                     [-Delta.conj(), -(K_b - Mu_b)]]) 
                assert np.allclose(H.real,H.conj().T.real)
                _ns,_taus,_kappa = self.get_ns_taus_kappa(np.asarray(H))
                return np.concatenate((_ns[0].ravel(),_ns[1].ravel(),_taus[0].ravel(),_taus[1].ravel(),_kappa.ravel()))
            rets = mquad(f,-kc,kc,abs_tol=abs_tol)/2/kc
            rets = rets.reshape(5,self.Nx)
            n_a = n_a + rets[0]
            n_b = n_b + rets[1]
            tau_a_ = tau_a_ + rets[2]
            tau_b_ = tau_b_ + rets[3]
            kappa_ = kappa_ + rets[4]
        return ((n_a/N_twist,n_b/N_twist),(tau_a_/N_twist,tau_b_/N_twist),kappa_/N_twist)

    def get_ns_taus_kappa_average_1d(self,mus,delta,ns=None,taus=None,kappa=None, N_twist = 8,abs_tol=1e-12):
        kc = np.sqrt(2 * self.m * self.E_c)/self.hbar
        twists = np.arange(0, N_twist)*2*np.pi/N_twist
        n_a,n_b,tau_a_,tau_b_,kappa_ =0,0,0,0,0
        for twist in twists:
            H = self.get_H(mus=mus,delta=delta,ns=ns,taus=taus,kappa=kappa,twist=twist)
            _ns,_taus,_kappa = self.get_ns_taus_kappa(H)
            n_a = n_a + _ns[0]
            n_b = n_b + _ns[1]
            tau_a_ = tau_a_ + _taus[0]
            tau_b_ = tau_b_ + _taus[1]
            kappa_ = kappa_ + _kappa
        return ((n_a/N_twist,n_b/N_twist),(tau_a_/N_twist,tau_b_/N_twist),kappa_/N_twist)

    def get_R(self, mus, delta, N_twist=1, ky=0, kz=0,twists=None):
        """Return the density matrix R."""
        Rs = []
        if twists is None:
            twists = np.arange(0, N_twist)*2*np.pi/N_twist

        for twist in twists:
            H = self.get_H(mus=mus, delta=delta,ky = ky, kz=kz, twist=twist)
            d, UV = np.linalg.eigh(H)
            R = UV.dot(self.f(d)[:, None]*UV.conj().T)
            Rs.append(R)
        R = sum(Rs)/len(Rs)
        return R
    def get_R_twist_average(self, mus, delta,ky=0, kz=0,abs_tol=1e-12):
        """Return the density matrix R."""
        R0 = 1.0
        def f(twist):
            H = self.get_H(mus=mus, delta=delta,ky=ky, kz=kz, twist=twist)
            d, UV = np.linalg.eigh(H)
            R = UV.dot(self.f(d)[:, None]*UV.conj().T)
            return R/R0
        R0 = f(0)
        R = R0 * mquad(f, -np.pi, np.pi, abs_tol=abs_tol)/2/np.pi
        return R
    def get_R_per_average(self,mus,delta, abs_tol=1e-12):
        kc = np.sqrt(2 * self.m * self.E_c)/self.hbar
        
        def f(kz=0):
            def g(ky=0):
                R = self.get_R(mus=mus,delta=delta,N_twist=4,ky=ky,kz=kz)
                return R
            R = mquad(g,-kc,kc,abs_tol=abs_tol) /2 /kc# may need to divide a factor of 2*kc?
            return R

        R = mquad(f,-kc,kc,abs_tol=abs_tol) /2 /kc# may need to divide a factor of 2*kc?
        return R
    def get_abnormal_energy_density(self,ns,taus,kappa):
        na,nb=ns
        ta,tb = taus
        N1 = self.hbar**2/self.m *(self._alpha_a(na,nb) * ta/2 + self._alpha_b(na,nb) * tb/2)
        N2 = -(6*np.pi**2*self.hbar**2*(na+nb))**(5/3)/20/self.m/np.pi**2
        p = self._get_p(ns)
        N3 = self._alpha(p) * ((1 + p)/2)**(5/3) + self._alpha(-p)*((1-p)/2)**(5/3)
        N = N1 + N2 * N3
        return N
    def get_energy_density(self, ns,taus,kappa):
        """return energy density for aslda"""
        n_a,n_b = ns
        p = self._get_p(ns)
        tau_a, tau_b = taus
        normal_ed = self.hbar**2 / self.m * (6 * np.pi**2 * (n_a + n_b))**(5/3)/20/np.pi**2 * self._G(p)
        """
        # Check if the difference of energy densities from aslda and normal phase is as expected
        tau_a = tau_a * 0
        tau_b = tau_b * 0
        N1 = (6*np.pi**2*self.hbar**2*(n_a+n_b))**(5/3)/20/self.m/np.pi**2
        N2 = self._alpha(p) *((1+p)/2)**(5/3) + self._alpha(-p) * ((1-p)/2)**(5/3)
        N3 = N1 * N2 
        aslda_ed = (self._alpha_a(n_a,n_b) * tau_a / 2 + self._alpha_b(n_a,n_b) * tau_b + self._D(n_a,n_b)) * self.hbar**2/self.m + self.g_eff * kappa.conj().T * kappa # dot product or element-wise?
        assert(np.allclose(normal_ed - aslda_ed,N3))
        """
        return normal_ed
    def gx(self,ns,taus,kappa):
        na,nb = ns
        """PRL 101, 215301 (2008):Unitary Fermi Supersolid: The Larkin-Ovchinnikov Phase"""
        ed = self.get_energy_density(ns=ns,taus=taus,kappa=kappa)
        gx53_= ed * 10 *self.m /3/self.hbar**2 /(6 * np.pi**2)**(2.0/3) / na**(5.0/3)
        return gx53_ **(3.0/5)
    #Try to use broyden method
    #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.broyden1.html
   