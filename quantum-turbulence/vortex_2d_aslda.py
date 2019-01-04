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
    
    def __init__(self, Nxy=(32, 32), Lxy=(2*np.pi,2 * np.pi), dx=None, T=0,E_c=np.inf):

        dy = dx
        if dx is None:
            dx, dy = np.divide(Lxy, Nxy)
        elif Lxy is None:
            Lxy = np.prod(Nxy, dx)
        elif Nxy is None:
            Nxy = np.ceil(Lxy / dx).astype(int)
        Nx, Ny = Nxy
        Lx, Ly = Lxy
        self.xy = ((np.arange(Nx) * dx - Lx / 2)[:, None], (np.arange(Ny) * dy - Ly / 2)[None, :])
        self.kxy = (2*np.pi * np.fft.fftfreq(Nx, dx)[:, None],2*np.pi * np.fft.fftfreq(Ny, dy)[None, :])
        self.dx = dx
        self.Nxy = tuple(Nxy)
        self.Lxy = tuple(Lxy)
        self.T = T
        self.E_c = E_c
        self.gamma = -11.11
        self.g_eff = 1.0
        # External potential
        self.v_ext = self.get_v_ext()

    def fft2(self, y):
            return np.fft.fftn(y, axes=(0,1))
            
    def ifft2(self, y):
            return np.fft.ifftn(y, axes=(0,1))
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
        "[Numerical Test Status:Pass]"
        return -11.11

    def _alpha_a(self,n_a,n_b):
        return _alpha(self._get_p((n_a,n_b)))
     
    def _alpha_b(self,n_a,n_b):
        return _alpha(-self._get_p((n_a,n_b)))

    def _alpha_p(self,n_a,n_b):
        return 0.5 * (self._alpha_a(n_a,n_b) + self._alpha_b(n_a,n_b))

    def _alpha_m(self,n_a,n_b):
        return 0.5 * (self._alpha_a(n_a,n_b) - self._alpha_b(n_a,n_b))

    def _C(self,n_a,n_b):
        return self.alpha_p(n_a,n_b) * (n_a + n_b)**(1.0/3) / self._gamma(self._get_p(n_a,n_b))

    def _D(self,n_a,n_b):
        "[Numerical Test Status:Pass]"
        N1 = (6 * np.pi**2 *(n_a + n_b))**(5.0/3)/20/np.pi**2
        p = self._get_p((n_a,n_b))
        N2 = self._G(p) - self._alpha(p) * ((1+p)/2.0)**(5.0/3) - self._alpha(-p) * ((1-p)/2.0)**(5/3)
        return N1 * N2

    def _Dp(self,p):
        "[Numerical Test Status:Pass]"
        return self._G(p) - self._alpha(p) * ((1+p)/2.0)**(5.0/3) - self._alpha(-p) * ((1-p)/2.0)**(5/3)

    def _get_dD_dp(self,p):
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

    def get_alphas_p(self,ns = None):
        p = self._get_p(ns)
        p2 = p**2
        p4 = p2**2
        # too make code faster, equation (98) is divived into even part and odd part
        alpha_even,alpha_odd = 1.0094 + 0.532 * p2 * (1 - p2 + p4 / 3.0), 0.156 * p * (1 - 2.0 * p2 / 3.0 + p4 / 5.0)

        alpha_a,alpha_b =  alpha_odd + alpha_even, -alpha_odd + alpha_even #fixed an error here
        alpha_p = alpha_even # it's defined as (alpha_a + alpha_b) /2 , which is just alpha_even
        return (alpha_a,alpha_b,alpha_p,p)
    def _get_dD_dn(self,ns):
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


        n2 = n**2
        p = self._get_p(ns)
        p2 = p**2

        dp_n_a = 2*nb/n2 # dp/dna
        dp_n_b = -2*na/n2# dp/dnb
        
        dD_p = self._get_dD_dp(p=p)
        N0 = (6 * np.pi**2) ** (5.0/3) / 20 / np.pi**2
        N1 = self._D(na,nb) / 0.6 / n
        N2 = N0 * n**(5/3) * dD_p
        dD_n_a = N1 + N2 * dp_n_a
        dD_n_b = N1 + N2 * dp_n_b
        return (dD_n_a,dD_n_b)

    def get_nabla(self, twist=(0,0)):
        k_bloch = np.divide(twist, self.Lxy)
        kxy = [_k + _kb for _k, _kb in zip(self.kxy, k_bloch)]
        mat_shape = (np.prod(self.Nxy),)*2
        tensor_shape = self.Nxy + self.Nxy
        nabla = np.eye(mat_shape[0]).reshape(tensor_shape)
        nabla  = self.ifft2(-1j*sum(_k  for _k in self.kxy)[:, :,  None, None]*self.fft2(nabla)).reshape((np.prod(self.Nxy),)*2).reshape(mat_shape)
        return nabla

    def get_D2(self, twist=(0,0)):
        """return the second order derivative operator matrix"""
        k_bloch = np.divide(twist, self.Lxy)
        kxy = [_k + _kb for _k, _kb in zip(self.kxy, k_bloch)]
        mat_shape = (np.prod(self.Nxy),)*2
        tensor_shape = self.Nxy + self.Nxy
        K = np.eye(mat_shape[0]).reshape(tensor_shape)
        D2 = self.ifft2(sum(-_k**2 for _k in self.kxy)[:, :,  None, None]*self.fft2(K)).reshape((np.prod(self.Nxy),)*2).reshape(mat_shape)
        return D2

    def get_Ks(self, twist=(0,0)):
        """return the original kinetic density matrix for homogeneous system"""
        D2 = self.get_D2(twist)
        K = -D2  * self.hbar**2/2/self.m
        return (K,K)

    def get_modified_K(self, D2,alpha):
        """"return a modified kinetic density  matrix"""
        "[Numerical Test Status:Pass]"
        A = np.diag(alpha.ravel())
        K = (D2.dot(A) - np.diag(D2.dot(alpha.ravel())) + A.dot(D2)) / 2
        return K

    def get_modified_Ks(self,alpha_a,alpha_b,twist=(0,0)):
        """return the modified kinetic density  matrix"""
        "[Numerical Test Status:Pass]"
        D2 = self.get_D2(twist=twist)
        # K( A U') = [(A u')'= (A u)'' - A'' u + A u'']/2
        K_a = - self.hbar**2/2/self.m * self.get_modified_K(D2,alpha_a)
        K_b = - self.hbar**2/2/self.m * self.get_modified_K(D2,alpha_b)
        return (K_a,K_b)

    def get_modified_Vs(self,delta, ns=None, taus=None,nu=0, alphas = None,twist=(0,0)):
        """get the modified V functional terms"""
        return self.v_ext
        if ns == None or taus == None or alphas == None:
            return self.v_ext
        U_a, U_b = self.v_ext
        k_bloch = np.divide(twist, self.Lxy)
        kxy = [_k + _kb for _k, _kb in zip(self.kxy, k_bloch)]
        if(alphas == None):
            alphas = self.get_alphas_p(ns)
        tau_a, tau_b = taus
        tau_p, tau_m = tau_a + tau_b,tau_a - tau_b
        na,nb = ns
        n = na + nb
        n2 = n**2
        alpha_a, alpha_b, alpha_p,p = alphas
        p2,p3,p4,p5,p6 = p**2,p**3,p**4,p**5, p**6
        # alpha_p = 1.094 - 0.532*p2 + 0.532*p4 - 0.177333*p6
        alpha_m = 0.156*p + 0.104*p3 + 0.0312*p5
        dp_n_a = 2*nb/n2
        dp_n_b = -2*na/n2
        dalpha = -1.064*p5 + 0.156*p4 + 2.128*p3 + 0.312*p2-1.064*p+0.156
        dalpha_p = -1.064*p5 + 2.128*p3 - 1.064*p
        dalpha_m = 0.156*(p2+1.)**2
        dalpha_p_n_a, dalpha_p_n_b, dalpha_m_a, dalpha_m_b= dalpha_p * dp_n_a, dalpha_p * dp_n_b, dalpha_m * dp_n_a, dalpha_m * dp_n_b # apply chain rule
        # partial C / partial a and b
        dC_n_a,dC_n_b = (dalpha_p * dp_n_a + alpha_p * n **(-2/3) / 3)/self.gamma,(dalpha_p * dp_n_b + alpha_p * n **(-2/3) / 3)/self.gamma # comman term can be compute just once, do it later
        dD_n_a,dD_n_b = self._get_dD_dn(ns=ns)
        C1 = self.hbar**2 /2/self.m 
        V_a = dalpha_m_a * tau_m * C1  + dalpha_p_n_a * (tau_p * C1 - delta.conj().T * nu / alpha_p) + dC_n_a + dD_n_a + U_a # the common term can be compute just once
        V_b = dalpha_m_b * tau_m * C1 + dalpha_p_n_b * (tau_p * C1 - delta.conj().T * nu / alpha_p) + dC_n_b + dD_n_b + U_b
        return (V_a,V_b)

    def get_Lambda(self,k0,kc,dim = 2):
        """return the renomalization condition parameter Lambda"""
        if dim ==3:
            Lambda = self.m / self.hbar**2/2/np.pi**2 *(1.0 - k0/kc/2*np.log((kc+k0)/(kc-k0)))
        elif dim == 2:
            Lambda = self.m /self.hbar**2/4/np.pi *np.log((kc/k0)**2 - 1)
        elif dim == 1:
            Lambda = self.m/self.hbar**2/2/np.pi * np.log((kc-k0)/(kc+k0))/k0
        return Lambda

    def get_effective_g(self,ns,Vs,mus,alpha_p,dim = 2):
        """get the effective g"""
        V_a, V_b = Vs
        mu_p = (np.sum(mus) - V_a + V_b) / 2
        k0 = (2*self.m/self.hbar**2*mu_p/alpha_p)**0.5
        kc = (2*self.m/self.hbar**2 * (self.E_c + mu_p)/alpha_p)**0.5
        C = alpha_p * (np.sum(ns)**(1.0/3))/self.gamma
        Lambda = self.get_Lambda(k0=k0,kc=kc,dim=dim)
        g = alpha_p/(C - Lambda)
        return g

    def get_Ks_Vs(self, delta, mus=(0,0), ns=None, taus=None, nu=0, twist=(0, 0)): 
        """Return the kinetic energy and modifled potential matrics."""
        alphas = self.get_alphas_p(ns)
        if alphas == None or ns == None or taus == None:
            return (self.get_Ks(twist=twist), self.get_modified_Vs(delta=delta,ns=ns,taus=taus,nu=nu,alphas=alphas,twist=twist))

        alpha_a, alpha_b, alpha_p,p = alphas

        if type(alpha_a) != type(np.array):
            alpha_a = np.eye(self.Nxy[0]) * alpha_a # this will be problematic if Nx != Ny
            alpha_b = np.eye(self.Nxy[0]) * alpha_b
        K_a,K_b = self.get_modified_Ks(alpha_a=alpha_a,alpha_b=alpha_b,twist = twist)
        V_a, V_b = self.get_modified_Vs(delta=delta,ns=ns,taus=taus,nu=nu,alphas=alphas,twist=twist)
        self.g_eff = self.get_effective_g(ns = ns, Vs=(V_a,V_b), mus = mus,alpha_p = alpha_p)
        return ((K_a, K_b), (V_a, V_b))

            
    def get_v_ext(self):
        """Return the external potential."""
        #v_a = (-self.V0 * (1-((1+np.cos(2*np.pi * self.cells*self.x/self.L))/2)**self.power))
        #v_b = 0 * self.x
        #return v_a, v_b
        return (0, 0)

    def get_ns_taus_nu(self, H, Ec=0): # Ec not used yet
        """Return the n_a, n_b"""
        #print(H[:np.prod(self.Nxy), np.prod(self.Nxy):].real)
        Nx, Ny = self.Nxy
        # Es and psi contain redudant information, we may only need the first half with eith nagetive or positive energy
        Es, psi = np.linalg.eigh(H) 
        # after some debugging, fix the error when doing matrix manipulations.
        us, vs = psi.reshape(2, Nx*Ny , Nx*Ny*2)
        us,vs = us.T,vs.T
        #print(vs[0].real)
        # density
        n_a, n_b = np.sum(np.abs(us[i])**2 * self.f(Es[i])  for i in range(len(us))).reshape(self.Nxy)/self.dx**2, np.sum(np.abs(vs[i])**2 * self.f(-Es[i])  for i in range(len(vs))).reshape(self.Nxy)/self.dx**2
        #Tau terms
        nabla = self.get_nabla()
        tau_a = np.sum(np.abs(nabla.dot(us[i].ravel()))**2 * self.f(Es[i]) for i in range(len(us))).reshape(self.Nxy)/self.dx**2 # should divided by a factor dx^2?????
        tau_b = np.sum(np.abs(nabla.dot(vs[i].ravel()))**2 * self.f(-Es[i]) for i in range(len(vs))).reshape(self.Nxy)/self.dx**2
        nu = 0.5 * np.sum(us[i]*vs[i].conj() *(self.f(Es[i]) - self.f(-Es[i])) for i in range(len(us))).reshape(self.Nxy)/self.dx**2
        return ((n_a, n_b),(tau_a,tau_b),nu)  # divided by a factor, not sure if wrong or right, check later !!!

    def f(self, E, E_c=None):
        """Return the Fermi-Dirac distribution at E."""
        if E_c is None:
            E_c = self.E_c
        if self.T > 0:
            f = 1./(1+np.exp(E/self.T))
        else:
            f = (1 - np.sign(E))/2
        return np.where(abs(E)<E_c, f, 0)

    def get_H(self, mus, delta, ns=None,taus=None, nu=0, twist=(0,0)):
        """Return the single-particle Hamiltonian with pairing. """
        zero = np.zeros_like(sum((np.prod(self.Nxy),np.prod(self.Nxy))))
        Delta = np.diag((delta.reshape(self.Nxy) + zero).ravel()) # I do not understand why we just take the diagnal term of Delta?
        mu_a, mu_b = mus
        mu_a += zero
        mu_b += zero
        (K_a, K_b),(V_a,V_b) = self.get_Ks_Vs(delta = delta,nu=nu,taus=taus,ns=ns,twist=twist)
        Mu_a, Mu_b = np.diag((mu_a - V_a).ravel()), np.diag((mu_b - V_b).ravel())
        H = np.bmat([[K_a - Mu_a, Delta], # I remove the minus sign for Delta, need to check
                     [Delta.conj(), -(K_b - Mu_b)]]) # H is 512 * 512?
        assert np.allclose(H,H.conj().T)
        return np.asarray(H)

    def get_R(self, mus, delta, N_twist=1, twists=None):
        """Return the density matrix R."""
        N = self.Nxy
        Rs = []
        if twists is None:
            twists = itertools.product(
                (np.arange(0, N_twist)*2*np.pi/N_twist, )*2)

        for twist in twists:
            H = self.get_H(mus=mus, delta=delta, twist=twist)
            d, UV = np.linalg.eigh(H)
            R = UV.dot(self.f(d)[:, None]*UV.conj().T)
            # R_ = np.eye(2*N) - UV.dot(self.f(-d)[:, None]*UV.conj().T)
            # assert np.allclose(R, R_)
            Rs.append(R)
        R = sum(Rs)/len(Rs)
        return R

    def get_R_twist_average(self, mus, delta, abs_tol=1e-12):
        """Return the density matrix R."""
        R0 = 1.0
        def f(twist):
            H = self.get_H(mus=mus, delta=delta, twist=twist)
            d, UV = np.linalg.eigh(H)
            R = UV.dot(self.f(d)[:, None]*UV.conj().T)
            return R/R0
        R0 = f(0)
        R = R0 * mquad(f, -np.pi, np.pi, abs_tol=abs_tol)/2/np.pi
        return R

    def get_LDA(mu_eff, delta):
        """Return the LDA solution"""
