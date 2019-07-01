import numpy as np
from abc import ABC, abstractmethod

hbar=m=1


class IFunctional(ABC):
    """Interface for functional"""

    @abstractmethod
    def get_alphas(self, ns, d=0):
        """
        Parameters
        ----------------
        ns: densities (na, nb)
        d: order of derivative
        """
        pass

    @abstractmethod
    def get_p(self, ns, d=0):
        """
        Parameters
        ----------------
        ns: densities (na, nb)
        d: order of derivative
        """
        pass

    @abstractmethod
    def get_C(self, ns, d=0):
        """
        Parameters
        ----------------
        ns: densities (na, nb)
        d: order of derivative
        """
        pass

    @abstractmethod
    def get_D(self, ns, d=0):
        """
        Parameters
        ----------------
        ns: densities (na, nb)
        d: order of derivative
        """
        pass

    @abstractmethod    
    def get_beta(self, ns, d=0):
        """
        Parameters
        ----------------
        ns: densities (na, nb)
        d: order of derivative
        """
        pass


class FunctionalBdG(IFunctional):

    def _gamma(self, p=None):
        return -11.11  # -11.039 in Aureal's code

    def _get_p(self, ns):
        """return p for a given ns"""
        if ns is None:
            return 0
        n_a, n_b = ns
        n_p, n_m = n_a + n_b, n_a - n_b
        p = n_m /n_p
        return p

    def _dp_dn(self, ns):
        na, nb = ns
        n = na + nb
        n2 = n*n
        dp_n_a, dp_n_b= 2*nb/n2, -2*na/n2
        return (dp_n_a, dp_n_b)

    def _alpha(self, p=0):
        """return alpha"""
        alpha_a, alpha_b, alpha_odd, alpha_even = self._get_alphas_p(p)
        alpha = alpha_odd + alpha_even
        return alpha

    def _get_alphas(self, ns=None):
        """
        return alpha_a, alpha_b
        --------------
        Note: this method should be 
        """
        p = self._get_p(ns)
        alpha_a, alpha_b, alpha_even, alpha_odd = self._get_alphas_p(p)
        # alpha_p is defined as (alpha_a + alpha_b) /2 , it's alpha_even
        return (alpha_a, alpha_b)

    def _alpha_a(self, ns):
        return self._alpha(self._get_p(ns))
     
    def _alpha_b(self, ns):
        return self._alpha(-self._get_p(ns))

    def _alpha_p(self, p):
        return 0.5*(self._alpha(p) + self._alpha(-p))

    def _alpha_m(self, p):
        return 0.5*(self._alpha(p) - self._alpha(-p))

    def _get_alphas_p(self, p):
        """"[overridden in Children]"""
        alpha_even = 1
        alpha_odd = 0
        alpha_a, alpha_b = alpha_odd + alpha_even, -alpha_odd + alpha_even
        return (alpha_a, alpha_b, alpha_even, alpha_odd)

    def _Beta(self, ns):
        """"[overridden in Children]"""
        return 0
     
    def _dBeta_dp(self, p):
        """"[overridden in Children]"""
        return 0

    def _dalpha_dp(self, p):
        """"[overridden in Children]"""
        return 0

    def _dalpha_p_dp(self, p):
        """"[overridden in Children]"""
        return 0

    def _dalpha_m_dp(self, p):
        """"[overridden in Children]"""
        return 0

    def _C(self, ns):
        """return C tilde"""
        p = self._get_p(ns)
        return self._alpha_p(p)*(sum(ns))**(1.0/3)/self._gamma()

    def _dC_dn(self, ns):
        """"[overridden in Children]"""
        return (0, 0)

    def _D(self, ns):
        C1_ = (6*np.pi**2*(sum(ns)))**(5.0/3)/20/np.pi**2
        C2_ = self._Beta(ns=ns)
        return C1_*C2_*2**(-2.0/3)

    def _dD_dn(self, ns):
        """Return the derivative `dD(n_a,n_b)/d n_a and d n_b` """
        n_p = sum(ns)
        p = self._get_p(ns)
        dp_n_a, dp_n_b = self._dp_dn(ns)
        dBeta_p = self._dBeta_dp(p=p)
        C0_ = (6*np.pi**2)**(5.0/3)/20/np.pi**2
        C1_ = self._D(ns)/0.6/n_p
        C2_ = C0_*n_p**(5/3)*dBeta_p
        dD_n_a = C1_ + C2_*dp_n_a*2**(-2.0/3)
        dD_n_b = C1_ + C2_*dp_n_b*2**(-2.0/3)
        return (dD_n_a, dD_n_b)

    def _get_Lambda(self, k0, k_c, alpha, dim=1):
        """
        return the renormalization condition parameter Lambda
        # [check] be careful the alpha may be different for a and b
        """
        if dim ==3:
            Lambda = m*k_c/hbar**2/2/np.pi**2*(1.0 - k0/k_c/2.0*np.log((k_c + k0)/(k_c - k0)))
        elif dim == 2:
            Lambda = m/hbar**2/4/np.pi*np.log((k_c/k0)**2 - 1)
        elif dim == 1:
            Lambda = m/hbar**2/2/np.pi*np.log((k_c - k0)/(k_c + k0))/k0
        return Lambda/alpha  # do not forget effective mess inverse factor

    def _g_eff(self, mus_eff, ns, E_c, k_c=None, dim=3, **args):
        """1
            get the effective g
            equation (87c) in page 42
            -----------
            Note: mus_eff=(mu_eff_a, mu_eff_b)
            Should make sure we have consistent convention
        """
        alpha_p = sum(self.get_alphas(ns))/2.0
        mu_p = abs(sum(mus_eff))/2  # [check] mus_eff may be negative???
        k0 = (2*m/hbar**2*mu_p/alpha_p)**0.5
        if k_c is None:
            k_c = (2*m/hbar**2*(E_c + mu_p)/alpha_p)**0.5
        C = self.get_C(ns=ns)  # (97)
        # [check] be careful the alpha may be different for a and b
        Lambda = self._get_Lambda(k0=k0, k_c=k_c, dim=dim, alpha=alpha_p)
        g = alpha_p/(C - Lambda)  # (84)
        return g

    def get_alphas(self, ns, d=0):
        """IFunctional interface implementation
        parameters:
        ------------------
        if d==1, return the first order derivatives
        of alpha_p and alpha_m over n_a, and n_b,
        NOT alpha_a and alpha_b over n_a, and n_b
        """
        if d==0:
            alpha_a, alpha_b = self._get_alphas(ns=ns)
            return (alpha_a, alpha_b)
        elif d==1:
            p = self._get_p(ns=ns)
            dp_dn_a, dp_dn_b = self._dp_dn(ns=ns)
            dalpha_dp = self._dalpha_p_dp(p=p)
            dalpha_dm = self._dalpha_m_dp(p=p)
            dalpha_p_dn_a, dalpha_p_dn_b = dalpha_dp*dp_dn_a, dalpha_dp*dp_dn_b
            dalpha_m_dn_a, dalpha_m_dn_b = dalpha_dm*dp_dn_a, dalpha_dm*dp_dn_b
            return (dalpha_p_dn_a, dalpha_p_dn_b, dalpha_m_dn_a, dalpha_m_dn_b)
        else:
            raise ValueError(f"d={d} is not supported value")

    def get_p(self, ns, d=0):
        """IFunctional interface implementation"""
        if d==0:
            return self._dp_dn(ns=ns)
        elif d==1:
            return self._dp_dn(ns=ns)
        else:
            raise ValueError(f"d={d} is not supported value")

    def get_C(self, ns, d=0):
        """IFunctional interface implementation"""
        if d==0:
            return self._C(ns=ns)
        elif d==1:
            return self._dC_dn(ns)
        else:
            raise ValueError(f"d={d} is not supported value")

    def get_D(self, ns, d=0):
        """IFunctional interface implementation"""
        if d==0:
            return self._D(ns)
        elif d==1:
            return self._dD_dn(ns=ns)
        else:
            raise ValueError(f"d={d} is not supported value")
    
    def get_beta(self, ns, d=0):
        if d==0:
            return self._Beta(ns=ns)
        elif d==1:
            p = self._get_p(ns=ns)
            dp_n_a, dp_n_b = self._dp_dn(ns=ns)
            dBeta_dp = self._dBeta_dp(p=p)
            dBeta_dn_a, dBeta_p_dn_b = dBeta_dp*dp_n_a, dBeta_dp*dp_n_b
            return (dBeta_dn_a, dBeta_p_dn_b)
        else:
            raise ValueError(f"d={d} is not supported value")


class FunctionalSLDA(FunctionalBdG):
    
    def _G(self, p):
        "[Numerical Test Status:Pass]"
        return 0.357 + 0.642*p**2

    def _dG_dp(self, p):
        "[Numerical Test Status:Pass]"
        return 1.284*p

    def _Beta(self, ns):
        p = self._get_p(ns)
        "[Numerical Test Status:Pass]"
        return self._Beta_p(p)

    def _Beta_p(self, p):
        "[Numerical Test Status:Pass]"
        return ((self._G(p) - self._alpha(p)*((1+p)/2.0)**(5.0/3)
                - self._alpha(-p)*((1-p)/2.0)**(5.0/3))*2**(2/3.0))

    def _dBeta_dp(self, p):
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
        dB_dp = (1.284*p + (-0.62345 + 0.7127*p2 - 0.9987*p4
                 + 0.42823*p6)*pm + (0.20411*p - 0.51741*p3 + 0.26962*p5)*pp)
        return dB_dp*2**(2/3.0)

    def _dC_dn(self, ns):
        """return dC / dn"""
        "[Numerical Test Status:Pass]"
        n = sum(ns)
        p = self._get_p(ns)
        dp_n_a, dp_n_b = self._dp_dn(ns)
        dC_dn_a = self._alpha_p(p)*n**(-2/3)/3 + n**(1/3)*self._dalpha_p_dp(p)*dp_n_a
        dC_dn_b = self._alpha_p(p)*n**(-2/3)/3 + n**(1/3)*self._dalpha_p_dp(p)*dp_n_b
        return (dC_dn_a/self._gamma(), dC_dn_b/self._gamma())

    def _get_alphas_p(self, p):
        """"[overridden in Children]"""
        ones = np.ones_like(p)
        alpha_even = 1.094*ones  # 1.14 in Aureal's Matlab code
        alpha_odd = 0
        alpha_a, alpha_b = alpha_odd + alpha_even, -alpha_odd + alpha_even
        return (alpha_a, alpha_b, alpha_even, alpha_odd)

    
class FunctionalASLDA(FunctionalSLDA):
   
    def _get_alphas_p(self, p):
        p2 = p**2
        p4 = p2**2
        alpha_even = 1.094 - 0.532*p2*(1 - p2 + p4/3.0)
        alpha_odd = 0.156*p*(1 - 2.0*p2/3.0 + p4/5.0)
        alpha_a, alpha_b = alpha_odd + alpha_even, -alpha_odd + alpha_even
        return (alpha_a, alpha_b, alpha_even, alpha_odd)

    def _dalpha_dp(self, p):
        """return dalpha / dp"""
        return -((266*p - 39)*(p**2 - 1)**2)/250.0

    def _dalpha_p_dp(self, p):
        """return dalpha_p / dp"""
        return -1.064*p*(p**2 - 1.0)**2

    def _dalpha_m_dp(self, p):
        """return dalpha_m / dp"""
        return 0.156*(p**2 - 1.0)**2 