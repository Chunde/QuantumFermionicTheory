from mmf_hfb.xp import xp
from enum import Enum


class FunctionalType(Enum):
    BDG = 1
    SLDA = 2
    ASLDA = 3


class FunctionalBdG(object):

    def __init__(self):
        self.FunctionalType = FunctionalType.BDG
        self.gamma = self._gamma()


    def _gamma(self, p=None):
        return -11.11

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
        """return alpha_a, alpha_b, alpha_p"""
        p = self._get_p(ns)
        alpha_a, alpha_b, alpha_even, alpha_odd = self._get_alphas_p(p)
        # alpha_p is defined as (alpha_a + alpha_b) /2 , it's alpha_even
        alpha_p = alpha_even
        return (alpha_a, alpha_b, alpha_p)

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

    def _Beta(self, p):
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
        return self._alpha_p(p)*(sum(ns))**(1.0/3)/self.gamma

    def _dC_dn(self, ns):
        """"[overridden in Children]"""
        return 0

    def _D(self, ns):
        C1_ = (6*xp.pi**2*(sum(ns)))**(5.0/3)/20/xp.pi**2
        p = self._get_p(ns)
        C2_ = self._Beta(p)
        return C1_*C2_*2**(-2.0/3)


    def _dD_dn(self, ns):
        """Return the derivative `dD(n_a,n_b)/d n_a and d n_b` """
        n_p = sum(ns)
        p = self._get_p(ns)
        dp_n_a, dp_n_b = self._dp_dn(ns)
        dBeta_p = self._dBeta_dp(p=p)
        N0 = (6*xp.pi**2)**(5.0/3)/20/xp.pi**2
        C1_ = self._D(ns)/0.6/n_p
        C2_ = N0*n_p**(5/3)*dBeta_p
        dD_n_a = C1_ + C2_*dp_n_a*2**(-2.0/3)
        dD_n_b = C1_ + C2_*dp_n_b*2**(-2.0/3)
        return (dD_n_a, dD_n_b)

    def _g_eff(self, delta, kappa, **args):
        """
            get the effective g
            equation (87c) in page 42
        """
        g_eff = -delta/kappa
        return g_eff

class FunctionalSLDA(FunctionalBdG):
    
    def __init__(self):
        FunctionalBdG.__init__(self)
        self.FunctionalType = FunctionalType.SLDA

    def _G(self, p):
        "[Numerical Test Status:Pass]"
        return 0.357 + 0.642*p**2

    def _dG_dp(self, p):
        "[Numerical Test Status:Pass]"
        return 1.284*p

    def _Beta(self, p):
        "[Numerical Test Status:Pass]"
        return (self._G(p) - self._alpha(p)*((1+p)/2.0)**(5.0/3) - self._alpha(-p)*((1-p)/2.0)**(5.0/3)) * 2**(2/3.0)

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
        dB_dp = 1.284*p + (-0.62345 + 0.7127*p2 - 0.9987*p4 + 0.42823*p6)*pm + (0.20411*p - 0.51741*p3 + 0.26962*p5)*pp 
        return dB_dp * 2**(2/3.0)

    def _dC_dn(self, ns):
        """return dC / dn"""
        "[Numerical Test Status:Pass]"
        n = sum(ns)
        p = self._get_p(ns)
        dp_n_a, dp_n_b = self._dp_dn(ns)
        dC_dn_a = self._alpha_p(p)*n**(-2/3)/3 + n**(1/3)*self._dalpha_p_dp(p)*dp_n_a
        dC_dn_b = self._alpha_p(p)*n**(-2/3)/3 + n**(1/3)*self._dalpha_p_dp(p)*dp_n_b
        gamma = self.gamma
        return (dC_dn_a/gamma, dC_dn_b/gamma)

    def _get_alphas_p(self, p):
        """"[overridden in Children]"""
        alpha_even = 1.094
        alpha_odd = 0
        alpha_a, alpha_b = alpha_odd + alpha_even, -alpha_odd + alpha_even
        return (alpha_a, alpha_b, alpha_even, alpha_odd)


class FunctionalASLDA(FunctionalSLDA):

    def __init__(self):
        FunctionalSLDA.__init__(self)
        self.FunctionalType=FunctionalType.ASLDA
    
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

    def _g_eff(self, mus_eff, ns, Vs, alpha_p, dim, **args):
        """
            get the effective g
            equation (87c) in page 42
        """
        V_a, V_b = Vs
        mu_p = (sum(mus_eff) - V_a + V_b) / 2
        k0 = (2*self.m/self.hbar**2*mu_p/alpha_p)**0.5
        k_c = (2*self.m/self.hbar**2 * (self.E_c + mu_p)/alpha_p)**0.5
        C = alpha_p * (sum(ns)**(1.0/3))/self.gamma
        Lambda = self._get_Lambda(k0=k0, k_c=k_c, dim=dim)
        g = alpha_p/(C - Lambda)
        return g
