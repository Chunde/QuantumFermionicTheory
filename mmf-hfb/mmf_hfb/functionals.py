import numpy as np


class FunctionalBdG(object):
    """D0 is the factor used to vary the D term weight"""
    m=hbar=D0=1
    
    def _gamma(self, p=None):
        """
        the gamma term only shows up in C
        if C is set to constant, it may have
        no effect on the result.
        """
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
        NOTE:
        """
        p = self._get_p(ns)
        alpha_a, alpha_b, alpha_even, alpha_odd = self._get_alphas_p(p)
        # alpha_p is defined as (alpha_a + alpha_b) /2 , it's alpha_even
        return (alpha_a, alpha_b)

    def _alpha_a(self, ns):
        return self.get_alpha(self._get_p(ns))
     
    def _alpha_b(self, ns):
        return self.get_alpha(-self._get_p(ns))

    def _alpha_p(self, p):
        """(alpa_a + alpha_b)/ 2"""
        return 0.5*(self.get_alpha(p) + self.get_alpha(-p))

    def _alpha_m(self, p):
        return 0.5*(self.get_alpha(p) - self.get_alpha(-p))

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
        """
        return C tilde
        NOTE: This only works for Unitray Regime
        """
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

    def _get_Lambda(self, k0, k_c, alpha=1, dim=1):
        """
        return the renormalization condition parameter Lambda
        # [check] be careful the alpha may be different for a and b
        """
        assert np.all(k_c > k0)
        if dim ==3:
            Lambda = self.m*k_c/self.hbar**2/2/np.pi**2*(
                1.0 - k0/k_c/2.0*np.log((k_c + k0)/(k_c - k0)))
        elif dim == 2:
            Lambda = self.m/self.hbar**2/4/np.pi*np.log((k_c/k0)**2 - 1)
        elif dim == 1:
            Lambda = self.m/self.hbar**2/2/np.pi*np.log((k_c - k0)/(k_c + k0))/k0
        return Lambda/alpha  # do not forget effective mess inverse factor

    def get_effective_g(self, mus_eff, ns, E_c, k_c=None, dim=3, **args):
        """1
            get the effective g
            equation (87c) in page 42
            -----------
            Note: mus_eff=(mu_eff_a, mu_eff_b)
            Should make sure we have consistent convention
        """
        alpha_p = sum(self.get_alphas(ns))/2.0
        Lambda = self.get_Lambda(
            mus_eff=mus_eff, alpha_p=alpha_p, E_c=E_c, k_c=k_c, dim=dim)
        C = self.get_C(ns=ns)  # (97)
        g = alpha_p/(C - Lambda)  # (84)
        return g
    
    def get_Lambda(self, mus_eff, alpha_p, E_c, k_c=None, dim=3):
        mu_p = abs(sum(mus_eff))/2  # [check] mus_eff may be negative???
        k0 = (2*self.m/self.hbar**2*mu_p/alpha_p)**0.5
        if k_c is None:
            k_c = (2*self.m/self.hbar**2*(E_c + mu_p)/alpha_p)**0.5
        return self._get_Lambda(k0=k0, k_c=k_c, dim=dim, alpha=alpha_p)

    def get_alpha(self, p, d=0):
        """IFunctional interface implementation
        parameters:
        ------------------
        if d==1, return partial alpha_p over partial p
        if d==1, return partial alpha_m over partial p
        """
        if d==0:
            return self._alpha(p)
        elif d==1:
            return self._dalpha_p_dp(p=p)
        elif d==-1:
            return self._dalpha_m_dp(p=p)
        else:
            raise ValueError(f"d={d} is not supported value")

    def get_alphas(self, ns, d=0):
        """IFunctional interface implementation
        parameters:
        ------------------
        if d==1, return the first order derivatives
        of alpha_p and alpha_m over n_a, and n_b,
        NOT alpha_a and alpha_b over n_a, and n_b
        """
        p = self.get_p(ns)
        if d==0:
            return (self.get_alpha(p), self.get_alpha(-p))
        elif d==1:
            dp_dn_a, dp_dn_b = self._dp_dn(ns=ns)
            dalpha_dp = self.get_alpha(p=p, d=1)
            dalpha_dm = self.get_alpha(p=p, d=-1)
            dalpha_p_dn_a, dalpha_p_dn_b = dalpha_dp*dp_dn_a, dalpha_dp*dp_dn_b
            dalpha_m_dn_a, dalpha_m_dn_b = dalpha_dm*dp_dn_a, dalpha_dm*dp_dn_b
            return (dalpha_p_dn_a, dalpha_p_dn_b, dalpha_m_dn_a, dalpha_m_dn_b)
        else:
            raise ValueError(f"d={d} is not supported value")

    def get_p(self, ns, d=0):
        """IFunctional interface implementation"""
        if d==0:
            return self._get_p(ns=ns)
        elif d==1:
            return self._dp_dn(ns=ns)
        else:
            raise ValueError(f"d={d} is not supported value")

    def get_C(self, ns, d=0):
        """
        IFunctional interface implementation
        -----------
        Note: This is only valid for unitary case
        Test: Not done yet.
        """
        if d==0:
            return self._C(ns=ns)
        elif d==1:
            return self._dC_dn(ns)
        else:
            raise ValueError(f"d={d} is not supported value")

    def get_D(self, ns, d=0):
        """IFunctional interface implementation"""
        if d==0:
            return self.D0*self._D(ns)
        elif d==1:
            return self.D0*np.array(self._dD_dn(ns=ns))
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

    def get_Vs(self, **args):
        return self.get_Vext(**args)


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
        # assert self._alpha(p) == self.get_alpha(p)
        # assert self._alpha(-p) == self.get_alpha(-p)
        # tmp = ((self._G(p) - self.get_alpha(p=p)*((1+p)/2.0)**(5.0/3)
        #         - self.get_alpha(p=-p)*((1-p)/2.0)**(5.0/3))*2**(2/3.0))

        return ((self._G(p) - self.get_alpha(p)*((1+p)/2.0)**(5.0/3)
                - self.get_alpha(-p)*((1-p)/2.0)**(5.0/3))*2**(2/3.0))

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
        C1_ = self._alpha_p(p=p)*n**(-2/3)/3.0
        C2_ = n**(1/3)*self.get_alpha(p=p, d=1)
        dC_dn_a = C1_ + C2_*dp_n_a
        dC_dn_b = C1_ + C2_*dp_n_b
        return (dC_dn_a/self._gamma(), dC_dn_b/self._gamma())

    def _get_alphas_p(self, p):
        """"[overridden in Children]"""
        ones = np.ones_like(p)
        alpha_even = 1.0*ones  # 1.094*ones  # 1.14 in Aureal's Matlab code
        alpha_odd = 0
        alpha_a, alpha_b = alpha_odd + alpha_even, -alpha_odd + alpha_even
        return (alpha_a, alpha_b, alpha_even, alpha_odd)

    def get_Vs(self, delta=0, ns=None, taus=None, nu=None, **args):
        """
        return the modified V functional terms
        """
        if ns is None or taus is None:
            return self.get_Vext()
        U_a, U_b = self.get_Vext()  # external trap
        tau_a, tau_b = taus
        tau_p, tau_m = tau_a + tau_b, tau_a - tau_b

        alpha_p = sum(self.get_alphas(ns=ns))/2.0
        dalpha_p_dn_a, dalpha_p_dn_b, dalpha_m_dn_a, dalpha_m_dn_b=self.get_alphas(
            ns=ns, d=1)
        dC_dn_a, dC_dn_b = self.get_C(ns=ns, d=1)
        dD_dn_a, dD_dn_b = self.get_D(ns=ns, d=1)
       
        C0_ = self.hbar**2/self.m
        C1_ = C0_/2.0
        C2_ = tau_p*C1_ + np.conj(delta).T*nu/alpha_p
        C3_ = abs(delta)**2/alpha_p
        V_a = dalpha_m_dn_a*tau_m*C1_ + dalpha_p_dn_a*C2_ + dC_dn_a*C3_ + C0_*dD_dn_a + U_a
        V_b = dalpha_m_dn_b*tau_m*C1_ + dalpha_p_dn_b*C2_ + dC_dn_b*C3_ + C0_*dD_dn_b + U_b
        return np.array([V_a, V_b])


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
