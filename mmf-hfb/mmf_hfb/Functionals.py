import numpy as np

class Functionals(object):
    def __init__(self):
        self.gamma = -11.11

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
        p2, p4 = p**2,p**4
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