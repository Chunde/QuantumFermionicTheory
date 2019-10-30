import math
import numpy as np
import scipy as sp
import scipy.special


class HarmonicOscillator(object):
    """
    1D quantum harmonic Oscillator class
    to give exact wave function
    """
    w = m = hbar= 1

    def __init__(self, w=1, m=1, dim=1):
        """support 1d, will be generalized later"""
        assert dim == 1
        self.w = w
        self.m = m
    
    def get_wf(self, x, n=0):
        """return the wavefunction"""
        C1 = 1/np.sqrt(2**n*math.factorial(n))*(self.m*self.w/np.pi/self.hbar)**0.25
        C2 = np.exp(-1*self.m*self.w*x**2/2/self.hbar)
        Hn = sp.special.eval_hermite(n, np.sqrt(self.m*self.w/self.hbar)*x)
        return C1*C2*Hn
    
    def get_E(self, n):
        """return eigen value"""
        return self.hbar*self.w*(n+0.5)
