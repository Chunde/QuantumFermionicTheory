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


class HarmonicOscillator2D(object):
    """
    1D quantum harmonic Oscillator class
    to give exact wave function
    """
    w = m = hbar= 1

    def __init__(self, w=1, m=1):
        """support 1d, will be generalized later"""
        self.w = w
        self.m = m
        self.h = HarmonicOscillator(w=w, m=m)
    
    def get_wf(self, r, n=0, m=0):
        """return the wavefunction"""
        x = self.h.get_wf(x=r, n=n)
        y = self.h.get_wf(x=r, n=m)
        xx, yy = np.meshgrid(x, y, sparse=True)
        return xx*yy
    
    def get_E(self, n, m):
        """return eigen value"""
        return self.hbar*self.w*(n + m + 1)


def get_2d_ho_wf_p(n, m, rs):
    """
    return 2d radial wave function for a
    harmonic oscillator.
    ------------------------------------
    n = E -1
        e.g if E=1, to select the corresponding
        wavefunction, use n=E-1=0, and m = 0
    m is used to pick the degerated wavefunciton
    m <=n
    """
    assert n < 4 and n >=0
    assert m <=n
    P, pi = 1, np.pi
    C= (pi)**0.5
    if n ==1:  # E=2
        P = rs
    elif n == 2:  # E=3
        P=rs**2
        C=(2*pi)**0.5
        if m == 1:
            P=P-1
            C = pi**0.5
    elif n == 3:  #  E=4
        P = rs**3
        C= (6*pi)**0.5
        if m == 1 or m==2:
            P=P - rs/2
            C= (17*pi/4)**0.5
    return P*np.exp(-rs**2/2)/C


def get_2d_ho_wf(n, m, rs, p=False):
    """
    return 2d radial wave function for a 
    harmonic oscillator.
    ------------------------------------
    n = E -1
        e.g if E=1, to select the corresponding
        wavefunction, use n=E-1=0, and m = 0
    m is used to pick the degerated wavefunciton
    m <=n
    """
    if p:
        return get_2d_ho_wf(n=n, m=m, rs=rs)
    assert n < 4 and n >=0
    assert m <=n
    P, pi = 1, np.pi
    C= (pi**0.5/2)**0.5
    if n ==1:  # E=2
        P = rs
        C=(pi**0.5/4)**0.5
    elif n == 2:  # E=3
        P=rs**2
        if m == 1:
            P=P-1
        C = (3*pi**0.5/8)**0.5
    elif n == 3:  #  E=4
        P = rs**3
        C= (15*pi**0.5/16)**0.5
        if m == 1 or m==2:
            P= P - rs/2
            C= (5*pi**0.5/8)**0.5
    return P*np.exp(-rs**2/2)/C
