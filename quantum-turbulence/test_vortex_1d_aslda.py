# this file will eventually be changed to a test file for pytest, now just for debugging in Visual Studio
# Chunde's comment will start with a single sharp '#', to track modifications.
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import quad
from uncertainties import ufloat
from importlib import reload  # Python 3.4+
import numpy as np
import homogeneous;reload(homogeneous)
import bcs;reload(bcs)
from bcs import BCS
import vortex_1d_aslda;reload(vortex_1d_aslda)
import itertools  
hbar = 1
m = 1


class Lattice(BCS):
    """Adds optical lattice potential to species a with depth V0."""
    cells = 1.0
    t = 0.0007018621290128983
    E0 = -0.312433127299677
    power = 4
    V0 = -10.5
    
    def __init__(self, cells=1, N=2**5, L=10.0,  mu_a=1.0, mu_b=1.0, v0=0.1, V0=-10.5, power=2,**kw):
        self.power = power
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.v0 = v0
        self.V0 = V0
        self.cells = cells
        BCS.__init__(self, L=cells*L, N=cells*N, **kw)
    
    def get_v_ext(self):
        v_a =  (-self.V0 * (1-((1+np.cos(2*np.pi * self.cells*self.x/self.L))/2)**self.power))
        v_b = 0 * self.x
        return v_a, v_b
    
# a modified class from ASLDA with different alphas which are constant, so their derivatives are zero
class ASLDA_(vortex_1d_aslda.ASLDA):
    def get_alphas(self, ns = None):
        return super().get_alphas(ns)
        #alpha_a,alpha_b,alpha_p =np.ones(self.Nx),np.ones(self.Nx),np.ones(self.Nx)
        #return (alpha_a,alpha_b,alpha_p)       

def iterate(lda, mudelta, na_avg=0.5, nb_avg=0.5, N_twist=0, **kw):
    mu_a, mu_b, na,nb, mu_a_eff, mu_b_eff, delta,taus,kappa = mudelta
    mus = (mu_a_eff, mu_b_eff)
    if na is None:
        na = np.ones(lda.Nx) * na_avg
    if nb is None:
        nb = np.ones(lda.Nx) * nb_avg
    #H = lda.get_H(mus=mus,delta = delta,ns=(na,nb),taus = taus,kappa=kappa)
    ns_,taus_,kappa_ = lda.get_ns_taus_kappa_average_3d(mus=mus,delta = delta,ns=(na,nb),taus = taus,kappa=kappa) #lda.get_ns_taus_kappa(H) 
    plt.plot(lda.x,ns_[0].real)
    plt.show()
    gx = lda.gx(ns_,taus_,kappa_)
    na_,nb_ = ns_ # the new densities are not used in the iteration, just used for compute new mus
    nomral_na = na_.mean() / (na_.mean() + nb_.mean())
    nomral_nb = nb_.mean() / (na_.mean() + nb_.mean())
    mu_a = mu_a*(1 + (na_avg - nomral_na))
    mu_b = mu_b*(1 + (nb_avg - nomral_nb))
    lr = 0.4
    lo = 1 - lr
    v_a,v_b = lda.get_modified_Vs(delta=delta,ns=(na_,nb_),taus=taus_,kappa=kappa_,alphas=lda.get_alphas((na_,nb_)))
    mu_a_eff = mu_a + v_a * nb
    mu_b_eff = mu_b + v_b * na
    #delta = lda.g_eff*kappa 
    print(mu_a_eff.real.max(),mu_b_eff.real.max(),delta.real.max(), na_.real.mean(), nb_.real.mean(),gx.real.mean())
    return (mu_a, mu_b,lo*na + lr*na_,lo*nb + lr*nb_, mu_a_eff, mu_b_eff, delta,taus_,kappa_)




def test_ASLDA_Homogenous():
    L = 0.46
    N = 128
    N_twist = 32
    delta = 1.0
    mu_eff = 1.0
    v_0, n, mu, e_0 = homogeneous.get_BCS_v_n_e(delta=delta, mu_eff=mu_eff)

    print("Test 1d lattice with homogeneous system")
    b = ASLDA_(T=0,Nx=N,Lx = L)
    R = b.get_R(mus=(mu_eff  * np.ones((N),), mu_eff  * np.ones((N),)), delta=delta * np.ones((N),), N_twist=N_twist)
    na = np.diag(R)[:N]/b.dx
    nb = (1 - np.diag(R)[N:])/b.dx
    kappa = np.diag(R[:N, N:])/b.dx
    print((n, na[0].real + nb[0].real), (delta, v_0*kappa[0].real))
    assert np.allclose(n, na[0].real + nb[0].real,atol=0.001)
    assert np.allclose(delta, v_0*kappa[0].real,atol=0.001)
    print("Test 1d lattice with homogeneous system")
    ns,taus,kappa = b.get_ns_taus_kappa_average_1d(mus=(mu_eff  * np.ones((N),), mu_eff  * np.ones((N),)), delta=delta * np.ones((N),), N_twist=N_twist)
    na,nb = ns
    print((n, na[0].real + nb[0].real), (delta, v_0*kappa[0].real))
    assert np.allclose(n, na[0].real + nb[0].real,atol=0.001)
    assert np.allclose(delta, v_0*kappa[0].real,atol=0.001)
    print("Test 1d lattice plus 1d integral over y with homogeneous system")
    k_c = abs(b.kx).max()
    E_c = (b.hbar*k_c)**2/2/b.m # 3 dimension, so E_c should have a factor of 3
    b.E_c = E_c
    ns,taus,kappa = b.get_ns_taus_kappa_average_2d(mus=(mu_eff  * np.ones((N),), mu_eff  * np.ones((N),)), delta=delta * np.ones((N),), N_twist=N_twist)
    na,nb = ns
    print((n, na[0].real + nb[0].real), (delta, v_0*kappa[0].real))
    assert np.allclose(n, na[0].real + nb[0].real,atol=0.001)
    assert np.allclose(delta, v_0*kappa[0].real,atol=0.001)
    print("Test 1d lattice plus 2d integrals over y and  z with homogeneous system")
    ns,taus,kappa = b.get_ns_taus_kappa_average_3d(mus=(mu_eff  * np.ones((N),), mu_eff  * np.ones((N),)), delta=delta * np.ones((N),), N_twist=N_twist)
    na,nb = ns
    print((n, na[0].real + nb[0].real), (delta, v_0*kappa[0].real))
    assert np.allclose(n, na[0].real + nb[0].real,atol=0.001)
    assert np.allclose(delta, v_0*kappa[0].real,atol=0.001)

def test_ASLDA_unitary():
    L = 0.46
    N = 128
    N_twist = 32
    b = ASLDA_(T=0,Nx=N,Lx = L)
    k_c = abs(b.kx).max()
    E_c = (b.hbar*k_c)**2/2/b.m # 3 dimension, so E_c should have a factor of 3
    mu_eff = 1.0
    n = 1.0
    k_F = np.sqrt(2*m*E_c)
    n_F = k_F**3/3/np.pi**2
    E_FG = 2./3*n_F*E_c

    delta = 1#0.68640205206984016444108204356564421137062514068346 * E_c
    mu_eff = 0.59060550703283853378393810185221521748413488992993*E_c
    v_0, n, mu, e_0 = homogeneous.get_BCS_v_n_e(delta=delta, mu_eff=mu_eff)

    l = Lattice(T=0.0, N=N, L=L, v0=v_0, V0=0)
    R = l.get_R(mus=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
    na = np.diag(R)[:N]/b.dx
    nb = (1 - np.diag(R)[N:])/b.dx
    kappa = np.diag(R[:N, N:])/b.dx
    print((n, na[0].real + nb[0].real), (delta, v_0*kappa[0].real))
    #assert np.allclose(n, na[0].real + nb[0].real,atol=0.1)
    #assert np.allclose(delta, v_0*kappa[0].real,atol=0.1)

    print("Test 1d lattice with homogeneous system")
    R = b.get_R(mus=(mu_eff  * np.ones((N),), mu_eff  * np.ones((N),)), delta=delta * np.ones((N),), N_twist=N_twist)
    na = np.diag(R)[:N]/b.dx
    nb = (1 - np.diag(R)[N:])/b.dx
    kappa = np.diag(R[:N, N:])/b.dx
    print((n, na[0].real + nb[0].real), (delta, v_0*kappa[0].real))
    #assert np.allclose(n, na[0].real + nb[0].real,atol=0.1)
    #assert np.allclose(delta, v_0*kappa[0].real,atol=0.1)
    print("Test 1d lattice with homogeneous system")
    ns,taus,kappa = b.get_ns_taus_kappa_average_1d(mus=(mu_eff  * np.ones((N),), mu_eff  * np.ones((N),)), delta=delta * np.ones((N),), N_twist=N_twist)
    na,nb = ns
    print((n, na[0].real + nb[0].real), (delta, v_0*kappa[0].real))
    #assert np.allclose(n, na[0].real + nb[0].real,atol=0.001)
    #assert np.allclose(delta, v_0*kappa[0].real,atol=0.001)
    print("Test 1d lattice plus 1d integral over y with homogeneous system")
    k_c = abs(b.kx).max()
    E_c = (b.hbar*k_c)**2/2/b.m # 3 dimension, so E_c should have a factor of 3
    b.E_c = E_c
    ns,taus,kappa = b.get_ns_taus_kappa_average_2d(mus=(mu_eff  * np.ones((N),), mu_eff  * np.ones((N),)), delta=delta * np.ones((N),), N_twist=N_twist)
    na,nb = ns
    print((n, na[0].real + nb[0].real), (delta, v_0*kappa[0].real))
    #assert np.allclose(n, na[0].real + nb[0].real,atol=0.001)
    #assert np.allclose(delta, v_0*kappa[0].real,atol=0.001)
    print("Test 1d lattice plus 2d integrals over y and  z with homogeneous system")
    ns,taus,kappa = b.get_ns_taus_kappa_average_3d(mus=(mu_eff  * np.ones((N),), mu_eff  * np.ones((N),)), delta=delta * np.ones((N),), N_twist=N_twist)
    na,nb = ns
    print((n, na[0].real + nb[0].real), (delta, v_0*kappa[0].real))
    #assert np.allclose(n, na[0].real + nb[0].real,atol=0.001)
    #assert np.allclose(delta, v_0*kappa[0].real,atol=0.001)
def test_iterate_ASLDA():
    grid_size = 64
    aslda = ASLDA_(Nx=grid_size)
    k_c = abs(aslda.kx).max()
    E_c = (aslda.hbar*k_c)**2/2/aslda.m # 3 dimension, so E_c should have a factor of 3
    aslda = ASLDA_(Nx=grid_size, Lx=0.4, E_c=E_c)
    mu_eff = 1.0
    n = 1.0
    k_F = np.sqrt(2*m*E_c)
    n_F = k_F**3/3/np.pi**2
    E_FG = 2./3*n_F*E_c
    mu = 1# 0.59060550703283853378393810185221521748413488992993*E_c
    delta = 1# 0.68640205206984016444108204356564421137062514068346*E_c
    qT = (mu, mu) +(None,None)+ (mu_eff*np.ones(aslda.Nx),)*2 + (delta * np.ones((aslda.Nx),), None,None)
    max_iteration = 5
    v_0, n, mu, e_0 = homogeneous.get_BCS_v_n_e(delta=delta, mu_eff=mu_eff)
    x = 0.5
    while max_iteration > 0:
        qT = iterate(lda=aslda,mudelta = qT, N_twist=np.inf,na_avg=1/(1+x), nb_avg=x/(1+x), abs_tol=1e-2)

if __name__ == '__main__':
    test_iterate_ASLDA()
