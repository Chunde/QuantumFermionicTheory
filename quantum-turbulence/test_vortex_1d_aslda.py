# this file will eventually be changed to a test file for pytest, now just for debugging in Visual Studio
# Chunde's comment will start with a single sharp '#', to track modifications.
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from uncertainties import ufloat
from importlib import reload  # Python 3.4+
import numpy as np
import homogeneous;reload(homogeneous)
import vortex_1d_aslda;reload(vortex_1d_aslda)
import itertools  
hbar = 1
m = 1

# a modified class from ASLDA with different alphas which are constant, so their derivatives are zero
class ASLDA_(vortex_1d_aslda.ASLDA):
    def get_alphas(self, ns = None):
        return super().get_alphas(ns)
        #alpha_a,alpha_b,alpha_p =np.ones(self.Nx),np.ones(self.Nx),np.ones(self.Nx)
        #return (alpha_a,alpha_b,alpha_p)       

def iterate(lda, mudelta, na_avg=0.5, nb_avg=0.5, N_twist=0, **kw):
    mu_a, mu_b, na,nb, mu_a_eff, mu_b_eff, delta,taus = mudelta
    mus = (mu_a_eff, mu_b_eff)
    if na is None:
        na = np.ones(lda.Nx) * na_avg
    if nb is None:
        nb = np.ones(lda.Nx) * nb_avg
    H = lda.get_H(mus=mus,delta = delta,ns=(na,nb),taus = taus)
    ns,taus,kappa = lda.get_ns_taus_kappa_average(mus=mus,delta = delta,ns=(na,nb),taus = taus) #lda.get_ns_taus_kappa(H) 
    gx = lda.gx(ns,taus,kappa)
    na_,nb_ = ns # the new densities are not used in the iteration, just used for compute new mus
    nomral_na = na_.mean() / (na_.mean() + nb_.mean())
    nomral_nb = nb_.mean() / (na_.mean() + nb_.mean())
    mu_a = mu_a*(1 - (na_avg - nomral_na))
    mu_b = mu_b*(1 - (nb_avg - nomral_nb))
    ns = (na,nb)
    v_a,v_b = lda.get_modified_Vs(delta=delta,ns=ns,taus=taus,kappa=kappa,alphas=lda.get_alphas(ns))
    mu_a_eff = mu_a + v_a * nb
    mu_b_eff = mu_b + v_b * na
    #delta = lda.g_eff*kappa # I do not update the delta.
    print(mu_a_eff.real.max(),mu_b_eff.real.max(),delta.real.max(), na_.real.mean(), nb_.real.mean(),gx.real.mean())
    return (mu_a, mu_b,na,nb, mu_a_eff, mu_b_eff, delta,taus)

def get_density(lda, mudelta, na_avg=0.5, nb_avg=0.5, N_twist=0, **kw):
    mu_a, mu_b, na,nb, mu_a_eff, mu_b_eff, delta,taus = mudelta
    mus = (mu_a_eff, mu_b_eff)
    if na is None:
        na = np.ones(lda.Nx) * na_avg
    if nb is None:
        nb = np.ones(lda.Nx) * nb_avg
    H = lda.get_H(mus=mus,delta = delta,ns=(na,nb),taus = taus)
    ns,taus,kappa = lda.get_ns_taus_kappa_average(mus=mus,delta = delta,ns=(na,nb),taus = taus) #lda.get_ns_taus_kappa(H) 
    na,nb = ns
    print(mu_a_eff.mean(),mu_b_eff.mean(),na.mean(),nb.mean())
    return (na.mean()+nb.mean())/2

def test_iterate_ASLDA():
    grid_size = 4
    aslda = ASLDA_(Nx=grid_size)
    k_c = abs(aslda.kx).max()
    E_c = 3*(aslda.hbar*k_c)**2/2/aslda.m # 3 dimension, so E_c should have a factor of 3
    aslda = ASLDA_(Nx=grid_size, Lx=0.4, E_c=E_c)
    mu_eff = 1.0
    n = 1.0
    k_F = np.sqrt(2*m*E_c)
    n_F = k_F**3/3/np.pi**2
    E_FG = 2./3*n_F*E_c
    mu = 0.59060550703283853378393810185221521748413488992993*E_c
    delta = 0.68640205206984016444108204356564421137062514068346*E_c * 0
    qT = (mu, mu) +(None,None)+ (mu_eff*np.ones(aslda.Nx),)*2 + (delta * np.ones((aslda.Nx),), None)
    max_iteration = 5
    v_0, n, mu, e_0 = homogeneous.get_BCS_v_n_e(delta=delta, mu_eff=mu_eff)
    x = 0.75
    while max_iteration > 0:
        qT = iterate(lda=aslda,mudelta = qT, N_twist=np.inf,na_avg=1/(1+x), nb_avg=x/(1+x), abs_tol=1e-2)

def test_aslda():
    e_F = 1.0
    k_F = np.sqrt(2*m*e_F)
    n_F = k_F**3/3/np.pi**2
    E_FG = 2./3*n_F*e_F
    mu = 0.59060550703283853378393810185221521748413488992993*e_F
    delta = 0.68640205206984016444108204356564421137062514068346*e_F

    Nx, Ny = 64, 64
    H = np.eye(Nx*Ny).reshape(Nx, Ny, Nx, Ny) # apply 2d dft to the first and second dimensions only
    U = np.fft.fftn(H, axes=[0,1]).reshape(Nx*Ny, Nx*Ny)
    psi = np.random.random((Nx, Ny)) # the wave function is 2d
    np.allclose(np.fft.fftn(psi).ravel(), U.dot(psi.ravel())) # the relation means : dft(H) . psi = dft(psi) ???

    s = vortex_2d_aslda.ASLDA(Nxy=(16,)*2)
    k_c = abs(s.kxy[0]).max()
    E_c = (s.hbar*k_c)**2/2/s.m
    s = vortex_2d_aslda.ASLDA(Nxy=(16,)*2, E_c=E_c)
    kw = dict(mus=(mu, mu), delta=delta)
    H = s.get_H(**kw)
    while(True):
        ns, taus,nu = s.get_ns_taus_nu(H)
        p = s.get_p(ns=ns)
        alphas=s.get_alphas(ns)
        H = s.get_H(mus=(mu,mu),delta = delta,ns = ns, taus=taus,nu = nu)

def test_gx_aslda():
    grid_size = 128
    xs = np.linspace(0,1,10)
    for x in xs: 
        nb_avg = x/(1+x)
        na_avg = 1 - nb_avg
        aslda = ASLDA_(Nx=grid_size)
        k_c = abs(aslda.kx).max()
        E_c = 3*(aslda.hbar*k_c)**2/2/aslda.m # 3 dimension, so E_c should have a factor of 3
        aslda = ASLDA_(Nx=grid_size, Lx=0.4, E_c=E_c)
        mu_eff = 4466
        n = 1.0
        k_F = np.sqrt(2*m*E_c)
        n_F = k_F**3/3/np.pi**2
        E_FG = 2./3*n_F*E_c
        mu = 0.59060550703283853378393810185221521748413488992993*E_c
        delta = 0.68640205206984016444108204356564421137062514068346*E_c
        qT = (mu, mu) +(None,None)+ (mu_eff*np.ones(aslda.Nx),)*2 + (delta * np.ones((aslda.Nx),), None)
        qT = iterate(lda=aslda,mudelta = qT, N_twist=np.inf,na_avg=0.5, nb_avg=0.5, abs_tol=1e-2)

def find_effective_mu(ns):
    grid_size = 128
    xs = np.linspace(0,1,10)
    na=ns
    nb=ns
    aslda = ASLDA_(Nx=grid_size)
    k_c = abs(aslda.kx).max()
    E_c = 3*(aslda.hbar*k_c)**2/2/aslda.m # 3 dimension, so E_c should have a factor of 3
    aslda = ASLDA_(Nx=grid_size, Lx=0.4, E_c=E_c)
    n = 1.0
    k_F = np.sqrt(2*m*E_c)
    n_F = k_F**3/3/np.pi**2
    E_FG = 2./3*n_F*E_c
    mu = 0.59060550703283853378393810185221521748413488992993*E_c
    delta = 0.68640205206984016444108204356564421137062514068346*E_c

    def _lam(mu_eff):
        qT = (mu, mu) +(None,None)+ (mu_eff*np.ones(aslda.Nx),)*2 + (delta * np.ones((aslda.Nx),), None)
        ns_ = get_density(lda=aslda,mudelta = qT, N_twist=np.inf,na_avg=0.5, nb_avg=0.5, abs_tol=1e-2)
        return ns- ns_

    mu_eff = brentq(_lam, 4466, 4470)
    print(mu_eff)

if __name__ == '__main__':
    #test_aslda()
    #test_derivative()
    #test_gx_aslda()
    #test_gx_aslda()
    #print(get_density(4466))
    #print(get_density(4470))
    #print(find_effective_mu(0.5))
    test_iterate_ASLDA()