# this file will eventually be changed to a test file for pytest, now just for debugging in Visual Studio
# Chunde's comment will start with a single sharp '#', to track modifications.
import numpy as np
from scipy.integrate import quad
from uncertainties import ufloat
from importlib import reload  # Python 3.4+
import numpy as np
import vortex_2d_aslda;reload(vortex_2d_aslda)

hbar = 1
m = 1


def test_iterate_ASLDA():
    def iterate(self, mudelta,   na_avg=0.5, nb_avg=0.5, N_twist=0, plot=False, **kw):
        mu_a, mu_b, mu_a_eff, mu_b_eff, delta = mudelta
        mus = (mu_a_eff, mu_b_eff)
        if np.isinf(N_twist):
            R = self.get_R_twist_average(mus=mus, delta=delta, **kw)
        else:
            R = self.get_R(mus=mus, delta=delta, N_twist=N_twist)

        #Modified these code for 2D
        na = np.diag(R)[:np.prod(self.Nxy)]/self.dx**2  #this demoninor should be checked again, physical meaning?
        nb = (1 - np.diag(R)[np.prod(self.Nxy):])/self.dx **2

        H = self.get_H(mus=mus,delta = delta)
        # Q: the way I calculate ns yields different results from the density R
        # A: Because when compute ns from R, twist may be applied, which will yield different result,
        #    after turn off the twist, the results agree
        ns,taus,nu = self.get_ns_taus_nu(H) 
        na,nb = ns
        mu_a = mu_a*(1 + (na_avg - na.mean()))
        mu_b = mu_b*(1 + (nb_avg - nb.mean()))

        kappa = np.diag(R[:np.prod(self.Nxy), np.prod(self.Nxy):])/self.dx
        v_a,v_b = self.get_v_ext()
        mu_a_eff = mu_a + v_a*nb
        mu_b_eff = mu_b + v_b*na
        delta = self.v0*kappa
        if plot:
            plt.clf()
            plt.plot(self.x, na)
            plt.plot(self.x, nb)
            plt.plot(self.x, delta)
            display(plt.gcf())
            print(delta.real.max(), na.real.mean(), nb.real.mean())
        else:
            display("{:.12f}, {:.12f}, {:.12f}".format(
                delta.real.max(), na.real.mean(), nb.real.mean()))
        return (mu_a, mu_b, mu_a_eff, mu_b_eff, delta)

    mu_eff = 1.0
    n = 1.0
    e_F = 1.0
    k_F = np.sqrt(2*m*e_F)
    n_F = k_F**3/3/np.pi**2
    E_FG = 2./3*n_F*e_F
    mu = 0.59060550703283853378393810185221521748413488992993*e_F
    delta = 0.68640205206984016444108204356564421137062514068346*e_F

    aslda = vortex_2d_aslda.ASLDA(T=0.0,Nxy=(2,)*2)
    qT = (mu, mu) + (mu_eff*np.ones(aslda.Nxy),)*2 + (np.ones(aslda.Nxy)*delta,)
    max_iteration = 5
    
    while max_iteration > 0:
        max_iteration -= 1
        qT = iterate(self=aslda,mudelta = qT, plot=False, N_twist=1,na_avg=n/2, nb_avg=n/2, abs_tol=1e-2)



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

    s = vortex_2d_aslda.ASLDA(Nxy=(2,)*2)
    k_c = abs(s.kxy[0]).max()
    E_c = (s.hbar*k_c)**2/2/s.m
    s = vortex_2d_aslda.ASLDA(Nxy=(2,)*2, E_c=E_c)
    kw = dict(mus=(mu, mu), delta=delta)
    #R = s.get_R(**kw)
    H = s.get_H(**kw)
    while(True):
        ns, taus,nu = s.get_ns_taus_nu(H)
        p = s.get_p(ns=ns)
        alphas=s.get_alphas(ns)
        H = s.get_H(mus=(mu,mu),delta = delta,ns = ns, taus=taus,nu = nu)


if __name__ == '__main__':
    #test_aslda()
    test_iterate_ASLDA()