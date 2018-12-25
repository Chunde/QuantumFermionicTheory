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


def test_iterate_ASLDA(self, mudelta,   na_avg=0.5, nb_avg=0.5, N_twist=1, plot=False, **kw):
    def iterate(self, mudelta,   na_avg=0.5, nb_avg=0.5, N_twist=1, plot=False, **kw):
        mu_a, mu_b, mu_a_eff, mu_b_eff, delta = mudelta
        mus = (mu_a_eff, mu_b_eff)
        if np.isinf(N_twist):
            R = self.get_R_twist_average(mus=mus, delta=delta, **kw)
        else:
            R = self.get_R(mus=mus, delta=delta, N_twist=N_twist)
        na = np.diag(R)[:l.N]/l.dx
        nb = (1 - np.diag(R)[l.N:])/l.dx

        mu_a = mu_a*(1 + (na_avg - na.mean()))
        mu_b = mu_b*(1 + (nb_avg - nb.mean()))

        kappa = np.diag(R[:l.N, l.N:])/l.dx
        mu_a_eff = mu_a + self.v0*nb
        mu_b_eff = mu_b + self.v0*na
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

    l = Lattice(T=0.0, N=N, L=L, v0=v_0, V0=0)
    qT = (mu, mu) + (mu_eff*np.ones(l.N),)*2 + (np.ones(l.N)*delta,)
    max_iteration = 5
    with NoInterrupt() as interrupted:
        while max_iteration > 0:
            max_iteration -= 1
            qT = l.iterate_full(qT, plot=False, N_twist=np.inf,na_avg=n/2, nb_avg=n/2, abs_tol=1e-2)



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
    ns, taus,nu = s.get_ns_taus_nu(H)
    p = s.get_p(ns=ns)
    alphas=s.get_alphas(ns)


if __name__ == '__main__':
    test_aslda()