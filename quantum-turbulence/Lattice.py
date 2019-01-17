from importlib import reload  # Python 3.4+
import bcs;reload(bcs)
import homogeneous;reload(homogeneous)
from bcs import BCS
import numpy as np
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
    
    def iterate_full(self, mudelta, na_avg=0.5, nb_avg=0.5,  N_twist=1, plot=False, **kw):
        mu_a, mu_b, mu_a_eff, mu_b_eff, delta = mudelta
        mus = (mu_a_eff, mu_b_eff)
        if np.isinf(N_twist):
            R = self.get_R_twist_average(mus=mus, delta=delta, **kw)
        else:
            R = self.get_R(mus=mus, delta=delta, N_twist=N_twist)
        na = np.diag(R)[:self.N]/self.dx
        nb = (1 - np.diag(R)[self.N:])/self.dx
        
        mu_a = mu_a*(1 + (na_avg - na.mean()))
        mu_b = mu_b*(1 + (nb_avg - nb.mean()))

        kappa = np.diag(R[:self.N, self.N:])/self.dx
        mu_a_eff = mu_a + self.v0*nb
        mu_b_eff = mu_b + self.v0*na
        delta = self.v0*kappa
        print("{:.12f}, {:.12f}, {:.12f}".format(delta.real.max(), na.real.mean(), nb.real.mean()))
        return (mu_a, mu_b, mu_a_eff, mu_b_eff, delta)

def TestWithHomogenous():
    delta = 1.0
    mu_eff = 1.0
    v_0, n, mu, e_0 = homogeneous.get_BCS_v_n_e(delta=delta, mu_eff=mu_eff)

    L = 0.46
    N = 8
    N_twist = 20#2**5
    for b in [bcs.BCS(T=0, N=N, L=L),
              Lattice(T=0.0, N=N, L=L, v0=v_0, V0=0)]:
        R = b.get_R(mus=(mu_eff, mu_eff), delta=delta, N_twist=N_twist)
        na = np.diag(R)[:N]/b.dx
        nb = (1 - np.diag(R)[N:])/b.dx
        kappa = np.diag(R[:N, N:])/b.dx
        print((n, na[0].real + nb[0].real), (delta, v_0*kappa[0].real))

if __name__ == '__main__':
    # Test - reproduce homogeneous results
    TestWithHomogenous()
    #L = 0.46
    #N = 128
    #delta = 1.0
    #mu_eff = 1.0
    #v_0, n, mu, e_0 = homogeneous.get_BCS_v_n_e(delta=delta, mu_eff=mu_eff)
    #l = Lattice(T=0.0, N=N, L=L, v0=v_0, V0=0)
    #qT = (mu, mu) + (mu_eff*np.ones(l.N),)*2 + (np.ones(l.N)*delta,)

    #while True:
    #    qT = l.iterate_full(qT, plot=False, N_twist=np.inf, na_avg=n* 2/5, nb_avg=n*3/5, abs_tol=1e-2)
