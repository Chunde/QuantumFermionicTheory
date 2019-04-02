import numpy as np
import pytest
from mmf_hfb import tf_completion as tf
from mmf_hfb import bcs, homogeneous
from scipy.optimize import brentq
from collections import namedtuple
import warnings

from mmf_hfb.FuldeFerrelState import FFState as FF
tf.MAX_DIVISION = 200


@pytest.fixture(params=[1.0])
def delta(request):
    return request.param

@pytest.fixture(params=[1, 2, 3])
def dim(request):
    return request.param


@pytest.fixture(params=[0, 0.2])
def q_dmu(request):
    return request.param


@pytest.fixture(params=[0, 0.2])
def dq_dmu(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def mu_delta(request):
    return request.param


@pytest.fixture(params=[0.1, 0.5])
def dmu_delta(request):
    return request.param


def test_Thermodynamic(delta, mu_delta, dmu_delta, q_dmu, dq_dmu, dim, k_c=200):
    if dim == 3:
        k_c = 50
    mu = mu_delta * delta
    dmu = dmu_delta * delta
    q = q_dmu * mu
    dq = dq_dmu * mu
    Thermodynamic(mu=mu, dmu=dmu, k_c=k_c, q=q, dq=dq, dim=dim, delta0=delta)

def get_e_n_analytically(mu, dmu, q=0, dq=0, dim=1):
    """"return the analytical energy and particle density"""
    if dim == 1:
        def f(e_F): #energy density
            return np.sqrt(2)/np.pi * e_F**1.5/3.0
        def g(e_F): #particle density
            return np.sqrt(2 * e_F)/np.pi
    elif dim == 2:
        def f(e_F):
            return  (e_F**2)/4.0/np.pi
        def g(e_F):
            return e_F/np.pi/2.0
    elif dim == 3:
        def f(e_F):
            return  (e_F**2.5)*2.0**1.5/10.0/np.pi**2
        def g(e_F):
            return  ((2.0 * e_F)**1.5)/6.0/np.pi**2
        
    kF_a, kF_b = np.sqrt(2.0 * (mu+dmu)),np.sqrt(2.0 * (mu-dmu))
    mu_a1, mu_b1, mu_a2, mu_b2 = (q+dq)**2/2.0, (q-dq)**2/2.0, (kF_a)**2/2.0, (kF_b)**2/2.0
    E_a, E_b = f(mu_a2) - f(mu_a1), f(mu_b2) - f(mu_b1)
    n_a, n_b = g(mu_a2) - g(mu_a1), g(mu_b2) - g(mu_b1)
    energy_density = E_a + E_b
    return namedtuple('analytical', ['e','n_a', 'n_b'])(energy_density,n_a, n_b)
    #return energy_density, (n_a, n_b)

def get_dE_dn(mu, dmu, dim, q=0, dq=0):
    """compute the dE/dn for free Fermi Gas"""
    dx = 1e-6
    e1, n1 = get_e_n_analytically(mu=mu + dx, dmu=dmu, dim=dim, q=q, dq=dq)
    e2, n2 = get_e_n_analytically(mu=mu - dx, dmu=dmu, dim=dim, q=q, dq=dq)
    return (e1-e2)/(sum(n1)-sum(n2))

def Thermodynamic(mu, dmu, delta0=1, dim=1, k_c=100, q=0, dq=0, T=0.0,a=0.8, b=1.2, dx=1e-3, N=10, bCheckAnalytically=True):
    #print(f"mu={mu}\tdmu={dmu}\tkc={k_c}\tq={q}\tdq={dq}\tdim={dim}")    
    ff = FF(mu=mu, dmu=dmu, delta=delta0, q=q, dq=dq, dim=dim, k_c=k_c, T=T, fix_g=True, bStateSentinel=True)
    bSuperfluidity = ff.check_superfluidity(mu=mu, dmu=dmu, q=q, dq=dq)
    if not bSuperfluidity:
        print("Not a superfluid state")
    def get_P(mu, dmu):
        delta = ff.solve(mu=mu, dmu=dmu, q=q, dq=dq, a=0.8*delta0, b=1.2*delta0)
        return ff.get_pressure(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)

    def get_E_n(mu, dmu):
        E = ff.get_energy_density(mu=mu, dmu=dmu, q=q, dq=dq)
        na, nb = ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
        return E, na, nb

    def get_ns(mu, dmu):
        return ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
    
    
    def f_ns_dmu(dx, n):
        na, nb = ff.get_densities(mu=mu, dmu=dmu, q=q, dq=dq)
        dn = (na - nb).n
        np0 = (na + nb).n
        def f(dmu_):
            na_, nb_ = ff.get_densities(mu=mu+dx, dmu=dmu_, q=q, dq=dq)
            if n == 0: # fix n_-
                dn_ = (na_ - nb_).n
                return dn - dn_
            elif n == 1: # fix nb
                return (nb - nb_).n
            elif n == 2: # fix na
                return (na - na_).n
            elif n == 3: # fix n_+
                np_=(na_ + nb_).n
                return np0 - np_
        try:
            return brentq(f, a*dmu , b*dmu)
        except:
            irs = np.linspace(a, b, N) * dmu
            for i in reversed(range(N)):
                try:
                    startPos = f(irs[i])
                    for j in reversed(range(i + 1, N)):
                        try:
                            endPos = f(irs[j])
                            if startPos * endPos < 0: # has solution
                                return brentq(f, irs[i], irs[j])
                        except:
                            continue
                except:
                    continue
            warnings.warn(f"Can't find a solution in that region, use the default value={dmu}")
            return dmu # when no solution is found
        
    
    # Check the mu=dE/dn
    if True:
        dmu1 = f_ns_dmu(dx, 0)
        dmu2 = f_ns_dmu(-dx, 0)
        E1, na1, nb1 = get_E_n(mu=mu+dx, dmu=dmu1)
        E0, na0, nb0 = get_E_n(mu=mu-dx, dmu=dmu2)
        n1, n0 = (na1 + nb1).n, (na0 + nb0).n
        mu_ = ((E1-E0)/(n1-n0)).n
        print(f"Fix dn:\t[dn1={(na1-nb1).n}\tdn0={(na0-nb0).n}]")
        print(f"Expected mu={mu}\tNumerical mu={mu_}")
        assert np.allclose((na1-nb1).n, (na0-nb0).n)
        assert np.allclose(mu,mu_, rtol=1e-4)
    
    if False:
        # Reason not to do this for superfluid: a small change
        # to mu may need a big change to dmu, which will 
        # make the state change to normal state, we may need 
        # some more accurate code to properly tune the dmu
        # when mu change slightly. 
        #--------------------------------------------
        # Check the dmu_- = dE/dn_-
        dmu1 = f_ns_dmu(dx, 3)
        dmu2 = f_ns_dmu(-dx, 3)
        E1, na1, nb1 = get_E_n(mu=mu+dx, dmu=dmu1)
        E0, na0, nb0 = get_E_n(mu=mu-dx, dmu=dmu2)

        #if bCheckAnalytically and not bSuperfluidity: # compared with analytical results
        #    a1 = get_e_n_analytically(mu=mu+dx, dmu=dmu1, q=q, dq=dq, dim=dim)

        #    assert np.allclose(E1.n, a1.e)
        #    assert np.allclose(na1.n, a1.n_a)
        #    assert np.allclose(nb1.n, a1.n_b)

        #    a0 = get_e_n_analytically(mu=mu-dx, dmu=dmu2, q=q, dq=dq, dim=dim)
        #    assert np.allclose(E1.n, a1.e)
        #    assert np.allclose(na0.n, a0.n_a)
        #    assert np.allclose(nb0.n, a0.n_b)

        print(f"Fix n:\t[n1={(na1+nb1).n}\tn0={(na0+nb0).n}]")
        dn1, dn0 = (na1 - nb1).n, (na0 - nb0).n
        dmu_ = ((E1-E0)/(dn1-dn0)).n
        print(f"Expected dmu={dmu}\tNumerical dmu={dmu_}")
        assert np.allclose((na1+nb1).n, (na0+nb0).n)
        #assert np.allclose(dmu, dmu_)
        #--------------------------------------------
        dmu1 = f_ns_dmu(dx, 1)
        dmu2 = f_ns_dmu(-dx, 1)
        E1, na1, nb1 = get_E_n(mu=mu+dx, dmu=dmu1)
        E0, na0, nb0 = get_E_n(mu=mu-dx, dmu=dmu2)
        print(f"Fix nb:\t[nb1={nb1.n}\tnb0={nb0.n}]")
        n1, n0 = (na1 + nb1).n, (na0 + nb0).n
        mu_a_ = ((E1-E0)/(na1-na0)).n
        print(f"Expected mu_a={mu + dmu}\tNumerical mu_a={mu_a_}")
        assert np.allclose(nb1.n, nb0.n)
        #assert np.allclose(mu + dmu,mu_a_)    
        #--------------------------------------------
        dmu1 = f_ns_dmu(dx, 2)
        dmu2 = f_ns_dmu(-dx, 2)
        E1, na1, nb1 = get_E_n(mu=mu+dx, dmu=dmu1)
        E0, na0, nb0 = get_E_n(mu=mu-dx, dmu=dmu2)
        print(f"Fix na:\t[na1={na1.n}\tna0={na0.n}]")
        n1, n0 = (na1 + nb1).n, (na0 + nb0).n
        mu_b_ = ((E1-E0)/(n1-n0)).n
        print(f"Expected mu_b={mu - dmu}\tNumerical mu_b={mu_b_}") 
        assert np.allclose(na1.n, na0.n)
        assert np.allclose(mu - dmu,mu_b_)
     
    
    n_a, n_b = get_ns(mu, dmu)
    n_a_ = (get_P(mu+dx/2, dmu+dx/2) - get_P(mu-dx/2, dmu - dx/2))/2/dx
    n_b_ = (get_P(mu+dx/2, dmu-dx/2) - get_P(mu-dx/2, dmu + dx/2))/2/dx
    print(f"Expected n_a={n_a.n}\tNumerical n_a={n_a_.n}")
    print(f"Expected n_b={n_b.n}\tNumerical n_b={n_b_.n}")
    assert np.allclose(n_a.n, n_a_.n)
    assert np.allclose(n_b.n, n_b_.n)


if __name__ == "__main__":
    test_Thermodynamic(delta = 1.0, mu_delta = 3, dmu_delta = 0.5, q_dmu = 0, dq_dmu = 0.5, dim = 1, k_c = 200)
    #Thermodynamic(mu=5, dmu=.5, k_c=500, q=0, dq=.0, dim=1)