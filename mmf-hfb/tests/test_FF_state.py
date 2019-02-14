import numpy as np
import pytest

from mmf_hfb.FuldeFerrelState import FFState as FF
@pytest.fixture(params=[1,2])
def d(request):
    return request.param

@pytest.mark.bench()
def test_thermodynamic_relations(d):
    mus = [2,4,5,6,8,10]
    dmus = [0.4,0.5,0.64]
    for mu in mus:
        for dmu in dmus:
            r = np.inf
            delta0 = 1

            ff = FF(dmu=dmu, mu=mu, delta=delta0, d=d) # in 1d
            def get_P(mu, dmu):
                mu_a = mu + dmu
                mu_b = mu - dmu
                delta = ff.solve(r=r, mu_a=mu_a, mu_b=mu_b)
                print(delta)
                return ff.get_pressure(mu_a=mu_a, mu_b=mu_b, delta=delta, r=r)    

            def get_E_n(mu, dmu=0):
                mu_a = mu + dmu
                mu_b = mu - dmu
                E = ff.get_energy_density(mu_a=mu_a, mu_b=mu_b, r=r)
                n = sum(ff.get_densities(mu_a=mu_a, mu_b=mu_b, r=r))
                return E, n

            def get_ns(mu, dmu):
                mu_a = mu + dmu
                mu_b = mu - dmu
                return ff.get_densities(mu_a=mu_a, mu_b=mu_b, r=r)   
            dx = 0.001
            E1, n1 = get_E_n(mu+dx)
            E0, n0 = get_E_n(mu-dx)

            print((E1-E0)/(n1-n0), mu) #[check] at some point n1 is equal to n0, causes error


            dx = 1e-3
            n_p = (get_P(mu+dx, dmu) - get_P(mu-dx, dmu))/2/dx
            n_a, n_b = get_ns(mu, dmu)

            assert np.allclose(n_p.n, (n_a+n_b).n)
            n_a_ = (get_P(mu+dx/2, dmu+dx/2) - get_P(mu-dx/2, dmu - dx/2))/2/dx
            n_b_ = (get_P(mu+dx/2, dmu-dx/2) - get_P(mu-dx/2, dmu + dx/2))/2/dx
            assert np.allclose(n_a.n, n_a_.n)
            assert np.allclose(n_b.n, n_b_.n)


if __name__ == "__main__":
    test_thermodynamic_relations(d=2)