import numpy as np
from mmf_hfb.functionals import FunctionalASLDA
import itertools  


def test_derivative():
    a = FunctionalASLDA()
    dx = 1e-6
    nas = np.linspace(0.1, 1, 10)
    nbs = np.linspace(0.1, 1, 10)
    nss = list(itertools.product(nas, nbs))
    ls = np.linspace(0, 10, 32)[1:]

    for ns in nss:
        na, nb = ns
        # N1 = (6 * np.pi**2 *(na + nb))**(5.0/3)/20/np.pi**2
        p = a._get_p(ns)

        # test _dalpha_dp
        dalpha_p = (a._alpha(p + dx) - a._alpha(p-dx))/2/dx
        dalpha_p_ = a._dalpha_dp(p)

        assert np.allclose(dalpha_p, dalpha_p_, atol=0.0005)

        # test _Beta
        dBeta_p = (a._Beta_p(p + dx) - a._Beta_p(p-dx))/2/dx
        dBeta_p_ = a._dBeta_dp(p=p)
        assert np.allclose(dBeta_p, dBeta_p_, atol=0.0005)

        # test get_dD_dn
        dD_n_a = (a._D((na + dx, nb))-a._D((na - dx, nb)))/2/dx
        dD_n_b = (a._D((na, nb + dx))-a._D((na, nb - dx)))/2/dx
        dD_n_a_, dD_n_b_ = a._dD_dn(ns)

        assert np.allclose(dD_n_a, dD_n_a_, atol=0.0005)
        assert np.allclose(dD_n_b, dD_n_b_, atol=0.0005)

        # test _dC_dn
        dC_n_a = (a._C((na + dx, nb))-a._C((na - dx, nb)))/2/dx
        dC_n_b = (a._C((na, nb + dx))-a._C((na, nb - dx)))/2/dx
        dC_n_a_, dC_n_b_ = a._dC_dn(ns)
        assert np.allclose(dC_n_a, dC_n_a_, atol=0.0005)
        assert np.allclose(dC_n_b, dC_n_b_, atol=0.0005)

        # test _dalpha_m_dp
        dalpha_m_dp = (a._alpha_m(p + dx) - a._alpha_m(p-dx))/2/dx
        dalpha_m_dp_ = a._dalpha_m_dp(p)
        dalpha_p_dp = (a._alpha_p(p + dx) - a._alpha_p(p-dx))/2/dx
        dalpha_p_dp_ = a._dalpha_p_dp(p)
        assert np.allclose(dalpha_m_dp, dalpha_m_dp_)
        assert np.allclose(dalpha_p_dp, dalpha_p_dp_)

        # test beta and and D
        D = a.get_D(ns=ns)
        beta = a.get_beta(ns=ns)
        assert np.allclose(beta * 0.3 * (3*np.pi**2)**(2.0/3)*sum(ns)**(5.0/3), D)

        # test the public interfaces
        # ......add more tests here......
        for l in ls:
            # test alpha(l * n_a, l*n_b)=alpha(n_a, n_b)
            assert np.allclose(
                a.get_alphas(ns=(ns[0]*l, ns[1]*l)), a.get_alphas(ns=ns), rtol=1e-16)
            # test D(l*n_a,l*n_b)=l**(5/3)D(n_a, n_b)
            assert np.allclose(
                a.get_D(ns=(ns[0]*l, ns[1]*l)), l**(5.0/3)*a.get_D(ns=ns), rtol=1e-16)
