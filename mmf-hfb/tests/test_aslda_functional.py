import numpy as np
from importlib import reload  # Python 3.4+
import numpy as np
from mmf_hfb import homogeneous;reload(homogeneous)
from mmf_hfb import vortex_2d_aslda;reload(vortex_2d_aslda)
import itertools  
hbar = 1
m = 1


def test_derivative():
    a = vortex_2d_aslda.ASLDA()
    dx = 1e-6
    nas = np.linspace(0.1,1,10)
    nbs = np.linspace(0.1,1,10)
    nss = list(itertools.product(nas, nbs))


    for ns in nss:
        na, nb = ns
        N1 = (6 * np.pi**2 *(na + nb))**(5.0/3)/20/np.pi**2
        p = a._get_p(ns)

        # test _dalpha_dp
        dalpha_p = (a._alpha(p + dx) - a._alpha(p-dx)) / 2 / dx
        dalpha_p_ = a._dalpha_dp(p)

        assert np.allclose(dalpha_p, dalpha_p_, atol=0.0005)

        # test _get_dD_dp
        dD_p =   (a._Dp(p + dx) - a._Dp(p-dx))/ 2 / dx
        dD_p_ = a._dD_dp(p=p)
        assert np.allclose(dD_p, dD_p_, atol=0.0005)

        # test get_dD_dn
        dD_n_a = (a._D(na + dx,nb)-a._D(na-dx,nb))/2/dx
        dD_n_b = (a._D(na,nb + dx)-a._D(na,nb - dx))/2/dx
        dD_n_a_, dD_n_b_ = a._dD_dn(ns)

        assert np.allclose(dD_n_a, dD_n_a_, atol=0.0005)
        assert np.allclose(dD_n_b, dD_n_b_, atol=0.0005)

        # test _dC_dn
        dC_n_a = (a._C(na + dx,nb)-a._C(na-dx,nb))/2/dx
        dC_n_b = (a._C(na,nb + dx)-a._C(na,nb - dx))/2/dx
        dC_n_a_, dC_n_b_ = a._dC_dn(ns)
        assert np.allclose(dC_n_a, dC_n_a_, atol=0.0005)
        assert np.allclose(dC_n_b, dC_n_b_, atol=0.0005)

        # test _dalpha_m_dp
        dalpha_m_dp = (a._alpha_m(p + dx) - a._alpha_m(p-dx))/2/dx
        dalpha_m_dp_ = a._dalpha_m_dp(p)
        dalpha_p_dp = (a._alpha_p(p + dx) - a._alpha_p(p-dx))/2/dx
        dalpha_p_dp_ = a._dalpha_p_dp(p)
        assert np.allclose(dalpha_m_dp,dalpha_m_dp_)
        assert np.allclose(dalpha_p_dp,dalpha_p_dp_)


if __name__ == '__main__':
    test_derivative()
