"""Test the 2d integ code."""
import numpy as np
import pytest
from mmfutils.testing import allclose
from mmf_hfb.FuldeFerrelState import FFState
import mmf_hfb.tf_completion as tf
from mmf_hfb.Integrates import dquad_kF, dquad_q

@pytest.fixture(params=[3, 3.5, 4])
def r(request):
    return request.param

def min_index(fs):
    min_value = fs[0]
    min_index = 0
    for i in range(1,len(fs)):
        if fs[i] < min_value:
            min_value = fs[i]
            min_index = i
    return min_index,min_value


def compare_two_integration_routines_2d():

    its = [tf.kappa_integrand,tf.n_m_integrand,tf.n_p_integrand,tf.nu_delta_integrand]
    mu = 10
    dmu = 0.4
    mu_a=mu + dmu
    mu_b=mu - dmu
    m_a = m_b= delta= hbar=1
    T=0
    args = dict(mu_a=mu_a, mu_b=mu_b, m_a=m_a, m_b=m_b, delta=delta, hbar=hbar, T=T)
    k_c=100
    q = 0
    def test_integrand(it):
        def integrand(kp, kz):
                k2_a = (kz+q)**2 + kp**2
                k2_b = (kz-q)**2 + kp**2
                return it(ka2=k2_a, kb2=k2_b, **args) /np.pi**2

        def func(kz, kp): 
            return integrand(kz,kp)

        kF = np.sqrt(2*mu)
        print("dquad_kF..............")
        v1 = dquad_kF(f=integrand, kF=kF, k_0=0, k_inf=k_c)/4
        print("dquad_q..............")
        v2 = dquad_q(func=func, mu_a=mu_a, mu_b=mu_b, delta=delta, 
                        q=q, hbar=hbar, m_a=m_a, m_b=m_b, k_0=0, k_inf=k_c)/4
        print(v1, v2)
        assert np.allclose(v1.n, v2.n)

    for it in its:
        test_integrand(it)

if __name__ == "__main__":
    compare_two_integration_routines_2d()