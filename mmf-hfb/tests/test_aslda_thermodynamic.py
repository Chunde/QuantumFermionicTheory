from mmf_hfb import bcs_aslda, homogeneous_aslda
import numpy as np
import pytest
import warnings
warnings.filterwarnings("ignore")


@pytest.mark.skip(reason="Not pass yet")
def test_homogeneous_aslda_thermodynamic(dx=1e-3):
    delta = 1.0
    mu = 10
    dmu = 0
    C = -0.54
    lda = homogeneous_aslda.SLDA(T=0, mu_eff=mu, dmu_eff=dmu, delta=delta, C=C, dim=3)

    def get_ns_e_p(mu, dmu, delta=delta, update_C=False):
        ns, e, p = lda.get_ns_e_p(mus=(mu, dmu), delta=delta, update_C=update_C)
        return ns, e, p

    ns1, e1, p1 = get_ns_e_p(mu=mu+dx, dmu=dmu, delta=None)
    print("-------------------------------------")

    ns2, e2, p2 = get_ns_e_p(mu=mu-dx, dmu=dmu, delta=None)
    print("-------------------------------------")

    n_p = (p1-p2)/2/dx
    mu_ = (e1-e2)/(sum(ns1) - sum(ns2))
    print("-------------------------------------")
    print(n_p, sum(ns1 + ns2)/2.0)
    print(mu_, mu)
    print("-------------------------------------")
    assert np.allclose(n_p, sum(ns1 + ns2)/2.0, rtol=1e-2)
    assert np.allclose(mu_, mu, rtol=1e-2)


if __name__ == "__main__":
    test_homogeneous_aslda_thermodynamic()
