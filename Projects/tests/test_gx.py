"""
This test file used to check the g(x) function defined
in the Fig. 14 of the book chapter. Here the g(x) given
by SLDA and ASLDA should be identical even the alpha terms
are different. [2019/08/18]
The above statement may be wrong![2019/09/23]
"""
from mmf_hfb.ClassFactory import ClassFactory, FunctionalType, KernelType, Solvers
import numpy as np


def g(e, ns):
    g = (e/0.6/0.5/(6*np.pi**2)**(2.0/3))/(ns[0]**(5.0/3))
    g=g**0.6
    return g


def test_g(mu_eff=10):
    args = dict(
        mu_eff=mu_eff, dmu_eff=0, delta=1, T=0, dim=3, k_c=100, verbosity=False)
    slda = ClassFactory(
        "LDA", functionalType=FunctionalType.SLDA,
        kernelType=KernelType.HOM, args=args)
    aslda = ClassFactory(
        "LDA", functionalType=FunctionalType.ASLDA,
        kernelType=KernelType.HOM, args=args)
    
    ns1, _, e1, _ = slda.get_ns_mus_e_p(
        mus_eff=(mu_eff, mu_eff), delta=0, solver=Solvers.BROYDEN1)
    ns2, _, e2, _ = aslda.get_ns_mus_e_p(
        mus_eff=(mu_eff, mu_eff), delta=0, solver=Solvers.BROYDEN1)
    g1 = g(e1, ns1)
    g2 = g(e2, ns2)
    assert np.allclose(ns1[0], ns1[0])
    assert np.allclose(ns2[0], ns2[0])
    assert np.allclose(g1, g2)
