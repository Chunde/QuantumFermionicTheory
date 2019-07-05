from mmf_hfb.ClassFactory import ClassFactory, FunctionalType, KernelType, Solvers
import numpy as np


if __name__ == "__main__":
    dx = 1e-3
    L = 0.46
    N = 16
    N_twist = 4
    delta = 1.0
    mu=np.pi
    dmu = 0
    LDA = ClassFactory(
        className="LDA",
        functionalType=FunctionalType.BDG,
        kernelType=KernelType.HOM)

    lda = LDA(
        Nxyz=(N, ), Lxyz=(L,), mu_eff=mu, dmu_eff=dmu,
        delta=delta, T=0, dim=3)
    
    def get_ns_e_p(mu, dmu, update_C=False):
        ns, e, p = lda.get_ns_e_p(
            mus=(mu, dmu), delta=delta, N_twist=N_twist, Laplacian_only=True,
            update_C=update_C, max_iter=32, solver=Solvers.BROYDEN1)
        return ns, e, p

    ns, e, p = get_ns_e_p(mu=mu, dmu=dmu, update_C=True)
    print("-------------------------------------")
    ns1, e1, p1 = get_ns_e_p(mu=mu+dx, dmu=dmu)
    print("-------------------------------------")
    ns2, e2, p2 = get_ns_e_p(mu=mu-dx, dmu=dmu)
    #print(p1, p2, e1, e2, ns1, ns2)
    n_p = (p1-p2)/2.0/dx
    mu_ = (e1-e2)/(sum(ns1) - sum(ns2))
    print(np.max(n_p), np.max(sum(ns)))
    print(np.max(mu_), mu)
    print("-------------------------------------")
    assert np.allclose(np.max(n_p).real, sum(ns), rtol=1e-2)
    assert np.allclose(np.max(mu_).real, mu, rtol=1e-2)
