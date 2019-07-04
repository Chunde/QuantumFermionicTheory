from mmf_hfb.Functionals import FunctionalBdG, FunctionalSLDA, FunctionalASLDA
from mmf_hfb.KernelBCS import KernelBCS
from mmf_hfb.KernelHomogeneouse import KernelHomogeneous
import numpy as np
from enum import Enum
import inspect


class Adapter(object):
    """
    the adapter used to connect functional and HFB kernel
    (see interface.py). In the factory method, a new class
    inherit from this class will be able to change the behavior 
    of both functional and kernel as any method defined in 
    this class can override method in other classes.
    """
    def get_C(self, ns, d=0):
        """override the C functional to support fixed C value"""
        if d==0:
            if self.C is None:
                return FunctionalBdG.get_C(self, ns=ns)
            return self.C

        if d==1:
            if self.C is None:
                return FunctionalBdG.get_C(self, ns=ns, d=1)
            return (0, 0)

    def get_ns_e_p(self, mus, delta, update_C, use_Broyden=False, **args):
        """
            compute then energy density for BdG, equation(77) in page 39
            Note:
                the return value also include the pressure and densities
            -------------
            mus = (mu, dmu)
        """
        mu, dmu = mus
        mu_a, mu_b = mu + dmu, mu - dmu
        ns, taus, nu, g_eff, delta, mu_a_eff, mu_b_eff = self.solve(
            mus=mus, delta=delta, use_Broyden=use_Broyden)
        alpha_a, alpha_b = self.get_alphas(ns=ns)
        D = self.get_D(ns=ns)
        energy_density = taus[0]/2.0 + taus[1]/2.0 + g_eff*abs(nu)**2
        if self.T !=0:
            energy_density = (
                energy_density
                +self.T*self.get_entropy(mus_eff=(mu_a_eff, mu_b_eff), delta=delta).n)
        energy_density = energy_density - D
        pressure = ns[0]*mu_a + ns[1]*mu_b - energy_density
        if update_C:
            self.C = self.get_C(ns)
        return (ns, energy_density, pressure)


class FunctionalType(Enum):
    """
    Functional type 
    """
    BDG=0
    SLDA=1
    ASLDA=2


class KernelType(Enum):
    BCS=0  # BCS kernel
    HOM=1  # Homogeneous kernel
    #  FFS=2  # FuldeFerrel kernel

def ClassFactory(className, functionalType=FunctionalType.BDG, kernelType=KernelType.HOM):
    """
    A function that create a new class that uses an adapter class
    to connect a functional class with a kernel class, the new class
    can implement ASLDA in either homogeneous case or BCS case
    """
    Functionals = [FunctionalBdG, FunctionalSLDA, FunctionalASLDA]
    Kernels = [KernelBCS, KernelHomogeneous]
    base_classes = (Adapter, Kernels[kernelType.value], Functionals[functionalType.value])

    def __init__(self, **args):
        for base_class in base_classes:
            sig = inspect.signature(base_class.__init__)
            if len(sig.parameters) > 3:
                base_class.__init__(self, **args)
            else:
                base_class.__init__(self)
    new_class = type(className, (base_classes), {"__init__": __init__})
    return new_class


if __name__ == "__main__":
    dx = 1e-3
    L = 0.46
    N = 16
    N_twist = 1
    delta = 1.0
    mu=10
    dmu = 0
    LDA = ClassFactory(
        className="LDA",
        functionalType=FunctionalType.BDG,
        kernelType=KernelType.HOM)

    lda = LDA(
            Nxyz=(N,), Lxyz=(L,), mu_eff=mu, dmu_eff=dmu,
            delta=delta, T=0, dim=3, C=-0.54)
    
    def get_ns_e_p(mu, dmu, update_C=False):
        ns, e, p = lda.get_ns_e_p(
            mus=(mu, dmu), delta=delta, N_twist=N_twist, Laplacian_only=True,
            update_C=update_C, max_iter=32, use_Broyden=True)
        return ns, e, p

    ns, e, p = get_ns_e_p(mu=mu, dmu=dmu, update_C=False)
    ns1, e1, p1 = get_ns_e_p(mu=mu+dx, dmu=dmu)
    ns2, e2, p2 = get_ns_e_p(mu=mu-dx, dmu=dmu)
    n_p = (p1-p2)/2.0/dx
    mu_ = (e1-e2)/(sum(ns1) - sum(ns2))
    print("-------------------------------------")
    print(np.max(n_p), np.max(sum(ns)))
    print(np.max(mu_), mu)
    print("-------------------------------------")
    assert np.allclose(np.max(n_p).real, sum(ns), rtol=1e-2)
    assert np.allclose(np.max(mu_[0]).real, mu, rtol=1e-2)
