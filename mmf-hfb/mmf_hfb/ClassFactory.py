from mmf_hfb.Functionals import FunctionalBdG, FunctionalSLDA, FunctionalASLDA
from mmf_hfb.KernelHomogeneouse import KernelHomogeneous
from mmf_hfb.KernelBCS import KernelBCS
from mmf_hfb import tf_completion as tf
from enum import Enum
import scipy.optimize
import numpy as np
import inspect


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


class Solvers(Enum):
    """
    solver types
    """
    NONE=None
    BROYDEN1=scipy.optimize.broyden1
    BROYDEN2=scipy.optimize.broyden2
    ANDERSON=scipy.optimize.anderson
    LINEARMIXING=scipy.optimize.linearmixing
    DIAGBROYDEN=scipy.optimize.diagbroyden


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

    def get_alphas(self, ns, d=0):
        if d==0:
            return (1.0, 1.0)
        elif d==1:
            return (0, 0, 0, 0)

    def fix_C(self, mu, dmu, delta, q=0, dq=0, **args):
        mu_a, mu_b = mu + dmu, mu -dmu
        args.update(m_a=self.m, m_b=self.m, T=self.T, dim=self.dim, k_c=self.k_c)
        self.C = tf.compute_C(mu_a=mu_a, mu_b=mu_b, delta=delta, q=q, dq=dq, **args).n

    def compute_dc(self, mus, delta, dq):
        pass

    def solve(
        self, mus, delta, fix_delta=False, rtol=1e-12,
            solver=None, verbosity=True, **args):
        """
        use solver or simple interation to solve the gap equation
        """
        mu, dmu = mus
        mu_a, mu_b = mu + dmu, mu - dmu
        V_a, V_b = self.get_Vs()
        mu_a_eff, mu_b_eff = mu_a + V_a, mu_b + V_b
        args.update(dim=self.dim, k_c=self.k_c, E_c=self.E_c)
        if fix_delta:
            delta = delta*np.ones_like(sum(self.xyz))

        def _fun(x):
            mu_a_eff, mu_b_eff, delta=x
            res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta, **args)
            ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
            args.update(ns=ns)
            V_a, V_b = self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu)
            mu_a_eff_, mu_b_eff_ = mu_a + V_a, mu_b + V_b
            g_eff = self._g_eff(mus_eff=(mu_a_eff_, mu_b_eff_), **args)
            delta_  = delta if fix_delta else g_eff*nu
            if verbosity:
                self.output_res(mu_a_eff_, mu_b_eff_, delta_, g_eff, ns, taus, nu)
            return np.array([mu_a_eff_, mu_b_eff_, delta_])

        if solver is None or type(solver).__name__ != 'function':
            while(True):  # use simple iteration if no solver is specified
                mu_a_eff_, mu_b_eff_, delta_ = _fun((mu_a_eff, mu_b_eff, delta))
                if (np.allclose(
                    mu_a_eff_, mu_a_eff, rtol=rtol) and np.allclose(
                        mu_b_eff_, mu_b_eff, rtol=rtol) and np.allclose(
                            delta, delta_, rtol=rtol)):
                    break
                delta, mu_a_eff, mu_b_eff = delta_, mu_a_eff_, mu_b_eff_
        else:
            def fun(x):
                return _fun(x) - x
            
            x0 = np.array([mu_a_eff, mu_b_eff, delta*np.ones_like(sum(self.xyz))])
            mu_a_eff, mu_b_eff, delta = solver(fun, x0)
        # if the delta is too small, that may mean not solution is found
        if delta < 1e-5:
            raise ValueError("Invalid delta")

        res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta, **args)
        ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
        args.update(ns=ns)
        g_eff = self._g_eff(mus_eff=(mu_a_eff, mu_b_eff), **args)
        return (ns, taus, nu, g_eff, delta, mu_a_eff, mu_b_eff)

    def get_ns_e_p(self, mus, delta, update_C=False, solver=None, **args):
        """
            compute then energy density for BdG, equation(77) in page 39
            Note:
                the return value also include the pressure and densities
            -------------
            mus = (mu, dmu)
        """
        mu, dmu = mus
        mu_a, mu_b = mu + dmu, mu - dmu
        if update_C:
            self.fix_C(mu=mu, dmu=0, delta=delta, **args)
        ns, taus, nu, g_eff, delta, mu_a_eff, mu_b_eff = self.solve(
            mus=mus, delta=delta, solver=solver, fix_delta=update_C, **args)
        # alpha_a, alpha_b = self.get_alphas(ns=ns)
        D = self.get_D(ns=ns)
        energy_density = taus[0]/2.0 + taus[1]/2.0 + g_eff*abs(nu)**2
        if self.T !=0:
            energy_density = (
                energy_density
                +self.T*self.get_entropy(mus_eff=(mu_a_eff, mu_b_eff), delta=delta).n)
        energy_density = energy_density - D
        pressure = ns[0]*mu_a + ns[1]*mu_b - energy_density
        return (ns, energy_density, pressure)


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
