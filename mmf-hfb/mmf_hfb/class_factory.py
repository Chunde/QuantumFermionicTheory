"""
This file contains classes used for creating LDA class
factory. The Adapter class glues a functional and a Kernel
to form a new class. The available kernels are BCS and Homogenous.
The available functional: BdG, SLDA and ASLDA
When a new class is created, it can be instantialized. The
factory class accept some more agent classes to fulfil other
functions. The idea of the factory class is to create a brand
new class without passing a functional, a kernel and other
classes as parameter, which can be tightly coupled. With the
factory class, new functionalities can be implemented elsewhere
and be loaded as an agent.
---------------------------------------------------------------
For functional type, please check the FunctionalType enum class
For kernel type, please check the KernelType enum class.
For agent class, please check the FFStateAgent class.
"""
from scipy.optimize import brentq
from enum import Enum
import scipy.optimize
import numpy as np
import inspect
from mmf_hfb.functionals import FunctionalBdG, FunctionalSLDA, FunctionalASLDA
from mmf_hfb.homogeneouse_kernel import homogeneous_kernel
from mmf_hfb.bcs_kernel import bcs_kernel
from mmf_hfb import tf_completion as tf

__all__ = ['FunctionalType', 'KernelType', 'Solvers', 'ClassFactory',
           'DefaultFunctionalAdapter']


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


class DefaultFunctionalAdapter(object):
    """
    the adapter used to connect functional and HFB kernel
    In the factory method, a new class inherit from
    this class will be able to change the behavior
    of both functional and kernel as any method defined in
    this class can override method in other classes.
    """
    def get_C(self, ns, d=0):
        """
        override the C functional to support fixed C value
        """
        if d==0:
            if self.C is None:
                return FunctionalBdG.get_C(self, ns=ns)
            return self.C

        if d==1:
            if self.C is None:
                return FunctionalBdG.get_C(self, ns=ns, d=1)
            return (0, 0)

    def fix_C_BdG(self, mu, dmu, delta, q=0, dq=0, **args):
        """
        fix the C value using BDG integrand
        """
        mu_a, mu_b = mu + dmu, mu -dmu
        args.update(
            m_a=self.m, m_b=self.m, T=self.T, dim=self.dim, k_c=self.k_c)
        self.C = tf.compute_C(
            mu_a=mu_a, mu_b=mu_b, delta=delta, q=q, dq=dq, **args).n
    
    def get_mus_bare(self, mus_eff, delta, **args):
        """
        return the bare mus for given effective mu
        Note: args may contains dq
            mus_eff=(mu_a_eff, mu_b_eff)
        """
        mu_a_eff, mu_b_eff = mus_eff
        res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta, **args)
        ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
        V_a, V_b = self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu)
        mu_a, mu_b = mu_a_eff + V_a, mu_b_eff + V_b
        return (mu_a, mu_b)

    def get_mus_eff(
        self, mus, delta, dq=0, ns=None, taus=None, nu=None, verbosity=True):
        """
        return the effective mus
        ----------------
        Note: mus is (mu_a, mu_b)
        """
        mu_a, mu_b = mus
        V_a, V_b = self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu)
        mu_a_eff, mu_b_eff = mu_a - V_a, mu_b - V_b
        x0 = np.array([mu_a_eff, mu_b_eff])

        def _fun(x):
            mu_a_eff, mu_b_eff = x
            res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta, dq=dq)
            ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
            V_a, V_b = self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu)
            mu_a_eff_, mu_b_eff_ = mu_a - V_a, mu_b - V_b
            if not ((mu_a_eff_ > 0) and (mu_b_eff_ > 0)):  # assume positive
                raise ValueError(f"Effective mu must be positive:{mu_a_eff_, mu_b_eff_}")
            x_ = np.array([mu_a_eff_, mu_b_eff_])
            if verbosity:
                print(x_, ns)
            return x_ - x
        return scipy.optimize.broyden1(_fun, x0)

    def _get_g(self, mus_eff, delta, dq):
        """
        compute g for give effective mus
        """
        res = self.get_densities(
            mus_eff=mus_eff, delta=delta, dq=dq,
            ns_flag=False, taus_flag=False)
        g_eff = delta/res.nu if np.alltrue(delta !=0) else 0*delta
        return g_eff

    def get_g(self, mus, delta, dq=0, ns=None, taus=None, nu=None):
        """
        compute g with given mus and delta
        Note: mus = (mu_a, mu_b)
        """
        mus_eff = self.get_mus_eff(
            mus=mus, delta=delta, dq=dq, ns=ns, taus=taus, nu=nu)
        return self._get_g(mus_eff=mus_eff, delta=delta, dq=dq)

    def _get_C(
            self, delta, mus=None, mus_eff=None, dq=0,
            ns=None, taus=None, nu=None, verbosity=False, **args):
        """
        return the C value when computing g
        Note: NOT the functional get_C(ns), mus = (mu_a, mu_b)
            
        """
        if mus_eff is None:
            mus_eff = self.get_mus_eff(
                mus=mus, delta=delta, dq=dq, ns=ns,
                taus=taus, nu=nu, verbosity=verbosity)
        g = self._get_g(mus_eff=mus_eff, delta=delta, dq=dq)
        alpha_p = sum(self.get_alphas(ns))/2.0
        Lambda = self.get_Lambda(
            mus_eff=mus_eff, alpha_p=alpha_p, E_c=self.E_c,
            k_c=self.k_c, dim=self.dim)
        C = alpha_p/g + Lambda
        return C

    def get_a_inv(self, C=None):
        """
        return the inverse scattering length
        ---------------
        Note: the may be a alpha_p term missing
        """
        if C is None:
            C=self.C
        return 4.0*np.pi*self.hbar**2*C/self.m
    
    def solve(
            self, mus, delta, fix_delta=False,
            solver=None, x0=None, verbosity=True, rtol=1e-12, **args):
        """
        use a solver or simple iteration to solve the gap equation
        return delta and effective mus
        Parameter
        -------------
        x0: init guess(optional)
        """
        if delta is None:
            fix_delta = False
            delta = self.delta  # initial guess
        mu_a, mu_b = mus
        V_a, V_b = self.get_Vs()
        mu_a_eff, mu_b_eff = mu_a + V_a, mu_b + V_b
        if fix_delta and len(np.ones_like(sum(self.xyz))) > 1:
            delta = delta*np.ones_like(sum(self.xyz))

        def _fun(x):
            mu_a_eff, mu_b_eff, delta=x
            res = self.get_densities(
                mus_eff=(mu_a_eff, mu_b_eff), delta=delta, **args)
            ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
            args.update(ns=ns)
            V_a, V_b = self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu)
            mu_a_eff_, mu_b_eff_ = mu_a - V_a, mu_b - V_b
            g_eff = self.get_effective_g(
                mus_eff=(mu_a_eff_, mu_b_eff_), **args)
            delta_ = delta if fix_delta else g_eff*nu
            if verbosity:
                self.output_res(mu_a_eff_, mu_b_eff_, delta_, g_eff, ns, taus, nu)
            return np.array([mu_a_eff_, mu_b_eff_, delta_])

        if x0 is None:
            x0 = np.array([mu_a_eff, mu_b_eff, delta*np.ones_like(sum(self.xyz))])
        if solver is None or type(solver).__name__ != 'function':
            while(True):  # use simple iteration if no solver is specified
                mu_a_eff_, mu_b_eff_, delta_ = _fun(x0)
                if (np.allclose(
                    mu_a_eff_, mu_a_eff, rtol=rtol) and np.allclose(
                        mu_b_eff_, mu_b_eff, rtol=rtol) and np.allclose(
                            delta, delta_, rtol=rtol)):
                    break
                delta, mu_a_eff, mu_b_eff = delta_, mu_a_eff_, mu_b_eff_
        else:
            def fun(x):
                return _fun(x) - x
            mu_a_eff, mu_b_eff, delta = solver(fun, x0, f_rtol=rtol)

        return (delta, mu_a_eff, mu_b_eff)

    def solve_delta(self, mus_eff, dq=0, **args):
        mu_a_eff, mu_b_eff = mus_eff
        """solve the gap equation for given effective mus and C"""
        def f(delta):
            res = self.get_densities(
                mus_eff=(mu_a_eff, mu_b_eff), delta=delta,
                taus_flag=False, nu_flag=False, **args)
            ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
            return (self._get_C(
                mus_eff=mus_eff, delta=delta, dq=dq, ns=ns,
                taus=taus, nu=nu, **args) - self.C)
        delta = brentq(f, a=0.8*self.delta, b=2*self.delta)
        return delta

    def get_energy_density_pressure(self, mus, mus_eff, delta, ns, taus, nu, g_eff):
        """return energy  density and pressure"""
        alpha_a, alpha_b = self.get_alphas(ns=ns)
        energy_density = alpha_a*taus[0]/2.0 + alpha_b*taus[1]/2.0 + g_eff*abs(nu)**2
        if self.T !=0:
            energy_density = (
                energy_density +self.T*self.get_entropy(mus_eff=mus_eff, delta=delta).n)
        energy_density = energy_density + self.get_D(ns=ns)
        pressure = ns[0]*mus[0]*alpha_a + ns[1]*mus[1]*alpha_b - energy_density
        return (energy_density, pressure)

    def get_ns_e_p(self, mus, delta, update_C=False, fix_delta=True, solver=None, **args):
        """
        compute then energy density for BdG, equation(77) in page 39
        Note:
            the return value also include the pressure and densities
        -------------
        mus = (mu_a, mu_b)
        """
        # fix_delta = (delta is not None)
        mu_a, mu_b = mus
        args.update(dim=self.dim, k_c=self.k_c, E_c=self.E_c)
        delta, mu_a_eff, mu_b_eff = self.solve(
            mus=mus, delta=delta, solver=solver, fix_delta=fix_delta, **args)
        mus_eff = (mu_a_eff, mu_b_eff)

        res = self.get_densities(mus_eff=mus_eff, delta=delta, **args)
        ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
        args.update(ns=ns)
        g_eff = self.get_effective_g(mus_eff=mus_eff, **args)
        if update_C:
            self.C = self._get_C(mus_eff=mus_eff, delta=delta, ns=ns, taus=taus, nu=nu)
        e_p = self.get_energy_density_pressure(
            mus=(mu_a, mu_b), mus_eff=mus_eff, delta=delta,
            ns=ns, taus=taus, nu=nu, g_eff=g_eff)
       
        return (ns,) + e_p

    def get_ns_mus_e_p(self, mus_eff, delta, solver=None, **args):
        """
        return ns, bare mu, e and p
        -----------
        Note: dq may be in args
        """
        args.update(dim=self.dim, k_c=self.k_c, E_c=self.E_c)
        if mus_eff is None:
            mus_eff = (self.mu_eff + self.dmu_eff, self.mu_eff - self.dmu_eff)

        if delta is None:
            # solve delta so that yields same C
            delta = self.solve_delta(mus_eff=mus_eff, **args)
            self._delta = delta

        res = self.get_densities(mus_eff=mus_eff, delta=delta, **args)
        ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
        V_a, V_b = self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu)
        mu_a, mu_b = mus_eff[0] + V_a, mus_eff[1] + V_b
        args.update(ns=ns, taus=taus, nu=nu)
        g_eff = self.get_effective_g(mus_eff=mus_eff, **args)
        e_p = self.get_energy_density_pressure(
            mus=(mu_a, mu_b), mus_eff=mus_eff, delta=delta,
            ns=ns, taus=taus, nu=nu, g_eff=g_eff)
        return (ns, (mu_a, mu_b), ) + e_p

    def get_pressure(
            self, mus_eff=None, mus=None, delta=None, q=0, dq=0,
            solver=Solvers.BROYDEN1, **args):
        """return the pressure only"""
        if mus is None:
            return self.get_ns_mus_e_p(
                mus_eff, delta, q=q, dq=dq, solver=solver, **args)[3]
        return self.get_ns_e_p(
            mus=mus, delta=delta, q=q, dq=dq, solver=solver, **args)[2]


def ClassFactory(
        className="LDA", AgentClass=(),
        functionalType=FunctionalType.SLDA,
        kernelType=KernelType.HOM, adapter=None,
        functionalIndex=None, kernelIndex=None, args=None):
    """
    A function that create a new class that uses an adapter class
    to connect a functional class with a kernel class, the new class
    can implement ASLDA in either homogeneous case or BCS case
    Paras:
    -----------
    ClassName: a name for new class
    AgentClass: one or multiple class(es) to inherit from
        (such as a FF state finder agent class used for searching
        FF states)
    functionType: a given functional class(Enum type)
    kernelType: a given kernel class(Enum type)
    args: the arguments used to instantiate a class, if None, the class
        type will be returned.
    -----------
    Note: if args is used to create an instance, it should include all
        parameters fed to all base classes.
    -----------
    # Example
    if __name__ == "__main__":
        mu_eff = 10
        dmu_eff = 0
        delta = 1
        args = dict(
            mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta,
            T=0, dim=3, k_c=50, verbosity=False)
        lda = ClassFactory(
            "LDA", functionalType=FunctionalType.BDG,
            kernelType=KernelType.HOM, args=args)
        lda.C = lda._get_C(mus_eff=(mu_eff, 0), delta=1)
        lda.fix_C_BdG(mu=mu_eff, dmu=0, delta=1)
    """
    Functionals = [FunctionalBdG, FunctionalSLDA, FunctionalASLDA]
    Kernels = [bcs_kernel, homogeneous_kernel]
    if kernelIndex is not None:
        return KernelType(kernelIndex)
    if functionalIndex is not None:
        return FunctionalType(functionalIndex)
    if adapter is None:
        adapter = DefaultFunctionalAdapter
    base_classes = AgentClass + (
        adapter, Kernels[kernelType.value],
        Functionals[functionalType.value])

    def __init__(self, **args):
        self.functional_index=functionalType.value
        self.kernel_index = kernelType.value
        self.functional=functionalType
        self.kernel = kernelType
        for base_class in base_classes:
            if len(inspect.getargspec(base_class.__init__)[0]) > 1:
                base_class.__init__(self, **args)
            else:
                base_class.__init__(self)
    new_class = type(className, (base_classes), {"__init__": __init__})
    if args is None:
        return new_class
    return new_class(**args)
