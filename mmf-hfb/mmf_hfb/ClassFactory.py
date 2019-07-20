"""
This file contains classes used for creating ASLDA related class
factory. The Adater class glues functional and Kernel(BCS or Homognesouse)
as a brand new class used for all kind of calculation. 
"""
from mmf_hfb.Functionals import FunctionalBdG, FunctionalSLDA, FunctionalASLDA
from mmf_hfb.KernelHomogeneouse import KernelHomogeneous
from mmf_hfb.KernelBCS import KernelBCS
from mmf_hfb import tf_completion as tf
from scipy.optimize import brentq
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

    def get_alpha(self, p, d=0):
        if d==0:
            return 1.0
        else:
            return 0

    def fix_C_BdG(self, mu, dmu, delta, q=0, dq=0, **args):
        """
        fix the C value using BDG integrand
        """
        mu_a, mu_b = mu + dmu, mu -dmu
        args.update(m_a=self.m, m_b=self.m, T=self.T, dim=self.dim, k_c=self.k_c)
        self.C = tf.compute_C(mu_a=mu_a, mu_b=mu_b, delta=delta, q=q, dq=dq, **args).n
    
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
        mu_a, mu_b = mu_a_eff - V_a, mu_b_eff - V_b
        return (mu_a, mu_b)

    def get_mus_eff(self, mus, delta, dq=0, ns=None, taus=None, nu=None, verbosity=True):
        """
        return the effective mus
        ----------------
        Note: mus is (mu, dmu), not (mu_a, mu_b)
        """
        mu_a, mu_b = mus[0]+mus[1], mus[0]-mus[1]
        V_a, V_b = self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu)
        mu_a_eff, mu_b_eff = mu_a + V_a, mu_b + V_b
        x0 = np.array([mu_a_eff, mu_b_eff])

        def _fun(x):
            mu_a_eff, mu_b_eff = x
            res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta, dq=dq)
            ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
            V_a, V_b = self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu)
            mu_a_eff_, mu_b_eff_ = mu_a + V_a, mu_b + V_b
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
        res = self.get_densities(mus_eff=mus_eff, delta=delta, dq=dq)
        _, _, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
        g_eff = delta/nu
        return g_eff

    def get_g(self, mus, delta, dq=0, ns=None, taus=None, nu=None):
        """
        compute g with given mus and delta
        Note: mus = (mu, dmu)
        """
        mus_eff = self.get_mus_eff(
            mus=mus, delta=delta, dq=dq, ns=ns, taus=taus, nu=nu)
        return self._get_g(mus_eff=mus_eff, delta=delta, dq=dq)

    def _get_C(
            self, delta, mus=None, mus_eff=None, dq=0,
            ns=None, taus=None, nu=None, verbosity=False):
        """
        return the C value when computing g
        Note: NOT the get_C(ns), mus = (mu, dmu)
            
        """
        if mus_eff is None:
            mus_eff = self.get_mus_eff(
                mus=mus, delta=delta, dq=dq, ns=ns, taus=taus, nu=nu, verbosity=verbosity)
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
        use a solver or simple interation to solve the gap equation
        Parameter
        -------------
        x0: init guess(optional)
        """
        if delta is None:
            fix_delta = False
            delta = self.delta  # initial guess
        mu, dmu = mus
        mu_a, mu_b = mu + dmu, mu - dmu
        V_a, V_b = self.get_Vs()
        mu_a_eff, mu_b_eff = mu_a + V_a, mu_b + V_b
        args.update(dim=self.dim, k_c=self.k_c, E_c=self.E_c)
        if fix_delta and len(np.ones_like(sum(self.xyz))) > 1:
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
            
            mu_a_eff, mu_b_eff, delta = solver(fun, x0)
        

        res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta, **args)
        ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
        args.update(ns=ns)
        g_eff = self._g_eff(mus_eff=(mu_a_eff, mu_b_eff), **args)
        return (ns, taus, nu, g_eff, delta, mu_a_eff, mu_b_eff)

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

    def get_ns_e_p(self, mus, delta, update_C=False, fix_delta=True, solver=None, **args):
        """
        compute then energy density for BdG, equation(77) in page 39
        Note:
            the return value also include the pressure and densities
        -------------
        mus = (mu, dmu)
        """
        # fix_delta = (delta is not None)
        mu, dmu = mus
        mu_a, mu_b = mu + dmu, mu - dmu
        if update_C:
            self.fix_C(mu=mu, dmu=0, delta=delta, **args)
        ns, taus, nu, g_eff, delta, mu_a_eff, mu_b_eff = self.solve(
            mus=mus, delta=delta, solver=solver, fix_delta=fix_delta, **args)
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

    def get_ns_mus_e_p(self, mus_eff, delta, solver=None, **args):
        """
        return ns, bare mu, e and p
        -----------
        Note: dq may be in args
        """
        if mus_eff is None:
            mus_eff = (self.mu_eff + self.dmu_eff, self.mu_eff - self.dmu_eff)

        if delta is None:
            # solve delta so that yields same C
            delta = self.solve_delta(mus_eff=mus_eff, **args)
            self._delta = delta

        mu_a_eff, mu_b_eff = mus_eff
        res = self.get_densities(mus_eff=(mu_a_eff, mu_b_eff), delta=delta, **args)
        ns, taus, nu = (res.n_a, res.n_b), (res.tau_a, res.tau_b), res.nu
        V_a, V_b = self.get_Vs(delta=delta, ns=ns, taus=taus, nu=nu)
        mu_a, mu_b = mu_a_eff - V_a, mu_b_eff - V_b
        D = self.get_D(ns=ns)
        g_eff = 0 if nu==0 else delta/nu
        energy_density = taus[0]/2.0 + taus[1]/2.0 + g_eff*abs(nu)**2
        if self.T !=0:
            energy_density = (
                energy_density
                +self.T*self.get_entropy(mus_eff=(mu_a_eff, mu_b_eff), delta=delta).n)
        energy_density = energy_density - D
        pressure = ns[0]*mu_a + ns[1]*mu_b - energy_density
        return (ns, (mu_a, mu_b), energy_density, pressure)


def ClassFactory(
        className, AgentClass=(),
        functionalType=FunctionalType.ASLDA,
        kernelType=KernelType.HOM, args=None):
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
    """
    Functionals = [FunctionalBdG, FunctionalSLDA, FunctionalASLDA]
    Kernels = [KernelBCS, KernelHomogeneous]
    base_classes = AgentClass + (
        Adapter, Kernels[kernelType.value],
        Functionals[functionalType.value])

    def __init__(self, **args):
        for base_class in base_classes:
            sig = inspect.signature(base_class.__init__)
            if len(sig.parameters) > 3:
                base_class.__init__(self, **args)
            else:
                base_class.__init__(self)
    new_class = type(className, (base_classes), {"__init__": __init__})
    if args is None:
        return new_class
    return new_class(**args)
