from abc import ABC, abstractmethod


class IFunctional(ABC):
    """
    Interface for functional
    Note: Interface is unnecessary for Python
    because python function call is made by looking up
    a table with based on base classes order
    """
    @abstractmethod
    def get_alpha(self, p, d=0):
        """
        Parameters
        ----------------
        p: polarization in range of (0, 1)
        d: order of derivative(-1, 0, 1)
        """
        pass

    @abstractmethod
    def get_alphas(self, ns, d=0):
        """
        Parameters
        ----------------
        ns: densities (na, nb)
        d: order of derivative
        Note: call the get_alpha
        """
        pass

    @abstractmethod
    def get_p(self, ns, d=0):
        """
        Parameters
        ----------------
        ns: densities (na, nb)
        d: order of derivative
        """
        pass

    @abstractmethod
    def get_C(self, ns, d=0):
        """
        Parameters
        ----------------
        ns: densities (na, nb)
        d: order of derivative
        """
        pass

    @abstractmethod
    def get_D(self, ns, d=0):
        """
        Parameters
        ----------------
        ns: densities (na, nb)
        d: order of derivative
        """
        pass

    @abstractmethod    
    def get_beta(self, ns, d=0):
        """
        Parameters
        ----------------
        ns: densities (na, nb)
        d: order of derivative
        """
        pass
    @abstractmethod
    def get_Vs(self, **args):
        """
        return the Vs(effective potential)
        """
        pass


class IHFBKernel(ABC):

    @abstractmethod
    def get_densities(self, mus_eff, delta, **args):
        """
        return the densities(may include ns, taus, js, nu)
        ----------------
        mus_eff: the effective chemical potential(mu_a, mu_b)
        delta: the gap or coupling strength
        args: other parameters
        """
        pass

    @abstractmethod
    def get_ns_e_p(self, mus, delta, **args):
        """
        return the particle densities(n_a, n_b)
        energy density and pressure
        """
        pass