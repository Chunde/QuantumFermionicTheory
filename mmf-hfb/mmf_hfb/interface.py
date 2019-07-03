from abc import ABC, abstractmethod


class IFunctional(ABC):
    """Interface for functional"""

    @abstractmethod
    def get_alphas(self, ns, d=0):
        """
        Parameters
        ----------------
        ns: densities (na, nb)
        d: order of derivative
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


class IHFB(ABC):

    @abstractmethod
    def get_densities(self, mus_eff, delta, **args):
        """
        return the densities(may include ns, taus, js, nu)
        ----------------
        mus_eff: the effective chemical potential(mu_a, mu_b)
        delta: the gap or coupling strength
        args: other paramenters
        """
        pass
