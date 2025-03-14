"""
A class used to perform Fulde Ferrell states related
calculation. This class can be discarded as the FFStateAgent
class is now used more often due to the fact it can use
different functional. For the BdG functional, both classes
will yield same parameter regime.
"""
import numpy as np
from mmf_hfb import tf_completion as tf
from scipy.optimize import brentq
import warnings
tf.MAX_DIVISION = 500
MAX_ITERATION = 100


class FFState(object):
    def __init__(self, mu, dmu, delta=1, q=0, dq=0, m=1, T=0, hbar=1, g=None,
                 k_c=None, dim=1, fix_g=True):
        """
        Arguments
        ---------
        fix_g : bool
           If fix_g is False, the class will fix C_tilde when
           performing the non-linear iterations, otherwise the
           coupling constant g_c will be fixed.  Note: this g_c will
           depend on the cutoff k_c.
        """
        if g is not None:
            fix_g = True
        if k_c is None:
            k_c = 10*mu
        self.fix_g = fix_g
        self.dim = dim
        self.T = T
        self.mus = (mu + dmu, mu - dmu)
        self.m = m
        self.delta = delta
        self.hbar = hbar
        if dim == 1:
            k_c = np.inf
        self.k_c = k_c
        self._tf_args = dict(m_a=1, m_b=1, dim=dim, hbar=hbar, T=T, k_c=k_c)
        if fix_g:
            self._g = self.get_g(
                mu=mu, dmu=dmu, delta=delta, q=q, dq=dq) if g is None else g
            self.lock_g = False if g is None else True
        else:
            self._C = tf.compute_C(mu_a=mu, mu_b=mu,
                                   delta=delta, q=q, dq=dq, **self._tf_args).n

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, val):
        if self.lock_g is False:
            self._g = val

    def f(self, mu, dmu, delta, q=0, dq=0, **kw):
        args = dict(self._tf_args)
        args.update(kw)
        if self.fix_g:
            return self._g - self.get_g(
                mu=mu, dmu=dmu, q=q, dq=dq, delta=delta, **args)

        return tf.compute_C(
            mu_a=mu+dmu, mu_b=mu-dmu, delta=delta,
            q=q, dq=dq, **args).n - self._C

    def get_g(self, delta, mu=None, dmu=None, q=0, dq=0, **kw):
        assert (mu is None) == (dmu is None)
        if mu is None:
            mu_a, mu_b = self.mus
            mu, dmu = (mu_a + mu_b)/2.0, (mu_a - mu_b)/2.0
        args = dict(self._tf_args, q=q, dq=dq, delta=delta)
        args.update(kw, mu_a=mu+dmu, mu_b=mu-dmu)
        nu_delta = tf.integrate_q(tf.nu_delta_integrand, **args)
        g = 1./nu_delta.n
        return g

    def _get_Lambda(self, k0, k_c, dim=3):
        """return the renormalization condition parameter Lambda"""
        if dim == 3:
            Lambda = k_c/self.hbar**2/2/np.pi**2*(
                1.0 - k0/k_c/2*np.log((k_c+k0)/(k_c-k0)))
        elif dim == 2:
            Lambda = 1/self.hbar**2/4/np.pi*np.log((k_c/k0)**2 - 1)
        elif dim == 1:
            Lambda = 1/self.hbar**2/2/np.pi*np.log((k_c-k0)/(k_c+k0))/k0
        return Lambda  # do not forget effective mess inverse factor

    def get_a_inv(self, delta, mu=None, dmu=None, q=0, dq=0, k_c=None, **kw):
        """return the inverse of scattering length"""
        assert (mu is None) == (dmu is None)
        if mu is None:
            mu_a, mu_b = self.mus
            mu, dmu = (mu_a + mu_b)/2.0, (mu_a - mu_b)/2.0
        if k_c is None:
            k_c = self.k_c
        args = dict(self._tf_args, q=q, dq=dq, delta=delta)
        args.update(kw, mu_a=mu+dmu, mu_b=mu-dmu, k_c=k_c)
        nu_delta = tf.integrate_q(tf.nu_delta_integrand, **args)  # 1/g
        k0 = (2*mu)**0.5/self.hbar
        Lambda = self._get_Lambda(k0=k0, k_c=k_c, dim=self.dim)
        a_inv = (nu_delta + Lambda)*4*np.pi*self.hbar**2
        return a_inv

    def get_current(
            self, mu=None, dmu=None, q=0, dq=0, delta=None, k_c=None):
        """
        return the currents of two the components
        return value: (j_a, j_b, j_p, j_m)
        """
        assert (mu is None) == (dmu is None)
        if mu is None:
            mu_a, mu_b = self.mus
            mu, dmu = (mu_a + mu_b)/2.0, (mu_a - mu_b)/2.0

        if delta is None:
            delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq,
                               a=self.delta * 0.8, b=self.delta * 1.2)
        mu_a, mu_b = mu + dmu, mu - dmu
        args = dict(self._tf_args)
        if k_c is not None:
            args.update(k_c=k_c)
        args.update(q=q, dq=dq)
        return tf.compute_current(
            mu_a=mu_a, mu_b=mu_b, delta=delta, **args)

    def get_densities(
            self, mu=None, dmu=None, q=0,
            dq=0, delta=None, k_c=None):
        """return the densities of two the components"""
        assert (mu is None) == (dmu is None)
        if mu is None:
            mu_a, mu_b = self.mus
            mu, dmu = (mu_a + mu_b)/2.0, (mu_a - mu_b)/2.0
        if delta is None:
            delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq,
                               a=self.delta * 0.8, b=self.delta * 2)

        args = dict(self._tf_args, mu_a=mu + dmu, mu_b=mu - dmu, delta=delta,
                    q=q, dq=dq)
        if k_c is not None:
            args['k_c'] = k_c
        n_p = tf.integrate_q(tf.n_p_integrand, **args)
        n_m = tf.integrate_q(tf.n_m_integrand, **args)
        n_a, n_b = (n_p + n_m)/2, (n_p - n_m)/2
        return n_a, n_b

    def _get_effective_delta(
            self, mu, dmu, g=None, q=0, dq=0,
            k_c=np.inf, a=0.8, b=1.2):
        if g is None:
            g = self._g

        def f(delta):
            return self.get_g(delta=delta, mu=mu, dmu=dmu, q=q, dq=dq)
        return brentq(f, a*self.delta, b*self.delta)

    def _get_bare_mus(self, mu_eff, dmu_eff, delta=None, q=0, dq=0):
        if self.dim != 1:
            return (mu_eff, dmu_eff)
        if delta is None:
            delta = self.delta
        mu_a_eff, mu_b_eff = mu_eff + dmu_eff, mu_eff - dmu_eff
        args = dict(
            self._tf_args, mu_a=mu_a_eff, mu_b=mu_b_eff,
            delta=delta, q=q, dq=dq)
        n_p = tf.integrate_q(tf.n_p_integrand, **args).n
        n_m = tf.integrate_q(tf.n_m_integrand, **args).n
        n_a, n_b = (n_p + n_m)/2, (n_p - n_m)/2
        mu_a, mu_b = mu_a_eff + self._g * n_b, mu_b_eff + self._g * n_a
        return (mu_a + mu_b)/2, (mu_a - mu_b)/2

    def _get_effective_mus(
            self, mu, dmu, mus_eff=None, delta=None, q=0, dq=0,
            k_c=np.inf, rtol=1e-6, update_g=False):
        """
        return effective mu, dmu
        ---------
        update_g: bool
            if it's true, the g_c will be update as the iteration goes
            this can be done just once time for each instance, but
            can be multiple times as needed
        """
        if self.dim != 1:
            return (mu, dmu)

        self._tf_args.update(k_c=k_c)
        update_delta = False
        if delta is None:
            if mus_eff is not None:
                delta = self.solve(
                    mu=mus_eff[0], dmu=mus_eff[1], q=q, dq=dq,
                    a=self.delta * 0.8, b=self.delta * 1.2)
                mu_a, mu_b = mus_eff[0] + mus_eff[1], mus_eff[0] - mus_eff[1]
            else:
                delta = self.solve(
                    mu=mu, dmu=dmu, q=q, dq=dq,
                    a=self.delta * 0.8, b=self.delta * 1.2)
            update_delta = True
        args = dict(self._tf_args, mu_a=mu + dmu, mu_b=mu - dmu, delta=delta,
                    q=q, dq=dq)
        if k_c is not None:
            args['k_c'] = k_c

        n_p = tf.integrate_q(tf.n_p_integrand, **args).n
        n_m = tf.integrate_q(tf.n_m_integrand, **args).n
        n_a, n_b = (n_p + n_m)/2, (n_p - n_m)/2
        # for 1d case, we need to solve the densities self-consistently
        error = 1.0
        mu_a0, mu_b0 = mu + dmu, mu - dmu
        itr = 0
        while(error > rtol):
            itr = itr + 1
            mu_a_eff = mu_a0 - self.g * n_b
            mu_b_eff = mu_b0 - self.g * n_a
            args.update(mu_a=mu_a_eff, mu_b=mu_b_eff, delta=delta)
            if update_g:
                self.g = delta/tf.integrate_q(tf.nu_integrand, **args).n
            n_p = tf.integrate_q(tf.n_p_integrand, **args).n
            n_m = tf.integrate_q(tf.n_m_integrand, **args).n
            n_a, n_b = (n_p + n_m)/2, (n_p - n_m)/2
            mu_a, mu_b = mu_a_eff + self.g*n_b, mu_b_eff + self.g*n_a
            mu_ = (mu_a + mu_b) / 2
            error = np.abs(mu_ - mu)/mu
            mu_eff = (mu_a_eff + mu_b_eff)/2.0
            dmu_eff = (mu_a_eff - mu_b_eff)/2.0
            if update_delta:
                self.delta = self.solve(
                    mu=mu_eff, dmu=dmu_eff, q=q, dq=dq,
                    a=delta*0.8, b=delta*1.2)
            if itr > MAX_ITERATION:
                warnings.warn(
                    f"Reach max iteration without "
                    + f"converging to the desired accuracy:{error}")
                break
        mu_a_eff = mu_a - self._g*n_b
        mu_b_eff = mu_b - self._g*n_a
        return ((mu_a_eff + mu_b_eff)/2, (mu_a_eff - mu_b_eff)/2)

    def get_ns_p_e_mus_1d(
            self, mu, dmu, mus_eff=None, delta=None,
            q=0, dq=0, k_c=np.inf, update_g=False):
        """
        return the particle densities, pressure, energy density,
            effective mus for 1d using self-consistent method
        Arguments
        ---------
        mu: The bare mu (mu_a + mu_b)/2
        dmu: The bare chemical potential difference (mu_a - mu_b)/2
        delta: Nonable
            if delta is None, its value will be solved using mu and dmu
        mus_eff :(mu, dmu)
            Effective mu, dmu that can be used to evaluate
            the gap equation in the beginning of the iteration if
            it's not None.
        update_g: bool
            Indicate if the g_c should be updated in the iteration
            By default its value is False, means the g_c will be fixed
        """
        assert self.dim == 1
        mu_eff, dmu_eff = self._get_effective_mus(
            mu=mu, dmu=dmu, mus_eff=mus_eff,
            delta=delta, q=q, dq=dq, k_c=k_c, update_g=update_g)
        if delta is None:
            delta = self.delta
        mu_a_eff = mu_eff + dmu_eff
        mu_b_eff = mu_eff - dmu_eff
        args = dict(self._tf_args, mu_a=mu_a_eff, mu_b=mu_b_eff, delta=delta,
                    q=q, dq=dq)
        n_p = tf.integrate_q(tf.n_p_integrand, **args).n
        n_m = tf.integrate_q(tf.n_m_integrand, **args).n
        n_a, n_b = (n_p + n_m)/2, (n_p - n_m)/2
        print(
            f"mu_eff={mu_eff}, dmu_eff={dmu_eff}, delta={self.delta}, "
            + f"n_a={n_a}, n_b={n_b}, g_c={self._g}")
        kappa = tf.integrate_q(tf.kappa_integrand, **args).n
        e = kappa + self._g * n_a * n_b
        p = (mu+dmu) * n_a + (mu-dmu) * n_b - e
        return (n_a, n_b, e, p, (
            (mu_a_eff + mu_b_eff)/2,
            (mu_a_eff - mu_b_eff)/2))

    def get_FFG_energy_density(self, mu, dmu, q=0, dq=0):
        """return the Free Fermi Gas(FFG) energy density(delta=0)"""
        mu_a, mu_b = mu + dmu, mu - dmu
        if self.dim == 1:
            energy_density = np.sqrt(2)/np.pi*(mu_a**1.5 + mu_b**1.5)/3.0
            # na, nb=np.sqrt(2 *mu_a)/np.pi, np.sqrt(2 * mu_b)/np.pi
        elif self.dim == 2:
            energy_density = (mu_a**2 + mu_b**2)/4.0/np.pi
            # na, nb =mu_a/np.pi/2.0, mu_b/np.pi/2.0
        elif self.dim == 3:
            energy_density = (mu_a**2.5 + mu_b**2.5)*2.0**1.5/10.0/np.pi**2
            #  na=nb = ((2.0*mu_a)**1.5)/6.0/np.pi**2
        return energy_density

    def get_energy_density(self, mu=None, dmu=None, q=0, dq=0, delta=None,
                           n_a=None, n_b=None, k_c=None, use_kappa=True):
        assert (mu is None) == (dmu is None)
        if mu is None:
            mu_a, mu_b = self.mus
            mu, dmu = (mu_a + mu_b)/2.0, (mu_a - mu_b)/2.0
        if delta is None:
            delta = self.solve(mu=mu, dmu=dmu, q=q, dq=dq,
                               a=self.delta * 0.8, b=self.delta * 1.2)
        if n_a is None:
            n_a, n_b = self.get_densities(
                mu=mu, dmu=dmu, delta=delta, q=q, dq=dq, k_c=k_c)

        args = dict(
            self._tf_args, mu_a=mu + dmu,
            mu_b=mu - dmu, delta=delta, q=q, dq=dq)
        # kappa only for the situation where the gap equation is satisfied
        if use_kappa:
            kappa = tf.integrate_q(tf.kappa_integrand, **args)
        else:
            tau_p = tf.integrate_q(tf.tau_p_integrand, **args)
            nu = tf.integrate_q(tf.nu_integrand, **args)
            kappa = tau_p/2 + self._g * abs(nu)**2
        if self.fix_g:
            g_c = self._g
        else:
            g_c = 1./self._C
        if self.dim == 1:
            return kappa + g_c * n_a * n_b
        return kappa

    def get_pressure(
            self, mu=None, dmu=None, mu_eff=None,
            dmu_eff=None, q=0, dq=0, delta=None,
            use_kappa=True):
        """
        return the pressure
        -------------------
        NOTE: the mus and effective mus only differ for 1D
            because the Hartree term is non zero
        """
        assert (mu is None) == (dmu is None)
        assert (mu_eff is None) == (dmu_eff is None)
        if mu is None:
            mu, dmu = self._get_bare_mus(
                mu_eff=mu_eff, dmu_eff=dmu_eff, delta=delta, q=q, dq=dq)
        if mu_eff is None:
            mu_eff, dmu_eff = self._get_effective_mus(
                mu=mu, dmu=dmu, q=q, dq=dq)
        if delta is None:
            delta = self.solve(
                mu=mu_eff, dmu=dmu_eff, q=q, dq=dq,
                a=self.delta*0.8, b=self.delta*1.2)
        n_a, n_b = self.get_densities(
            mu=mu_eff, dmu=dmu_eff, delta=delta, q=q, dq=dq)
        energy_density = self.get_energy_density(
            mu=mu_eff, dmu=dmu_eff, delta=delta, q=q, dq=dq,
            n_a=n_a, n_b=n_b, use_kappa=use_kappa)
        mu_a, mu_b = mu + dmu, mu - dmu
        pressure = mu_a*n_a + mu_b*n_b - energy_density
        return pressure

    def solve(
            self, mu=None, dmu=None, q=0, dq=0,
            a=None, b=None, throwException=False, **args):
        """
        On problem with brentq is that it requires very smooth function with a
        and b having different sign of values, this can fail frequently if our
        integration is not with high accuracy. Should be solved in the future.
        """
        assert (mu is None) == (dmu is None)
        if mu is None:
            mu_a, mu_b = self.mus
            mu, dmu = (mu_a + mu_b)/2.0, (mu_a - mu_b)/2.0

        if a is None:
            a = 0.0001
        if b is None:
            b = self.delta
        # if dq > self.delta:
        #     return 0
        def f(delta):
            return self.f(mu=mu, dmu=dmu, delta=delta, q=q, dq=dq)

        self.delta_ex = None  # a another possible solution
        if throwException:
            delta = brentq(f, a, b)
        else:
            try:
                delta = brentq(f, a, b)
            except ValueError:
                ds = np.linspace(a, b, 40)
                f0 = f(ds[-1])
                index0 = -1
                delta = 0
                for i in reversed(range(0, len(ds)-1)):
                    f_ = f(ds[i])
                    if f0*f_ < 0:
                        delta = brentq(f, ds[index0], ds[i])
                        # save the extra delta
                        if f_*f(ds[0]) < 0:  # another solution
                            delta_ = brentq(f, ds[0], ds[i])
                            self.delta_ex = delta_

                            p_ = self.get_pressure(
                                mu_eff=mu, dmu_eff=dmu,
                                delta=delta_, q=q, dq=dq)
                            p = self.get_pressure(
                                mu_eff=mu, dmu_eff=dmu,
                                delta=delta, q=q, dq=dq)
                            print(
                                f"q={dq}: Delta={delta_}/{delta},"
                                + f",Pressue={p_.n}/{p.n}")
                            if p_ > p:
                                self.delta_ex = delta
                                delta = delta_
                        break
                    else:
                        f0 = f_
                        index0 = i
                if (
                    delta == 0 and (f(
                        0.999*self.delta)*f(
                            1.001*self.delta) < 0)):
                    delta = brentq(f, 0.999*self.delta, 1.001*self.delta)
        return delta
