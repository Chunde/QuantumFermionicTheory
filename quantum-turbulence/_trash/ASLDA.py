r"""This module defines the ASLDA density functional.
"""
from __future__ import division
__all__ = ['ASLDAFunctional', 'ASLDAFunctionalFit']

from copy import deepcopy
import warnings

import scipy.optimize
sp = scipy

import numpy
np = numpy
from numpy import pi, inf

from mmf.objects import Container, process_vars #, StateVars
from mmf.objects import Required, Delegate, Computed, ClassVar #, Excluded

import mmf.fit.lsq as lsq
import mmf.math.interp
from mmf.math.interp import tabulate #, UnivariateSpline, EvenUnivariateSpline

from slda.interfaces import implements, IState
import slda.TF

from _functionals import UnitaryFunctional, TwoSpeciesState
from SSLDA import SSLDAFunctional

_PI2 = pi*pi
_ABS_TOL = slda.TF._ABS_TOL
_REL_TOL = slda.TF._REL_TOL
_L = slda.TF._L
_Tinv = slda.TF._Tinv
_TINY = np.finfo(float).tiny
_EPS = np.finfo(float).eps
_OPTIMIZED = True

#------------------------------------------------
# these are the original Carlson et al estimates
#------------------------------------------------
Carlson_param = {'xi_thermo': 0.44,
                 'zeta_thermo': 0.44,
                 'eta_thermo': 0.486,
                 'alpha': 1.1175,
                 'beta_thermo': -0.51955,
                 'gammainv_thermo': -0.095509}

#--------------------------------------
# these are Aurel's best estimates
#--------------------------------------
Aurel_param = {'xi_thermo': 0.42,
               'zeta_thermo': 0.42,
               'eta_thermo': 0.504,
               'alpha': 1.14,
               'beta_thermo': -0.55269,
               'gammainv_thermo': -0.090585}

###############################
Best_param = {'xi_thermo': 0.42,                 # +- 0.01
              'zeta_thermo': 0.42,               # +- 0.01
              'eta_thermo': 0.504,               # +- 0.024
              'm_thermo': 0.871,                 # +- 0.027
              'alpha': 1.131,             # +- 0.034 
              'beta_thermo': -0.545,             # +- 0.032
              'gammainv_thermo': -0.0908}        # +- 0.0098

# My parameters based on fitting xi_thermo = 0.40(1) and eta_thermo = 0.504(24)
Best_param_ = {'xi_thermo': 0.40,                 # +- 0.01
               'zeta_thermo': 0.40,               # +- 0.01
               'eta_thermo': 0.504,               # +- 0.024
               'm_thermo': 0.914491,              # +- 0.013942
               'alpha': 1./0.914491,       # +- 
               'beta_thermo': -0.517201,          # +- 0.018489
               'gammainv_thermo': -0.086408}        # +- 0.007677

Best_param = {'xi_thermo': 0.42,
              'zeta_thermo': 0.42,
              'eta_thermo': 0.504,
              'alpha': 1.14,
              'beta_thermo': -0.55269,
              'gammainv_thermo': -0.090585}

class ASLDAState(TwoSpeciesState):
    r"""State representing a two-component unitary gas with the
    densities required for the ASLDA functional.

    An assortment of common :attr:`_state vars` are provided here.
    Not all are used.
    """
    implements(IState)
    _state_vars = [
        ('n_p_x', None, "Density of species a"),
        ('n_m_x', None, "Density of species b"),
        ('delta_alpha_x', None,\
             "Gap parameter Delta/alpha"),
        ('tau_a_x', NotImplemented,\
             "Optional: Energy for species a."),
        ('tau_b_x', NotImplemented,\
             "Optional: Energy for species a."),
        ('k_m_x', NotImplemented,\
             "Optional: Energy a - b."),
        ('k_p_x', NotImplemented,\
             "Optional: Convergent energy."),
        ]
    process_vars()

class ASLDAFunctional(UnitaryFunctional):
    r"""Asymmetric Superfluid Local Density Approximation (ASLDA)
    Density Functional for two species of fermions.

    This describes a density functional where

    .. math::
       \alpha_{a}(n_+, n_-) &= \alpha(p),\\
       \alpha_{b}(n_+, n_-) &= \alpha(-p),\\
       D(n_+, n_-) &= \beta(p)\frac{(3\pi^2 n_{+})^{5/3}}{10\pi^2},\\
       C(n_+, n_-) &= \gamma^{-1}n_{+}^{1/3}
    
    where the functions :math:`\alpha(p)`, :math:`\beta(p)`, and the
    parameter :math:`\gamma` are determined by fitting the Monte Carlo
    data for the quasiparticle dispersion relationships and the energy
    of the normal state.

    For the convenience of fitting, we use the variable

    .. math::
       p = \frac{n_-}{n_+} = \frac{1 - x}{1 + x} \in [-1, 1].

    Notes
    -----
    The form specified here consists of three homogeneous phases:
       `y` < :attr:`y_0`:
          A fully polarized phase.
       :attr:`y_0` < `y` < :attr:`y_c`:
          A partially polarized normal phase.
       :attr:`y_c` < `y`: A fully paired superfluid fully characterized by
          :attr:`xi_thermo`.

    Examples
    --------

    >>> f = ASLDAFunctionalFit()
    >>> mu_p = (1.0 + 2.5)/2
    >>> mu_m = (1.0 - 2.5)/2
    >>> ans = f.TF_densities(mu_p=mu_p, mu_m=mu_m, delta_alpha=0.0)
    >>> abs(ans.F/ans.P - 3./2.) < _ABS_TOL
    True
    >>> ans = f.TF_densities(mu_p=mu_p, mu_m=mu_m, delta_alpha=1.0)
    >>> abs(ans.F/ans.P - 3./2.) < _ABS_TOL
    True
    """
    _state_vars = [
        ('alpha_p', Required,\
             r"""Interpolated function :math:`\alpha(p)`.

             It is intended that this be implemented in terms of an
             interpolation, and that it will return the derivative if
             passed a parameter `nu`.
             """),
        ('_G_p', Required,\
             r"""Interpolated function :math:`G(x) = G(p(x))`.

             In terms of the dimensionless function :math:`g(x)`
             representing the energy density as a function of
             :math:`x=n_b/n_a`, this is:

             .. math::
                G(p) &= \left(\frac{g(x)}{1+x}\right)^{5/3}
                      = \left(\frac{(1 + p) g(p)}{2}\right)^{5/3}.

             It is intended that this be implemented in terms of an
             interpolation, and that it will return the derivative if
             passed a parameter `nu`.
             """),
        ('spline_opt', Delegate(mmf.math.interp.Options, []),\
             r"""Options such as tolerances to use for spline
             approximation of :meth:`h_y`.  These are
             only used by :meth:`grand_canonical` for computing Thomas-
             Fermi approximations, so high accuracy will not affect
             the functional.  See :class:`mmf.interp.Options` for
             details."""),
        ('spline_opt.abs_tol', 1e-3),
        ('spline_opt.rel_tol', 1e-3),
        ('spline_opt.n_max', 1000),
        ('spline_opt.plot', False),
        ('xi_thermo', Computed,\
             r"""Energy of the superfluid phase compared to the
             non-interacting normal phase at the same density in the
             thermodynamic limit."""),
        ('eta_thermo', Computed,\
             r"""`eta_thermo = delta/eF`, the size of the gap in terms of the
             Fermi energy in the symmetric phase in the thermodynamic limit."""),
        ('gammainv_thermo', Required,\
             r"""Pairing interaction parameter."""),
        ('y_0', Computed,\
             r"""Critical `y` below which the system is completely
             polarized."""),
        ('y_c', Computed,\
             r"""Critical `y` at which there is a first order
             transition between the fully paired SF state and the
             other phases."""),
        ('y_LO_N', -0.215,\
             r"""Critical `y` separating the Normal and LO phases.
             Used if :attr:`LOFF` is `True`"""),
        ('y_LO_SF', 0.02,\
             r"""Critical `y` separating the LO and superfluid phase.
             Used if :attr:`LOFF` is `True`"""),
        ('y_dh_LO', [[0.5], [0.009]],\
             r"""Data to parametrize LOFF state.  The first is a list
             of interior points as a linear interpolation between
             :attr`y_LO_N` and :attr:`y_LO_SF`.  The second is the
             increase `dh` in `h`.  Used if :attr:`LOFF`  is `True`"""),
        ('LOFF', False,\
             r"""If `True`, then include approximate LOFF phase in LDA
             calculations."""),
        ('_LOFF_poly', Computed,\
            r"""Array representing polynomial that is used by
            :meth:`_dh_LOFF` to parametrize the LDA LOFF correction.
            `dh = x*p(x)` where `p` is the polynomial, `0 <= x <= 1`
            is a linear parameter interpolating between :attr:`y_LO_N`
            and :attr:`y_LO_SF`.  This ensures a second order
            transition at :attr:`y_LO_N`."""),
        ('x_c', Computed,\
             r"""Critical `x<1` at which the first order
             transition between the fully paired SF state and the
             other phases starts."""),
        ('_h_y_normal_spline', Computed,\
             r"""Spline approximation to :meth:`_h_y_normal` for
             speed."""),
        ('no_SF', False, r"""If `True`, then the pairing correlations
             are turned off."""),
        ('_keys', ClassVar(['n_p_x', 'n_m_x', 'delta_alpha_x', 'k_m_x',
                            'k_p_x'])),
        ('_State', ClassVar(ASLDAState)),
        ('_ycs_a', Computed),
        ('_ycs_b', Computed),
        ('_hs_a', Computed),
        ('_hs_b', Computed),
        ]
    process_vars()

    def __init__(self, *v, **kw):
        # Compute xi_thermo, eta_thermo
        n_p = 2**(3/2)/3/np.pi**2
        res = slda.gap_equations.gap_equation_2(n_p=n_p, delta_alpha=1.0,
                                                functional=self)
        self.xi_thermo = res.xi
        self.eta_thermo = res.eta
        C = res.tC_delta/res.delta/self.alpha_p(0)
        assert (abs(self.gammainv_thermo - C/n_p**(1/3)) < 1e-12)
        
        # Compute location of first order transition to the SF state.
        gSF = (2*self.xi_thermo)**(3/5)
        def f(x):
            g = self._g_x_normal(x)
            dg = self._g_x_normal(x, d=1)
            return (1-x)*dg + g - gSF

        try:
            self.x_c = sp.optimize.brentq(f, 0, 1)
        except ValueError:
            warnings.warn("Could not find `x_c`... Don't trust" +
                          "thermodynamic functions!")
            return

        y_c = self._y_x_normal(self.x_c)
        y_0 = self._y_x_normal(0.0)
        
        #self._ycs_a = np.array([y_0, y_c])
        #self._ycs_b = np.array([y_0, y_c])
        # Need to deal with archiving methods for these to work
        #self._hs_a = [self.h_y, self.h_y, self.h_y]
        #self._hs_b = [self.h_y, self.h_y, self.h_y]
        y_ = np.hstack([[y_0, y_c, self.y_LO_SF, self.y_LO_N]])
        y_.sort()
        h_ = self._h_y_normal(y_)
        self.spline_opt.xy = (y_,h_)
        self._h_y_normal_spline = tabulate(self._h_y_normal,
                                           y_.min(), y_.max(),
                                           self.spline_opt)
        self.y_0 = y_0
        self.y_c = y_c

        if self.LOFF:
            # Modify to include an LDA LOFF state.  We use a
            # polynomial fit to the provided data for the
            # parameter x \in [0,1] that interpolates linearly between
            # y_LO_N and y_LO_SF.  By starting the interpolation with
            # x^2 we ensure a second order transition at y_LO_N.

            dx_dy = 1/(self.y_LO_SF - self.y_LO_N)
            x = np.hstack([[0],self.y_dh_LO[0],[1]])
            dh = np.hstack([[0],self.y_dh_LO[1],[0]])
            y = x/dx_dy + self.y_LO_N
            # Start with dh = 0 so we can compute background h without
            # LOFF.
            self._LOFF_poly = np.polyfit(x, 0*x, len(x)-1)
            h = self.h_y(y)
            h_n = self._h_y_normal(y)
                        
            # Now set transition to LO/SF.  h_y will now return only
            # the normal state, so we must include the difference with
            # the superfluid state in the polynomial.  We add these to
            self.y_c = self.y_LO_SF
            dh = dh + (h - h_n)

            # Now fit polynomial of one degree less to ensure a smooth
            # transition at x=0 where the transition between N and LO occurs.
            dh_x = dh/x
            dh_x[0] = 0
            self._LOFF_poly = np.polyfit(x, dh_x, len(x)-1)
    
    def _Q_p(self, p, nu=0):
        r"""
        Examples
        --------
        >>> f = ASLDAFunctionalFit()
        >>> f._Q_p(-1.0)
        0.0
        >>> f._Q_p(0.0)         # 1./0.914/2**(5./3.)
        0.3446...
        >>> print f._Q_p(1.0)
        1.0

        >>> from mmf.math.differentiate import differentiate
        >>> diff = np.vectorize(lambda f, x: differentiate(f, x, h0=0.1))
        >>> p = np.linspace(-0.9, 0.9, 11)
        >>> (abs(diff(f._Q_p, p) - f._Q_p(p, nu=1)) < 5e-11).all()
        True
        >>> (abs(differentiate(f._Q_p, -1, dir=+1, h0=1e-4)
        ...  - f._Q_p(-1, nu=1)) < 1e-5)
        True
        >>> (abs(differentiate(f._Q_p, 1, dir=-1, h0=1e-4)
        ...  - f._Q_p(1, nu=1)) < 1e-5)
        True
        """
        
        if 0 == nu:
            return self.alpha_p(p)*np.power((1 + p)/2, 5/3)
        elif 1 == nu:
            return (self.alpha_p(p, nu=1)*np.power((1 + p)/2, 5/3)
                    + 5/6*self.alpha_p(p)*np.power((1 + p)/2, 2/3))
        else:
            raise NotImplementedError
        
    ############################################
    # Full thermodynamic functions
    def _grand_canonical(self, mu_p, mu_m):
        r"""Return `(n_p, n_m, delta_alpha, P, E, v_p, v_m)`.  May assume
        that `mu_a > mu_b`

        Examples
        --------
        >>> f = ASLDAFunctionalFit()
        >>> res = f.grand_canonical(-10,-10)
        >>> [abs(float(np.round(getattr(res, k)))) for k in res]
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        >>> map(lambda x:round(x, 12), f._grand_canonical(0, 10))
        [1.5104..., 1.5104..., 0.0, 6.0416..., 9.0624..., -2.710..., 2.710...]

        >>> f._grand_canonical(np.array([1,2.5]), np.array([0,0.5]))
        (array([ 0.37810619,  1.49459593]),
         array([ 0., 0.]),
         array([ 1.23432925,  3.08582312]),
         array([ 0.15124247,  1.49459593]),
         array([ 0.22686371,  2.2418939 ]),
         array([-1.22964033, -3.07410083]),
         array([ 0.,  0.]))

        (array([ 0.18905309,  0.74729797]),
         array([ 0.18905309,  0.74729797]),
         array([ 1.23432925,  3.08582312]),
         array([ 0.15124247,  1.49459593]),
         array([ 0.22686371,  2.2418939 ]),
         array([-1.22964033, -3.07410083]), 
         array([-1.22964033, -3.07410083]))

        (array([ 0.18880088,  0.746301  ]),
         array([ 0.18880088,  0.746301  ]),
         array([ 1.2332312 ,  3.08307799]),
         array([ 0.1510407,  1.492602 ]), 
         array([ 0.22656105,  2.238903  ]),
         array([-1.22765686, -3.06914215]),
         array([-1.22765686, -3.06914215]))

        (array([ 0.18905316, 0.74729823]),
         array([ 0.18905316, 0.74729823]),
         array([ 1.23432189, 3.08580472]),
         array([ 0.15124253, 1.49459646]),
         array([ 0.22686379, 2.24189469]),
         array([-1.22960998, -3.07402495]),
         array([-1.22960998, -3.07402495]))

        This should give xi_thermo

        >>> c = 3.0/10.0*(3.0*np.pi**2)**(2.0/3.0)
        >>> mu = np.array(10.0)
        >>> res = f.grand_canonical(mu, mu)
        >>> n = res.n_p
        >>> _xi = 3.0*mu/5.0/c/(n**(2.0/3.0))
        >>> np.allclose(_xi, f.xi_thermo)
        True

        """
        _optimized = _OPTIMIZED
        mu_a = mu_p + mu_m
        mu_b = mu_p - mu_m
        n_p, n_m, P, E = self._thermodynamics(mu_p=mu_p, mu_m=mu_m)
        n_a = (n_p + n_m)/2
        n_b = (n_p - n_m)/2
        n_a = abs(n_a) + _TINY
        n_b = abs(n_b) + _TINY
        n_p = n_a + n_b
        n_m = n_a - n_b

        if not _optimized:
            C = self.C(n_p=n_p, n_m=n_m)
            D = self.D(n_p=n_p, n_m=n_m)
            alpha_p, alpha_m = self.alpha(n_p=n_p, n_m=n_m)
        else:
            # Optimized version

            # Use this to deal with fluctuations in tails.
            p = self._p(n_p=n_p, n_m=n_m)
            
            alpha_a = self.alpha_p(p)
            alpha_b = self.alpha_p(-p)
            
            G_p = self._G_p(p)
            G_m = self._G_p(-p)

            C = self.gammainv_thermo*n_p**(1/3)
            
            Q_a = alpha_a*((1 + p)/2)**(5/3)
            Q_b = alpha_b*((1 - p)/2)**(5/3)

            beta_p = 2**(2/3)*((G_p + G_m)/2 - Q_a - Q_b)
            
            # This is A/n_p and D/n_p to avoid some problems when n_p
            # is small.
            A_n_p = (3*_PI2)**(5/3)/10/_PI2*n_p**(2/3)
            D_n_p = A_n_p*beta_p
            D = A_n_p*beta_p*n_p

        eF = (3*pi**2*n_p)**(2/3)/2
        tau_a = (6*np.pi**2*n_a)**(5/3)/10/np.pi**2
        tau_b = (6*np.pi**2*n_b)**(5/3)/10/np.pi**2
        alpha_p = (alpha_a + alpha_b)/2.0

        y = mu_b/mu_a

        SF = self.y_c < y
        if self.no_SF:
            SF *= False

        delta = np.where(SF, self.eta_thermo*eF, 0)
        k_m = np.where(SF, 0, tau_a - tau_b)/2
        k_p = np.where(SF, (E - D)/alpha_p, (tau_a + tau_b)/2)
        
        if not _optimized:
            # Slow version
            v_a, v_b = self.get_v(n_a=n_a, n_b=n_b,
                                  k_p=k_p, k_m=k_m,
                                  delta_alpha=delta/alpha_p)
        else:
            # Optimized version
            dalpha_a =  self.alpha_p(p, nu=1)
            dalpha_b =  -self.alpha_p(-p, nu=1)

            # These are dp/dn_a*n_p and dp/dn_b*n_p to avoid problems
            # when n_p is small.
            dp_an_p = -(p - 1)
            dp_bn_p = -(p + 1)
            
            dQ_a_p = (dalpha_a*(1 + p)**(5/3)
                     + 5/3*alpha_a*(1 + p)**(2/3))/2**(5/3)

            dQ_b_p = (dalpha_b*(1 - p)**(5/3)
                      - 5/3*alpha_b*(1 - p)**(2/3))/2**(5/3)

            dG_p_p = self._G_p(p, nu=1)
            dG_m_p = -self._G_p(-p, nu=1)

            dbeta_p = 2**(2/3)*((dG_p_p + dG_m_p)/2 - dQ_a_p - dQ_b_p)
            
            dD_a = 5*D_n_p/3 + np.where(p < 1 , A_n_p*dbeta_p*dp_an_p, 0)
            dD_b = 5*D_n_p/3 + np.where(p > -1, A_n_p*dbeta_p*dp_bn_p, 0)
            
            dC_a = dC_b = C/3/n_p
            
            # Back to common code
            alpha_p = (alpha_a + alpha_b)/2.0
            #alpha_m = (alpha_a - alpha_b)/2.0

            delta2 = abs(delta)**2
            v_a = (dD_a - delta2*dC_a)
            v_b = (dD_b - delta2*dC_b)

            if not self.no_dalpha:
                dalpha_a_a = dalpha_a*dp_an_p/n_p
                dalpha_a_b = dalpha_a*dp_bn_p/n_p
                dalpha_b_a = dalpha_b*dp_an_p/n_p
                dalpha_b_b = dalpha_b*dp_bn_p/n_p
                
                dalpha_p_a = (dalpha_a_a + dalpha_b_a)/2.0
                dalpha_m_a = (dalpha_a_a - dalpha_b_a)/2.0
                dalpha_p_b = (dalpha_a_b + dalpha_b_b)/2.0
                dalpha_m_b = (dalpha_a_b - dalpha_b_b)/2.0

                # This step removes spurious tails
                _a = mmf.math.step(n_p, self.n_unit_mass, self.n_unit_mass/2)
                v_a += _a*(dalpha_p_a*k_p + 
                           dalpha_m_a*k_m -
                           dalpha_p_a/alpha_p*delta2*C)
                v_b += _a*(dalpha_p_b*k_p + 
                           dalpha_m_b*k_m -
                           dalpha_p_b/alpha_p*delta2*C)

            n_a -= _TINY
            n_b -= _TINY
            n_p -= _TINY

        n_p = n_a + n_b
        n_m = n_a - n_b
        v_p = (v_a + v_b)/2
        v_m = (v_a - v_b)/2
        return n_p, n_m, k_p, k_m, delta/alpha_p, P, E, v_p, v_m

    def g_x(self, x):
        r"""Return `g(x) = f^(3/5)(x)` for `x` in `[0,1]` including
        mixed phases etc.

        Parameters
        ----------
        x : float
           `x = n_b/n_a`.  Must be in the range [0,1].  To deal with
           `x > 1` note that `g(x) = x g(1/x)`.
        """ 
        g1 = (2*self.xi_thermo)**(3/5)
        if x is 1:
            # Special case 1 so that we can get this value even if
            # _x_c is not yet defined.
            return g1
        else:
            return np.where(self.x_c <= x,
                            ((g1*(x - self.x_c) 
                              + self._g_x_normal(self.x_c)*(1 - x))/
                             (1 - self.x_c)), # Maxwell construction
                            self._g_x_normal(x))
    def dg_x(self, x):
        r"""Return `dg(x)/dx` including mixed phases

        Parameters
        ----------
        x : float
           `x = n_b/n_a`.  Must be in the range [0,1].  To deal with
           `x > 1` note that `g'(x) = g(1/x) - g'(1/x)/x`.
        """ 
        g1 = self._g_x(1)
        gc = self._g_x_normal(self.x_c)
        return np.where(self.x_c < x,
                        (g1 - gc)/(1 - self.x_c),
                        self._g_x_normal(x, d=1))

    def h_y(self, y, d=0, compute=False):
        r"""Return `h(y)` given `y = mu_b/mu_a`.

        Parameters
        ----------
        y : float
           `y = mu_b/mu_a`.  Must be in the range [-inf, 1].
        d : int
           Order of derivative to compute
        compute : bool
           If `True`, then perform the exact calculation, otherwise
           used the spline approximation calculated during `__init__`.
    
        Notes
        -----
        .. math::
           P(\mu_a, \mu_b) = \tfrac{2}{5}\beta\left[\mu_a h(y)\right]^{5/2},
           \qquad \beta = \frac{1}{6\pi^2}\left[\frac{2m}{\hbar^2}\right]^{3/2}
        """
        if 0 == d:
            if compute:
                h_y_normal = self._h_y_normal
            else:
                h_y_normal = self._h_y_normal_spline

            if self.no_SF:
                h = np.where(y < self.y_0,
                             1.0,
                             h_y_normal(y))
            else:
                h = np.where(y < self.y_0,
                             1.0,
                             np.where(y <= self.y_c,
                                      h_y_normal(y),
                                      self._h_y_SF(y)))
                if self.LOFF:
                    h = np.where(
                        np.logical_and(self.y_LO_N < y,
                                       y <= self.y_LO_SF),
                        h + self._dh_LOFF(y),
                        h)
            return h
        elif 1 == d:
            if compute:
                dh_y_normal = self._dh_y_normal
            else:
                def dh_y_normal(y):
                    return self._h_y_normal_spline(y, nu=1)

            if self.no_SF:
                dh = np.where(y < self.y_0,
                              0.0,
                              dh_y_normal(y))
            else:
                dh = np.where(y < self.y_0,
                              0.0,
                              np.where(y < self.y_c,
                                       dh_y_normal(y),
                                       self._h_y_SF(y, d=1)))
                if self.LOFF:
                    dh = np.where(
                        np.logical_and(self.y_LO_N < y,
                                       y < self.y_LO_SF),
                        dh + self._dh_LOFF(y,d=1),
                        dh)
            return dh
        else:
            raise NotImplementedError("Only `d` = 0 or 1 supported.")

    h_a_y = h_y
    h_b_y = h_y

    def _dh_LOFF(self, y, d=0):
        r"""Return `dh` in the LO region.  Uses :attr:`_LOFF_poly`."""
        dx_dy = 1./(self.y_LO_SF - self.y_LO_N)
        x = (y - self.y_LO_N)*dx_dy
        if 0 == d:
            return np.polyval(self._LOFF_poly, x)*x
        elif 1 == d:
            tmp = (np.polyval(np.polyder(self._LOFF_poly), x)*x
                   + np.polyval(self._LOFF_poly, x))
            return tmp*dx_dy
        else:
            raise NotImplementedError("Only `d` = 0 or 1 supported.")

    def y_x(self, x):
        r"""Return `y = mu_b/mu_a` given `x = n_b/n_a`.
        
        Parameters
        ----------
        x : float
           `x = n_b/n_a`.  Must be in the range [0,1].  To deal with
           `x > 1` note that `y(x) = 1/y(1/x)`.

        Notes
        -----
        .. math::
           y = \frac{g'(x)}{g(x) - xg'(x)} =
               \frac{1}{\frac{5f(x)}{3f'(x)} - x}
        """
        return np.where(x <= self.x_c,
                        self._y_x_normal(x),
                        self.y_c)

    def x_y(self, y):
        r"""Return `x = mu_b/mu_a` given `y = n_b/n_a`.

        Parameters
        ----------
        y : float
           `x = mu_b/mu_a`.  Must be in the range [-inf, 1].
        """
        return np.where(y < self.y_0,
                        0.0,
                        np.where(y < self.y_c,
                                 self._x_y_normal(y),
                                 1.0))

    def _beta_p(self, p, nu=0):
        r"""
        Examples
        --------
        >>> f = ASLDAFunctionalFit()
        >>> round(f._beta_p(-1.0), 12)
        -0.0
        >>> f._beta_p(0.0)      # doctest: +ELLIPSIS
        -0.526...
        >>> round(f._beta_p(1.0), 12)
        -0.0

        >>> from mmf.math.differentiate import differentiate
        >>> diff = np.vectorize(lambda f, p: differentiate(f, p, h0=0.1))
        >>> p = np.linspace(-0.9, 0.9, 11)
        >>> (abs(diff(f._beta_p, p) - f._beta_p(p, nu=1)) < 5e-11).all()
        True
        >>> (abs(differentiate(f._beta_p, -1, dir=+1, l=4, h0=0.1)
        ...  - f._beta_p(-1, nu=1)) < 1e-6)
        True
        >>> (abs(differentiate(f._beta_p, 1, dir=-1, l=4, h0=0.1)
        ...  - f._beta_p(1, nu=1)) < 1e-6)
        True
        """
        if 0 == nu:
            b_p = ((self._G_p(p) + self._G_p(-p))/2 
                   - self._Q_p(p) - self._Q_p(-p))
            return 2**(2/3)*b_p
        elif 1 == nu:
            db_p = ((self._G_p(p, nu=1) - self._G_p(-p, nu=1))/2 
                    - self._Q_p(p, nu=1) + self._Q_p(-p, nu=1))
            return 2**(2/3)*db_p            
        else:
            raise NotImplementedError
        
    ##########################################################
    def alpha(self, n_p, n_m, k_p=None, k_m=None, delta_alpha=None):
        r"""There is no dependence on `k_p` or `k_m` so we provide a
        default version here."""
        p = self._p(n_p=n_p, n_m=n_m)
        alpha_a = self.alpha_p(p)
        alpha_b = self.alpha_p(-p)
        alpha_p = (alpha_a + alpha_b)/2
        alpha_m = (alpha_a - alpha_b)/2

        if True:
            _alpha_p, _alpha_m = UnitaryFunctional.alpha(
                self, n_p=n_p, n_m=n_m, k_p=0, k_m=0, delta_alpha=0)
            assert np.allclose(_alpha_p, alpha_p)
            assert np.allclose(_alpha_m, alpha_m)

        return (_alpha_p, _alpha_m)
        
    def D(self, n_p, n_m):
        r"""Return LDA energy density for the density-only dependent
        portion of the LDA density functional.

        This should be of dimension :math:`n^{5/3}`.
        """
        p = self._p(n_p=n_p, n_m=n_m)
        A = (3*_PI2*n_p)**(5/3)/10/_PI2
        D = A*self._beta_p(p)
        return D

    def dD_n_p(self, n_p, n_m):
        r"""Return the derivative `dD(n_p, n_m)/dn_p`.

        This should be of dimension :math:`n^{2/3}`.
        D = #(n_p)^(5/3)*beta(p)
        p = n_m/n_p

        """
        p = self._p(n_p=n_p, n_m=n_m)
        A_n_p = (3*_PI2)**(5/3)/10/_PI2*n_p**(2/3)
        D_n_p = A_n_p*self._beta_p(p)
        dp_dn_p = -p/n_p
        
        return 5*D_n_p/3 + np.where(abs(p) < 1, 
                                    A_n_p*self._beta_p(p, nu=1)*dp_dn_p, 0)
    def dD_n_m(self, n_p, n_m):
        r"""Return the derivative `dD(n_p, n_m)/dn_m`.

        This should be of dimension :math:`n^{2/3}/m`.
        """
        p = self._p(n_p=n_p, n_m=n_m)
        A_n_p = (3*_PI2)**(5/3)/10/_PI2*n_p**(2/3)
        dp_dn_m = 1/n_p

        return np.where(abs(p) < 1, 
                        A_n_p*self._beta_p(p, nu=1)*dp_dn_m, 0)

    def tC(self, n_p, n_m, k_p=None, k_m=None, delta_alpha=None):
        r"""Return the effective coupling.

        .. math::
           tC = \frac{\alpha_+(0)}{\gamma} n^{1/3}

        This should be of dimension :math:`m n^{1/3}`.
        """
        alpha_p0 = self.alpha_p(0)
        C = self.gammainv_thermo*n_p**(1/3)
        tC = alpha_p0*C
        return tC

    def dtC_n_p(self, n_p, n_m, k_p=None, k_m=None, delta_alpha=None):
        dtC_n_p = self.tC(n_p, n_m)/3/n_p
        return dtC_n_p

    def dtC_n_m(self, n_p, n_m, k_p=None, k_m=None, delta_alpha=None):
        return 0*n_p

    def dtC_k_p(self, n_p, n_m, k_p=None, k_m=None, delta_alpha=None):
        return 0.0*n_p

    def dtC_k_m(self, n_p, n_m, k_p=None, k_m=None, delta_alpha=None):
        return 0.0*n_p

    def E(self, n_p, n_m, k_p, k_m):
        r"""Express the energy in terms of :attr:`alpha_p` and
        :meth:`D`."""
        p = self._p(n_p=n_p, n_m=n_m)
        alpha_a = self.alpha_p(p)
        alpha_b = self.alpha_p(-p)
        alpha_p = (alpha_a + alpha_b)/2
        alpha_m = (alpha_a - alpha_b)/2
        return alpha_p*k_p + alpha_m*k_m + self.D(n_p=n_p, n_m=n_m)

    def dE_n_p(self, n_p, n_m, k_p, k_m):
        p = self._p(n_p=n_p, n_m=n_m)

        # d\alpha_a/dp, dalpha_b/d_p etc.
        dalpha_a = self.alpha_p(p, nu=1)
        dalpha_b = -self.alpha_p(-p, nu=1)
        dalpha_p = (dalpha_a + dalpha_b)/2
        dalpha_m = (dalpha_a - dalpha_b)/2


        # This is dp/dn_p*n_p to avoid problems when n_p is small.
        dp_pn_p = -p

        dalpha_p_p = dalpha_p*dp_pn_p/n_p
        dalpha_m_p = dalpha_m*dp_pn_p/n_p

        return (dalpha_p_p*k_p + dalpha_m_p*k_m 
                + self.dD_n_p(n_p=n_p, n_m=n_m))

    def dE_n_m(self, n_p, n_m, k_p, k_m):
        p = self._p(n_p=n_p, n_m=n_m)

        # dalpha_a/dp, dalpha_b/dp etc.
        dalpha_a = self.alpha_p(p, nu=1)
        dalpha_b = -self.alpha_p(-p, nu=1)
        dalpha_p = (dalpha_a + dalpha_b)/2
        dalpha_m = (dalpha_a - dalpha_b)/2

        # This is dp/dn_m*n_p to avoid problems when n_p is small.
        dp_mn_p = 1.0

        dalpha_p_m = dalpha_p*dp_mn_p/n_p
        dalpha_m_m = dalpha_m*dp_mn_p/n_p
        
        return (dalpha_p_m*k_p + dalpha_m_m*k_m 
                + self.dD_n_m(n_p=n_p, n_m=n_m))

    def dE_k_p(self, n_p, n_m, k_p, k_m):
        p = self._p(n_p=n_p, n_m=n_m)
        alpha_a = self.alpha_p(p)
        alpha_b = self.alpha_p(-p)
        alpha_p = (alpha_a + alpha_b)/2
        return alpha_p
        
    def dE_k_m(self, n_p, n_m, k_p, k_m):
        p = self._p(n_p=n_p, n_m=n_m)
        alpha_a = self.alpha_p(p)
        alpha_b = self.alpha_p(-p)
        alpha_m = (alpha_a - alpha_b)/2
        return alpha_m

    def plot(self):
        import pylab
        giorgini = np.array([[0.212,   0.867,  0.002],
                             [0.259,   0.851,  0.003],
                             [0.576,   0.853,  0.005],
                             [0.704,   0.883,  0.004],
                             [0.818,   0.950,  0.006],
                             [1.000,   1.118,  0.010]])
        soon_yong = np.array([[1.000, 0.8200, 0.026],
                              [0.947, 0.8822, 0.033],
                              [0.842, 0.8940, 0.043],
                              [0.737, 0.8480, 0.017],
                              [0.632, 0.8226, 0.0136],
                              [0.526, 0.7950, 0.017],
                              [0.421, 0.7780, 0.010],
                              [0.316, 0.7900, 0.007],
                              [0.211, 0.8370, 0.008],
                              [0.105, 0.8830, 0.004],
                              [0.000, 1.0000, 0.000]])
        def f(x):
            n_a = 1.0
            n_b = x*n_a
            n_p = n_a + n_b
            n_m = n_a - n_b
            alpha_p, alpha_m = self.alpha(n_p=n_p, n_m=n_m)
            return (alpha_a + x**(5./3.)*alpha_b
                    +
                    10.0/3.0/(6.0*_PI2)**(2./3.)*self.D(n_p=n_p, 
                                                        n_m=n_m)/n_a**(5./3.))

        x = np.linspace(0, 1, 1000)
        pylab.clf()
        pylab.subplot(221)
        # pylab.plot(x, f(x))
        # pylab.errorbar(giorgini[:, 0], giorgini[:, 1],
        #                giorgini[:, 2], fmt='g')
        # pylab.errorbar(soon_yong[:, 0], soon_yong[:, 1],
        #                soon_yong[:, 2], fmt='r')


        p = (1-x)/(1+x)
        g_s = self._G_p(p)**(3/5)*(1+p)/2
        pylab.plot(x, f(x)**(3./5.))
        pylab.plot(x, g_s,':')
        
        pylab.errorbar(giorgini[:, 0], giorgini[:, 1]**(3./5.),
                       giorgini[:, 2], fmt='g')
        pylab.errorbar(soon_yong[:, 0], soon_yong[:, 1]**(3./5.),
                       soon_yong[:, 2], fmt='r')

        pylab.subplot(222)
        x = np.linspace(0, 3, 1000)
        alpha_p, alpha_m = self.alpha(n_p=x + 1, n_m= x - 1)
        alpha_a = alpha_p + alpha_m
        alpha_b = alpha_p - alpha_m
        pylab.plot(x, 0*x + alpha_a)
        pylab.plot(x, 0*x + alpha_b)
        pylab.plot(x, self.D(n_p=x+1, n_m=x-1))
        pylab.plot(x, self.D(n_p=1+x, n_m=1-x))

        pylab.subplot(223)
        yFP = np.linspace(-1,self.y_0,100)
        hFP = self.h_y(yFP)
        LOFF = self.LOFF

        self.LOFF = False
        yN = np.linspace(self.y_0,self.y_c,100)
        hN = self.h_y(yN)
        ySF = np.linspace(self.y_c,1,100)
        hSF = self.h_y(ySF)

        self.LOFF = True
        yLO = np.linspace(self.y_LO_N,self.y_LO_SF,100)
        hLO = self.h_y(yLO)

        self.LOFF = LOFF

        ys = [yFP, yN, yLO, ySF]
        hs = [hFP, hN, hLO, hSF]

        fmts = ['b-', 'g-', 'r-', 'k-']
        for n, y in enumerate(ys):
            pylab.plot(y, hs[n], fmts[n])
        pylab.axis([-1,0.3,0.8,1.5])
        
        pylab.xlabel('y')
        pylab.ylabel('h')

    def _plots(self):
        r"""Generate plots for documentation."""
    
class ASLDAFunctionalFit(ASLDAFunctional):
    r"""ASLDA functional with parameters determined by fitting
    quasi-particle data and the energy of the normal polarized
    phase.

    The functional form is determined by the following procedure:

    1) Choose a parametrization for the effective masses.  We either
       use the data :attr:`fit_m_a_eff` and an interpolation or use a
       constant value.  The latter greatly simplifies the calculation
       and we have yet to explore how well the approximation
       compares.  To fix the effective mass to a constant value, set
       the parameter :attr:`constant_mass`.
    2) Determine the paring interaction parameter :math:`\gamma` and
       the self-interaction parameter :math:`\beta` for the superfluid
       phase.  Once the effective mass is fixed, this can be done by
       fitting the energy :attr:`xi_thermo` =
       :math:`=\xi=E/E_F=\mu/\epsilon_F` and the gap parameter
       :attr:`eta_thermo` = :math:`\eta=\Delta/\epsilon_{F}`.  If one wants
       to also find the effective mass, all three parameters can be
       fit from the quasiparticle dispersions :attr:`qp_disp`.
    3) Determine the self-energy interaction for the normal state by
       fitting the Monte-Carlo data for this :attr:`f` =
       :math:`E/E_F`.

    Notes
    -----
    Although the thermodynamic functions are best defined in terms of
    :math:`x = n_b/n_a`, it is not good to fit the data in terms of
    this asymmetric and unbounded coordinate.  Instead we use

    .. math::
       p = \frac{n_a - n_b}{n_b + n_a} = \frac{1 - x}{x + 1} \in [-1,1]

    In general we fit the functions with cubic splines in this
    variable.

    1) Fitting the mass:

       If we allow the mass to vary, we presently use the following
       three points: In a completely polarized gas, the majority
       species is unaffected, so :math:`\alpha_{p=1}^{-1} = 1` while
       the minority species has mass :math:`\alpha_{p=-1}^{-1} =
       1.20` from an essentially exact calculation.  Fitting the
       quasiparticle dispersions in the superfluid state to the FNGF
       Monte-Carlo  gives :math:`\alpha_{p=0}^{-1} = 0.914[14]`:
       
       >>> f = ASLDAFunctionalFit()
       >>> 1/f.alpha_p([-1,0,1])
       array([ 1.2 ,  0.91...,  1.   ])

       We fit the three points `(p, alpha(p))` with a parabola in the
       coordinate `p`:
       
       .. plot::
          :width: 50%
          :include-source:

          from slda.functionals.ASLDA import ASLDAFunctionalFit
          f = ASLDAFunctionalFit()
          p = np.linspace(-1, 1, 100)

          plt.plot(p, f.alpha_p(p))
          plt.errorbar(f.fit_m_a_eff[:,0],
                       1./f.fit_m_a_eff[:,1],
                       f.fit_m_a_eff[:,2]/f.fit_m_a_eff[:,1]**2, fmt='+')
          plt.xlabel(r'$$p = (n_a - n_b)/(n_a + n_b)$$')
          plt.ylabel(r'$$\alpha = m/m_{eff}$$')
          plt.axis([-1.02, 1.02, 0.5, 1.2])

       >>> f = ASLDAFunctionalFit()
       >>> res = f._get_alpha_p_results()
       >>> print("alpha_p = \n%s" % (res.poly,))
       alpha_p =
                6           5          4          3          2
       -0.1774 p + 0.03125 p + 0.5323 p - 0.1042 p - 0.5323 p + 0.1562 p + 1.094
       
    2) Fitting the superfluid:
    
       Once we fix the mass, the two parameters :math:`\gamma` and
       :math:`\beta` in the fully paired superfluid phase are fixed.
       If we also wish to determine the mass, we must fit the
       quasiparticle dispersion relationship.  We do this with a
       weighted least-squares fit to the BCS form for the three
       parameters :math:`\alpha`, :math:`\mu`, and :math:`\Delta`.

       .. math::
          \frac{E_k}{\epsilon_{F}} = \sqrt{\left[
             \alpha k^2 + \left(\frac{U}{\epsilon_{F}} - \xi\right)
          \right]^2 + \eta^2}

       The combination :math:`U/\epsilon_{F} - \xi` is a function of
       :math:`\eta = \Delta/\epsilon_{F}` through the equation for
       the total density :math:`n_{+} = k_{F}^3/3\pi^2`:

       .. math::
          \frac{1}{3\pi^2} = \int\dbar^{3}{k}\; \left\{1 - 
              \frac{k^2}{\sqrt{
                  \left[\alpha k^2 - \left(\frac{U}{\epsilon_{F}} - \xi\right)
                  \right]^2 + \eta^2}}\right\}

       The results of the full fit are:

       >>> res = ASLDAFunctionalFit()._get_symmetric_parameters();\
       ... print("Effective mass:     m = %f+-%f"%(res.m_thermo));\
       ... print("eta:              eta = %f+-%f"%(res.eta_thermo));\
       ... print("beta:            beta = %f+-%f"%(res.beta_thermo));\
       ... print("bbar=U/eF:       bbar = %f+-%f"%(res.betabar));\
       ... print("gammainv:    gammainv = %f+-%f"%(res.gammainv_thermo));\
       ... print("xi_N:            xi_N = %f+-%f"%(res.xi_N));\
       ... print("Quality of fit:     Q = %f"%(res.Q));\
       ... print("Reduced chi^2: chi2_r = %f"%(res.chi2_r))
       Effective mass:     m = 0.913989+-0.013816
       eta:              eta = 0.493292+-0.011756
       beta:            beta = -0.526257+-0.018012
       bbar=U/eF:       bbar = -0.491063+-0.018278
       gammainv:    gammainv = -0.090671+-0.007719
       xi_N:            xi_N = 0.567848+-0.024453
       Quality of fit:     Q = 0.520664
       Reduced chi^2: chi2_r = 1.075495

       If we fix the mass to unity, then we simply fit the two
       parameters :math:`\beta` and :math:`\gamma` to :math:`\xi` and
       :math:`\eta`:

       >>> res = ASLDAFunctionalFit(constant_mass=1)\
       ... ._get_symmetric_parameters();\
       ... print("eta:           eta = %f+-%f"%(res.eta_thermo));\
       ... print("beta:         beta = %f+-%f"%(res.beta_thermo));\
       ... print("bbar=U/eF:    bbar = %f+-%f"%(res.betabar));\
       ... print("gammainv: gammainv = %f+-%f"%(res.gammainv_thermo));\
       ... print("xi_N:         xi_N = %f+-%f"%(res.xi_N))
       eta:           eta = 0.504000+-0.024000
       beta:         beta = -0.402402+-0.023778
       bbar=U/eF:    bbar = -0.372056+-0.022489
       gammainv: gammainv = -0.074894+-0.010649
       xi_N:         xi_N = 0.597598+-0.023778

       .. plot::
          :width: 50%
          :include-source:
        
          from slda.functionals.ASLDA import ASLDAFunctionalFit
          f0 = ASLDAFunctionalFit(constant_mass=1)
          f1 = ASLDAFunctionalFit()

          # Plot qp data
          res = f1._get_symmetric_parameters()
          k2_kF2 = f1.fit_qp_disp[:,0]
          Eqp_eF = f1.fit_qp_disp[:,1]
          err = f1.fit_qp_disp[:,2]
          plt.errorbar(k2_kF2, Eqp_eF, err, fmt="k+")

          # Plot fits
          x = np.linspace(0, max(k2_kF2), 100)
          fmt = {f0: 'r--', f1: 'b-'}
          label = {f0: 'm=1', f1: 'm=1/alpha(p)'}
          for f in [f0, f1]:          
              # Plot data curve
              res = f._get_symmetric_parameters()
              Eqp = np.sqrt((x/res.m_thermo[0] + (res.betabar[0] - f.fit_xi[0]))**2 
                             + res.eta_thermo[0]**2)
              plt.plot(x, Eqp, fmt[f], label=label[f])

          x_min = x[np.argmin(Eqp)]
          plt.errorbar([x_min], [f.fit_eta[0]], [f.fit_eta[1]],
                       fmt='bo')

          plt.xlabel(r'k^2/k_F^2')
          plt.ylabel(r'E_k/\epsilon_F')

          plt.axis([-0.02,1.2,0.35,1.02])

       The blue curve is the fit including the mass as a parameter
       whereas the red dashed curve is the dispersion relationship
       obtained by holding the mass fixed to $m=1$ and fitting for the
       measured gap and energy.  The blue dot and associated error
       bars is the extracted values for the gap parameter $\eta$ from the fit
       (placed at the minimum of the dispersion relationship).

       The full fit is significant if the error bars on the dispersion
       points for low momenta can be trusted to include both
       systematic and statistical uncertainties, otherwise, the $m=1$
       curve fits the low-energy points well.

    3) Fitting the self-energy for the Normal state:

       Once the mass function has been parametrized, we can solve for
       the normal state energy density:

       .. math::
             \mathcal{E}_{N}[n_{a},n_{b}] &=
             \frac{\alpha_{a}(6\pi^2n_{a})^{5/3}}{20\pi^2}
             + \frac{\alpha_{b}(6\pi^2n_{b})^{5/3}}{20\pi^2} 
             + D
             = \frac{(6\pi^2)^{5/3}(n_+)^{5/3}}{20\pi^2}G(p),\\
             G(p) &= \left(\frac{1+p}{2}\right)^{5/3} f(x).

       We interpolate the function `G` because it is independent of
       the mass parameterization and is finite for all `p` (unlike
       `g(x)` which is convex or `f(x)` which are both infinite in
       domain and range).

       .. math::
          G(p) = \alpha(p)\left(\frac{1+p}{2}\right)^{5/3}
               + \alpha(-p)\left(\frac{1-p}{2}\right)^{5/3}
               + 2^{-2/3}\beta(x)
               = \left(\frac{1+p}{2}g(x)\right)^{5/3}
               = \left(\frac{1+p}{2}\right)^{5/3}f(x).

       We use the data $(x, f(x))$ from fixed-node Monte-Carlo
       calculations of the normal state, but we only use the data in
       the regions of high polarization.

       >>> res = ASLDAFunctionalFit()._get_G_p_results();\
       ... print("Fit:             G_p =\n%s" % (res.poly,));\
       ... print("Quality of fit:  Q = %f"%(res.Q))
       Fit:             G_p =
               2
       0.6425 p + 0.3575
       Quality of fit:  Q = 0.999...
       >>> res = ASLDAFunctionalFit(constant_mass=1)._get_G_p_results();\
       ... print("Fit:             G_p =\n%s" % (res.poly,));\
       ... print("Quality of fit:  Q = %f"%(res.Q))
       Fit:             G_p =
               2
       0.6423 p + 0.3577
       Quality of fit:  Q = 0.891259

       There is a slight difference in the fits and qualities when
       $m=1$ is held fixed because the point used in the SF phase
       differs slightly:

       >>> ASLDAFunctionalFit()._get_symmetric_parameters().xi_N
       (0.5678475448..., 0.02445287...)
       >>> ASLDAFunctionalFit(constant_mass=1)._get_symmetric_parameters().xi_N
       (0.5975982461..., 0.023778147...)

       .. plot::
          :width: 100%
          :include-source:

          from slda.functionals.ASLDA import ASLDAFunctionalFit          
          f0 = ASLDAFunctionalFit(constant_mass=1)
          f1 = ASLDAFunctionalFit()

          fig = plt.figure()
          fig.set_figwidth(2*fig.get_figwidth())
          plt.subplot(121)
          plt.subplot(122)

          # Plot Giorgini data
          x = f1.fit_f[:,0]
          f_x = f1.fit_f[:,1]
          err = f1.fit_f[:,2]
          plt.errorbar(x, f_x, err, fmt="+r")

          colour = {f0: 'b', f1:'g'}
          label = {f0: 'm=1', f1:'m=1/alpha(p)'}
          for f in [f0, f1]:
              # Plot data used to fit:
              res = f1._get_G_p_results()

              plt.subplot(122)
              plt.errorbar(res.x1, res.f1, res.f1_err,
                           fmt="+"+colour[f])

              style = '-'
              res = f._get_G_p_results()
              f.initialize()
              G_p = f._G_p
              pp = np.linspace(-1.0,1.0,100)
              xx = (1-pp)/(1+pp)

              # Plot all of Giorgini data
              x = f.fit_f[:,0]
              f_x = f.fit_f[:,1]
              f_err = f.fit_f[:,2]
              p = (1-x)/(1+x)
              fact = ((1+p)/2.)**(5./3.)
              G = f_x*fact
              G_err = f_err*fact
              plt.subplot(121)
              plt.errorbar(p, G, G_err, fmt="+r")

              # Plot data used for interpolation on top.
              plt.errorbar(res.p, res.G, res.E, fmt="+"+colour[f])

              # Plot curve
              plt.plot(pp, G_p(pp),style+colour[f],
                       label=label[f])

              plt.subplot(122)
              x = np.linspace(0, 1, 100)
              p = (1 - x)/(1 + x)
              fact = ((1 + p)/2.)**(5./3.)
              plt.plot(x, G_p(p)/fact, style+colour[f],
                       label=label[f])

          plt.subplot(122)
          plt.errorbar([1.0], [f.fit_xi[0]*2], [f.fit_xi[1]*2],
                       fmt='ok', label=r'$\xi$')
          f_BCS = (f.alpha_p(p) + f.alpha_p(-p)*x**(5./3.))
          plt.plot(x, f_BCS, 'y:', label='BCS')
          plt.plot([1.0], [0.590605507032838533783938*2], 'oy',
                   label=r'$$\xi_{BCS}$$')

          plt.subplot(121)
          plt.xlabel('$$p=(n_a - n_b)/(n_a + n_b)$$')
          plt.ylabel('G')

          plt.legend(loc='upper center')

          plt.subplot(122)
          plt.xlabel("$$x=n_b/n_a$$")
          plt.ylabel("$$f(p)=g^{5/3}(p)$$")
          plt.legend(loc='upper right')
          plt.axis([-0.02,1.02,0.75,1.5])

       The yellow curve corresponds to the BCS result for comparison.
       The circles at `x=1` correspond to the paired superfluid
       state.  The slight, almost imperceptible differences between
       the two fits is due to the slightly different value of $\xi_N$
       used as discussed above.  Since the errors in this point are
       large, the variance here does not really affect the fit.

    Examples
    --------
    >>> f = ASLDAFunctionalFit(constant_mass=0.8)
    >>> f.alpha_p(1.0), f.alpha_p(0.0), f.alpha_p(-1.0)      # 1./0.8
    (1.25, 1.25, 1.25)
    >>> f = ASLDAFunctionalFit()
    >>> print f.alpha_p(1.0)
    1.0
    >>> print f.alpha_p(0.0)      # 1./0.914
    1.094...
    >>> print f.alpha_p(-1.0)      # 1./1.20
    0.83333...

    This should give the polaron bound Y_0 ~ -0.58(1).  We don't fit
    this point yet, so we get the old value -0.54

    >>> (f._G_p(1) - 3./5*2*f._G_p(1, nu=1))/f._G_p(1)
    -0.54...
    """
    _state_vars = [
        ('no_dalpha', False,\
             "If true, then dalpha terms are ignored."),
        ('fit_xi', np.array([0.40, 0.01]),\
             """Energy of paired system wrt free system.  Used
             to define the superfluid part of the functional."""),
        ('fit_xi_N', np.array([1.118/2.0, 0.010/2.0]),\
             """Energy of interacting normal system wrt free
             system.  Used to define the effective mass.  Data here is
             from Giorgini"""),
        ('fit_eta', np.array([0.504, 0.024]),\
             """`delta/e_F` for symmetric superfluid phase. `[eta,
             err]`.  Use to determine the functional parameter C.
             Present value is from Carlson and Reddy."""),
        ('fit_m_a_eff', np.array([[-1.0, 1.20, 0.0], # Polaron
                                  #[0.0, 0.914, 0.014],  # SF.  We fit
                                  [1.0, 1.000, 1e-12],  # Free majority
                                  ]),
             r"""Array of points specifying the effective mass
             `m_a(p)`. Columns are `[p, m_a(p), err]`.  These are used to
             define the functional.  The default data are from Lobo et
             al."""),
        ('_fit_m_a_eff', Computed, 
             r"""This is the actual set of values used to fit.  It may
             include a fitted value of the superfluid state if this is
             excluded from :attr:`fit_m_a_eff`."""),
        ('constant_mass', None,\
             """A quick shortcut to enforce a constant mass: specify
             it here."""),
        ('fit_qp_disp', np.array([
            [0.0,        0.6*1.7055655,  0.6*0.043087971],
            [0.2525766,  0.6*1.3321364,  0.6*0.028725314],
            [0.50515321, 0.6*0.95152603, 0.6*0.028725314],
            [0.75772981, 0.6*0.83662478, 0.6*0.035906643],
            [1.0103064,  0.6*0.9018,     0.6*0.035906643]]),\
             r"""Array of points specifying the quasi-particle
             dispersion relationship in the symmetric superfluid
             phase.  Columns are `[k**2/k_F**2, E/e_{F}, err]`.  These
             are used to define the mass `m_eff(x=1)`.  Present data is
             from Carlson and Reddy."""),
        ('fit_f', np.array([[  0.000, 1.000, 1e-12, 1.0],
                            [  0.212, 0.867, 0.002, 1.0],
                            [  0.259, 0.851, 0.003, 1.0],
                            [  0.576, 0.853, 0.005, 0.0],
                            [  0.704, 0.883, 0.004, 0.0],
                            [  0.818, 0.950, 0.006, 0.0],
                            [  1.000, 1.118, 0.010, 0.0]]),\
             r"""Array of points specifying the self-energy
             `f(x) = E(x)/E_FG(x)` of the normal state. Columns are
             `[x, f(x), err]`. These are used to define the functional
             by interpolating the function `f(z)` where `z =
             (x-1)/(x+1)`.  The present data is from Giorgini et al.

             The last column contains a weighting factor.  The points
             beyond 0.5 are not used because there may be some
             contamination from the superfluid state in these numbers.
             The main constraint comes from the symmetric superfluid
             state properties."""),
        ('alpha_p', Computed),
        ('_G_p', Computed),
        ('_symmetric_parameters', Computed),
        ('gammainv_thermo', Computed)]
    process_vars()

    def __init__(self, *varargin, **kwargs):
        self._define_interpolations()
        if self.constant_mass is None:
            for na, nb, p in zip([0,1,1], [1,0,1], [-1,1,0]):
                ma = self._fit_m_a_eff[:,1][
                    np.where(self._fit_m_a_eff[:,0] == p)[0]][0]
                mb = self._fit_m_a_eff[:,1][
                    np.where(self._fit_m_a_eff[:,0] == -p)[0]][0]
            alpha_p, alpha_m = self.alpha(n_p=na + nb, n_m=na - nb)
            alpha_a = alpha_p + alpha_m
            alpha_b = alpha_p - alpha_m
            assert (abs(1.0/alpha_a - ma)<1e-12).all()
            assert (abs(1.0/alpha_b - mb)<1e-12).all()

        ASLDAFunctional.__init__(self, *varargin, **kwargs)

    def _get_symmetric_parameters(self, abs_tol=_ABS_TOL, rel_tol=_REL_TOL,
                                  _cache=[None, None]):
        r"""Return superfluid parameters by matching quasiparticle
        dispersions and xi_thermo.

        Examples
        --------
        >>> f1 = ASLDAFunctionalFit()
        >>> f2 = ASLDAFunctionalFit(constant_mass=True)
        >>> str(f1._symmetric_parameters) == str(f2._symmetric_parameters)
        True        
        """
        if (self.constant_mass is not None and 
            self.constant_mass is not True):
            # Accumulate the approximate errors in quadrature.
            m_thermo = self.constant_mass
            def f(x):
                xi_thermo, eta_thermo = x
                params = SSLDAFunctional.get_parameters(
                    m_thermo=m_thermo, xi_thermo=xi_thermo,
                    eta_thermo=eta_thermo,
                    abs_tol=abs_tol, rel_tol=rel_tol)
                betabar = xi_thermo - params.mu_0
                return np.array([params.beta_thermo, params.gammainv_thermo,
                                 betabar])
            x = np.array([self.fit_xi[0], self.fit_eta[0]])
            dx = np.array([self.fit_xi[1], self.fit_eta[1]])
            xp = x + dx
            xm = x - dx
            dy2 = 0
            for n in xrange(len(x)):
                x_ = x.copy()
                x_[n] = xp[n]
                yp = f(x_)
                x_[n] = xm[n]
                ym = f(x_)
                dy2 = dy2 + (yp - ym)**2/4

            beta_thermo, gammainv_thermo, betabar = f(x)
            dbeta_thermo, dgammainv_thermo, dbetabar = np.sqrt(dy2)
            xi_N = 1/m_thermo + beta_thermo
            dxi_N = dbeta_thermo

            ans = Container(m_thermo=(m_thermo, 0),
                            eta_thermo=tuple(self.fit_eta.tolist()),
                            beta_thermo=(beta_thermo, dbeta_thermo),
                            betabar=(betabar, dbetabar),
                            gammainv_thermo=(gammainv_thermo, dgammainv_thermo),
                            xi_N=(xi_N, dxi_N))
            return ans
        else:
            # Cache results for speed.
            try:
                if ((_cache[0][0] == self.fit_qp_disp).all() and
                    (_cache[0][1] == self.fit_xi).all() and
                    (_cache[0][2] == self.fit_eta).all()):
                    return _cache[1]
            except:
                pass
            def m_eff(xi_thermo=0.4):
                """Returns the inverse effective mass with errors from the
                specified parameters by fitting the quasiparticle
                spectrum."""

                # Add to the data the value of eta_thermo
                data_x = np.hstack((self.fit_qp_disp[:, 0], [inf]))
                data_y = np.hstack((self.fit_qp_disp[:, 1], [self.fit_eta[0]]))
                data_err = np.hstack((self.fit_qp_disp[:, 2], [self.fit_eta[1]]))

                def Eqp(k2_kF2, p):
                    """Return the energy of the quasiparticles as a function
                    of k2_kF2 = k^2/kF^2 in units of eF."""
                    m_thermo, eta_thermo = p
                    params = SSLDAFunctional.get_parameters(
                        m_thermo=m_thermo, xi_thermo=xi_thermo,
                        eta_thermo=eta_thermo,
                        abs_tol=abs_tol, rel_tol=rel_tol)

                    beta_thermo = params.beta_thermo
                    gammainv_thermo = params.gammainv_thermo
                    mu_0 = params.mu_0
                    assert abs((beta_thermo
                                - ((3*_PI2)**(2/3)/6*gammainv_thermo
                                   *eta_thermo**2))
                               - (xi_thermo - mu_0)) < 1e-11
                    # No factor of 2 in kinetic term because factor of 2
                    # in eF cancels.
                    ans = np.sqrt((k2_kF2/m_thermo - mu_0)**2 
                                  + abs(eta_thermo)**2)

                    # Special point to add eta_thermo constraint
                    ans = np.where(k2_kF2 == inf, eta_thermo, ans)
                    return ans
                [M, D], [dM, dD], Q, C = lsq.leastsq(Eqp,
                                                     [1.0, self.fit_eta[0]],
                                                     data_x, data_y,
                                                     data_err)
                chi2 = (((Eqp(data_x, [M,D]) -
                          data_y)/data_err)**2).sum()
                chi2_r = chi2/(len(data_x) - 2 - 1)

                return ([M, dM], [D, dD], Q, chi2_r, Eqp)
            xi_thermo = self.fit_xi[0]
            dxi_thermo = self.fit_xi[1]

            # The determination of the effective mass should not depend on
            # xi_thermo at all.
            if False:
                dm_dxi_thermo = (m_eff(xi_thermo+dxi_thermo)[0][0]
                          - m_eff(xi_thermo-dxi_thermo)[0][0])/2.0/dxi_thermo
                assert abs(dm_dxi_thermo) < 1e-12
                dD_dxi_thermo = (m_eff(xi_thermo+dxi_thermo)[1][0]
                          - m_eff(xi_thermo-dxi_thermo)[1][0])/2.0/dxi_thermo
                assert abs(dD_dxi_thermo) < 1e-12

            ([m_thermo, dm_thermo], 
             [eta_thermo, deta_thermo], Q, chi2_r, Eqp) = m_eff()

            dM = dm_thermo
            def beta_gammainv_thermo(m_thermo, xi_thermo, eta_thermo):
                params = SSLDAFunctional.\
                    get_parameters(m_thermo=m_thermo, xi_thermo=xi_thermo,
                                   eta_thermo=eta_thermo, 
                                   abs_tol=abs_tol, rel_tol=rel_tol)
                return np.array([params.beta_thermo, params.gammainv_thermo])

            beta_thermo, gammainv_thermo = beta_gammainv_thermo(
                m_thermo, xi_thermo, eta_thermo)
            dbeta_gammainv_thermo_dm = (
                beta_gammainv_thermo(m_thermo + dM, xi_thermo, eta_thermo)
                -beta_gammainv_thermo(m_thermo - dM, xi_thermo, eta_thermo)
                )/2.0/dM
            dbeta_gammainv_thermo_dxi_thermo = (
                beta_gammainv_thermo(m_thermo, xi_thermo + dxi_thermo,
                                     eta_thermo)
                -beta_gammainv_thermo(m_thermo, xi_thermo - dxi_thermo,
                                      eta_thermo))/2.0/dxi_thermo
            dbeta_gammainv_thermo_dm = (
                beta_gammainv_thermo(m_thermo, xi_thermo, 
                                     eta_thermo + deta_thermo)
                -beta_gammainv_thermo(m_thermo, xi_thermo, 
                                      eta_thermo - deta_thermo)
                )/2.0/deta_thermo

            dbeta_thermo, dgammainv_thermo = np.sqrt(
                (dbeta_gammainv_thermo_dm*dm_thermo)**2 
                + (dbeta_gammainv_thermo_dxi_thermo*dxi_thermo)**2 
                + (dbeta_gammainv_thermo_dm*dm_thermo)**2)
            xi_N = 1./m_thermo + beta_thermo
            dxi_N = np.sqrt((dM/m_thermo/m_thermo)**2 + dbeta_thermo**2)

            betabar = (beta_thermo
                       - (3*_PI2)**(2./3.)/6*gammainv_thermo*eta_thermo**2)
            dbetabar = np.sqrt(dbeta_thermo**2 + 
                               (betabar - beta_thermo)**2*
                               ((dgammainv_thermo/gammainv_thermo)**2 
                                + (deta_thermo/eta_thermo)**2))
            ans = Container(m_thermo=(m_thermo, dM),
                            eta_thermo=(eta_thermo, deta_thermo),
                            beta_thermo=(beta_thermo, dbeta_thermo),
                            betabar=(betabar, dbetabar),
                            gammainv_thermo=(gammainv_thermo, dgammainv_thermo),
                            xi_N=(xi_N, dxi_N),
                            Q=Q, 
                            chi2_r=chi2_r,
                            Eqp=Eqp)
            _cache[0] = deepcopy([self.fit_qp_disp, self.fit_xi, self.fit_eta])
            _cache[1] = ans
            return ans

    def get_v(self, n_p, n_m, k_p, k_m, delta_alpha):
        r"""Optimized version."""
        _optimized = _OPTIMIZED
        if not _optimized:
            return UnitaryFunctional.get_v(self, n_p=n_p, n_m=n_m,
                                           k_p=k_p, k_m=k_m,
                                           delta_alpha=delta_alpha)
        
        mesg_threshold = 20

        # Optimized version
        n_p = abs(n_p) + _TINY
        n_m = np.minimum(np.maximum(-n_p, n_m), n_p)
        p = self._p(n_p=n_p, n_m=n_m)
        
        alpha_a = self.alpha_p(p)
        alpha_b = self.alpha_p(-p)

        dalpha_a = self.alpha_p(p, nu=1)
        dalpha_b = -self.alpha_p(-p, nu=1)
        
        # These are dp/dn_a*n_p and dp/dn_b*n_p to avoid problems
        # when n_p is small.
        dp_an_p = -(p - 1)
        dp_bn_p = -(p + 1)

        dalpha_a_a = dalpha_a*dp_an_p/n_p
        dalpha_a_b = dalpha_a*dp_bn_p/n_p
        dalpha_b_a = dalpha_b*dp_an_p/n_p
        dalpha_b_b = dalpha_b*dp_bn_p/n_p

        Q_a = alpha_a*((1 + p)/2)**(5/3)
        Q_b = alpha_b*((1 - p)/2)**(5/3)

        dQ_a_p = (dalpha_a*(1 + p)**(5/3)
                  + 5/3*alpha_a*(1 + p)**(2/3))/2**(5/3)

        dQ_b_p = (dalpha_b*(1 - p)**(5/3)
                  - 5/3*alpha_b*(1 - p)**(2/3))/2**(5/3)

        G_p = self._G_p(p)
        G_m = self._G_p(-p)

        dG_p_p = self._G_p(p, nu=1)
        dG_m_p = -self._G_p(-p, nu=1)

        beta_p = 2**(2/3)*((G_p + G_m)/2 - Q_a - Q_b)
        dbeta_p = 2**(2/3)*((dG_p_p + dG_m_p)/2 - dQ_a_p - dQ_b_p)

        # This is A/n_p and D/n_p to avoid some problems when n_p
        # is small.
        A_n_p = (3*_PI2)**(5/3)/10/_PI2*n_p**(2/3)
        D_n_p = A_n_p*beta_p

        dD_a = 5*D_n_p/3 + np.where(p < 1 , A_n_p*dbeta_p*dp_an_p, 0)
        dD_b = 5*D_n_p/3 + np.where(p > -1, A_n_p*dbeta_p*dp_bn_p, 0)
            
        C = self.gammainv_thermo*n_p**(1/3)
        dC_a = dC_b = C/3/n_p

        alpha_p = (alpha_a + alpha_b)/2.0
        delta = delta_alpha*alpha_p
        #alpha_m = (alpha_a - alpha_b)/2.0

        delta2 = abs(delta)**2

        v_a = (dD_a - delta2*dC_a)
        v_b = (dD_b - delta2*dC_b)

        if not self.no_dalpha or mesg_threshold <= self.verbosity:
            dalpha_p_a = (dalpha_a_a + dalpha_b_a)/2.0
            dalpha_m_a = (dalpha_a_a - dalpha_b_a)/2.0
            dalpha_p_b = (dalpha_a_b + dalpha_b_b)/2.0
            dalpha_m_b = (dalpha_a_b - dalpha_b_b)/2.0
            dv_a = (dalpha_p_a*k_p + 
                    dalpha_m_a*k_m -
                    dalpha_p_a/alpha_p*delta2*C)
            dv_b = (dalpha_p_b*k_p + 
                    dalpha_m_b*k_m -
                    dalpha_p_b/alpha_p*delta2*C)

        if not self.no_dalpha:
            # This step removes spurious tails
            _a = mmf.math.step(n_p, self.n_unit_mass, self.n_unit_mass/2)
            v_a += _a*dv_a
            v_b += _a*dv_b

        if mesg_threshold <= self.verbosity:
            err_a = abs(dv_a/v_a)
            err_b = abs(dv_b/v_b)
            max_err = max(np.hstack([err_a, err_b]))
            print("Maximum dalpha contribution: %f%%"%(100*max_err))

        v_p = (v_a + v_b)/2
        v_m = (v_a - v_b)/2
        return v_p, v_m


    def _get_alpha_p_results(self):
        r"""Helper function to determine the `alpha_p` interpolation.

        Examples
        --------
        >>> f1 = ASLDAFunctionalFit()
        >>> f2 = ASLDAFunctionalFit(constant_mass=True)
        >>> np.allclose(f1.alpha_p(0) - f2.alpha_p(0), 1e-12)
        True
        """
        p = np.array(self.fit_m_a_eff[:, 0])
        m = np.array(self.fit_m_a_eff[:, 1])
        m_err = np.array(self.fit_m_a_eff[:, 2])
        if not (p == 0.0).any():
            p = np.hstack([[0.0], p])
            inds = p.argsort()
            p = p[inds]
            m = np.hstack([[self._symmetric_parameters.m_thermo[0]],
                           m])[inds]
            m_err = np.hstack([[self._symmetric_parameters.m_thermo[1]],
                               m_err])[inds]
        if self.constant_mass is not None:
            if self.constant_mass is not True:
                m[:] = self.constant_mass
                m_err[:] = 0.0
            else:
                m[:] = self._symmetric_parameters.m_thermo[0]
                m_err[:] = self._symmetric_parameters.m_thermo[1]

        if self.constant_mass is not None:
            def constant_alpha(p, nu=0):
                if nu > 0:
                    return 0.0
                else:
                    return 1./m[0]
            y = y_err = a = C = Q = F = poly = None
            alpha_p = constant_alpha
        else:
            y = 1/m
            y_err = abs(m_err/m)*y

            a = C = Q = F = None
            if False:
                # This is a least squares fit, but since there are
                # only three points, a polynomial fit suffices.
                a, C, Q, F = lsq.lsqf([lambda x:1, lambda x:x, lambda x:x*x],
                                      p, y, y_err)
                poly = [a[2], a[1], a[0]]
            elif False:
                # Simple quadratic fit to data:
                poly = np.polyfit(p, y, 2)
            elif False:
                # Least squares fit including zero derivative
                # constraints at the ends to improve numerics.
                p = np.asarray(p)
                _A = np.vstack([p*0 + 1,
                                p*(1-p*p/3),
                                p*p*(1-p*p/2)])
                a = np.linalg.solve(_A.T, y)
                #print a
                #a, C, Q, F = lsq.lsqf([lambda x:1, 
                #                       lambda x:x*(1-x*x/3), 
                #                       lambda x:x*x*(1-x*x/2)],
                #                      p, y, 1e-12*y_err)
                poly = [-a[2]/2, -a[1]/3, a[2], a[1], a[0]]
            else:
                # Least squares fit including zero derivative and
                # second derivative constraints at the ends to improve
                # numerics.
                p = np.asarray(p)
                _A = np.vstack([p*0 + 1,
                                p*(1 - p*p*(2/3 - p*p/5)),
                                p*p*(1-p*p*(1 - p*p/3))])
                a = np.linalg.solve(_A.T, y)
                poly = [a[2]/3, a[1]/5, -a[2], -2*a[1]/3, a[2], a[1], a[0]]

                
            def alpha_p(p, nu=0):
                r"""Interpolated quadratic polynomial approximation for
                `alpha(p)`."""
                return np.polyval(np.polyder(poly, nu), p)

        return Container(p=p, a=a, alpha=y, err=y_err, m=m, m_err=m_err,
                         Q=Q, F=F, poly=np.poly1d(poly, variable='p'),
                         C=C, alpha_p=alpha_p)

            #self.alpha_p = mmf.math.interp.UnivariateSpline(
            #    x=p, y=y, k=2, w=1/y_err, s=None)

    def _define_alpha_p(self):
        r"""Define the interpolating functions for the inverse
        effective mass `alpha_a(p)`.

        If the flag :attr:`constant_mass` is set to a value, then this is
        used instead.

        Examples
        --------
        >>> from mmf.math.differentiate import differentiate
        >>> diff = np.vectorize(lambda f, x: differentiate(f, x, h0=0.1))
        >>> f = ASLDAFunctionalFit()
        >>> p = np.linspace(-1.0, 1.0, 11)
        >>> (abs(diff(f.alpha_p, p) - f.alpha_p(p, nu=1)) < 2e-12).all()
        True

        >>> f = ASLDAFunctionalFit(constant_mass=0.8)
        >>> f.alpha(n_p=1, n_m=0)
        (1.25, 0.0)
        """
        res = self._get_alpha_p_results()
        self._fit_m_a_eff = np.vstack([res.p, res.m, res.m_err]).T
        self.alpha_p = res.alpha_p
            
    def _get_G_p_results(self):
        r"""Helper function to compute results related to the `G_p`
        interpolation."""
        x = self.fit_f[:, 0]
        f = self.fit_f[:, 1]
        f_err = self.fit_f[:, 2]/self.fit_f[:, 3]

        # Select only data with finite weights
        inds = np.where(f_err < inf)
        x1 = x[inds]
        f1 = f[inds]
        f1_err = f_err[inds]

        # Add symmetric point
        xi_N, dxi_N = self._get_symmetric_parameters().xi_N
        x1 = np.hstack([x1, 1.0])
        f1 = np.hstack([f1, 2*xi_N])
        f1_err = np.hstack([f1_err, 2*dxi_N])
        p1 = np.divide(1 - x1, 1 + x1)

        inds = np.argsort(p1)
        x1 = x1[inds]
        p1 = p1[inds]
        f1 = f1[inds]
        f1_err = f1_err[inds]

        G1 = f1*np.power((1 + p1)/2, 5/3)
        E1 = f1_err*np.power((1 + p1)/2, 5/3)

        # Add flipped points (assume last point is z=0.0 so we exclude
        # this)
        inds = np.where(p1 > 0.0)
        p2 = -np.flipud(p1[inds])
        G2 = np.flipud(G1[inds])
        E2 = np.flipud(E1[inds])

        p = np.hstack([p2, p1])
        G = np.hstack([G2, G1])
        E = np.hstack([E2, E1])

        # Now do a polynomial fit to an even polynomial
        a, C, Q, F = lsq.lsqf([lambda x:1, lambda x:x*x],
                              p, G, E)
        poly = [a[1], 0, a[0]]
        def G_p(p, nu=0):
            r"""Interpolated quadratic polynomial approximation for
            `G(p)`."""
            return np.polyval(np.polyder(poly, nu), p)

        return Container(p=p, G=G, E=E, x1=x1, f1=f1, f1_err=f1_err,
                         Q=Q, F=F, a=a, C=C, G_p=G_p,
                         poly=np.poly1d(poly, variable='p'))

    def _define_G_p(self):
        r"""Define the interpolating functions for the self energy
        corrections.

        Examples
        --------
        >>> from mmf.math.differentiate import differentiate
        >>> diff = np.vectorize(lambda f,
        ...                     x: differentiate(f, x, h0=0.01, nmax=50))
        >>> f = ASLDAFunctionalFit()
        >>> p = np.linspace(-1.0, 1.0, 11)
        >>> ((abs(diff(f._G_p, p) - f._G_p(p, nu=1))) < 5e-11).all()
        True

        >>> print f._G_p(-1.0)
        1.0...
        >>> f._G_p(0.0)
        0.357...
        >>> print f._G_p(1.0)
        1.0...
        >>> np.allclose(f._G_p(0.5), f._G_p(-0.5))
        True

        >>> from mmf.math.differentiate import differentiate
        >>> diff = np.vectorize(lambda f, p: differentiate(f, p, h0=0.05))
        >>> p = np.linspace(-0.9, 0.9, 11)
        >>> np.allclose(diff(f._G_p, p), f._G_p(p, nu=1))
        True
        """
        res = self._get_G_p_results()
        self._G_p = res.G_p
        # self._G_p = mmf.math.interp.EvenUnivariateSpline(
        #    x=res.p, y=res.G, w=1/res.E, s=None)

    def _define_interpolations(self):
        r"""Define the interpolating functions for the inverse
        effective mass alpha_a(x) etc."""
        self._symmetric_parameters = self._get_symmetric_parameters()
        self._define_alpha_p()
        self._define_G_p()
        self.gammainv_thermo = self._symmetric_parameters.gammainv_thermo[0]
