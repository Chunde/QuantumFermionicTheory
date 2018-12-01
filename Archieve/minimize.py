"""Robust GPE minimizer"""
import numpy as np

import scipy.optimize
sp = scipy


class Minimize(object):
    def __init__(self,
                 state,
                 real=False,
                 x_scale=1.0,  E_scale=1.0):
        """
        state : IState
           Initial state.  In addition to the methods provided by the IState
           interface, this class uses the following special functions:

           `state.get_N()` : Returns the particle number of the state.  Used
              for normalization.
           `state.bcast` : Used to broadcast scaling factors etc. for the
              particle numbers so that they can scale the whole state.
           `state.get_Hy(subtract_mu)` : Return the Hamiltonian applied to the
              state.
           `state.metric`
           `state.get_energy()` : Return the energy to be minimized.
           `state.plot()` : Plot the state (for interactive used and debugging.)

        real : bool
           If `True`, then the solver will only consider real ansatz,
           assuming that the ground state is real, or that any phases are
           exactly provided by `P`.  Note: if the actual state should also
           be real (i.e. if `P` is also real) then this should be
           indicated in `state.dtype`.

        To Do
        -----
        - Check memory usage and avoid unecessary copies.
        """
        self.state = state
        self.x_scale = x_scale
        self.E_scale = E_scale
        self.real = real
        self.init()

    def init(self):
        self._x_dtype = float if self.real else complex

    def unpack(self, x, state):
        """Unpack `x` into `state` including factors of x_scale

        Note: Do not scale or otherwise mutate `x`.
        """
        x = np.asarray(x, dtype=float).view(self._x_dtype)

        state[...] = x.reshape(state[...].shape)

        state *= self.x_scale
        return state

    def pack(self, psi):
        """This is not symmetric with `unpack` as it does not scale
        the solution by any factors.  (This is because we unpack both
        states and Hpsi which have different scalings.)"""
        psi = np.asarray(psi, self.state.dtype).ravel()

        if self.real:
            assert np.allclose(psi.imag, 0)
            psi = psi.real

        return psi.view(dtype=float).ravel()

    def minimize(self,
                 fix_N=True,
                 plot=False,
                 _test=False,
                 _debug=False,
                 callback=None,
                 method='L-BFGS-B',
                 **kw):
        """Return the state with minimized energy.

        Developer notes:

        1. Do not mutate `state0`.  Use `state` as a working copy if a
           true state is needed.
        2. Be careful which version of `state` or `state0` you use as the
           argument to `get_Hy` and `get_energy` since this determines the
           non-linear portions.  In general one should use `state`, but
           if one implements a minimization on the norm of `Hy` then for
           the derivative one must be careful.

        """
        state0 = self.state
        state = self.state.copy()
        N = np.asarray(state.get_N())

        if _debug:
            return locals()
        psi = state[...]

        x0 = self.pack(psi)/self.x_scale
        state = self.unpack(x0, state)

        def f_df(x, state=state):
            """Computes both f and df since these are always called in pairs."""
            state = self.unpack(x, state)

            if fix_N:
                s = np.sqrt(N/state.get_N())
                state *= s

            Hpsi = state.get_Hy(subtract_mu=fix_N)

            if fix_N:
                Hpsi *= s

            Hpsi *= 2.0 * state.metric  # for real x rather than psi

            E = state.get_energy()
            if plot:
                state.plot()

            f = E/self.E_scale
            df = self.pack(Hpsi)
            df *= self.x_scale/self.E_scale
            return f, df

        _cache = [0, None, None]

        def _f(x):
            if not np.allclose(x, _cache[0]):
                _cache[0] = x.copy()
                _cache[1:] = f_df(x)
            return _cache[1]

        def _df(x):
            if not np.allclose(x, _cache[0]):
                _cache[0] = x.copy()
                _cache[1:] = f_df(x)
            return _cache[2]

        if _test:  # Check derivative
            assert self.check_derivative(f=_f, df=_df, x=x0)

        _x = [x0]
        options = dict(disp=0)
        options.update(kw)

        callback_ = None
        if callback is not None:
            def callback_(x):
                if _test:  # Check derivative
                    self.check_derivative(f=_f, df=_df, x=x0)
                state = state0.copy()
                state = self.unpack(x, state)
                if fix_N:
                    state *= np.sqrt(N/state.get_N())
                return callback(state)

        res = sp.optimize.minimize(fun=_f,
                                   jac=_df,
                                   x0=_x[0],
                                   method=method,
                                   callback=callback_,
                                   options=options)
        res.f = _f
        res.df = _df
        state.minimize_res = res

        if not res.success:
            warnings.warn(res.message)

        _x[0] = res.x
        state = self.unpack(_x[0], state)

        if fix_N:
            state *= np.sqrt(N/state.get_N())

        state.minimize_res = res
        return state, res.x, _f, _df

    def check_derivative(self, f, df, x, rtol=1e-4):
        from mmfutils.math.differentiate import differentiate

        np.random.seed(11)
        dx = (np.random.random(x.shape) - 0.1)
        dx /= np.linalg.norm(dx)

        def _f(h):
            return f(x + dx*h)

        d_f = differentiate(_f, h0=0.1)
        _f_x = f(x)             # Do this to populate f_df cache
        _df_x = df(x)
        _df_x_ = np.dot(_df, dx)
        return np.allclose(_df_x, _df_x_, rtol=rtol)
