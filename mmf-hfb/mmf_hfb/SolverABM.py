import numpy as np
import math
from collections import namedtuple


def axpy(y, x, a):
    # generalized vector addition
    return y + a*x


class EvolverABM(object):

    def __init__(
            self, y, dt, dy_dt, t=None,
            normalize=True, no_runge_kutta=False,
            history_step=10, **kw):
        self.y = y
        self.N = math.sqrt((y.conj().dot(y)).real)
        self.t = t if t is not None else 0
        self.dt = dt
        self.dy_dt = dy_dt
        self.normalize = normalize
        self.no_runge_kutta = no_runge_kutta
        self.ys = None
        self.history_step = history_step
        self.replay_ys = []
        self.replay_ts = []
        self.dcps = None
        self.dys = None
        self.init()

    def _add_replay(self):
        r"""save current time and y"""
        self.replay_ts.append(self.t)
        self.replay_ys.append(self.y)

    def get_y(self):
        r"""Return a copy of the current state `y`."""
        return self.y.copy()

    def get_dy(self, y=None, t=None, dy=None):
        r"""return dy/dt"""
        return self.dy_dt(psi=y, t=t, dy=dy)

    def evolve(self, steps=None):
        r"""Evolve the system by `steps`."""
        self.steps = 0
        t0 = float(self.t)
        assert steps > 1
        self.do_step(first=True)
        for _ in range(1, steps-1):
            self.do_step()
        self.do_step(final=True)
        assert np.allclose(self.t, t0 + steps * self.dt)
        if self.replay_ts[-1] != self.t:
            self._add_replay()
        return self.y
        
    def init(self):
        y0 = self.y
        dt = self.dt
        if self.no_runge_kutta:
            self.ys = [y0, y0.copy()]
            self.dcps = [_y*(161/170*0) for _y in self.ys]
            self.dys = [_y*0 for _y in [y0]*4]
        else:
            self.ys = [y0]
            self.dcps = [161/170*0*y0]
            self.dys = []
        self._add_replay()
        # Coefficients for the ABM method
        h = dt
        self._ap = h/48*np.array([119, -99, 69, -17], dtype=float)
        _tmp = h * 161./48./170.
        self._am = _tmp*(17)
        self._ac = _tmp*np.array([-68, 102, -68, 17], dtype=float)

    def do_step(self, first=None, final=None):
        if len(self.dys) < 4:
            self.do_step_runge_kutta()
            self.ys = self.ys[:2]            # Only keep two previous steps
            if len(self.dys) == 4:
                # Only allocate these here.  Not exactly sure what
                # values to use.
                self.dcps = [0*_y for _y in self.ys]
        else:
            self.do_step_ABM()
        if self.normalize:
            self.y = self.N*self.y/math.sqrt((self.y.conj().dot(self.y)).real)
        self.steps += 1
        if self.steps % self.history_step == 0:
            self._add_replay()
        
    def do_step_runge_kutta(self):
        r"""4th order Runge Kutta for the first four steps to populate the
        predictor/corrector arrays."""
        t = float(self.t)
        h = self.dt
        ys = self.ys
        dys = self.dys

        y = self.ys[0].copy()
        if len(self.dys) < len(self.ys):
            dy = self.get_dy(y=y)
            dys.insert(0, dy)
        else:
            dy = self.dys[0]

        f0 = dy
        y = axpy(y, dy, h/2.)
        f1 = self.get_dy(y, t=t + h/2.)
        y = axpy(y, dy, -h/2.)
        y = axpy(y, f1, h/2.)
        f2 = self.get_dy(y, t=t + h/2.)
        y = axpy(y, f1, -h/2.)
        y = axpy(y, f2, h)
        f1 = axpy(f1, f2, -2.)
        f3 = self.get_dy(y, dy=f2, t=t + h)
        y = axpy(y, f1, h/3.)
        y = axpy(y, f0, h/6.)
        y = axpy(y, f3, h/6.)
        del f0, f1, f2, f3
        self.t += h
        dy = self.get_dy(y=y)
        ys.insert(0, y)
        dys.insert(0, dy)

    def do_step_ABM(self):
        r"""Perform one step of the ABM method."""
        t = float(self.t)
        dt = self.dt
        ys = self.ys            # Slightly faster to make these local
        dcps = self.dcps
        dys = self.dys
        y = ys.pop()
        y *= 0.5
        y = axpy(y, x=ys[0], a=0.5)
        for _i in range(4):
            y = axpy(y, x=dys[_i], a=self._ap[_i])
        y = axpy(y, x=dcps[0], a=1)
        dcp = dcps.pop()
        # Compute m' in next dcp array, then update
        dcp = self.get_dy(y=y, t=t+dt, dy=dcp)
        dcp *= self._am
        for _i in range(4):
            dcp = axpy(dcp, x=dys[_i], a=self._ac[_i])
        y = axpy(y, x=dcp, a=1)
        y = axpy(y, x=dcps[0], a=-1)
        self.t += dt
        dy = dys.pop()
        dy = self.get_dy(y=y, dy=dy)
        ys.insert(0, y)
        dys.insert(0, dy)
        dcps.insert(0, dcp)
        self.y = y
        

def ABMEvolverAdapter(fun, t_span, dt, y0, beta_t=0.1, history_step=100, **args):
    """
    An adapter function used to make ABMEvolver be compatible
    with the ivp_solver convention
    """
    success = True
    dt = beta_t*dt
    total_step = int(t_span[1]/dt)
    if history_step > total_step:
        history_step = total_step
    e = EvolverABM(y=y0, dt=dt, dy_dt=fun, history_step=history_step, **args)
    e.evolve(steps=total_step)
    res = namedtuple('res', ['success', 't', 'y', 'nfev'])
    return res(success=success, t=e.replay_ts, y=np.array(e.replay_ys).T, nfev=total_step)