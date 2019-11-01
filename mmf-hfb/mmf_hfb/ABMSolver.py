from pytimeode.evolvers import EvolverSplit, EvolverABM
from collections import namedtuple


def ABMEvolver(fun, t_span, dt, y0, history_step=10, **args):
    success = True
    t = [0]
    y = [y0]
    total_step = int(t_span[1]/dt)
    if history_step > total_step:
        history_step = total_step
    data_N = total_step//history_step
    y=y0
    for _ in range(data_N):
        e = EvolverABM(y, dt=dt, **args)

    res = namedtuple(
                'res', ['success', 't', 'y'])
    return res(success=success, t=t, y=y)