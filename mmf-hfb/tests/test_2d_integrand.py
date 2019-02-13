"""Test the 2d integ code."""
import numpy as np

from scipy.optimize import brentq

import pytest

from mmfutils.testing import allclose

from mmf_hfb import bcs, homogeneous
from mmf_hfb.FuldeFerrelState import FFState


@pytest.fixture(params=[3, 3.5, 4])
def r(request):
    return request.param

def test_compute_delta_n(r, d=2 ,mu=10, dmu=0.4):
    # return (1,2,3) # for quick debug
    ff = FFState(dmu=dmu, mu=mu, d=d)
    ds = np.linspace(0,1.5,10)
    fs = [ff.f(delta=delta, r=r, mu_a=mu+dmu, mu_b=mu-dmu) for delta in ds]
    index, value = min_index(fs)
    delta = 0
    if value < 0:
        delta = ff.solve(r=r,a= ds[index], mu_a=mu+dmu, mu_b=mu-dmu)
        if fs[0] > 0:
            smaller_delta = ff.solve(r=r,a=ds[0],b=ds[index], mu_a=mu+dmu, mu_b=mu-dmu)
            print(f"a smaller delta={smaller_delta} is found for r={r}")
            p1 = ff.get_pressure(delta=delta,r=r, mu_a=mu+dmu, mu_b=mu-dmu)
            p2 = ff.get_pressure(delta=smaller_delta, r=r, mu_a=mu+dmu, mu_b=mu-dmu)
            if(p2 > p1):
                delta = smaller_delta
    na,nb = ff.get_densities(delta=delta, r=r, mu_a=mu+dmu, mu_b=mu-dmu)
    return (delta, na, nb)

if __name__ == "__main__":
    test_compute_delta_n(3.6)