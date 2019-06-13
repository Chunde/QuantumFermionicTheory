import numpy as xp
import numpy
xp

def allclose(a, b, rtol=1e-5):
    if xp==numpy:
        return numpy.allclose(a,b,rtol=rtol)
    return xp.allclose(a.get(), b.get())