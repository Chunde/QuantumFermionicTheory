# Units
class Units(object):
    hbar = 1.0
    micron = 1.0
    mm = 1e3*micron
    cm = 1e4*micron
    nm = 1e-3*micron
    meter = 1e3*mm

    u = 1.0                        # AMU
    kg = u/1.660539040e-27
    G = 1.0                        # Gauss

    m = 86.909187*u
    a = 100.40*nm                  # |F=1, m=-1>
    Hz = hbar/micron**2/u/63507.799258914903398
    kHz = 1e3*Hz
    s = 1./Hz
    ms = 1e-3*s