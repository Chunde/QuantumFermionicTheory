from mmf_hfb.BCSCooling import BCSCooling
import matplotlib.pyplot as plt
import numpy as np
def Normalize(psi):
    return psi/psi.dot(psi.conj())**0.5

def Prob(psi):
    return np.abs(psi)**2

def test_derivative_cooling():
    args = dict(N=128, dx=0.1, beta_0=1, divs=(1, 1), beta_K=0, beta_V=0, beta_D=1)
    b = BCSCooling(**args)
    k0 = 2*np.pi/b.L
    x = b.xyz[0]
    V = 0  # x**2/2
    H0 = b._get_H(mu_eff=0, V=0)
    H1 = b._get_H(mu_eff=0, V=V)
    U0, E0 = b.get_U_E(H0, transpose=True)
    U1, E1 = b.get_U_E(H1, transpose=True)
    psi_1 = Normalize(np.cos(k0*x))
    assert np.allclose(Prob(psi_1), Prob(U0[1]))
    assert np.allclose(E0[1], k0**2/2.0)


    psi_0 = U0[1]
    ts, psis = s.solve([psi_0], T=10, rtol=1e-5, atol=1e-6, V=V, method='BDF')
    psi0 = U1[0]
    E0, _ = s.get_E_Ns([psi0], V=V)
    Es = [s.get_E_Ns([_psi], V=V)[0] for _psi in psis[0]]
    plt.subplot(121)
    plt.plot(ts[0][:-2], (Es[:-2] - E0)/abs(E0))
    plt.subplot(122)
    l, = plt.plot(x, psi0)  # ground state
    plt.plot(x, psis[0][0], "+", c=l.get_c())
    plt.plot(x, psis[0][-1], '--', c=l.get_c())
    plt.show()


if __name__ == "__main__":
    test_derivative_cooling()