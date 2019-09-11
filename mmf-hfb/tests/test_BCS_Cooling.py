from mmf_hfb.BCSCooling import BCSCooling
import matplotlib.pyplot as plt


def get_V(x):
    return x**2/2


def PlayCooling(bcs, psis0, psis, V=None, N_data=10, N_step=100, **kw):
    x = bcs.xyz[0]
    if V is None:
        V = get_V(x)
    E0, _ = bcs.get_E_Ns(psis0, V=V)
    Es, cs= [], []
    for _n in range(N_data):
        psis = bcs.step(psis, V=V, n=N_step)
        # assert np.allclose(psis[0].dot(psis[1].conj()), 0)
        E, _ = bcs.get_E_Ns(psis, V=V)
        Es.append(abs(E - E0)/E0)
        for psi in psis:
            ax, = plt.plot(x, abs(psi)**2)
            cs.append(ax.get_c())
        for i, psi in enumerate(psis0):
            plt.plot(x, abs(psi)**2, '+', c=cs[i])
        # for i, psi in enumerate(psis):
        #    dpsi = bcs.Del(psi, n=1)
        #   plt.plot(x, abs(dpsi)**2,'--', c=cs[i])
        plt.title(f"E0={E0},E={E}")
        plt.show()
    return psis


def Cooling(bcs, beta_0=1, N=1, **args):
    V = get_V(bcs.xyz[0])
    H0 = bcs._get_H(mu_eff=0, V=0)  # free particle
    H1 = bcs._get_H(mu_eff=0, V=V)  # harmonic trap
    U0, _ = bcs.get_U_E(H0, transpose=True)
    U1, _ = bcs.get_U_E(H1, transpose=True)
    psis0 = U1[:N]
    psis = U0[:N]
    psis=PlayCooling(bcs=bcs, psis0=psis0, psis=psis, V=V, **args)


if __name__ == "__main__":
    Nx = 64
    L = 23.0
    dx = L/Nx
    bcs = BCSCooling(N=Nx, L=None, dx=dx, beta_0=1, beta_V=1, beta_K=0, smooth=True) #, divs=(1, 1))
    #bcs.erase_max_ks()
    Cooling(bcs=bcs, N_data=10, N_step=100)