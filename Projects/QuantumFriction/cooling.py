"""
Methods that will be called from notebook to play
Animation for cooling procedure. Put this method
in a file to make a notebook cleaner.
"""
import os
import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt
from mmf_hfb.potentials import HarmonicOscillator

current_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, current_dir)
from bcs_cooling import BCSCooling
from IPython.display import clear_output

eps = 7./3 - 4./3 - 1  # machine precision


def check_uv_ir_error(psi, plot=False):
    """check if a lattice configuration(N, L, dx) is good"""
    psi_k = np.fft.fft(psi)
    psi_log = np.log10(abs(psi)+eps)
    psi_log_k = np.log10(abs(psi_k)+eps)
    if plot:
        l, = plt.plot(psi_log_k)
        plt.plot(psi_log, '--', c=l.get_c())
        print(np.min(psi_log), np.min(psi_log_k))
    # assert the log10 value to be close to machine precision
    #assert np.min(psi_log)<-15
    assert np.min(psi_log_k) < -15


def get_occupancy(psis0, psis, dx=1):
    """return occupancy"""
    def p(psis0):
        ps =[np.abs(psis0.conj().dot(psi))**2 for psi in psis]
        return sum(ps)*dx
    return [p(psi0) for psi0 in psis0]


def plot_occupancy_n(ts, nss):
    """plot occupancy as cooling proceeds"""
    if len(nss) == 0:
        return
    num_plt = len(nss[0])
    datas = []
    for i in range(num_plt):
        datas.append([])
    for ns in nss:
        for i in range(num_plt):
            v = ns[i]
            datas[i].append(v)
    for i, data in enumerate(datas):
        plt.plot(ts, data, label=f'{i}')
    plt.axhline(1, linestyle='dashed')
    plt.xlabel("Time(s)")
    plt.title("Occupancy")
    plt.legend()


def plot_occupancy_k(b, psis):
    n_k = 0
    k0 = 2*np.pi/b.dxyz[0]
    ks = b.kxyz[0]/k0
    for psi in psis:
        n_k += abs(np.fft.fft(psi))**2
    n_k = np.fft.fftshift(n_k)
    ks = np.fft.fftshift(ks)
    plt.plot(ks, n_k)
    plt.xlabel(r"$2\pi k/dx$")
    plt.title(r"$n_k$")


def plot_psis(b, psis0, psis, E, E0):
    x = b.xyz[0]
    cs = []
    for psi in psis:
        ax, = plt.plot(x, abs(psi)**2)
        cs.append(ax.get_c())
        for i, psi in enumerate(psis0):
            if i < len(cs):
                plt.plot(x, abs(psi)**2, '+', c=cs[i])
            else:
                plt.plot(x, abs(psi)**2, '+')
        plt.title(
            f"E0={E0:5.4},E={E:5.4}, $" + r"\beta_0$" +f"={b.beta_0}, "
            +r"$\beta_V$"+f"={b.beta_V}, "+r" $\beta_K$" +f"={b.beta_K}"
            +r" $\beta_D$" +f"={b.beta_D}")


def PlayCooling(
        b, psis0, psis, N_data=10, N_step=100,
        plot=True, plot_n=True, plot_k=True, save_file_name=None,
        log_E=True, **kw):
    E0, N0 = b.get_E_Ns(psis0[:len(psis)])
    Es, Ns = [], []
    plt.rcParams["figure.figsize"] = (18, 6)
    plot_n_k = plot_n and plot_k
    for _n in range(N_data):
        Ps = get_occupancy(psis0, psis)
        Ns.append(Ps)

        psis = b.step(psis, n=N_step)
        E, N = b.get_E_Ns(psis)
        Es.append(abs(E - E0))
        ts = b.dt*N_step*np.array(list(range(len(Es))))
        if plot:
            plt.subplot(131)
            if plot_n_k:
                plot_occupancy_n(ts=ts, nss=Ns)
            else:
                plot_psis(b=b, psis0=psis0, psis=psis, E0=E0, E=E)
            plt.subplot(132)
            if plot_k:
                plot_occupancy_k(b=b, psis=psis)
            else:
                plot_occupancy_n(ts=ts, nss=Ns)

            plt.subplot(133)
            if log_E:
                plt.semilogy(ts, Es)
            else:
                plt.plot(ts, Es)
            plt.xlabel("time(s)")
            plt.title(r"$(E-E_0)/E_0$")
            plt.axhline(0, linestyle='dashed')
            if _n == N_data - 1 and save_file_name is not None:
                plt.savefig(save_file_name, bbox_inches='tight')
            clear_output(wait=True)
            plt.show()
    return psis, Es, Ns


def get_box_wf(n, L, x):
    n = n+1
    k_n = n*np.pi/L
    if n%2 == 1:
        return (1/L)**0.5*np.cos(k_n*x)
    return (1/L)**0.5*np.sin(k_n*x)


def cooling(
        Nx=128, Lx=23, init_state_ids=None, V0=1,
        beta_0=1, N_state=1, plot_k=True, **args):
    """
    N_state: integer if init_state_ids is not provided
        , it will use the first N_state states, and
        also check the ground states with that numbers.
    init_state_ids: list, a list of initial states indics

    """
    L = Lx
    dx = L/Nx
    b = BCSCooling(N=Nx, L=None, dx=dx, **args)
    x = b.xyz[0]
    V = V0*x**2/2
    b.V = V
    if False:
        # the wavefuntions here are not in a box
        # they are free particle wavefunctions
        H0 = b._get_H(mu_eff=0, V=0)  # free particle
        H1 = b._get_H(mu_eff=0, V=V)  # harmonic trap
        U0, Es0 = b.get_psis_es(H0, transpose=True)
        U1, Es1 = b.get_psis_es(H1, transpose=True)
        if init_state_ids is None:
            psis = U0[:N_state]  # change the start states here if needed.
        else:
            assert len(init_state_ids) <= N_state
            psis = [U0[id] for id in init_state_ids]
        psis0 = U1[:N_state]  # the ground states for the harmonic potential
    else:
        h = HarmonicOscillator()
        if init_state_ids is None:
            psis = [b.Normalize(get_box_wf(n=i, L=L, x=x), dx=1) for i in range(N_state)]  # change the start states here if needed.
        else:
            assert len(init_state_ids) <= N_state
            psis = [b.Normalize(get_box_wf(n=i, L=L, x=x), dx=1) for i in init_state_ids]
        psis0 = [b.Normalize(h.get_wf(n=i, x=x), dx=1) for i in range(N_state)] # the ground states for the harmonic potential
    # for i in range(len(psis0)):
    #     check_uv_ir_error(psis0[i], plot=False)
    # for i in range(len(psis)):
    #     check_uv_ir_error(psis[i], plot=False)
    return [x, PlayCooling(b=b, psis0=psis0, psis=psis, plot_k=plot_k, **args)]

if __name__ == "__main__":
    cooling(
        N_state=3, Nx=256, N_data=25, start_state=2,
        N_step=100, beta_V=1, beta_K=1, beta_D=0)
