
import os
import sys
import inspect
import time
import numpy as np
import matplotlib.pyplot as plt
import xlwt
import argparse
import pandas as pd
from os.path import join
currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from bcs_cooling import BCSCooling
from abm_solver import ABMEvolverAdapter


def Normalize(psi, dx=0.1):
    return psi/(psi.dot(psi.conj())*dx)**0.5


def Prob(psi):
    return np.abs(psi)**2


def random_gaussian_mixing(x, dx=0.1):
    n = np.random.randint(1, 10)
    cs = np.random.random(n)
    ns = np.random.randint(1, 10, size=n)
    ys = sum([c*np.exp(-x**2/n**2) for (c, n) in zip(cs, ns)])
    return Normalize(ys, dx=dx)


def get_init_states(N=128, dx=0.1):
    b = BCSCooling(N=N, dx=dx)
    x = b.xyz[0]
    H0 = b._get_H(mu_eff=0, V=0)
    U0, _ = b.get_psis_es(H0, transpose=True)
    psi_standing_wave = Normalize(U0[1], dx=dx)
    psi_gaussian_mixing = random_gaussian_mixing(x, dx=dx)
    psi_uniform = Normalize(U0[0], dx=dx)
    psi_bright_soliton = Normalize(np.exp(-x**2/2.0)*np.exp(1j*x), dx=dx)
    return dict(
        ST=psi_standing_wave, GM=psi_gaussian_mixing,
        UN=psi_uniform, BS=psi_bright_soliton)


def get_potentials(x):
    V0 = 0*x
    V_HO = x**2/2
    V_PO = V0 + np.random.random()*V_HO + abs(x**2)*np.random.random()
    return dict(V0=V0, HO=V_HO, PO=V_PO)


class TestCase(object):

    def __init__(
            self, g, N, dx, eps=1e-2, E_E0=None,
            psi_init=None, psi_ground=None, E0=None, T_max=10,
            T_ground_state=20, V_key=None, use_abm=True, **args):
        """
        Arguments
        ---------
        E_E0: float
            the target energy E/E0, should be no smaller than 1 or None
            used to set the target energy level in BCSCooling object
        """
        args.update(g=g, N=N, dx=dx)
        b = BCSCooling(**args)
        self.use_abm = use_abm
        self.solver = ABMEvolverAdapter if use_abm else None
        self.x = b.xyz[0]
        self.b = b
        self.V_key = V_key
        self.eps = eps
        self.T_max = T_max
        self.g = g
        self.N = N
        self.dx = dx
        if psi_init is None:
            psi_init = random_gaussian_mixing(self.x)
        self.psi_init = psi_init + 1j*0
        if psi_ground is None:
            psi_ground = self.get_ground_state(
                T=T_ground_state, psi_init=self.psi_init)
        self.psi_ground = psi_ground
        self.E0 = E0
        self.E_E0 = E_E0 if E_E0 >= 1 else None

    def get_ground_state(self, psi_init, T=None, plot=False):
        b = BCSCooling(
            N=self.b.N, dx=self.b.dx, g=self.g, V=self.b.V, beta_0=-1j)
        H = b._get_H(mu_eff=0, V=self.b.V)
        U, _ = b.get_psis_es(H, transpose=True)
        psi0 = Normalize(U[0], dx=b.dx)
        self.psi_0 = psi0
        if self.g == 0:
            return psi0
        else:
            # imaginary cooling
            if T is None:
                Ts = list(range(2, self.T_max, 1))
            else:
                Ts = [T]
            for T in Ts:
                print(f"Imaginary cooling with T={T}")
                _, psis, _ = b.solve(
                    [psi_init], T=T, solver=self.solver,
                    rtol=1e-5, atol=1e-6, method='BDF')
                assert len(psis) > 2
                E1, _ = b.get_E_Ns(psis[-1])
                E2, _ = b.get_E_Ns(psis[-2])
                if np.isnan(E1) or np.isnan(E2):
                    print(f"Value error: E1={E1}, E2={E2}")
                    raise ValueError("Failed to get ground state.")
                print((E2 - E1)/E1)
                return psis[-1][0]

    def run(
            self, N_T=10, T=None, plot=False,
            plot_log=True, plot_dE=False, verbose=True):
        b, x = self.b, self.x
        E0, _ = b.get_E_Ns([self.psi_ground])
        self.E0 = E0
        self.b.E_stop = None if self.E_E0 is None else E0*self.E_E0
        if T is None:
            Ts = (np.array(list(range(N_T)))+1)*0.5
        else:
            Ts = [T]
        args = dict(rtol=1e-5, atol=1e-6, solver=self.solver, method='BDF')
        self.physical_time = []
        self.wall_time = []
        self.Es = []
        self.psis = []
        self.nfevs = []
        for T in reversed(Ts):
            start_time = time.time()
            ts, psis, nfev = b.solve([self.psi_init], T=T, **args)
            wall_time = time.time() - start_time
            E, _ = b.get_E_Ns(psis[0])
            self.E_init = E
            E, _ = b.get_E_Ns(psis[-1])
            self.wall_time.append(wall_time)
            self.physical_time.append(T)
            self.Es.append(E)
            self.nfevs.append(nfev)
            self.psis.append(psis[-1][0])
            if verbose:
                print(f"physical time:{T}, wall time:{wall_time},dE:{(E-E0)/abs(E0)} ")

            if plot:
                Es = [b.get_E_Ns([_psi])[0] for _psi in psis]
                plt.subplot(131)
                plt.plot(x, self.b.V, label=self.V_key)
                plt.title(f"g={b.g}")
                plt.legend()
                plt.subplot(132)
                plt.plot(x, Prob(psis[0]), "+", label='init')
                plt.plot(x, Prob(psis[-1]), '--', label="final")
                plt.plot(x, Prob(self.psi_ground), label='Ground')
                plt.title(
                    f"E0={self.E0:5.4},E={E:5.4}, $" + r"\beta_0$" + f"={b.beta_0}, "
                    + r"$\beta_V$"+ f"={b.beta_V}, " + r" $\beta_K$" + f"={b.beta_K}"
                    + r" $\beta_D$" + f"={b.beta_D}" + r" $\beta_Y$" + f"={b.beta_Y}")
                plt.legend()
                plt.subplot(133)
                alt_text = "ABM" if self.use_abm else "IVP"
                if plot_log:
                    plt.semilogy(
                        ts[0][:-1], (Es[:-1] - E0)/abs(E0), label=f"E({alt_text})")
                else:
                    plt.plot(ts[0][:-1], (Es[:-1] - E0)/abs(E0), label=f"E({alt_text})")
                if plot_dE:
                    dE_dt = [-1*b.get_dE_dt([_psi]) for _psi in psis]
                    plt.plot(ts[0][:-1], dE_dt[:-1], label='-dE/dt')
                    plt.axhline(0, linestyle='dashed')
                plt.title(
                    f"i={((Es[0] - E0)/abs(E0)):5.4}, f={((Es[-1] - E0)/abs(E0)):5.4}")
                plt.legend()
                plt.axhline(0, linestyle='dashed')
                plt.show()
            if abs((E - E0)/E0) > self.eps:
                break
                
    def plot(self, id=0):
        E = self.Es[id]
        plt.plot(self.x, Prob(self.psis[id]), '+', label="final")
        plt.plot(self.x, Prob(self.psi_init), "--", label='init')
        plt.plot(self.x, Prob(self.psi_ground), label='Ground')
        b = self.b
        plt.title(
            f"E0={self.E0:5.4},E={E:5.4}, $" + r"\beta_0$" +f"={b.beta_0}, "
            + r"$\beta_V$"+ f"={b.beta_V}, " + r" $\beta_K$" + f"={b.beta_K}"
            + r" $\beta_D$" + f"={b.beta_D}" + r" $\beta_Y$" + f"={b.beta_Y}")
        plt.legend()


def benchmark_test_excel(
        N=128, dx=0.1, g=0, Ts=[5], trail=1, use_abm=False,
        beta_0=1, beta_Ks=[0], beta_Vs=[10], beta_Ds=[0], beta_Ys=[0],
        ground_state="Gaussian", iState="ST", V="HO",
        time_out=120, Ti=10, last_file=None,
        save_interval=5, verbose=False, E_E0=0):
    """
    this function is provided to perform test the cooling vs wall time
    calling this function will create a excel file that summarize the
    results, including all parameters used for each case.
    """
    print(
        f"N={N}, dx={dx}, g={g}, Ts={Ts}, trail={trail}, use_abm={use_abm},"
        + f"beta_0={beta_0}, beta_Ks={beta_Ks}, beta_Vs={beta_Vs},"
        + f"beta_Ds={beta_Ds}, beta_Ys={beta_Ys},ground_state={ground_state}, "
        + f"iState={iState}, V_key={V},"
        + f"time_out={time_out}, T_ground_state={Ti}"
        + f",save_interval={save_interval}, verbose={verbose}")
    # create an excel table to store the result
    file_stem = (
        f"TestCase_N[{N}]_dx[{dx}]_g[{g}]_T[{5}]_Tr[{trail}]"
        + f"_IS[{iState}]_V[{V}]_PID=[{os.getpid()}]_"
        + time.strftime("%Y_%m_%d_%H_%M_%S"))
    file_name = file_stem+".xls"
    output = xlwt.Workbook(encoding='utf-8')
    sheet = output.add_sheet("overall", cell_overwrite_ok=True)
    col = 0
    row = 0
    headers = [
        "Trail", "Time", "N", "dx", "beta_0", "beta_V", "beta_K",
        "beta_D", "beta_Y", "g", "V", "gState", "iState",
        "E0", "Ei", "Ef", "Evolver",
        "cooling", "pTime", "nfev", "wTime"]
    for value in headers:
        sheet.write(row, col, value)
        col += 1
    psis_init = get_init_states(N=N, dx=dx)
    psi_init = psis_init[iState]
    b = BCSCooling(N=N, dx=dx)
    x = b.xyz[0]
    Vs = get_potentials(x)
    args = dict(
        N=N, dx=dx, eps=1e-1, T_ground_state=Ti, V=Vs[V],
        V_key=V, g=g, psi_init=psi_init,
        use_abm=use_abm, check_dE=False, E_E0=E_E0, time_out=time_out)
    t=TestCase(ground_state_eps=1e-1, beta_0=beta_0, **args)
    row = 1
    counter = 0
    for beta_Y in beta_Ys:
        t.b.beta_Y = beta_Y
        for beta_D in beta_Ds:
            t.b.beta_D = beta_D
            for beta_K in beta_Ks:
                t.b.beta_K = beta_K
                for beta_V in beta_Vs:
                    t.b.beta_V = beta_V
                    if beta_V + beta_K + beta_Y + beta_D == 0:
                        continue
                    for T in Ts:
                        if verbose:
                            print(
                                f"Trail#={trail}: beta_V={beta_V}, beta_K={beta_K},"
                                + f"beta_D={beta_D}, beta_Y={beta_Y},"
                                + f"g={g}, T={T}, V={V}, N={N},dx={dx}")
                        if beta_V==0 and beta_K==0 and beta_Y==0:
                            continue
                        try:
                            t.run(T=T, plot=False, verbose=verbose)
                        except ValueError as e:
                            print('Exception: ' + str(e))
                            continue
                        wall_time = t.wall_time[-1]
                        nfev = t.nfevs[-1]
                        E0 = t.E0
                        Ei, Ef = t.E_init, t.Es[-1]
                        dEf = (Ef - E0)/E0
                        col = 0
                        values = [
                            trail, time.strftime("%Y/%m/%d %H:%M:%S"), N, dx,
                            beta_0, beta_V, beta_K, beta_D, beta_Y, g, V,
                            ground_state, iState, E0, Ei, Ef]
                        for value in values:
                            sheet.write(row, col, value)
                            col += 1
                        Evoler = "ABM" if t.use_abm else "IVP"
                        sheet.write(row, col, Evoler)
                        col += 1
                        if abs(dEf) < 1:
                            sheet.write(row, col, "Cooled")
                        elif abs((Ef - Ei)/Ei) < 0.01:
                            sheet.write(row, col, "Failed")
                        else:
                            sheet.write(row, col, "Partially Cooled")
                        col += 1
                        sheet.write(row, col, T)
                        col += 1
                        sheet.write(row, col, nfev)
                        col += 1
                        sheet.write(row, col, wall_time)
                        col += 1
                        row += 1
                        counter += 1
                        if counter % save_interval == 0:
                            output.save(file_name)
                            print(
                                f"{counter}: E0={E0}, Ei={Ei},"
                                + f"Ef={Ef}: Saved to {file_name}")
    # convert to csv files
    output.save(file_name)
    data_xls = pd.read_excel(file_name, 'overall', index_col=None)
    data_xls.to_csv(file_stem+".csv", encoding='utf-8')


def do_case_test_excel(
        beta_Vs=None, beta_Ks=None, beta_Ds=None, beta_Ys=None,
        N_beta_V=25, min_beta_V=20, max_beta_V=100,
        N_beta_K=11, min_beta_K=0, max_beta_K=100,
        N_beta_D=1, min_beta_D=0, max_beta_D=0,
        N_beta_Y=1, min_beta_Y=0, max_beta_Y=0,
        trails=None, min_T=1, max_T=5, N_T=20, **args):
    """
    a function benchmarks on wall time for given set of parameters.
    change parameters below as needed.
    """
    if beta_Vs is None:
        beta_Vs = np.linspace(min_beta_V, max_beta_V, N_beta_V)
    if beta_Ks is None:
        beta_Ks = np.linspace(min_beta_K, max_beta_K, N_beta_K)
    if beta_Ds is None:
        beta_Ds = np.linspace(min_beta_D, max_beta_D, N_beta_D)
    if beta_Ys is None:
        beta_Ys = np.linspace(min_beta_Y, max_beta_Y, N_beta_Y)
    Ts = np.linspace(min_T, max_T, N_T)
    if trails is None:
        trails = 3
    for trail in range(trails):
        benchmark_test_excel(
            trail=trail, Ts=Ts, beta_Vs=beta_Vs, beta_Ks=beta_Ks,
            beta_Ds=beta_Ds, beta_Ys=beta_Ys, **args)


if __name__ == "__main__":
    # do_case_test_excel(
    # beta_Vs=[65], beta_Ks=[0], Ts=[5], E_E0=1.5, time_out=300)
    
    parser = argparse.ArgumentParser(description='cooling Case Data Generation')
    parser.add_argument('--N', type=int, default=128,
    help='lattice point number')
    parser.add_argument(
        '--trails', type=int, default=1,
        help='trail number used to track different runs')
    parser.add_argument(
        '--dx', type=float, default=0.2,
        help='An optional integer positional argument')
    parser.add_argument(
        '--g', type=float, default=0, help='Interaction')
    parser.add_argument(
        '--E_E0', type=float, default=0,
        help='target energy over ground energy')
    parser.add_argument(
        '--iState', default="ST", help='Initial State Type: ST/BS/UN/GM')
    parser.add_argument('--V', default="HO", help='Potential Type: HO/V0')
    # beta_Vs settings
    parser.add_argument(
        '--N_beta_V', type=int, default=21, help='Number of beta_Vs')
    parser.add_argument(
        '--min_beta_V', type=float, default=0, help='min value of beta_Vs')
    parser.add_argument(
        '--max_beta_V', type=float, default=100, help='max value of beta_Vs')
    # beta_Vs settings
    parser.add_argument(
        '--N_beta_K', type=int, default=21, help='Number of beta_Ks')
    parser.add_argument(
        '--min_beta_K', type=float, default=0, help='min value of beta_Ks')
    parser.add_argument(
        '--max_beta_K', type=float, default=100, help='max value of beta_Ks')
    # beta_Ds settings
    parser.add_argument(
        '--N_beta_D', type=int, default=1, help='Number of beta_Ds')
    parser.add_argument(
        '--min_beta_D', type=float, default=0, help='min value of beta_Ds')
    parser.add_argument(
        '--max_beta_D', type=float, default=0, help='max value of beta_Ds')
    # beta_Ys settings
    parser.add_argument(
        '--N_beta_Y', type=int, default=1, help='Number of beta_Ys')
    parser.add_argument(
        '--min_beta_Y', type=float, default=0, help='min value of beta_Ys')
    parser.add_argument(
        '--max_beta_Y', type=float, default=0, help='max value of beta_Ys')
    # Ts settings
    # parser.add_argument('--Ts', nargs='+', default=[], help='Number of T')
    parser.add_argument('--N_T', type=int, default=25, help='Number of T')
    parser.add_argument('--min_T', type=float, default=0, help='min value of T')
    parser.add_argument('--max_T', type=float, default=5, help='max value of T')
    # Time out
    parser.add_argument('--time_out', type=float, default=60, help='time out')
    # Time for imaginary cooling to find ground state
    parser.add_argument(
        '--Ti', type=float, default=20.0, help='imaginary cooling Max time')
    parser.add_argument(
        '--use_abm', type=bool, default=False, help='use ABM or not:True/False')
    parser.add_argument(
        '--save_interval', type=int, default=5, help='write file interval')
    parser.add_argument(
        '--verbose', dest='verbose', type=lambda x: bool(True if x=='True' else False))
    args = vars(parser.parse_args())
    try:
        do_case_test_excel(**args)
    except ValueError:
        parser.print_help()
