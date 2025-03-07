
import os
import sys
import time
import inspect
import argparse
import matplotlib.pyplot as plt
import numpy as np
import xlwt
import pandas as pd
currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from bcs_cooling import BCSCooling
from abm_solver import ABMEvolverAdapter


def Normalize(psi, dx=0.1):
    return psi/(psi.dot(psi.conj())*dx)**0.5


def Prob(psi):
    return np.abs(psi)**2


class TestCaseBCS(object):

    def __init__(
            self, N, dx, eps=1e-2, E_E0=None,
            N_state=2, T_max=10, use_abm=True, **args):
        args.update(N=N, dx=dx)
        b = BCSCooling(**args)
        self.N = N
        self.dx = dx
        x = b.xyz[0]
        V = x**2/2
        b.V = V
        H0 = b._get_H(mu_eff=0, V=0)  # free particle
        H1 = b._get_H(mu_eff=0, V=V)  # harmonic trap
        U0, _ = b.get_psis_es(H0, transpose=True)
        U1, Es1 = b.get_psis_es(H1, transpose=True)
        self.psis_init = [self.Normalize(psi) for psi in U0[:N_state]]
        self.psis_ground = [self.Normalize(psi) for psi in U1[:N_state]]
        self.E0 = sum(Es1[:N_state])
        self.use_abm = use_abm
        self.solver = ABMEvolverAdapter if use_abm else None
        self.x = x
        self.b = b
        self.eps = eps
        self.T_max = T_max
        self.N_state = N_state
        self.E_E0 = E_E0 if E_E0 >= 1 else None

    def Normalize(self, psi):
        return psi/(psi.dot(psi.conj())*self.dx)**0.5

    def run(
            self, N_T=10, T=None, plot=False, plot_log=True,
            plot_dE=False, verbose=True):
        b, x = self.b, self.x
        E0, _ = b.get_E_Ns(self.psis_ground)
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
            ts, psis, nfev = b.solve(self.psis_init, T=T, **args)
            wall_time = time.time() - start_time
            E, _ = b.get_E_Ns(psis[0])
            self.E_init = E
            E, _ = b.get_E_Ns(psis[-1])
            self.wall_time.append(wall_time)
            self.physical_time.append(T)
            self.Es.append(E)
            self.nfevs.append(nfev)
            self.psis.append(psis[-1])
            if verbose:
                print(f"physical time:{T}, wall time:{wall_time},dE:{(E-E0)/abs(E0)} ")
            
            if plot:
                Es = [b.get_E_Ns([_psi])[0] for _psi in psis]
                plt.subplot(131)
                plt.plot(x, self.b.V, label="HO")
                plt.title(f"Potential")
                plt.legend()
                plt.subplot(132)
                plt.plot(x, Prob(psis[0]), "+", label='init')
                plt.plot(x, Prob(psis[-1]), '--', label="final")
                plt.plot(x, Prob(self.psis_ground), label='Ground')
                plt.title(
                    f"E0={self.E0:5.4},E={E:5.4}, $" + r"\beta_0$" +f"={b.beta_0}, "
                    +r"$\beta_V$"+ f"={b.beta_V}, "+ r" $\beta_K$" + f"={b.beta_K}"
                    +r" $\beta_D$" + f"={b.beta_D}"+ r" $\beta_Y$" + f"={b.beta_Y}")
                plt.legend()
                plt.subplot(133)
                alt_text = "ABM" if self.use_abm else "IVP"
                if plot_log:
                    plt.semilogy(
                        ts[0][:-1], (Es[:-1] - E0)/abs(E0), label=f"E({alt_text})")
                else:
                    plt.plot(ts[0][:-1], (Es[:-1] - E0)/abs(E0), label=f"E({alt_text})")
                if plot_dE:
                    dE_dt= [-1*b.get_dE_dt([_psi]) for _psi in psis[0]]
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
        E=self.Es[id]
        plt.plot(self.x, Prob(self.psis[id]), '+', label="final")
        plt.plot(self.x, Prob(self.psis_init[id]), "--", label='init')
        plt.plot(self.x, Prob(self.psis_ground[id]), label='Ground')
        b=self.b
        plt.title(
            f"E0={self.E0:5.4},E={E:5.4}, $" + r"\beta_0$" +f"={b.beta_0}, "
            +r"$\beta_V$"+ f"={b.beta_V}, "+ r" $\beta_K$" + f"={b.beta_K}"
            +r" $\beta_D$" + f"={b.beta_D}"+ r" $\beta_Y$" + f"={b.beta_Y}")
        plt.legend()


def compute_t(t, beta_V, beta_K, beta_D, beta_Y, T, verbose=False, **args):
    if t is None:
        t = TestCaseBCS(**args)
    t.b.beta_V = beta_V
    t.b.beta_K = beta_K
    t.b.beta_D = beta_D
    t.b.beta_Y = beta_Y
    t.run(T=T, plot=False, verbose=verbose)
    return t.wall_time[-1], t.nfevs[-1], t.E0, t.E_init, t.Es[-1]


def benchmark_test_excel(
        N=128, dx=0.1, Ts=[5], trail=1, N_state=2, use_abm=False,
        beta_0=1, beta_Ks=[0], beta_Vs=[10], beta_Ds=[0], beta_Ys=[0],
        time_out=120, save_interval=5, verbose=False, E_E0=0):
    """
    this function is provided to perform test the cooling vs wall time
    calling this function will create a excel file that summarize the
    results, including all parameters used for each case.
    """
    print(
        f"N={N}, dx={dx}, N_state={N_state},Ts={Ts}, trail={trail}, use_abm={use_abm},"
        +f"beta_0={beta_0}, beta_Ks={beta_Ks}, beta_Vs={beta_Vs},"
        +f"beta_Ds={beta_Ds}, beta_Ys={beta_Ys},time_out={time_out}"
        +f",save_interval={save_interval}, verbose={verbose}")
    # create an excel table to store the result
    file_stem = (
        f"TestCaseBCS_N[{N}]_dNs[{N_state}]_x[{dx}]_T[{5}]_Tr[{trail}]"
        +f"_PID=[{os.getpid()}]_"  + time.strftime("%Y_%m_%d_%H_%M_%S"))
    file_name = file_stem+".xls"
    output = xlwt.Workbook(encoding='utf-8')
    sheet = output.add_sheet("overall", cell_overwrite_ok=True)
    col = 0
    row = 0
    headers = [
        "Trail", "Time", "N", "dx", "N_state", "beta_0", "beta_V", "beta_K",
        "beta_D", "beta_Y", "E0", "Ei", "Ef", "Evolver",
        "cooling", "pTime", "nfev", "wTime"]
    for value in headers:
        sheet.write(row, col, value)
        col += 1

    args = dict(
        beta_0=beta_0, N=N, dx=dx, eps=1e-1, use_abm=use_abm, N_state=N_state,
        check_dE=False, E_E0=E_E0, time_out=time_out)
    t = TestCaseBCS(**args)
    row = 1
    counter = 0
    for beta_Y in beta_Ys:
        for beta_D in beta_Ds:
            for beta_K in beta_Ks:
                for beta_V in beta_Vs:
                    if beta_V + beta_K + beta_Y + beta_D == 0:
                        continue
                    for T in Ts:
                        if verbose:
                            print(
                                f"Trail#={trail}: beta_V={beta_V}, beta_K={beta_K},"
                                +f"beta_D={beta_D}, beta_Y={beta_Y},"
                                +f"T={T}, N={N},dx={dx}, N_state={t.N_state}")
                        if beta_V == 0 and beta_K== 0 and beta_Y==0:
                            continue
                        try:
                            res = compute_t(
                                t, beta_V=beta_V, beta_K=beta_K,
                                beta_D=beta_D, beta_Y=beta_Y,
                                T=T, verbose=verbose)
                        except ValueError as e:
                            print('Exception: ' + str(e))
                            continue
                        wall_time, nfev, E0, Ei, Ef = res
                        print(res)
                        col = 0
                        values = [
                            trail, time.strftime("%Y/%m/%d %H:%M:%S"), N, dx, t.N_state,
                            beta_0, beta_V, beta_K, beta_D, beta_Y, E0, Ei, Ef]
                        for value in values:
                            sheet.write(row, col, value)
                            col += 1
                        Evoler = "ABM" if t.use_abm else "IVP"
                        sheet.write(row, col, Evoler)
                        col + =1
                        if abs((Ef - E0)/E0) < 1:
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
                                +f"Ef={Ef}: Saved to {file_name}")
    # convert to csv files
    output.save(file_name)
    data_xls = pd.read_excel(file_name, 'overall', index_col=None)
    data_xls.to_csv(file_stem+".csv", encoding='utf-8')


def do_case_test_excel(
        beta_Vs=None, beta_Ks=None, beta_Ds=None, beta_Ys=None,  
        N_beta_V=25, min_beta_V=20, max_beta_V=100,
        N_beta_K=11, min_beta_K=0, max_beta_K=100,
        N_beta_D=1, min_beta_D=0, max_beta_D=0,
        N_beta_Y=1, min_beta_Y=0, max_beta_Y=0, Ts=None,
        min_T=1, max_T=5, N_T=20, trails=None, **args):
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
    if Ts is None:
        Ts = np.concatenate(
            [np.linspace(0.001, 0.99, 20), np.linspace(min_T, max_T, N_T)])
    if trails is None:
        trails = 3
    beta_Ds = [0]
    beta_Ys = [0]
    for trail in range(trails):
        benchmark_test_excel(
            trail=trail, Ts=Ts, beta_Vs=beta_Vs, beta_Ks=beta_Ks,
            beta_Ds=beta_Ds, beta_Ys=beta_Ys, **args)


if __name__ == "__main__":
    # do_case_test_excel(N_state=1, beta_Vs=[65], beta_Ks=[0], Ts=[5], time_out=300)
    parser = argparse.ArgumentParser(description='cooling Case Data Generation')
    parser.add_argument('--N', type=int, default=128, help='lattice point number')
    parser.add_argument(
        '--trails', type=int, default=1, help='trail number used to track different runs')
    parser.add_argument(
        '--dx', type=float, default=0.2, help='An optional integer positional argument')
    parser.add_argument('--N_state', type=int, default=128, help='Number of states')
    parser.add_argument(
        '--E_E0', type=float, default=0, help='target energy over ground energy')
    # beta_Vs settings
    parser.add_argument('--N_beta_V', type=int, default=21, help='Number of beta_Vs')
    parser.add_argument(
        '--min_beta_V', type=float, default=0, help='min value of beta_Vs')
    parser.add_argument(
        '--max_beta_V', type=float, default=100, help='max value of beta_Vs')
    # beta_Vs settings
    parser.add_argument('--N_beta_K', type=int, default=21, help='Number of beta_Ks')
    parser.add_argument(
        '--min_beta_K', type=float, default=0, help='min value of beta_Ks')
    parser.add_argument(
        '--max_beta_K', type=float, default=100, help='max value of beta_Ks')
    # beta_Ds settings
    parser.add_argument('--N_beta_D', type=int, default=1, help='Number of beta_Ds')
    parser.add_argument(
        '--min_beta_D', type=float, default=0, help='min value of beta_Ds')
    parser.add_argument(
        '--max_beta_D', type=float, default=0, help='max value of beta_Ds')
    # beta_Ys settings
    parser.add_argument('--N_beta_Y', type=int, default=1, help='Number of beta_Ys')
    parser.add_argument(
        '--min_beta_Y', type=float, default=0, help='min value of beta_Ys')
    parser.add_argument(
        '--max_beta_Y', type=float, default=0, help='max value of beta_Ys')
    # Ts settings
    # example: --Ts 5

    parser.add_argument('--Ts',  nargs='+', default=[5], help='Number of T')
    parser.add_argument('--N_T', type=int, default=25, help='Number of T')
    parser.add_argument('--min_T', type=float, default=0, help='min value of T')
    parser.add_argument('--max_T', type=float, default=5, help='max value of T')

    parser.add_argument('--time_out', type=float, default=120, help='time out')
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
