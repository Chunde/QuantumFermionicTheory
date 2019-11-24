from mmf_hfb.BCSCooling import BCSCooling
import numpy as np
import xlwt
import time
import argparse
import os
import pandas as pd


class TestCase2D(object):
    def __init__(self, T=5, g=0, **args):
        b = BCSCooling(N=32, dx=0.1, beta_0=-1j, g=g, dim=2, **args)
        x, y =b.xyz
        V = sum(_x**2 for _x in b.xyz)
        b.V = np.array(V)/2
        x0 = 0.5
        phase = ((x-x0) + 1j*y)*((x+x0) - 1j*y)
        psi_init = 1.0*np.exp(1j*np.angle(phase))
        _, psis, _ = b.solve([psi_init], T=T, rtol=1e-5, atol=1e-6)
        psi_ground = psis[-1][0]
        E0 = b.get_E_Ns([psi_ground])[0]
        self.E0 = E0
        self.psi_ground=psi_ground
        self.psi_init = psi_init
        self.b = b
        
    def get_E_Tw(self, beta_V, beta_K=0, beta_D=0, beta_Y=0, T=5):
        b = self.b
        b.beta_V = beta_V
        b.beta_K = beta_K
        b.beta_D = beta_D
        b.beta_Y = beta_Y
        start_time = time.time()
        _, psis, nfev = b.solve([self.psi_init], T=T, rtol=1e-5, atol=1e-6)
        wall_time = time.time() - start_time
        Ei = b.get_E_Ns(psis[0])[0]
        Ef = b.get_E_Ns(psis[-1])[0]
        return (Ei, Ef, wall_time, nfev)


def benchmark_test_excel(
        g=0, Ts=[5], trail=1, use_abm=False,
        beta_0=1, beta_Ks=[0], beta_Vs=[10], beta_Ds=[0], beta_Ys=[0],
        time_out=120, T_ground_state=10, last_file=None,
        save_interval=5, verbose=False):
    """
    this function is provided to perform test the cooling vs wall time
    calling this function will create a excel file that summarize the
    results, including all parameters used for each case.
    """
    print(
        f" g={g}, Ts={Ts}, trail={trail}, use_abm={use_abm},"
        +f"beta_0={beta_0}, beta_Ks={beta_Ks}, beta_Vs={beta_Vs},"
        +f"beta_Ds={beta_Ds}, beta_Ys={beta_Ys} "
        +f"time_out={time_out}, T_ground_state={T_ground_state}"
        +f",save_interval={save_interval}, verbose={verbose}")
    # create an excel table to store the result
    file_stem = (
        f"TestCase2D_g[{g}]_T[{5}]_Tr[{trail}]_PID=[{os.getpid()}]_"
        +time.strftime("%Y_%m_%d_%H_%M_%S"))
    file_name = file_stem+".xls"
    output = xlwt.Workbook(encoding='utf-8')
    sheet = output.add_sheet("overall", cell_overwrite_ok=True)
    col = 0
    row = 0
    headers = [
        "Trail", "Time", "beta_0", "beta_V", "beta_K",
        "beta_D", "beta_Y", "g", "E0", "Ei", "Ef", "Evolver",
        "Cooling", "pTime", "nfev", "wTime"]
    for value in headers:
        sheet.write(row, col, value)
        col += 1
    
    args = dict(g=g, use_abm=use_abm, check_dE=False, time_out=time_out)
    t=TestCase2D(T=T_ground_state, **args)
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
                                +f"beta_D={beta_D}, beta_Y={beta_Y}, g={g}, T={T}")
                        if beta_V == 0 and beta_K== 0 and beta_Y==0:
                            continue
                        try:
                            Ei, Ef, wall_time, nfev= t.get_E_Tw(
                                T=T, beta_V=beta_V, beta_K=beta_K,
                                beta_D=beta_D, beta_Y=beta_Y)
                        except ValueError as e:
                            print('Exception: '+ str(e))
                            continue
                        E0 = t.E0
                        dEf = (Ef - E0)/E0
                        col = 0
                        values = [
                            trail, time.strftime("%Y/%m/%d %H:%M:%S"),
                            beta_0, beta_V, beta_K, beta_D, beta_Y, g,
                            E0, Ei, Ef]
                        for value in values:
                            sheet.write(row, col, value)
                            col += 1
                        Evoler = "ABM" if use_abm else "IVP"
                        sheet.write(row, col, Evoler)
                        col+=1
                        if abs(dEf) < 1:
                            sheet.write(row, col, "Cooled")
                        elif abs((Ef - Ei)/Ei)<0.01:
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
        beta_Vs=None, beta_Ks=None, Ts=None,
        g=1, beta_0=1, N_beta_V=10, N_beta_K=11,
        min_beta_V=10, max_beta_V=100, min_beta_K=0, max_beta_K=100,
        min_T=1, max_T=5, N_T=20, trails=None, time_out=600, Ti=20,
        use_abm=False, save_interval=5, verbose=False):
    """
    a function benchmarks on wall time for given set of parameters.
    change parameters below as needed.
    """
    if beta_Vs is None:
        beta_Vs = np.linspace(min_beta_V, max_beta_V, N_beta_V)
    if beta_Ks is None:
        beta_Ks = np.linspace(min_beta_K, max_beta_K, N_beta_K)
    if Ts is None:
        Ts = np.concatenate(
            [np.linspace(0.001, 0.99, 20), np.linspace(min_T, max_T, N_T)])
    if trails is None:
        trails = 3
    beta_Ds = [0]
    beta_Ys = [0]
    for trail in range(trails):
        benchmark_test_excel(
            g=g, trail=trail, Ts=Ts, use_abm=use_abm, time_out=time_out,
            beta_Vs=beta_Vs, beta_Ks=beta_Ks, beta_Ds=beta_Ds, beta_0=beta_0,
            beta_Ys=beta_Ys, T_ground_state=Ti,
            save_interval=save_interval, verbose=verbose)


if __name__ == "__main__":
    # do_case_test_excel(beta_Vs=[65], beta_Ks=[0], Ts=[5], time_out=300)
    parser = argparse.ArgumentParser(description='Cooling Case Data Generation(2D)')
    parser.add_argument(
        '--trails', type=int, default=1, help='trail number used to track different runs')
    parser.add_argument(
        '--g', type=float, default=0, help='Interaction')
    parser.add_argument('--N_beta_V', type=int, default=10, help='Number of beta_Vs')
    parser.add_argument(
        '--min_beta_V', type=float, default=10, help='min value of beta_Vs')
    parser.add_argument(
        '--max_beta_V', type=float, default=100, help='max value of beta_Vs')
    parser.add_argument('--N_beta_K', type=int, default=11, help='Number of beta_Ks')
    parser.add_argument(
        '--min_beta_K', type=float, default=0, help='min value of beta_Ks')
    parser.add_argument(
        '--max_beta_K', type=float, default=100, help='max value of beta_Ks')
    parser.add_argument('--N_T', type=int, default=25, help='Number of T')
    parser.add_argument('--min_T', type=float, default=1, help='min value of T')
    parser.add_argument('--max_T', type=float, default=5, help='max value of T')
    parser.add_argument('--time_out', type=float, default=60, help='time out')
    parser.add_argument(
        '--Ti', type=float, default=20.0, help='imaginary cooling Max time')
    parser.add_argument(
        '--use_abm', type=bool, default=False,
        help='use ABM or not:True/False')
    parser.add_argument(
        '--save_interval', type=int, default=5, help='write file interval')
    parser.add_argument(
        '--verbose', dest='verbose', type=lambda x: bool(True if x=='True' else False))
    args = vars(parser.parse_args())
    try:
        do_case_test_excel(**args)
    except ValueError:
        parser.print_help()
