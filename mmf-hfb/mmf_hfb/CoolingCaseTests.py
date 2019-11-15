from mmf_hfb.BCSCooling import BCSCooling
from mmf_hfb.ParallelHelper import PoolHelper
from mmf_hfb.SolverABM import ABMEvolverAdapter
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import json
import xlwt
import xlrd
import time
import os
import datetime
import inspect
import glob


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


def unpack(psi):
    return dict(r=psi.real.tolist(), i=psi.imag.tolist())


def dict_to_complex(psi_data):
    psi_r = psi_data['r']
    psi_i = psi_data['i']
    return np.array(psi_r)+1j*np.array(psi_i)


class TestCase(object):
    
    def __init__(
            self, g, N, dx, eps=1e-2,
            psi_init=None, psi_ground=None, E0=None, T_max=10,
            T_ground_state=20, V_key=None, use_abm=True, **args):
        args.update(g=g, N=N, dx=dx)
        b = BCSCooling(**args)
        self.use_abm = use_abm
        self.solver = ABMEvolverAdapter if use_abm else None
        self.x = b.xyz[0]
        self.b=b
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
        self.E0=E0

    def get_ground_state(self, psi_init, T=None, plot=False):
        b = BCSCooling(N=self.b.N, dx=self.b.dx, g=self.g, V=self.b.V, beta_0=-1j)
        H = b._get_H(mu_eff=0, V=self.b.V)
        U, E = b.get_U_E(H, transpose=True)
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
                print(f"Imaginary Cooling with T={T}")
                _, psiss = b.solve(
                    [psi_init], T=T, solver=self.solver,
                    rtol=1e-5, atol=1e-6, method='BDF')
                psis = psiss[0]
                assert len(psis) > 2
                E1, _ = b.get_E_Ns([psis[-1]])
                E2, _ = b.get_E_Ns([psis[-2]])
                if np.isnan(E1) or np.isnan(E2):
                    print(f"Value error: E1={E1}, E2={E2}")
                    raise ValueError("Failed to get ground state.")
                print((E2 - E1)/E1)
                return psis[-1]

    def run(self, N_T=10, T=None, plot=False, plot_log=True, plot_dE=False):
        b, x = self.b, self.x
        E0, _ = b.get_E_Ns([self.psi_ground])
        print(f"E0={E0}")
        self.E0 = E0
        if T is None:
            Ts = (np.array(list(range(N_T)))+1)*0.5
        else:
            Ts = [T]
        args = dict(rtol=1e-5, atol=1e-6, solver=self.solver, method='BDF')
        self.physical_time = []
        self.wall_time = []
        self.Es = []
        self.psis = []
        for T in reversed(Ts):
            start_time = time.time()
            ts, psiss = b.solve([self.psi_init], T=T, **args)
            wall_time = time.time() - start_time
            E, _ = b.get_E_Ns([psiss[0][0]])
            self.E_init = E
            E, _ = b.get_E_Ns([psiss[0][-1]])
            self.wall_time.append(wall_time)
            self.physical_time.append(T)           
            self.Es.append(E)
            self.psis.append(psiss[0][-1])
            print(f"physical time:{T}, wall time:{wall_time},dE:{(E-E0)/abs(E0)} ")
            
            if plot:
                Es = [b.get_E_Ns([_psi])[0] for _psi in psiss[0]]
                plt.subplot(131)
                plt.plot(x, self.b.V, label=self.V_key)
                plt.title(f"g={b.g}")
                plt.legend()
                plt.subplot(132)
                plt.plot(x, Prob(psiss[0][0]), "+", label='init')
                plt.plot(x, Prob(psiss[0][-1]), '--', label="final")
                plt.plot(x, Prob(self.psi_ground), label='Ground')
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
                    dE_dt= [-1*b.get_dE_dt([_psi]) for _psi in psiss[0]]
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
        psi = self.psis[id]
        plt.plot(self.x, Prob(psi), "--", label='init')
        plt.plot(self.x, Prob(self.psis[id]), '+', label="final")
        plt.plot(self.x, Prob(self.psi_ground), label='Ground')
        b=self.b
        plt.title(
            f"E0={self.E0:5.4},E={E:5.4}, $" + r"\beta_0$" +f"={b.beta_0}, "
            +r"$\beta_V$"+ f"={b.beta_V}, "+ r" $\beta_K$" + f"={b.beta_K}"
            +r" $\beta_D$" + f"={b.beta_D}"+ r" $\beta_Y$" + f"={b.beta_Y}")
        plt.legend()


def get_cooling_potential_setting(beta_V=60, beta_K=75, beta_D=1000, beta_Y=1):
    cooling_para_list=[
        dict(beta_V=beta_V, beta_K=beta_K, beta_D=beta_D, beta_Y=beta_Y),
        dict(beta_V=beta_V, beta_K=beta_K, beta_D=beta_D, beta_Y=0),
        dict(beta_V=beta_V, beta_K=beta_K, beta_D=0, beta_Y=beta_Y),
        dict(beta_V=beta_V, beta_K=0, beta_D=beta_D, beta_Y=beta_Y),
        dict(beta_V=0, beta_K=beta_K, beta_D=beta_D, beta_Y=beta_Y),
        dict(beta_V=beta_V, beta_K=beta_K, beta_D=0, beta_Y=0),
        dict(beta_V=beta_V, beta_K=0, beta_D=beta_D, beta_Y=0),
        dict(beta_V=0, beta_K=beta_K, beta_D=beta_D, beta_Y=0),
        dict(beta_V=beta_V, beta_K=0, beta_D=0, beta_Y=beta_Y),
        dict(beta_V=0, beta_K=beta_K, beta_D=0, beta_Y=beta_Y),
        dict(beta_V=0, beta_K=0, beta_D=beta_D, beta_Y=beta_Y),
        dict(beta_V=0, beta_K=beta_K, beta_D=0, beta_Y=0),
        dict(beta_V=0, beta_K=0, beta_D=beta_D, beta_Y=0),
        dict(beta_V=0, beta_K=0, beta_D=0, beta_Y=beta_Y),   
        dict(beta_V=beta_V, beta_K=0, beta_D=0, beta_Y=0)]
    return cooling_para_list


def get_potentials(x):
    V0 = 0*x
    V_HO = x**2/2
    V_PO = V0 + np.random.random()*V_HO + abs(x**2)*np.random.random()
    return dict(V0=V0, HO=V_HO, PO=V_PO)


def SaveTestCase(ts, Vs, psi_init):
    """Save the test case data to a json file"""
    file_name = "CoolingTestData_"+time.strftime("%Y_%m_%d_%H_%M_%S.json") 
    for key in Vs:
        Vs[key]=Vs[key].tolist()
    output = dict(Vs=Vs, psi_init=unpack(psi_init))
    cases = []
    for t in ts:
        if t is None:
            continue
        V = t.V
        b = t.b
        if isinstance(t.V, np.ndarray):
            V = V.tolist()
        dic = dict(
            dx=t.dx, dt=b.dt, N=t.N, E0=t.E0, max_T=t.max_T, g=t.g, eps=t.eps,
            beta_0=b.beta_0, beta_V=b.beta_V, beta_K=b.beta_K, dE_dt=b.dE_dt,
            beta_D=b.beta_D, beta_Y=b.beta_Y, V_key=t.V_key, use_abm=t.use_abm,
            psi0=unpack(t.psi_ground))
        data = []
        for (E, T, Tw, psi) in zip(t.Es, t.physical_time, t.wall_time, t.psis):
            data.append(dict(E=E, T=T, Tw=Tw, psi=unpack(psi)))
        dic['data']=data
        cases.append(dic)
    output["cases"] = cases
    with open(file_name, 'w') as wf:
        json.dump(output, wf)
    print(f"file {file_name} saved")
    return output


def load_json_data(file=None, filter="CoolingTestData*.json"):
    """load a json file """
    if file is None:
        current_dir = join(
            os.path.dirname(
                os.path.abspath(
                    inspect.getfile(inspect.currentframe()))), "..")
        pattern = join(current_dir, filter)
        files = glob.glob(pattern)
        if len(files) > 0:
            file = files[0]
    if os.path.exists(file):
        with open(file, 'r') as rf:
            ret = json.load(rf)
            return ret
    return None


def deserialize_object(json_object):
    """
        Deserialize objects from a given json object
        return a list of TestCase objects
    """
    def parse_time_data(testCase, res):
        """parse wave function and its energy"""
        psis = []
        wall_time = []
        physical_time = []
        Es = []
        for data in res:
            Es.append(data['E'])
            physical_time.append(data['T'])
            wall_time.append(data['Tw'])
            psis.append(dict_to_complex(data['psi']))
        testCase.wall_time = wall_time
        testCase.physical_time = physical_time
        testCase.Es = Es
        testCase.psis = psis
    
    Vs = json_object['Vs']
    psi_init = dict_to_complex(json_object['psi_init'])
    cases = json_object['cases']
    testCases = []
    for args in cases:
        V = Vs[args['V_key']]
        args['psi_ground'] = dict_to_complex(args['psi0'])
        args.update(V=V, psi=psi_init)
        t = TestCase(**args)
        parse_time_data(t, res=args['data'])
        testCases.append(t)
    return testCases


def DumpTestPara(pid, para):
    file_name = f"DumpData_{pid}_" + time.strftime("%Y_%m_%d_%H_%M_%S.json")
    with open(file_name, 'w') as wf:
        psi_init = para['psi']
        V = para['V']
        del para['psi']
        del para['V']
        para['V'] = unpack(V)
        para['psi'] = unpack(psi_init)
        json.dump(para, wf)
        print(f"file {file_name} saved")


def loadDumpedParas(files=None):
    if files is None:
        current_dir = join(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))), "..")
        pattern = join(current_dir, "DumpData*.json")
        files = glob.glob(pattern)
    rets = []
    for file in files:
        if os.path.exists(file):
            try:
                print(file)
                with open(file, 'r') as rf:
                    ret = json.load(rf)
                    ret['psi'] = dict_to_complex(ret['psi'])
                    ret['V']=dict_to_complex(ret['V'])
                    rets.append(ret)
            except:
                print("Error@" + file)
    return rets


def test_case_worker(para):
    pid = os.getpid()
    start_time = time.time()
    print(f"{pid} start time:{datetime.datetime.now()}")
    try:
        t = TestCase(**para)
        t.run()
        print(f"{pid} Wall time:{time.time() - start_time}")
        return t
    except:
        DumpTestPara(pid=pid, para=para)
        print(f"{pid} Wall time:{time.time() - start_time}")
        return None


def get_init_states(N=128, dx=0.1):
    b = BCSCooling(N=N, dx=dx)
    x = b.xyz[0]
    V = x**2/2
    H0 = b._get_H(mu_eff=0, V=0)
    H1 = b._get_H(mu_eff=0, V=V)
    U0, E0 = b.get_U_E(H0, transpose=True)
    U1, E1 = b.get_U_E(H1, transpose=True)
    psi_standing_wave=Normalize(U0[1],dx=dx)
    psi_gaussian_mixing = random_gaussian_mixing(x, dx=dx)
    psi_uniform = Normalize(U0[0], dx=dx)
    psi_bright_soliton = Normalize(np.exp(-x**2/2.0)*np.exp(1j*x), dx=dx)
    return dict(
        ST=psi_standing_wave, GM=psi_gaussian_mixing,
        UN=psi_uniform, BS=psi_bright_soliton)


def benchmark_test_json():
    if True:  # set to false if want to debug overflow by loading dumped file
        N, dx = 128, 0.1
        args = dict(
            N=N, dx=dx,
            beta0=1, beta_K=0, beta_V=0, beta_D=0, beta_Y=0,
            T=0, divs=(1, 1), check_dE=False)
        b = BCSCooling(**args)
        x = b.xyz[0]
        psi_init = random_gaussian_mixing(x)
        Vs = get_potentials(x)
        cooling_para_list = get_cooling_potential_setting()
        paras = []
        use_abm = True
        dE_dt = 1
        for g in [0, 1]:
            for key in Vs:
                for para in reversed(cooling_para_list):
                    args = dict(
                        N=N, dx=dx, eps=1e-1, V=Vs[key], V_key=key, beta_0=1, g=g,
                        dE_dt=dE_dt, psi=psi_init, use_abm=use_abm, check_dE=False)
                    args.update(para)
                    paras.append(args)
        testCases = PoolHelper.run(test_case_worker, paras=paras, poolsize=10)
        SaveTestCase(ts=testCases, Vs=Vs, psi_init=psi_init)
    else:
        paras = loadDumpedParas()
        Vs = []
        if len(paras) > 0:
            psi_init = paras[0]['psi']
        else:
            psi_init = []
        testCases = PoolHelper.run(test_case_worker, paras=paras, poolsize=1)


def write_sheet(sheet, last_file):
    try:
        if last_file is not None:  # load last saved
            book = xlrd.open_workbook(last_file)
            table = book.sheet_by_name("overall")
            nrows = table.nrows
            ncols = table.nrows
            last_row = sheet.nrows
            for r in range(1, nrows):
                for c in range(ncols):
                    sheet.write(last_row, c, table.cell(r, c).value)
                last_row += 1
    except:
        pass


def benchmark_test_excel(
        N=128, dx=0.1, g=0, Ts=[5], trails=1, use_abm=False,
        beta_0=1, beta_Ks=[0], beta_Vs=[10], beta_Ds=[0], beta_Ys=[0],
        ground_state="Gaussian", init_state_key="ST", V_key="HO",
        time_out=120, T_ground_state=10, last_file=None):
    """
    this function is provided to perform test the cooling vs wall time
    calling this function will create a excel file that summarize the 
    results, including all parameters used for each case.
    """
    # create an excel table to store the result
    file_name = (
        f"TestCase_N{N}_dx{dx}_g{g}_T{5}_Trails{trails}"
        +f"_IS={init_state_key}_V={V_key}_"
        +time.strftime("%Y_%m_%d_%H_%M_%S.xls"))
    output = xlwt.Workbook(encoding='utf-8')
    sheet = output.add_sheet("overall", cell_overwrite_ok=True)
    col = 0
    row = 0
    headers = [
        "Trail#", "Time", "N", "dx", "beta_0", "beta_V", "beta_K",
        "beta_D", "beta_Y", "g", "V", "Ground State", "Init State",
        "E0(Ground)", "Ei(Init)", "Ef(Final)", "Evolver",
        "Cooling Effect", "Physical Time", "Wall Time"]
    for value in headers:
        sheet.write(row, col, value)
        col += 1
    write_sheet(sheet, last_file=last_file)
    psis_init = get_init_states()
    psi_init = psis_init[init_state_key]
    b = BCSCooling(N=N, dx=dx)
    x = b.xyz[0]
    Vs = get_potentials(x)
    args = dict(
        N=N, dx=dx, eps=1e-1, T_ground_state=T_ground_state, V=Vs[V_key], V_key=V_key,
        g=g, psi_init=psi_init, use_abm=use_abm, check_dE=False, time_out=time_out)
    t=TestCase(ground_state_eps=1e-1, beta_0=beta_0, **args)
    row = 1
    for trail in range(trails):
        for beta_Y in beta_Ys:
            t.b.beta_Y = beta_Y
            for beta_D in beta_Ds:
                t.b.beta_D = beta_D
                for beta_K in beta_Ks:
                    t.b.beta_K = beta_K
                    for beta_V in beta_Vs:
                        t.b.beta_V = beta_V
                        for T in Ts:
                            print(
                                f"Trail#={trail}: beta_V={beta_V}, beta_K={beta_K},"
                                +f"beta_D={beta_D}, beta_Y={beta_Y},"
                                +f"g={g}, T={T}, V={V_key}, N={N},dx={dx}")
                            try:
                                if beta_V == 0 and beta_K== 0 and beta_Y==0:
                                    continue
                                t.run(T=T, plot=False)
                                wall_time = t.wall_time[-1]
                                E0 = t.E0
                                Ei, Ef = t.E_init, t.Es[-1]
                                dEf = (Ef - E0)/E0
                                col = 0
                                values = [
                                    trail, time.strftime("%Y/%m/%d %H:%M:%S"), N, dx,
                                    beta_0, beta_V, beta_K, beta_D, beta_Y, g, V_key,
                                    ground_state, init_state_key, E0, Ei, Ef]
                                for value in values:
                                    sheet.write(row, col, value)
                                    col += 1
                                Evoler = "ABM" if t.use_abm else "IVP"
                                sheet.write(row, col, Evoler)
                                col+=1
                                if abs(dEf) < 1:
                                    sheet.write(row, col, "Cooled")
                                elif abs((Ef - Ei)/Ei)<0.01:
                                    sheet.write(row, col, "Failed")
                                else:
                                    sheet.write(row, col, "Partially Cooled")
                                col+=1
                                sheet.write(row, col, T)
                                col+=1
                                sheet.write(row, col, wall_time)
                                col+=1
                                row+=1
                                output.save(file_name)
                                print(f"E0={E0}, Ei={Ei}, Ef={Ef}: Saved to {file_name}")
                            except:
                                continue


def do_case_test_excel():
    """
    a function benchmarks on wall time for given set of parameters.
    change parameters below as needed.
    """
    N=128
    dx=0.2
    g = -1
    beta_0=1
    time_out=60
    use_abm=False
    beta_Vs = np.linspace(10, 100, 10)
    beta_Ks = np.linspace(0, 100, 11)
    Ts = np.linspace(0.001, 5, 20)
    beta_Ds = [0]
    beta_Ys = [0]
    for init_state_key in get_init_states():
        for V_key in ["HO"]:
            benchmark_test_excel(
                N=N, dx=dx, g=g, trails=1, Ts=Ts, use_abm=use_abm, time_out=time_out,
                beta_Vs=beta_Vs, beta_Ks=beta_Ks, beta_Ds=beta_Ds, beta_0=beta_0,
                beta_Ys=beta_Ys, init_state_key=init_state_key, V_key=V_key)


if __name__ == "__main__":
    do_case_test_excel()
