from mmf_hfb.BCSCooling import BCSCooling
from mmf_hfb.ParallelHelper import PoolHelper
from mmf_hfb.SolverABM import ABMEvolverAdapter
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import json
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
            self, V, g, N, dx, eps=1e-2, ground_state_eps=1e-5,
            psi_init=None, psi_ground=None, E0=None, T_max=10,
            T_ground_state=None, V_key=None, use_abm=True, **args):
        args.update(g=g, N=N, dx=dx)
        b = BCSCooling(**args)
        self.use_abm = use_abm
        self.solver = ABMEvolverAdapter if use_abm else None
        self.x = b.xyz[0]
        self.b=b
        self.V = V
        self.V_key = V_key
        self.eps = eps
        self.ground_state_eps = ground_state_eps
        self.T_max = T_max
        self.g = g
        self.N = N
        self.dx = dx
        if psi_init is None:
            psi_init = random_gaussian_mixing(self.x)
        self.psi_init = psi_init
        if psi_ground is None:
            psi_ground = self.get_ground_state(
                T=T_ground_state, psi_init=psi_init)
        self.psi_ground = psi_ground
        self.E0=E0

    def get_ground_state(self, psi_init, T=None,  plot=False):
        b = BCSCooling(N=self.b.N, dx=self.b.dx, beta_0=-1j)
        H = b._get_H(mu_eff=0, V=self.V)
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
                    [psi_init], T=T, V=self.V, solver=self.solver,
                    rtol=1e-5, atol=1e-6, method='BDF')
                psis = psiss[0]
                assert len(psis) > 2
                E1, _ = b.get_E_Ns([psis[-1]], V=self.V)
                E2, _ = b.get_E_Ns([psis[-2]], V=self.V)
                if np.isnan(E1) or np.isnan(E2):
                    print(f"Value error: E1={E1}, E2={E2}")
                    raise ValueError("Failed to get ground state.")
                print((E2 - E1)/E1)
                if abs((E2 - E1)/E1) < self.ground_state_eps:
                    return psis[-1]
            raise ValueError("Failed to cool down to ground state.")

    def run(self, N_T=10, T=None, plot=False, plot_log=True, plot_dE=False):
        b, x = self.b, self.x
        E0, _ = b.get_E_Ns([self.psi_ground], V=self.V)
        print(f"E0={E0}")
        self.E0 = E0
        if T is None:
            Ts = (np.array(list(range(N_T)))+1)*0.5
        else:
            Ts = [T]
        args = dict(rtol=1e-5, atol=1e-6, V=self.V, solver=self.solver, method='BDF')
        self.physical_time = []
        self.wall_time = []
        self.Es = []
        self.psis = []
        for T in reversed(Ts):
            start_time = time.time()
            ts, psiss = b.solve([self.psi_init], T=T, **args)
            wall_time = time.time() - start_time
            E, _ = b.get_E_Ns([psiss[0][-1]], V=self.V)
            self.wall_time.append(wall_time)
            self.physical_time.append(T)
            self.Es.append(E)
            self.psis.append(psiss[0][-1])
            print(f"physical time:{T}, wall time:{wall_time},dE:{(E-E0)/abs(E0)} ")
            
            if plot:
                Es = [b.get_E_Ns([_psi], V=self.V)[0] for _psi in psiss[0]]
                plt.subplot(131)
                plt.plot(x, self.V, label=self.V_key)
                plt.title(f"g={b.g}")
                plt.legend()
                plt.subplot(132)
                plt.plot(x, Prob(psiss[0][0]), "+", label='init')
                plt.plot(x, Prob(psiss[0][-1]), '--',label="final")
                plt.plot(x, Prob(self.psi_ground), label='Ground')
                plt.title(
                    f"E0={self.E0:5.4},E={E:5.4}, $" + r"\beta_0$" +f"={b.beta_0}, "
                    +r"$\beta_V$"+ f"={b.beta_V}, "+ r" $\beta_K$" + f"={b.beta_K}"
                    +r" $\beta_D$" + f"={b.beta_D}"+ r" $\beta_Y$" + f"={b.beta_Y}")
                plt.legend()
                plt.subplot(133)
                alt_text = "ABM" if self.use_abm else "IVP"
                if plot_log:
                    plt.semilogy(ts[0][:-1], (Es[:-1] - E0)/abs(E0), label=f"E({alt_text})")
                else:
                    plt.plot(ts[0][:-1], (Es[:-1] - E0)/abs(E0), label=f"E({alt_text})")
                if plot_dE:
                    dE_dt= [-1*b.get_dE_dt([_psi], V=self.V) for _psi in psiss[0]]
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
    return dict(HO=V_HO, PO=V_PO)


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
    psi_random = Normalize(np.random.random(N), dx=dx)
    psi_standing_wave=Normalize(U0[1],dx=dx)
    psi_gaussian_mixing = random_gaussian_mixing(x, dx=dx)
    psi_uniform = Normalize(U0[0], dx=dx)
    return dict(RN=psi_random, ST=psi_standing_wave, GM=psi_gaussian_mixing, UN=psi_uniform)


def test_scatch():
    import matplotlib.pyplot as plt
    psis_init = get_init_states()

    b = BCSCooling(N=128, dx=0.1)
    x = b.xyz[0]
    psi_init= psis_init['UN']
    Vs = get_potentials(x)
    V=Vs['HO']
    args = dict(N=128, dx=0.1, eps=1e-1, V=0*V, V_key='0', g=1, psi=psi_init, use_abm=False, check_dE=False)
    t=TestCase(ground_state_eps=1e-4, **args)
    plt.figure(figsize=(18,5))
    t.b.beta_V= 0
    t.b.beta_K = 300
    t.run(T=5, plot=True, plot_log=False)

if __name__ == "__main__":
    test_scatch()
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
        use_abm=True
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
