from mmf_hfb.BCSCooling import BCSCooling
from mmf_hfb.ParallelHelper import PoolHelper
from mmf_hfb.SolverABM import ABMEvolverAdapter
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import os
import datetime


class TestCase(object):
    
    def __init__(self, V, g, N, dx, eps=1e-2, psi=None, max_T=10, **args):
        args.update(g=g, N=N, dx=dx)
        b = BCSCooling(**args)        
        self.x = b.xyz[0]
        self.b=b
        self.V = V
        self.eps = eps
        self.max_T = max_T
        self.g = g
        self.N = N
        self.dx = dx
        self.psi = psi if psi is not None else TestCase.random_gaussian_mixing(self.x)
        self.psi0 = self.get_ground_state()

    def random_gaussian_mixing(x):
        n = np.random.randint(1, 10)
        cs = np.random.random(n)
        ns = np.random.randint(1, 10, size=n)    
        ys = sum([c*np.exp(-x**2/n**2) for (c, n) in zip(cs, ns)])
        return TestCase.Normalize(ys)

    def Normalize(psi, dx=0.1):
        return psi/(psi.dot(psi.conj())*dx)**0.5

    def Prob(self, psi):
        return np.abs(psi)**2

    def get_ground_state(self):
        b = self.b
        H = b._get_H(mu_eff=0, V=self.V)
        U, E = b.get_U_E(H, transpose=True)
        psi0 = TestCase.Normalize(U[0], dx=b.dx)
        self.psi_0 = psi0
        if self.g == 0:            
            return psi0
        else:
            # imaginary cooling
            for T in range(2, 10, 1):
                print(f"Imaginary Cooling with T={T}")
                ts, psiss = b.solve(
                    [psi0], T=5, beta_0=-1j, V=self.V, solver=ABMEvolverAdapter)
                psis = psiss[0]
                assert len(psis) > 2
                E1, _ = b.get_E_Ns([psis[-1]], V=self.V)
                E2, _ = b.get_E_Ns([psis[-2]], V=self.V)
                if np.isnan(E1) or np.isnan(E2):
                    print(f"Value error: E1={E1}, E2={E2}")
                    raise ValueError("Failed to get ground state.")
                print((E2 - E1)/E1)
                if abs((E2 - E1)/E1)<1e-5:
                    return psis[-1]
            raise Exception("Failed to cool down to ground state.")

    def run(self):
        b = self.b
        E0, _ = b.get_E_Ns([self.psi0], V=self.V)
        print(f"E0={E0}")
        self.E0 = E0
        Ts = (np.array(list(range(10)))+1)*0.5
        args = dict(rtol=1e-5, atol=1e-6, V=self.V, solver=ABMEvolverAdapter)
        self.physical_time = []
        self.wall_time = []
        self.Es = []
        self.psis = []
        for T in reversed(Ts):
            start_time = time.time()
            ts, psiss = b.solve([self.psi], T=T, **args)
            wall_time = time.time()-start_time
            E, _ = b.get_E_Ns([psiss[0][-1]], V=self.V)           
            self.wall_time.append(wall_time)
            self.physical_time.append(T)
            self.Es.append(E)
            self.psis.append(psiss[0][-1])
            print(f"physical time:{T}, wall time:{wall_time},dE:{(E-E0)/abs(E0)} ")
            if abs((E - E0)/E0) > self.eps:
                break
    
    def plot(self, id=0):
        E=self.Es[id]
        psi = self.psis[id]
        plt.plot(self.x, TestCase.Prob(psi), "--", label='init')
        plt.plot(self.x, TestCase.Prob(self.psis[id]), '+', label="final")
        plt.plot(self.x, TestCase.Prob(self.psi0), label='Ground')
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


def SaveTestCase(ts):
    """Save the test case data to a json file"""
    file_name = "CoolingTestData_"+time.strftime("%Y_%m_%d_%H_%M_%S.json")
    output = []

    def unpack(psi):
        return dict(r=psi.real.tolist(), i=psi.imag.tolist())
    for t in ts:
        if t is None:
            continue
        V = t.V
        b = t.b
        if isinstance(t.V, np.ndarray):
            V = V.tolist()
        dic = dict(
            dx=t.dx, N=t.N, E0=t.E0, max_T=t.max_T, g=t.g, eps=t.eps,
            beta_0=b.beta_0, beta_V=b.beta_V, beta_K=b.beta_K,
            beta_D=b.beta_D, beta_Y=b.beta_Y, V=V, psi0=unpack(t.psi0))
        data = []
        for (E, T, Tw, psi) in zip(t.Es, t.physical_time, t.wall_time, t.psis):
            data.append(dict(E=E, T=T, Tw=Tw, psi=unpack(psi)))
        dic['data']=data
        output.append(dic)
    with open(file_name, 'w') as wf:
        json.dump(output, wf)
    print(f"file {file_name} saved")
    return output


def test_case_worker(para):
    id = os.getpid()
    start_time = time.time()
    print(f"{id} start time:{datetime.datetime.now()}")
    try:
        t = TestCase(**para)
        t.run()
        print(f"{id} Wall time:{time.time() - start_time}")
        return t
    except:
        print(para)
        print(f"{id} Wall time:{time.time() - start_time}")
        return None

    
if __name__ == "__main__":
    N, dx = 128, 0.1
    args = dict(
        N=N, dx=dx,
        beta0=1, beta_K=0, beta_V=0, beta_D=0, beta_Y=0,
        T=0, divs=(1, 1), check_dE=False)
    b = BCSCooling(**args)
    x = b.xyz[0]
    psi_init = TestCase.random_gaussian_mixing(x)
    Vs = get_potentials(x)
    cooling_para_list = get_cooling_potential_setting()
    paras = []
    for g in [0, 1]:
        for key in Vs:
            for para in reversed(cooling_para_list):
                args = dict(
                    N=N, dx=dx, eps=1e-1, V=Vs[key], V_name=key, beta_0=1, g=g,
                    psi=psi_init, check_dE=False)
                args.update(para)
                paras.append(args)
    
    testCases = PoolHelper.run(test_case_worker, paras=paras, poolsize=10)
    SaveTestCase(ts=testCases)
