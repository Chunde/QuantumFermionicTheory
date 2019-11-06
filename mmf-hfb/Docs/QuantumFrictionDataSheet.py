# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import mmf_setup;mmf_setup.nbinit()
import matplotlib.pyplot as plt
# %pylab inline --no-import-all
from nbimports import *
import numpy as np

# # Define Some Helper Functions

# +
from mmf_hfb.BCSCooling import BCSCooling
from mmf_hfb.SolverABM import ABMEvolverAdapter
from mmf_hfb.Cooling import Cooling
from mmf_hfb.Potentials import HarmonicOscillator
from IPython.display import display, clear_output
import time
np.random.seed(1)

def Normalize(psi, dx=0.1):
    return psi/(psi.dot(psi.conj())*dx)**0.5

def Prob(psi):
    return np.abs(psi)**2
   
def check_uv_ir_error(x, psi, plot=False):
    """check if a lattice configuration(N, L, dx) is good"""
    psi_k = np.fft.fft(psi)
    psi_log = np.log10(abs(psi))
    psi_log_k = np.log10(abs(psi_k))
    if plot:
        l, =plt.plot(x,psi_log_k)
        plt.plot(x, psi_log,'--', c=l.get_c())
        print(np.min(psi_log), np.min(psi_log_k))
    assert np.min(psi_log_k) < -15


# -

# # Lattice Configuration

N, dx = 128, 0.1
args = dict(N=N, dx=dx, divs=(1, 1), beta0=1, beta_K=0, beta_V=0, beta_D=0, beta_Y=0, T=0, check_dE=False)
b = BCSCooling(**args)
x = b.xyz[0]
gaussian = np.exp(-x**2)
check_uv_ir_error(x=x, psi=gaussian, plot=True)


# # Compute Ground State
# * For any potential $V$, and any interaction $g$

# +
def get_ground_state(V, **args):
    b = BCSCooling(**args)
    x = b.xyz[0]
    H = b._get_H(mu_eff=0, V=V)
    U, E = b.get_U_E(H, transpose=True)
    return Normalize(U[0], dx=b.dx)

def random_gaussian_mixing(x):
    n = np.random.randint(1, 10)
    cs = np.random.random(n)
    ns = np.random.randint(1, 10, size=n)
    
    ys = sum([c*np.exp(-x**2/n**2) for (c, n) in zip(cs, ns)])
    return Normalize(ys)


# -

y = random_gaussian_mixing(x)
plt.plot(x,y)

# # Check Upper-Bound
# * Roughly check the max value of $\beta_0$, $\beta_V$, $\beta_K$, $\beta_D$, $\beta_Y$

beta_V=60
beta_K=75
beta_D=1000
beta_Y=1

# # List of Cooling Configuration

cooling_para_list=[dict(beta_V=beta_V, beta_K=beta_K, beta_D=beta_D, beta_Y=beta_Y),
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
             dict(beta_V=beta_V, beta_K=0, beta_D=0, beta_Y=0),
]

# ## List of Potentials

args = dict(N=N, dx=dx)
V0 = 0*x
V_HO = x**2/2
V_PO = V0 + np.random.random()*V_HO +  + abs(x**2)*np.random.random()
Vs = [0, V_HO, V_PO]
plt.plot(V_PO)


class TestCase(object):
    def __init__(self, V, g, N, dx, eps=1e-2, psi=None, max_T=10, **args):
        args.update(g=g, N=N, dx=dx)
        self.b = BCSCooling(**args)
        self.x = b.xyz[0]
        self.V = V
        self.eps = eps
        self.max_T = max_T
        self.g = g
        self.N = N
        self.dx = dx
        self.psi = psi if psi is not None else random_gaussian_mixing(self.x)
        self.psi0 = self.get_ground_state()       
        
    def get_ground_state(self):
        b = self.b
        H = b._get_H(mu_eff=0, V=self.V)
        U, E = b.get_U_E(H, transpose=True)
        psi0 = Normalize(U[0], dx=b.dx)
        self.psi_0 = psi0
        if self.g == 0:            
            return psi0
        else:
            # imaginary cooling
            for T in range(2, 10, 1):
                print(f"Imaginary Cooling with T={T}")
                ts, psiss = b.solve([psi0], T=5, beta_0=-1j, V=self.V, solver=ABMEvolverAdapter)
                psis = psiss[0]
                assert len(psis) > 2
                E1, _ = b.get_E_Ns([psis[-1]], V=self.V)
                E2, _ = b.get_E_Ns([psis[-2]], V=self.V)
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
            Es = [b.get_E_Ns([_psi], V=self.V)[0] for _psi in psiss[0]]
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
        plt.plot(self.x, Prob(psi), "--", label='init')
        plt.plot(self.x, Prob(self.psis[id]), '+', label="final")
        plt.plot(self.x, Prob(self.psi0), label='Ground')
        b=self.b
        plt.title(
            f"E0={self.E0:5.4},E={E:5.4}, $" + r"\beta_0$" +f"={b.beta_0}, "
            +r"$\beta_V$"+ f"={b.beta_V}, "+ r" $\beta_K$" + f"={b.beta_K}"
            +r" $\beta_D$" + f"={b.beta_D}"+ r" $\beta_Y$" + f"={b.beta_Y}")
        plt.legend()


psi_init = random_gaussian_mixing(x)
# args = dict(N=N, dx=dx, eps=1e-1, V=V_PO, beta_V=beta_V, g=1, psi=psi_init, check_dE=False)
# testCases = []
# t = TestCase(**args)
# t.run(plot=True)
# testCases.append(t)

psi_init = random_gaussian_mixing(x)
paras = []
for g in [0, 1]:
    for V in Vs:        
        for para in reversed(cooling_para_list):
            args = dict(N=N, dx=dx, eps=1e-1, V=V, beta_0=1, g=g, psi=psi_init, check_dE=False)
            args.update(para)
            paras.append(args)

from mmf_hfb.ParallelHelper import PoolHelper
def test_case_worker(para):
    t = TestCase(**para)
    t.run()
    return t
#res = PoolHelper.run(mqaud_worker_thread, paras=obj_twists_kp)


# +
#t0 = test_case_worker(paras[40])

# +
# res = PoolHelper.run(test_case_worker, paras=paras)
# -

testCases = []
for args in paras:
    t = test_case_worker(args)
    t.run()
    testCases.append(t)

t0 = testCases[0]

t0.V


# +
def show_des(t):
    b = t.b
    print(f"beta_0={b.beta_0} beta_V={b.beta_V} beta_K={b.beta_K} beta_D={b.beta_D} beta_Y={b.beta_Y}")
    print(t.E0, t.wall_time)
    print(f"{t.dE*100/t.E0}%100")
    
for t in testCases:
    show_des(t)


# -

def test_cooling(plot_dE=True, use_ABM=False, T=0.5, plt_log=True, **args):   
    b = BCSCooling(**args)
    solver = ABMEvolverAdapter if use_ABM else None
    h0 = HarmonicOscillator(w=1)
    h = HarmonicOscillator()
    da, db=b.divs    
    x = b.xyz[0]
    V = x**2/2
    H0 = b._get_H(mu_eff=0, V=0)
    H1 = b._get_H(mu_eff=0, V=V)
    U0, E0 = b.get_U_E(H0, transpose=True)
    U1, E1 = b.get_U_E(H1, transpose=True)
    psi0 = h.get_wf(x)
    psi0 = U1[0]
    psi0 = Normalize(psi0, dx=b.dx)
    psi = h0.get_wf(x, n=2)
    psi = random_gaussian_mixing(x) # U0[1]
    psi = Normalize(psi, dx=b.dx)
    #b.erase_max_ks()
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    N0 = psi0.conj().dot(psi0)
    ts, psiss = b.solve([psi], T=T, rtol=1e-5, atol=1e-6, V=V, solver=solver, method='BDF')
    E0, _ = b.get_E_Ns([psi0], V=V)
    Es = [b.get_E_Ns([_psi], V=V)[0] for _psi in psiss[0]]   
    plt.plot(x, Prob(psiss[0][0]), "+", label='init')
    plt.plot(x, Prob(psiss[0][-1]), '--',label="final")
    plt.plot(x, Prob(psi0), label='Ground')
    plt.legend()
    plt.subplot(122)
    if plt_log:
        plt.semilogy(ts[0][:-2], (Es[:-2] - E0)/abs(E0), label="E")
    else:
        plt.plot(ts[0][:-2], (Es[:-2] - E0)/abs(E0), label="E")
    if plot_dE:
        dE_dt= [-1*b.get_dE_dt([_psi], V=V) for _psi in psiss[0]]
        plt.plot(ts[0][:-2], dE_dt[:-2], label='-dE/dt')
        plt.axhline(0, linestyle='dashed')
    plt.legend()
    plt.axhline(0, linestyle='dashed')
    plt.show()    
    return psiss[0][-1]


# %%time 
args = dict(N=N, dx=dx,beta_V=beta_V, beta_K=0,  beta_D=0, beta_Y=0, T=5, divs=(1, 1), use_ABM=True, plt_log=True, check_dE=False)
psi = test_cooling(plot_dE=False, **args)


def test_cooling(plot_dE=True, use_ABM=False, T=0.5, **args):   
    b = BCSCooling(**args)
    solver = ABMEvolverAdapter if use_ABM else None
    h0 = HarmonicOscillator(w=1)
    h = HarmonicOscillator()
    da, db=b.divs    
    x = b.xyz[0]
    V = x**2/2
    H0 = b._get_H(mu_eff=0, V=0)
    H1 = b._get_H(mu_eff=0, V=V)
    U0, E0 = b.get_U_E(H0, transpose=True)
    U1, E1 = b.get_U_E(H1, transpose=True)
    psi0 = h.get_wf(x)
    psi0 = U1[0]
    psi0 = Normalize(psi0, dx=b.dx)
    psi = h0.get_wf(x, n=2)
    psi = random_gaussian_mixing(x) # U0[1]
    psi = Normalize(psi, dx=b.dx)
    #b.erase_max_ks()
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    N0 = psi0.conj().dot(psi0)
    ts, psiss = b.solve([psi], T=T, rtol=1e-5, atol=1e-6, V=V,solver=solver, method='BDF')
    E0, _ = b.get_E_Ns([psi0], V=V)
    Es = [b.get_E_Ns([_psi], V=V)[0] for _psi in psiss[0]]   
    plt.plot(x, Prob(psiss[0][0]), "+", label='init')
    plt.plot(x, Prob(psiss[0][-1]), '--',label="final")
    plt.plot(x, Prob(psi0), label='Ground')
    plt.legend()
    plt.subplot(1,2,2)
    plt.semilogy(ts[0][:-2], (Es[:-2] - E0)/abs(E0), label="E")
    if plot_dE:
        dE_dt= [-1*b.get_dE_dt([_psi], V=V) for _psi in psiss[0]]
        plt.plot(ts[0][:-2], dE_dt[:-2], label='-dE/dt')
        plt.axhline(0, linestyle='dashed')
    plt.legend()
    plt.axhline(0, linestyle='dashed')
    plt.show()    
    return psiss[0][-1]


# %%time 
args = dict(N=N, dx=dx, beta_0=1, beta_K=130, beta_V=0, beta_D=0, beta_Y=0, T=5, divs=(1, 1), use_ABM=True, check_dE=False)
psi = test_cooling(plot_dE=False, **args)


def Cooling(plot_dE=True, use_ABM=False, T=0.5, **args):
    b = BCSCooling(**args)
    solver = ABMEvolverAdapter if use_ABM else None
    h0 = HarmonicOscillator(w=1)
    h = HarmonicOscillator()
    da, db=b.divs    
    x = b.xyz[0]
    V =  x**2/2
    psi_0 = Normalize(np.sqrt(1.0 + 0.2*x*np.exp(-x**2)))
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    ts, psiss = b.solve([psi_0], T=T, rtol=1e-5, atol=1e-6, V=V,solver=solver, method='BDF')
    psi0 = h0.get_wf(x=x, n=0)
    N0 = psi0.conj().dot(psi0)
    Ns = [_psi.conj().dot(_psi) for _psi in psiss[0]]
    E0, _ = b.get_E_Ns([psi0], V=V)
    Es = [b.get_E_Ns([_psi], V=V)[0] for _psi in psiss[0]]
    dE_dt= [-1*b.get_dE_dt([_psi], V=V) for _psi in psiss[0]]
    plt.plot(x, Prob(psiss[0][0]), "+", label='init')
    plt.plot(x, Prob(psiss[0][-1]), '--',label="final")
    plt.plot(x, Prob(psi0), label='Ground')
    plt.legend()
    plt.subplot(1,2,2)
    plt.semilogy(ts[0][:-2], (Es[:-2] - E0)/abs(E0), label="E")
    if plot_dE:
        plt.plot(ts[0][:-2], dE_dt[:-2], label='-dE/dt')
        plt.axhline(0, linestyle='dashed')
    plt.legend()
    plt.axhline(0, linestyle='dashed')
    plt.show()    
    return psiss[0][-1]


# %%time 
args = dict(N=128, dx=0.1,beta_0=-1j, beta_K=0, beta_V=0, beta_D=0, beta_Y=0, T=5, divs=(1, 1), use_ABM=True, check_dE=False)
psi = Cooling(plot_dE=False, **args)

args = dict(N=4, g=1)
egs = [BCSCooling(beta_0=-1j, beta_V=0.0, beta_K=0.0, **args),
       BCSCooling(beta_0=0.0, beta_V=0.0, beta_K=1.0, **args),
       BCSCooling(beta_0=1.0, beta_V=0.0, beta_K=1.0, **args),      
       BCSCooling(beta_0=0.0, beta_V=1.0, beta_K=0.0, **args),
       BCSCooling(beta_0=1.0, beta_V=1.0, beta_K=0.0, **args),
       BCSCooling(beta_0=0.0, beta_V=1.0, beta_K=1.0, **args),
       BCSCooling(beta_0=1.0, beta_V=1.0, beta_K=1.0, **args)]
labels = ['Imaginary Time',
          'K', 'H+K',
          'V', 'H+V',
          'V+K', 'H+V+K']
eg = egs[0]
psi0 = 2*(np.random.random(eg.Nxyz[0]) + 1j*np.random.random(eg.Nxyz[0]) - 0.5 - 0.5j)
V = np.array(psi0)*0
x=egs[0].xyz[0]
#psi0 = 0*x + 1.5 + 1.5*np.exp(-x**2/2)
psi_ground = 0*psi0 + np.sqrt((abs(psi0)**2).mean())
E0, N0 = eg.get_E_Ns([psi_ground], V=V)
Es = [[] for _n in range(len(egs))]
psis = [psi0.copy() for _n in range(len(egs))]
t_max = 3.0
Nstep = 4
Ndata = int(np.round(t_max/eg.dt/Nstep))
ts = np.arange(Ndata)*Nstep*eg.dt
for _n in range(Ndata):
    for n, eg in enumerate(egs):
        ps = [psis[n]]
        ps = eg.step(psis=ps, n=Nstep, V=V)
        psis[n] = ps[0]
        E, N = eg.get_E_Ns(psis=ps, V=V) 
        Es[n].append(E/E0 - 1.0)
Es = np.asarray(Es)

plt.semilogy(ts, Es[1], c='C0', ls=':', label=labels[1])
plt.semilogy(ts, Es[2], c='C0', ls='-', label=labels[2])
plt.semilogy(ts, Es[3], c='C1', ls=':', label=labels[3])
plt.semilogy(ts, Es[4], c='C1', ls='-', label=labels[4])
plt.semilogy(ts, Es[5], c='C2', ls=':', label=labels[5])
plt.semilogy(ts, Es[6], c='C2', ls='-', label=labels[6])
plt.semilogy(ts, Es[0], c='k', ls='-', label=labels[0], scaley=False)
plt.xlabel("t")
plt.ylabel("E-E0")
plt.legend()


