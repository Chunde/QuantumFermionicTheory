# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import mmf_setup;mmf_setup.nbinit()
import matplotlib.pyplot as plt
# %pylab inline --no-import-all
from nbimports import *
import sys
import numpy as np
import inspect
from os.path import join
import json
import glob
import os
from IPython.display import display, clear_output
currentdir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, join(currentdir, '..','Projects', 'QuantumFriction'))
from mmf_hfb.potentials import HarmonicOscillator
from abm_solver import ABMEvolverAdapter
from bcs_cooling import BCSCooling
from cooling_case_tests import TestCase, Prob, Normalize, random_gaussian_mixing


# # 1D cooling

# +
def get_init_states(N=128, dx=0.1):
    b = BCSCooling(N=N, dx=dx)
    h = HarmonicOscillator()
    x = b.xyz[0]
    V = x**2/2
    H0 = b._get_H(mu_eff=0, V=0)
    H1 = b._get_H(mu_eff=0, V=V)
    U0, E0 = b.get_psis_es(H0, transpose=True)
    U1, E1 = b.get_psis_es(H1, transpose=True)
    psi_standing_wave=Normalize(U0[1],dx=dx)
    psi_gaussian_mixing = random_gaussian_mixing(x, dx=dx)
    psi_uniform = Normalize(U0[0], dx=dx)
    psi_bright_soliton = Normalize(np.exp(-x**2/2.0)*np.exp(1j*x), dx=dx)
    return dict(ST=psi_standing_wave, GM=psi_gaussian_mixing, UN=psi_uniform, BS=psi_bright_soliton)

def PN(psi, dx):
    return Prob(Normalize(psi, dx=dx))

def get_potentials(x):
    V0 = 0*x
    V_HO = x**2/2
    V_PO = V0 + np.random.random()*V_HO + abs(x**2)*np.random.random()
    return dict(V0=V0, HO=V_HO, PO=V_PO)
h = HarmonicOscillator()

# +
fontsize=18

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : fontsize}
matplotlib.rc('font', **font)
matplotlib.rcParams.update({'font.size': fontsize})

plt.figure(figsize=(16,8))
N = 513
dx=0.1/2
L = N*dx
xs = np.linspace(0,L,N)-L/2
ys = get_init_states(N=N, dx=dx)
plt.axhline(0, c='black', ls='dashed')
plt.axvline(0, c='black', ls='dashed')
i = 0
labels = ['-', '--', '-+','-o']
for key in ys:
    plt.plot(xs, ys[key].conj()*ys[key], labels[i], label=key)
    i += 1

plt.xlabel("x", fontsize=fontsize)
plt.ylabel(r"$|\psi(x)|^2$", fontsize=fontsize)
plt.legend(prop={"size":fontsize})
plt.savefig("initial_state_densities.pdf", bbox_inches='tight')


# -

# # Test Bed
# * code used to check results generated by the code in [CoolingCaseTests.py](CoolingCaseTests.py)

def Check_Test_Case(beta_H=1, beta_V=10, beta_K=0,  N=128, dx=0.2, g=1, Tp=20, Tg=20, V_key = 'V0', psi_key='ST'):
    psis_init = get_init_states(N=N, dx=dx)
    psi_init= psis_init[psi_key]
    b = BCSCooling(N=N, dx=dx)
    x = b.xyz[0]
    Vs = get_potentials(x)
    args = dict(
        N=N, dx=dx, eps=1e-1, T_ground_state=Tg, V=Vs[V_key],
        V_key=V_key, g=g, psi_init=psi_init, use_abm=False, check_dE=False)
    t=TestCase(ground_state_eps=1e-1, beta_0=1, beta_H=beta_H, **args)
    plt.figure(figsize=(15, 5))
    t.b.beta_V=beta_V
    t.b.beta_K=beta_K
    t.run(T=Tp, plot=True, plot_log=False)
    print(t.E_init, t.Es[-1])


# +
#Check_Test_Case(g=-1, Tg=5, Tp=5,beta_V=10, beta_K=0, psi_key="ST", V_key="HO")
# -

# ## Check Overall factor $\beta_H$ vs Wall Time

def test_wall_time(N=128, dx=0.2, beta_V=0, beta_K=0, beta_Y=0):
    psis_init = get_init_states(N=N, dx=dx)
    psi_init= psis_init["ST"]
    b = BCSCooling(N=N, dx=dx)
    x = b.xyz[0]
    V_key="HO"
    g = 0
    Vs = get_potentials(x)
    args = dict(
        N=N, dx=dx, eps=1e-1, T_ground_state=5, V=Vs[V_key],
        V_key=V_key, g=g, psi_init=psi_init, use_abm=False, check_dE=False)
    t=TestCase(ground_state_eps=1e-1, beta_0=1, **args)
    
    t.b.beta_V= beta_V
    t.b.beta_K = beta_K
    t.b.keta_Y = beta_Y
    
    for beta_H in [1, 2, 3, 4]:
        Es = []
        Tsw = []
        t.b.beta_H=beta_H
        Ts = np.linspace(0, 2, 20)/beta_H
        for T in Ts:
            t.run(T=T, plot=False, plot_log=False)
            Es.append(t.Es[-1])
            Tsw.append(t.wall_time[-1])
        plt.loglog(Tsw, (np.array(Es) - t.E0)/t.E0, label=f"beta_H={beta_H}")
    plt.xlabel("Wall Time")
    plt.ylabel("(E-E0)/E0")
    plt.legend()


def Test_Beta_H():
    plt.figure(figsize=(18, 5))
    plt.subplot(131)
    test_wall_time(beta_V=20, beta_K=0, beta_Y=0)
    plt.subplot(132)
    test_wall_time(beta_V=20, beta_K=20, beta_Y=0)
    plt.subplot(133)
    test_wall_time(beta_V=0, beta_K=20, beta_Y=0)
    clear_output()


# # 2D cooling
# * not that slow

def test_2d_Cooling():
    s = BCSCooling(N=32, dx=0.1, beta_0=-1.0j, beta_V=0.0, beta_K=0.0, g=0, dim=2)
    x, y = s.xyz
    V = sum(_x**2 for _x in s.xyz)
    s.V = np.array(V)/2
    x0 = 0.5
    phase = ((x-x0) + 1j*y)*((x+x0) - 1j*y)
    psi0 = s.Normalize(1.0*np.exp(1j*np.angle(phase)))
    ts, psis, _ = s.solve([psi0], T=5.0, rtol=1e-5, atol=1e-6)
    plt.subplot(121)
    s.plot(psis[-1][0], show_plot=False, show_title=False)
    plt.subplot(122)
    Es = [s.get_E_Ns(psi)[0] for psi in psis]
    plt.semilogy(ts, Es)


plt.figure(figsize=(16, 6))
test_2d_Cooling()

# # Load CVS file

import pandas as pd 
import sys

currentdir = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),"..","projects","FFStateData", "CoolingData")
#currentdir = 'E:\Physics\quantum-fermion-theories\mmf-hfb\FFStateData\CoolingData'

currentdir


# ## Excel to CVS
# * conver excel file to cvs

# +
def post_process(df0, index,  columns):
    """remove some uncessary columns"""
    if index > 0:
        df0.columns.values[columns - 1] = "wTime0"
        df0['wTime'] = df0[[f"wTime{id}" for id in range(index + 1)]].mean(axis=1)
        for id in range(index + 1):
            del df0[f"wTime{id}"]
    del df0['Trail'] 
    del df0['Time']
    if 'N' in df0:
        del df0['N']
    if 'dx' in df0:
        del df0['dx']
    del df0['beta_D']
    del df0['beta_Y']
    if 'gState' in df0:
        del df0['gState']
    del df0['cooling']
    del df0['Evolver']
    return df0
        
def MergeExcels(merge=True):
    dfs = []
    # change the data file path if location is different
    file_paths = glob.glob(join(currentdir, "kamiak", "BCS", "*.xls")) # 1D_without_E_E0_nfev
    file_paths.sort()
    files = [os.path.basename(file[:file.index("_PID")]) for file in file_paths]
    file0 = files[0]
    df0 = pd.read_excel(file_paths[0], sheet_name="overall")
    columns = len(df0.columns)
    index = 0
    for i, file_path in enumerate(file_paths):
        if i == 0:
            continue
        if file0 == files[i]:
            df = pd.read_excel(file_path, sheet_name="overall")
            last_col = df["wTime"]
            index += 1
            df0[f'wTime{index}'] = last_col.values
        else:           
            dfs.append(post_process(df0=df0, index=index, columns=columns))
            # next set
            file0 = files[i]
            df0 = pd.read_excel(file_paths[i], sheet_name="overall")
            columns = len(df0.columns)
            index = 0
    dfs.append(post_process(df0=df0, index=index, columns=columns))
    if merge:
        return pd.concat(dfs)
    return dfs


def convert2CSV():
    files = glob.glob(join(currentdir, "*.xls"))
    for file in files:
        csv_file = os.path.splitext(file)[0] + ".csv"
        if os.path.exists(csv_file):
            continue
        data_xls = pd.read_excel(file, 'overall', index_col=None)
        data_xls.to_csv(csv_file, encoding='utf-8')
        print(f"generated file:{cvs_file}")
        
        
def CombineExcelSheetsToCSV(file_path=None):
    if file_path is None:
        file_path = join(currentdir, 'CoolingTestData1D.xlsx')

    df = pd.concat(pd.read_excel(file_path, sheet_name=None), ignore_index=True)
    file_stem = file_path.split()[0] # not right
    df.to_csv( join(file_stem +".csv"), encoding='utf-8')
    return df


def ReadAllExcelFile():
    files = glob.glob(join(currentdir, "*.csv"))
    data = pd.concat([pd.read_csv(file) for file in files])
    data.to_excel(join(currentdir, "data.xlsx"), sheet_name='overall')
    return data


# -

# data = MergeExcels()
data = CombineExcelSheetsToCSV()
#data.to_csv("merged_data.csv", encoding='utf-8')

# ## Query

iStates =None
gs = None
if 'iState' in data:
    data_key = 'iState'
else:
    data_key = 'N_state'
if data_key in data:
    iStates = set(data[data_key])
if 'g' in data:
    gs = set(data['g'])
beta_Vs = set(data['beta_V'])
beta_Ks = set(data['beta_K'])


def find_best_betas(data, p=1.01):
    dfs = []
    df = data[data.E0*p > data.Ef] # find all rows with Ef in per*E0
    def _append_kv(df):
        if df.empty == False:
            df3 = df[df.beta_K==0] # with beta_K = 0
            if df3.empty == False:
                df_v = df3.loc[df3['wTime'].idxmin()]
                if df_v.empty == False:
                    dfs.append(df_v)
            df4 = df[df.beta_K!=0]
            if df4.empty == False:
                df_kv = df4.loc[df4['wTime'].idxmin()]
                if df_kv.empty == False:
                    dfs.append(df_kv)
                        
    def append_kv(df):
        if gs is None:
            _append_kv(df=df)
        else:
            for g in gs:
                df2 = df[df.g == g]
                _append_kv(df = df2)
            
    if iStates is not None:                    
        for iState in iStates:
            df1 = df[df[f"{data_key}"] == iState]
            append_kv(df=df1)
    else:
        append_kv(df=df)
    if len(dfs) == None:
        return (None, None)
    if len(dfs)==0:
        return (None, None)
    output =pd.concat(dfs, axis=1).transpose()   
    output.reset_index(drop=True, inplace=True)
    
    if iStates is not None:
        dict_kv = {}
        
        def find_dict_kvs(g_states):
            if g_states.empty:
                kvs.append((0, 0, 0))
            else:
                v_state = g_states[g_states.beta_K == 0]
                kv_state = g_states[g_states.beta_K !=0]
                if v_state.empty:
                    kvs.append((0, kv_state.beta_V.values[0], kv_state.beta_K.values[0]))
                elif kv_state.empty:
                    kvs.append((v_state.beta_V.values[0], 0, 0))
                else:
                    kvs.append((v_state.beta_V.values[0], kv_state.beta_V.values[0], kv_state.beta_K.values[0]))
                dict_kv[state] = kvs      
                
        for state in iStates:
            kvs = []
            df_states = output[output[f"{data_key}"] == state]
            if gs is not None:
                for g in [-1, 0, 1]:
                    g_states = df_states[df_states.g == g]
                    find_dict_kvs(g_states=g_states)
            else:
                    g_states = df_states
                    find_dict_kvs(g_states=g_states)
                      
        return (output,dict_kv)
    else:
        return (output, None)


# ## Plot $(E-E_0)/E_0$ vs Wall-Time

# +
fontsize=16
def get_Es_Ts(beta_K, beta_V, iState, V, g, use_nfev):
    sql = f"beta_K=={beta_K} and beta_V=={beta_V}"
    if g is not None:
        sql = sql + f" and g=={g}"
    if iState is not None:
        sql = sql + f" and {data_key}=='{iState}'"
    if V is not None:
        sql = sql + f" and V=='{V}'"
    res = data.query(sql)
    Ts = res['wTime']
    E0 = res['E0']
    Ef = res['Ef']
    if use_nfev and 'nfev' in res:
        Ts = res['nfev']
    dE = (Ef- E0)/E0
    return dE, Ts

def plot_Es_Ts(beta_K, beta_V, g, V, iState, use_nfev, line='-', style=None, c=None):
    Es, Ts = get_Es_Ts(beta_K=beta_K, beta_V=beta_V, V=V, g=g, iState=iState, use_nfev=use_nfev)
    if Ts is None or len(Ts) ==0:
        return (None, None, None)
    x = Ts
    y = Es
    state="State"
    if len(y) > 0:
        if style is None:
            l, = plt.plot(x, y, line, c=c, label=r"$\beta_V$"+f"={beta_V/100},"+r"$\beta_K$"+f"={beta_K/100},  g={g}")
        elif style=='log':
            l, = plt.loglog(x,y, line, c=c, label=r"$\beta_V$"+f"={beta_V/100},"+r"$\beta_K$"+f"={beta_K/100},  g={g}")
        elif style == 'semi':
            l, = plt.semilogy(x, y, line,c=c, label=r"$\beta_V$"+f"={beta_V/100},"+r"$\beta_K$"+f"={beta_K/100}, g={g}")
        else:
            l, =plt.plot(x, y, line,c=c, label=r"$\beta_V$"+f"={beta_V/100},"+r"$\beta_K$"+f"={beta_K/100},  g={g}")
        c = l.get_c()
    return (Es, Ts, c)

def BestPlot(dict_kvs, title=None, iState="ST", style="semi", V="HO", use_nfev=False):
    if iState is not None and dict_kvs is not None and iState in dict_kvs:        
        kvs =dict_kvs[iState]
    else:
        return
    #plt.figure(figsize=(10, 8))
    if gs is not None:
        for g in gs:
            v, v1, k1 = kvs[g + 1]
            res = plot_Es_Ts(beta_V=v, beta_K=0, iState=iState, g=g, V=V, style=style, use_nfev=use_nfev)
            plot_Es_Ts(beta_V=v1, beta_K=k1, iState=iState, g=g, V=V,c=res[2], line='--', style=style, use_nfev=use_nfev)
    else:
        v, v1, k1 = kvs[0]
        res = plot_Es_Ts(beta_V=v, beta_K=0, iState=iState, g=None, V=None, style=style, use_nfev=use_nfev)
        plot_Es_Ts(beta_V=v1, beta_K=k1, iState=iState, g=None, V=None,c=res[2], line='--', style=style, use_nfev=use_nfev)
    plt.ylabel("(E-E0)/E0", fontsize=fontsize)
    plt.xlabel("Wall Time", fontsize=fontsize)
    if title is None:
        title=iState
    plt.title(title, fontsize=fontsize)
    plt.legend(prop={"size":fontsize})


# -

# $\beta_V$, $\beta_K$, $V_c$, $K_c$

E_E0=1.01
use_nfev=False
output, dict_kvs = find_best_betas(data, p=E_E0)
plt.figure(figsize=(22,16))
for i, state in enumerate(iStates):
    plt.subplot(2, 2,i+1)
    BestPlot(dict_kvs, title = f"State:{state}",iState=state, style="semi", V="HO", use_nfev=use_nfev)
if use_nfev:
    plt.xlabel("nfev")
#plt.title(r"BCS with final energy $E/E0$<"+f"{E_E0} ")
plt.savefig(f"cooling_results_precision_{int(E_E0*100)}.pdf", bbox_inches='tight')


# ## Plot Wall-Time vs $\beta$ s

def find_Tw_kvs(data, p1=1.01, p2=1.011):  
    assert p2 > p1
    dfs = []
    dict_tkvs = {}
    df = data[data.E0*p2 > data.Ef] # find all rows with Ef in per*E0
    df = df[df.E0*p1 < df.Ef]
    for iState in iStates:
        df1 = df[df.iState == iState]
        res = []
        for g in [-1,0,1]:
            df2 = df1[df1.g == g]
            if df2.empty:
                res.append(((None,None),)*3) # wall time, beta_V, beta_V1, beta_K1
                continue
            df3 = df2[df2.beta_K == 0]
            df4 = df2[df2.beta_K != 0]
            if df3.empty:
                wall_time = df4['wTime'].values
                res.append(((None, None), (wall_time, df4['beta_V'].values), (wall_time, df4['beta_K'].values)))
            elif df4.empty:
                wall_time = df3['wTime'].values
                res.append(((wall_time, df3['wTime'].values), (wall_time, df3['beta_V'].values), (None, None)))
            else:
                wall_time = df3['wTime'].values
                wall_time1 = df4['wTime'].values
                res.append(((wall_time, df3['beta_V'].values), (wall_time1, df4['beta_V'].values), (wall_time1, df4['beta_K'].values)))
        dict_tkvs[iState] = res
    return dict_tkvs


find_Tw_kvs(data)["UN"][0][1]


def plot_Tw_kvs(data, state="UN", g=0,  p1=1.01, p2=1.011):
    dict_tkvs = find_Tw_kvs(data, p1=p1, p2=p2)
    rs = dict_tkvs[state]
    labels = [r"$\beta_V$", r"$\beta_V1$", r"$\beta_K1$"]
    for i, tb in enumerate(rs[g + 1]):
        ts, betas = tb
        if ts is not None:
            plt.plot(ts, betas, 'o', label=labels[i])
    plt.xlabel("Wall Time")
    plt.ylabel(r"$\beta s$")
    plt.legend()


plot_Tw_kvs(data, p1=1.01, p2=1.011)

# # BCS Cases

N=128
dx=0.2
eps=1e-2
N_state=2
T_max=10
use_abm=True
args = {}
args.update(N=N, dx=dx)
b = BCSCooling(**args)
x = b.xyz[0]
V = x**2/2
b.V = V
H0 = b._get_H(mu_eff=0, V=0)  # free particle
H1 = b._get_H(mu_eff=0, V=V)  # harmonic trap
U0, Es0 = b.get_psis_es(H0, transpose=True)
U1, Es1 = b.get_psis_es(H1, transpose=True)
psis_init = U0[:N_state]  # change the start states here if needed.
psis_ground = U1[:N_state]  # change the start states here if needed.
E0=sum(Es1[:N_state])

psi1, psi = psis_ground

Es1[0], Es1[1]

np.allclose(H1.dot(psi1), Es1[0]*psi1)


