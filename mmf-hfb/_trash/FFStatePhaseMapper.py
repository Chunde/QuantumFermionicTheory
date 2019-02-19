import numpy as np
import json
from json import dumps
import os
import time
import multiprocessing
import threading
from os.path import join
from multiprocessing import Process,Queue,Array,RLock,Lock
from mmf_hfb import FuldeFerrelState as FF


current_path = "d:\\"
if current_path == None:
    current_path = os.path.dirname(os.path.abspath(__file__))
input_file_dir = join(current_path,"input_files")
output_file_dir = join(current_path,"output_files")

filelock = Lock()
all_output_list = []
history_input_list =[]
waiting_input_list = []

def look_up_new_input_files(read_file_dir=input_file_dir):
    with filelock:
        if not os.path.exists(read_file_dir):
            os.makedirs(read_file_dir, exist_ok=True)
        if len(waiting_input_list) == 0:
            print(f"Looking for new conf files...")
            fileNames = os.listdir(read_file_dir)
            for i in range(len(fileNames)):
                if(fileNames[i] in history_input_list or fileNames[i] in waiting_input_list):
                    continue
                waiting_input_list.append(fileNames[i])
        if len(waiting_input_list) == 0:
            return None
        conf_file = waiting_input_list[0]
        waiting_input_list.remove(conf_file)
        history_input_list.append(conf_file)
        return conf_file

def search_maximum_state_thread(id):
    print(f"{id}:starting...")
    while(True):
        time.sleep(1);
        conf_file = look_up_new_input_files()
        if conf_file == None:
            print(f"{id}:No configure file, waiting...")
            continue

        outputs = []
        try:
            with open(join(input_file_dir,conf_file),'r',encoding ='utf-8') as rf:
                datas = json.load(rf)
                for data in datas:
                    mu_a,mu_b,delta,m,T,q = data["mu_a"],data["mu_b"],data["delta"],data["m"],data["T"],data["q"]
                    p = get_pressure(mu_a = mu_a,mu_b=mu_b,delta=delta,m=m,T=0,q=q).n
                    data["p"] = p
                    outputs.append(data)
        except:
            continue
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir, exist_ok=True)
        if len(outputs) > 0:
            try:
                with open(join(output_file_dir,conf_file),'w',encoding ='utf-8') as wf:
                    json.dump(outputs,wf, ensure_ascii=False)
                    print(f"{id}:output result for {conf_file} saved.")
            except:
                continue

def scane_phase_map():
    np.random.seed(1)
    m = hbar = kF = 1
    eF = (hbar*kF)**2/2/m
    nF = kF**3/3/np.pi**2
    mu = 0.59060550703283853378393810185221521748413488992993*eF
    delta = 0.68640205206984016444108204356564421137062514068346*eF
    args = dict(mu_a=mu, mu_b=mu, delta=delta, m_a=m, m_b=m, hbar=hbar, T=0.0)
    search_maximum_state_thread(1)
    thread_count = os.cpu_count()
    thred_count = 1 if thread_count < 1 else thread_count
    thread_list = []
    filelock.reset()
    for i in range(thread_count):
        t = threading.Thread(target=search_maximum_state_thread, args=[i]) # a process works much faster than a thread.
        thread_list.append(t)
    for i in range(len(thread_list)):
        thread_list[i].start()
    for i in range(len(thread_list)):# wait for all process
        thread_list[i].join()


if __name__ == "__main__":
    scane_phase_map()