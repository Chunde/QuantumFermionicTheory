import numpy as np
import json
from json import dumps
from os.path import join

current_path = "d:\\"
if current_path == None:
    current_path = os.path.dirname(os.path.abspath(__file__))
input_file_dir = join(current_path,"input_files")
output_file_dir = join(current_path,"output_files")


if __name__ == "__main__":
    q_min = 0
    q_max = 0.5
    q_num = 10

    qs = np.linspace(q_min,q_max,q_num)

    np.random.seed(1)
    m = hbar = kF = 1
    eF = (hbar*kF)**2/2/m
    nF = kF**3/3/np.pi**2
    mu = 0.59060550703283853378393810185221521748413488992993*eF
    delta = 0.68640205206984016444108204356564421137062514068346*eF
    #mu_a,mu_b,delta,m,T,q = data["mu_a"],data["mu_b"],data["delta"],data["m"],data["T"],delta["q"]
    outputs = []
    for q in qs:
        dic = {}
        dic['mu_a'] = mu
        dic['mu_b'] = mu
        dic['delta'] = delta
        dic['m'] = m
        dic['hbar'] = hbar
        dic['T']  = 0
        dic['q'] = q
        outputs.append(dic)
    conf_file = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(input_file_dir):
        os.makedirs(input_file_dir, exist_ok=True)
    with open(join(input_file_dir,conf_file),'w',encoding ='utf-8') as wf:
        json.dump(outputs,wf, ensure_ascii=False)
        print(f"{conf_file} saved.")
