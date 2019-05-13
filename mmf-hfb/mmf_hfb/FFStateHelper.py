
from mmf_hfb.FFStateFinder import FFStateFinder
from multiprocessing import Pool
import os
import inspect
from os.path import join
import json
import time
import glob


class FFStateHelper(object):
    def compute_pressure_current_worker(jsonData_file):
        """Use the FF State file to compute their current and pressure"""
        jsonData, fileName = jsonData_file
        filetokens = fileName.split("_")
        output_fileName = "FFState_J_P_" + "_".join(filetokens[1:]) + ".json"
        dim = jsonData['dim']
        delta = jsonData['delta']
        mu = jsonData['mu']
        dmu = jsonData['dmu']
        data = jsonData['data']
        ff = FFStateFinder(mu=mu, dmu=dmu, delta=delta, g=jsonData['g'],
                        dim=dim, prefix=f"{output_fileName}", timeStamp=False)
        if os.path.exists(ff._get_fileName()):
            return None
        print(f"Processing {ff._get_fileName()}")
        output1 = []
        output2 = []
        try:
            for item in data:
                dq1, dq2, d = item
                #if not (np.allclose(dq1, 0.042377468400988445) or np.allclose(dq2, 0.042377468400988445)):
                #    continue
                if dq1 is not None:
                    dic = {}
                    p1 = ff.get_pressure(delta=d, dq=dq1)
                    ja, jb, jp, jm = ff.get_current(delta=d, dq=dq1)

                    dic['d']=d
                    dic['q']=dq1
                    dic['p']=p1
                    dic['j']=jp.n
                    dic['ja']=ja.n
                    dic['jb']=jb.n
                    output1.append(dic)
                    print(dic)
                if dq2 is not None:
                    dic = {}
                    p2 = ff.get_pressure(delta=d, dq=dq2)
                    ja, jb, jp, jm = ff.get_current(delta=d, dq=dq2)
                    dic['d']=d
                    dic['q']=dq2
                    dic['p']=p2
                    dic['j']=jp.n
                    dic['ja']=ja.n
                    dic['jb']=jb.n
                    output2.append(dic)
                    print(dic)
            output =[output1, output2]
            ff.SaveToFile(output)
        except ValueError as e:
            print(f"Parsing file: {fileName}. Error:{e}")
        

    def compute_pressure_current(root=None):
        """compute current and pressure"""
        currentdir = root
        if currentdir is None:
            currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        pattern = join(currentdir, "data","FFState_[()_0-9]*.json")
        files = files=glob.glob(pattern)

        jsonObjects = []
        for file in files:
            if os.path.exists(file):
                with open(file, 'r') as rf:
                    jsonObjects.append((json.load(rf), os.path.splitext(os.path.basename(file))[0]))
        logic_cpu_count = os.cpu_count() - 1
        logic_cpu_count = 1 if logic_cpu_count < 1 else logic_cpu_count
        if False:  # Debugging
            for item in jsonObjects:
                FFStateHelper.compute_pressure_current_worker(item)
        with Pool(logic_cpu_count) as Pools:
            Pools.map(FFStateHelper.compute_pressure_current_worker, jsonObjects)

    def search_FFState_worker(dim_delta_mus):
        """worker thread"""
        dim, delta, mu, dmu=dim_delta_mus
        ff = FFStateFinder(delta=delta, dim=dim, dmu=dmu)
        ff.run(dl=0.001, du=0.2501, dn=40, ql=0, qu=1)
        
    def SearchFFState(delta=0.1, mu=10, dmus=None, dim=1):
        """Search FF State"""
        if dmus is None:
            dmus = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
        logic_cpu_count = os.cpu_count() - 1
        logic_cpu_count = 1 if logic_cpu_count < 1 else logic_cpu_count
        dim_delta_mus_list = [(dim, delta, mu, dmu) for dmu in dmus]
        with Pool(logic_cpu_count) as Pools:
            Pools.map(FFStateHelper.search_FFState_worker, dim_delta_mus_list)

    def search_single_configuration_1d():
        dim = 1
        mu = 10
        delta = 0.2  # when set g, delta is useless
        dmu = 0.6
        g = -10
        ff = FFStateFinder(delta=delta, dim=dim, mu=mu, dmu=dmu, g=g)
        ff.run(dl=0.001, du=15, dn=200, ql=0, qu=2)

    def search_single_configuration_3d():
        dim = 3
        mu = 10
        delta = 2.4
        dmu = 2.48
        ff = FFStateFinder(delta=delta, dim=dim, mu=mu, dmu=dmu)
        ff.run(dl=2.37, du=2.95, dn=100, ql=0, qu=.5)

    def sort_file(files=None, abs_file=False):
        #files = ["FFState_(3d_2.5_10_3.15)2019_05_06_23_23_35.json"]
        currentdir = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), "data")
        if files is None:
            pattern = join(currentdir,"FFState_[()d_0-9]*.json")
            files = glob.glob(pattern)
            abs_file = True

        if len(files) < 1:
            print("At least two files input")
            return
        for file in files:
            if not abs_file:
                file = join(currentdir, file)
            if os.path.exists(file):
                with open(file,'r+') as rf:
                    ret = json.load(rf)
                    data = FFStateFinder.sort_data(ret['data'])
                    ret['data']=data
                    rf.seek(0)
                    json.dump(ret, rf)
                    rf.truncate()  # truncate the rest of the old file.
                    print(f"{file} saved")

    def merge_files():
        files = ["FFState_(3d_2.4_10_2.85)2019_05_04_23_22_48.json", "FFState_(3d_2.4_10_2.85)2019_05_04_23_23_01.json"]
        if len(files) < 1:
            print("At least two files input")
            return
        currentdir = join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), "data")
        ts = time.strftime("%Y_%m_%d_%H_%M_%S.json")
        
        datas = []
        for file in files:
            file = join(currentdir, file)
            if os.path.exists(file):
                with open(file, 'r') as rf:
                    datas.append(json.load(rf))
        if len(datas) < 1:
            return
        filetokens = files[0].split(")")
        output_fileName = "_".join([filetokens[0] + ")", ts])

        output = datas[0]
        for i in range(1, len(datas)):
            output["data"].extend(datas[i]["data"])
        with open(join(currentdir, output_fileName),'w') as wf:
                json.dump(output, wf)

if __name__ == "__main__":
    ## Sort file with discontinuity
    #FFStateHelper.sort_file()
    ## Merge files with the same configuration
    # FFStateHelper.merge_files()
    ## Method: change parameters manually
    # FFStateHelper.search_single_configuration_1d()
    #FFStateHelper.search_single_configuration_3d()
    ## Method 2: Thread pool
    #dmus = np.array([0.11, 0.12, 0.13, 0.14, 0.15, 0.16]) * 2 + 2
    # FFStateHelper.SearchFFState(delta=2.1, mu=10, dmus=dmus, dim=1)
    ## Compute the pressure and current
    FFStateHelper.compute_pressure_current()
    