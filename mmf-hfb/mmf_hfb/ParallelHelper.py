from multiprocessing import Pool
import os

class PoolHelper(object):
    """a helper class for parallelization"""
    def run(fun, args, poolsize = None):
        if poolsize is None or poolsize < 1:
            logic_cpu_count = os.cpu_count() - 2
            poolsize = 1 if logic_cpu_count < 1 else logic_cpu_count
        res = None
        with Pool(poolsize) as Pools:
            res = Pools.map(fun, args)
        return res
