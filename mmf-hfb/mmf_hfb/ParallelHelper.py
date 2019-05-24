from multiprocessing import Pool
import os

class PoolHelper(object):
    """a helper class for parallelization"""
    def run(fun, paras, poolsize=None, **args):
        """
        invoke thread pools for parallelization
        ---------
        paras: list of parameters for the callee
        """
        if poolsize is None or poolsize < 1:
            logic_cpu_count = os.cpu_count() - 2
            poolsize = 1 if logic_cpu_count < 1 else logic_cpu_count
        poolsize = min(poolsize, len(paras))
        if poolsize > 1:
            res = None
            with Pool(poolsize) as Pools:
                res = Pools.map(fun, paras)
            return res
        return [fun(paras[0])]
