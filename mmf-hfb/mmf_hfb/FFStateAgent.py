from mmf_hfb.ParallelHelper import PoolHelper


class FFStateAgent(object):
    """
    An agent class used for searching FF states
    """

    def __init__(self, mu_eff, delta, dim=3):
        self.C = self._get_C(mus_eff=(mu_eff, mu_eff), delta=delta)
        self.mu_eff = mu_eff
        self.delta = delta
        self.dim = dim

    def run(self):
        pass

    def Search(self, delta_lower, delta_upper, q_lower, q_upper):
        """
        Search possible states in ranges of delta and q
        Paras:
        delta_lower: lower boundary for delta
        delta_upper: upper boundary for delta
        q_lower    : lower boundary for dq
        q_upper    : upper boundary for dq
        """
        pass
