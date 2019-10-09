"""
A helper package used for all kind of data sorting,
filtering and other manipulation.
"""


def ff_state_sort_data(rets):
    """
        sort data to make lines as long and smooth as possible
    """
    rets_ = []
    
    def p(ret):
        v1, v2, _ = ret
        if v1 is None and v2 is None:
            return 0
        if v1 is None or v2 is None:
            return 1
        return 2

    for ret in rets:
        if p(ret) > 0:
            rets_.append(ret)
    rets = rets_
    bflip = False
    for i in range(1, len(rets)):
        v1, v2, _ = rets[i]
        v1_, v2_, _ = rets[i-1]
        bflip = False
        p1 = p(rets[i])
        p2 = p(rets[i-1])
        if p1 > p2:
            if v1_ is None:
                if abs(v1 - v2_) < abs(v2 - v2_):
                    bflip = True
                    print("flipping data")
            if v2_ is None:
                if abs(v1 - v1_) > abs(v2 - v1_):
                    bflip = True
                    print("flipping data")
        elif p1 < p2:
            if v1 is None:
                if abs(v1_ - v2) < abs(v2_ - v2):
                    bflip = True
                    print("flipping data")
            if v2 is None:
                if abs(v1_ - v1) > abs(v2_ - v1):
                    bflip = True
                    print("flipping data")
        elif p1 == p2:
            if (v1 is None) !=(v1_ is None) or (v2 is None) != (v2_ is None):
                bflip=True
                print("flipping data")
        if bflip:
            rets[i] = [rets[i][1], rets[i][0], rets[i][2]]
    return rets
