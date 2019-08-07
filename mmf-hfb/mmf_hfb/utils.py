import numpy as np


def block(a11, a12, a21, a22):
    RowBlock1=np.concatenate((a11, a12), axis=1)
    RowBlock2=np.concatenate((a21, a22), axis=1)
    Block=np.concatenate((RowBlock1, RowBlock2), axis=0)
    return Block