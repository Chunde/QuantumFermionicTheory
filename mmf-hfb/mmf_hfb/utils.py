import numpy as np


def block(a11, a12, a21, a22):
    """
    used to stack four element to form a 2x2 matrix
    ---------
    Note: this can be used for numpy and cupy(if np->cp)
    """
    RowBlock1=np.concatenate((a11, a12), axis=1)
    RowBlock2=np.concatenate((a21, a22), axis=1)
    Block=np.concatenate((RowBlock1, RowBlock2), axis=0)
    return Block
    