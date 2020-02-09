import numpy as np


def block(M):
    """
    used to stack four element to form a 2x2 matrix
    ---------
    Note: this can be used for numpy and cupy(if np->cp)

    Examples
    --------
    >>> M = np.random.random((3, 3, 4))
    >>> Ms = [[M[..., 0], M[..., 1]],
    ...       [M[..., 2], M[..., 3]]]
    >>> np.allclose(np.bmat(Ms), block(Ms))
    True
    """
    return np.concatenate(
        [np.concatenate(_row, axis=1)
         for _row in M],
    axis=0)
