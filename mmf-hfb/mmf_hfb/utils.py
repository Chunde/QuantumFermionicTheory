import numpy as np
import json

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
            for _row in M], axis=0)


class JsonEncoderEx(json.JSONEncoder):
    """
    An customized encoder to address errors due to encoder
    in consistent between numpy and json package
    such as:
        TypeError: Object of type int32 is not JSON serializable
    
    -------
    json.dumps(data,cls=JsJsonEncoderExonEncoder)
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoderEx, self).default(obj)
