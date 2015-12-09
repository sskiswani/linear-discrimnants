import typing
from itertools import count

import numpy as np

narray = typing.Union[np.ndarray, list]
T = typing.TypeVar('T', int, float, narray)
collection = typing.Union[typing.Iterable[T], typing.List[T], narray, typing.Tuple]


def count_trials(num_samples: int, max_trials: int = 100000):
    for k in count():
        if k >= max_trials: break
        yield (k % num_samples)


def get(true_value, false_value, condition=None):
    if condition is None:
        return false_value if true_value is None else true_value
    return true_value if condition else false_value


def approx(a, b, eps: float = 1e-3, open: bool = False):
    if open:
        return 0 < np.abs(b - a) < eps
    return 0 <= np.abs(b - a) <= eps


def dist(x: narray, y: narray) -> float:
    """
    Distance between two numpy arrays.
    """
    return np.linalg.norm(x - y)


def signum(x):
    """
    True for x >= 0, False otherwise
    """
    return np.sign(x) >= 0


def npzip(a: narray, b: narray) -> narray:
    """
    Join two numpy arrays row-wise, such that the result is:
    [[a[0][0], ..., a[0][m], b[0][0] ..., b[0][m]],
     ...
     [a[n][0], ..., a[n][m], b[n][0] ..., b[n][m]]]
    :param a: The first array
    :param b: The second array
    :return: matrix of [a b]
    """
    (rows_a, cols_a) = a.shape
    (rows_b, cols_b) = b.shape

    result = np.zeros((rows_a + rows_b, cols_a + cols_b))
    result[:, :rows_a] = a
    result[:, rows_a:] = b

    return result


def keslers_construction(data: narray) -> narray:
    """
    Apply Kesler's Construction on the given data set.
    :param data: The data set to transform
    """
    labels = np.unique(data[:, 0])
    M = labels.shape[0]

    points = data[:, 1:]
    n, d = points.shape

    result = np.zeros((n * (M - 1), d * M))
    ki = 0
    for (i, x) in enumerate(points):
        lab = data[i, 0]
        for j in range(1, M + 1):
            if j == lab: continue

            # Create sample
            xk = np.zeros((d * M,))
            xk[d * (lab - 1): d * lab] = x
            xk[d * (j - 1): d * j] = -x

            # Add to construction
            result[ki, :] = xk
            ki += 1
    return result
