import numpy as np
from typing import Union

narray = Union[np.ndarray]

def approx(a, b, eps=1e-3):
    return 0 <= np.abs(b - a) <= eps


def dist(x:narray, y:narray) -> float:
    return np.linalg.norm(x-y)

def signum(x):
    return np.sign(x) >= 0
