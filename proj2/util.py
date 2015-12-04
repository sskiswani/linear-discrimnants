import numpy as np

def approx(a, b, eps=1e-3):
    return 0 <= np.abs(b - a) <= eps

def signum(a):
    return np.sign(a) >= 0