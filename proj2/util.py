import numpy as np

def within(x, y, eps=1e-11):
    return 0 <= np.abs(x - y) <= eps or np.allclose(x,y)