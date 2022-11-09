import numpy as np

def uniform(x, low, high):
    prior = np.ones_like(x)
    prior /= (high - low)
    prior *= (x > low) & (x < high)
    return prior