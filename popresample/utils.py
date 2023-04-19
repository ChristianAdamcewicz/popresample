import numpy as np
from scipy.integrate import cumtrapz


def it_sample(pdf, x):
    ''''''
    cdf = cumtrapz(pdf, x, initial=0)
    cdf /= np.max(cdf)
    cdf = np.nan_to_num(cdf)
    cdf_sample = np.random.rand()
    sample = np.interp(cdf_sample, cdf, x)
    return sample


def trapz_exp(log_y, x=None, dx=1.0, axis=-1):
    ''''''
    y = np.exp(log_y)
    y_int = np.trapz(y, x, dx, axis)
    return np.log(y_int)