"""
Miscellaneous helper functions.
"""
import numpy as np
from scipy.integrate import cumtrapz


def it_sample(pdf, x):
    """
    Inverse transform samples a given function pdf(x).
    
    Parameters
    ----------
    pdf: float, array-like
        The probability function to sample
        (does not need to be normalised)
    x: float, array-like
        Grid values associated with pdf values.
    
    Returns
    -------
    sample: float
        A sample from the given pdf.
    """
    cdf = cumtrapz(pdf, x, initial=0)
    cdf /= np.max(cdf)
    cdf = np.nan_to_num(cdf)
    cdf_sample = np.random.rand()
    sample = np.interp(cdf_sample, cdf, x)
    return sample


def trapz_exp(log_y, x=None, dx=1.0, axis=-1):
    """
    Evaluates an integral with a log input.
    
    Parameters
    ----------
    log_y: float, array-like
        The log function to be integrated.
    x: float, array-like
        Values to be integrated at.
    dx: float
        Spacing between points if x is not provided.
    axis: int
        Axis to integrate input over.
    
    Returns
    -------
    log_y_int: float
        The log of the integrated function.
    """
    y = np.exp(log_y)
    y_int = np.trapz(y, x, dx, axis)
    return np.log(y_int)


def effective_samples(weights):
    """
    Computes the effective number of samples given
    resampled weights.
    
    Parameters
    ----------
    weights: float, array-like
        Weights from resampling.
    
    Returns
    -------
    n_eff: float
        The number of effective samples.
    """
    w = np.array(weights)
    n_eff = np.sum(w)**2 / np.sum(w**2)
    return n_eff