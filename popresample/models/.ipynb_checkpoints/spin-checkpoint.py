"""
A place to store spin models.
"""
from ..cupy_utils import xp
from .model_utils import truncnorm


def gaussian_chi_eff(chi_eff, mu_chi_eff, log_sigma_chi_eff):
    """
    Truncated normal distribution for effective inspiral spin.
    
    Parameters
    ----------
    chi_eff: float, array-like
        Array of effective inspiral spin values.
    mu_chi_eff: float
        Mean of distribution.
    log_sigma_chi_eff: float
        log_10 of standard deviation.
    """
    prob = truncnorm(chi_eff, mu=mu_chi_eff, sigma=10**log_sigma_chi_eff,
                     high=1, low=-1)
    return prob
