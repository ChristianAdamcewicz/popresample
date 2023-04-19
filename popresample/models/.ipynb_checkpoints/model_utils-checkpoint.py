"""
Helper functions for population models.
"""
from ..cupy_utils import xp, trapz, erf


def powerlaw(xx, alpha, high, low):
    r"""
    Lifted from gwpopulation (https://github.com/ColmTalbot/gwpopulation)
    
    Power-law probability
    .. math::
        p(x) = \frac{1 + \alpha}{x_\max^{1 + \alpha} - x_\min^{1 + \alpha}} x^\alpha
        
    Parameters
    ----------
    xx: float, array-like
        The abscissa values (:math:`x`)
    alpha: float, array-like
        The spectral index of the distribution (:math:`\alpha`)
    high: float, array-like
        The maximum of the distribution (:math:`x_\min`)
    low: float, array-like
        The minimum of the distribution (:math:`x_\max`)
        
    Returns
    -------
    prob: float, array-like
        The distribution evaluated at `xx`
    """
    if xp.any(xp.asarray(low) < 0):
        raise ValueError(f"Parameter low must be greater or equal zero, low={low}.")
    if alpha == -1:
        norm = 1 / xp.log(high / low)
    else:
        norm = (1 + alpha) / (high ** (1 + alpha) - low ** (1 + alpha))
    prob = xp.power(xx, alpha)
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


def truncnorm(xx, mu, sigma, high, low):
    r"""
    Lifted from gwpopulation (https://github.com/ColmTalbot/gwpopulation)
    
    Truncated normal probability
    .. math::
        p(x) =
        \sqrt{\frac{2}{\pi\sigma^2}}
        \left[\text{erf}\left(\frac{x_\max - \mu}{\sqrt{2}}\right) + \text{erf}\left(\frac{\mu - x_\min}{\sqrt{2}}\right)\right]^{-1}
        \exp\left(-\frac{(\mu - x)^2}{2 \sigma^2}\right)
        
    Parameters
    ----------
    xx: float, array-like
        The abscissa values (:math:`x`)
    mu: float, array-like
        The mean of the normal distribution (:math:`\mu`)
    sigma: float
        The standard deviation of the distribution (:math:`\sigma`)
    high: float, array-like
        The maximum of the distribution (:math:`x_\min`)
    low: float, array-like
        The minimum of the distribution (:math:`x_\max`)
        
    Returns
    -------
    prob: float, array-like
        The distribution evaluated at `xx`
    """
    if sigma <= 0:
        raise ValueError(f"Sigma must be greater than 0, sigma={sigma}")
    norm = 2 ** 0.5 / xp.pi ** 0.5 / sigma
    norm /= erf((high - mu) / 2 ** 0.5 / sigma) + erf((mu - low) / 2 ** 0.5 / sigma)
    prob = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma ** 2))
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


def frank_copula(u, v, kappa):
    """
    Frank copula density function.
    
    Parameters
    ----------
    u: float, array-like
        CDF of first parameter.
    v: float, array-like
        CDF of second parameter.
    kappa: float
        Level of correlation
        
    Returns
    -------
    prob: float, array-like
        The distribution evaluated at (u,v)
    """
    if kappa == 0:
        prob = 1.
    else:
        expkap = xp.exp(kappa)
        expkapuv = expkap**(u + v)
        prob = kappa * expkapuv * (expkap - 1) / (expkap - expkap**u - expkap**v + expkapuv)**2
    return xp.nan_to_num(prob)
