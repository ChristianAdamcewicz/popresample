from ..cupy_utils import xp, trapz, erf


def powerlaw(xx, alpha, high, low):
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
    if sigma <= 0:
        raise ValueError(f"Sigma must be greater than 0, sigma={sigma}")
    norm = 2 ** 0.5 / xp.pi ** 0.5 / sigma
    norm /= erf((high - mu) / 2 ** 0.5 / sigma) + erf((mu - low) / 2 ** 0.5 / sigma)
    prob = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma ** 2))
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


def frank_copula(u, v, kappa):
    if kappa == 0:
        prob = 1.
    else:
        expkap = xp.exp(kappa)
        expkapuv = expkap**(u + v)
        prob = kappa * expkapuv * (expkap - 1) / (expkap - expkap**u - expkap**v + expkapuv)**2
    return xp.nan_to_num(prob)


def two_component_single(
    mass, alpha, mmin, mmax, lam, mpp, sigpp, gaussian_mass_maximum=100
):
    p_pow = powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=gaussian_mass_maximum, low=mmin)
    prob = (1 - lam) * p_pow + lam * p_norm
    return prob


