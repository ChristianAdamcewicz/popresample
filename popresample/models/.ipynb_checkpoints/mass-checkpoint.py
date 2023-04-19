"""
A place to store mass models.
"""
from ..cupy_utils import xp, trapz
from .model_utils import powerlaw, truncnorm


def two_component_single(
    mass, alpha, mmin, mmax, lam, mpp, sigpp, gaussian_mass_maximum=100):
    r"""
    Lifted from gwpopulation (https://github.com/ColmTalbot/gwpopulation)
    
    Power law model for one-dimensional mass distribution with a Gaussian component.
    .. math::
        p(m) &= (1 - \lambda_m) p_{\text{pow}} + \lambda_m p_{\text{norm}}
        p_{\text{pow}}(m) &\propto m^{-\alpha} : m_\min \leq m < m_\max
        p_{\text{norm}}(m) &\propto \exp\left(-\frac{(m - \mu_{m})^2}{2\sigma^2_m}\right)
        
    Parameters
    ----------
    mass: array-like
        Array of mass values (:math:`m`).
    alpha: float
        Negative power law exponent for the black hole distribution (:math:`\alpha`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum black hole mass (:math:`m_\max`).
    lam: float
        Fraction of black holes in the Gaussian component (:math:`\lambda_m`).
    mpp: float
        Mean of the Gaussian component (:math:`\mu_m`).
    sigpp: float
        Standard deviation of the Gaussian component (:math:`\sigma_m`).
    gaussian_mass_maximum: float, optional
        Upper truncation limit of the Gaussian component. (default: 100)
    """
    p_pow = powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=gaussian_mass_maximum, low=mmin)
    prob = (1 - lam) * p_pow + lam * p_norm
    return prob


class _SmoothedMassDistribution(object):
    """
    Lifted from gwpopulation (https://github.com/ColmTalbot/gwpopulation)
    
    Generic smoothed mass distribution base class.
    Implements the low-mass smoothing and power-law mass ratio
    distribution. Requires p_m1 to be implemented.
    """
    def __init__(self, mmin=2, mmax=100):
        self.mmin = mmin
        self.mmax = mmax
        self.m1s = xp.linspace(2, 100, 1000)
        self.qs = xp.linspace(0.001, 1, 500)
        self.dm = self.m1s[1] - self.m1s[0]
        self.dq = self.qs[1] - self.qs[0]
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.chi_effs = xp.linspace(-1, 1, 500)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def p_m1(self, *args, **kwargs):
        raise NotImplementedError

    def p_q(self, dataset, beta, mmin, delta_m):
        p_q = powerlaw(dataset["mass_ratio"], beta, 1, mmin / dataset["mass_1"])
        p_q *= self.smoothing(
            dataset["mass_1"] * dataset["mass_ratio"],
            mmin=mmin,
            mmax=dataset["mass_1"],
            delta_m=delta_m,
        )
        try:
            p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m)
        except (AttributeError, TypeError, ValueError):
            self._cache_q_norms(dataset["mass_1"])
            p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m)

        return xp.nan_to_num(p_q)

    def norm_p_q(self, beta, mmin, delta_m):
        """Calculate the mass ratio normalisation by linear interpolation"""
        if delta_m == 0.0:
            return 1
        p_q = powerlaw(self.qs_grid, beta, 1, mmin / self.m1s_grid)
        p_q *= self.smoothing(
            self.m1s_grid * self.qs_grid, mmin=mmin, mmax=self.m1s_grid, delta_m=delta_m
        )
        norms = trapz(p_q, self.qs, axis=0)

        all_norms = (
            norms[self.n_below] * (1 - self.step) + norms[self.n_above] * self.step
        )

        return all_norms

    def _cache_q_norms(self, masses):
        """
        Cache the information necessary for linear interpolation of the mass
        ratio normalisation
        """
        self.n_below = xp.zeros_like(masses, dtype=xp.int) - 1
        m_below = xp.zeros_like(masses)
        for mm in self.m1s:
            self.n_below += masses > mm
            m_below[masses > mm] = mm
        self.n_above = self.n_below + 1
        max_idx = len(self.m1s)
        self.n_below[self.n_below < 0] = 0
        self.n_above[self.n_above == max_idx] = max_idx - 1
        self.step = xp.minimum((masses - m_below) / self.dm, 1)

    @staticmethod
    def smoothing(masses, mmin, mmax, delta_m):
        """
        Apply a one sided window between mmin and mmin + delta_m to the
        mass pdf.
        The upper cut off is a step function,
        the lower cutoff is a logistic rise over delta_m solar masses.
        See T&T18 Eqs 7-8
        Note that there is a sign error in that paper.
        S = (f(m - mmin, delta_m) + 1)^{-1}
        f(m') = delta_m / m' + delta_m / (m' - delta_m)
        See also, https://en.wikipedia.org/wiki/Window_function#Planck-taper_window
        """
        window = xp.ones_like(masses)
        if delta_m > 0.0:
            smoothing_region = (masses >= mmin) & (masses < (mmin + delta_m))
            shifted_mass = masses[smoothing_region] - mmin
            if shifted_mass.size:
                exponent = xp.nan_to_num(
                    delta_m / shifted_mass + delta_m / (shifted_mass - delta_m)
                )
                window[smoothing_region] = 1 / (xp.exp(exponent) + 1)
        window[(masses < mmin) | (masses > mmax)] = 0
        return window


class SinglePeakSmoothedMassDistribution(_SmoothedMassDistribution):
    def __call__(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m):
        """
        Lifted from gwpopulation (https://github.com/ColmTalbot/gwpopulation)
        
        Powerlaw + peak model for two-dimensional mass distribution with low
        mass smoothing.
        https://arxiv.org/abs/1801.02699 Eq. (11) (T&T18)
        
        Parameters
        ----------
        dataset: dict
            Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
        alpha: float
            Powerlaw exponent for more massive black hole.
        beta: float
            Power law exponent of the mass ratio distribution.
        mmin: float
            Minimum black hole mass.
        mmax: float
            Maximum mass in the powerlaw distributed component.
        lam: float
            Fraction of black holes in the Gaussian component.
        mpp: float
            Mean of the Gaussian component.
        sigpp: float
            Standard deviation fo the Gaussian component.
        delta_m: float
            Rise length of the low end of the mass distribution.
            
        Notes
        -----
        The interpolation of the p(q) normalisation has a fill value of
        the normalisation factor for m_1 = 100.
        """
        p_m1 = self.p_m1(
            dataset,
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
            delta_m=delta_m,
        )
        p_q = self.p_q(dataset, beta=beta, mmin=mmin, delta_m=delta_m)
        prob = p_m1 * p_q
        return prob

    def p_m1(self, dataset, alpha, mmin, mmax, lam, mpp, sigpp, delta_m):
        p_m = two_component_single(
            dataset["mass_1"],
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
        )
        p_m *= self.smoothing(dataset["mass_1"], mmin=mmin, mmax=100, delta_m=delta_m)
        norm = self.norm_p_m1(
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
            delta_m=delta_m,
        )
        return p_m / norm

    def norm_p_m1(self, alpha, mmin, mmax, lam, mpp, sigpp, delta_m):
        """Calculate the normalisation factor for the primary mass"""
        if delta_m == 0.0:
            return 1
        p_m = two_component_single(
            self.m1s, alpha=alpha, mmin=mmin, mmax=mmax, lam=lam, mpp=mpp, sigpp=sigpp
        )
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=100, delta_m=delta_m)

        norm = trapz(p_m, self.m1s)
        return 