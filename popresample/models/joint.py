"""
A place to store custom joint models.
"""
from gwpopulation.cupy_utils import xp, trapz
from gwpopulation.utils import powerlaw, truncnorm
from gwpopulation.models.mass import two_component_single

from .model_utils import frank_copula, cumtrapz
from .mass import LegacySinglePeakSmoothedMassDistribution


class SPSMD_EffectiveCopula(LegacySinglePeakSmoothedMassDistribution):
    """
    Power-Law + Peak mass distribution and Gaussian effective inspiral spin
    distribution with a Frank copula density function correlating mass ratio
    and effective inspiral spin.
    """
    def __init__(self):
        super(SPSMD_EffectiveCopula, self).__init__()
        self.chi_effs = xp.linspace(-1,1,500)
    
    def __call__(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                 mu_chi_eff, log_sigma_chi_eff, kappa):
        """
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
        mu_chi_eff: float
            Mean of spin distribution.
        log_sigma_chi_eff: float
            log_10 of standard deviation of spin distribution.
        kappa: float
            Level of covariance between mass ratio and spin.
        """
        p_mass = super(SPSMD_EffectiveCopula, self).__call__(
            dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m)
        sigma_chi_eff = 10**log_sigma_chi_eff
        p_spin = truncnorm(dataset["chi_eff"], mu=mu_chi_eff, sigma=sigma_chi_eff,
                           high=1, low=-1)
        u, v = self.copula_coords(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                                  mu_chi_eff, sigma_chi_eff)
        prob = p_mass * p_spin * frank_copula(u, v, kappa)
        return prob
    
    def copula_coords(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                      mu_chi_eff, sigma_chi_eff):
        """
        Retrieves copula coordinates u(q) and v(chi_eff).
        """
        '''Get u(q)'''
        # p(m1) grid
        p_m = two_component_single(
            self.m1s, alpha=alpha, mmin=mmin, mmax=mmax, lam=lam, mpp=mpp, sigpp=sigpp)
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=100, delta_m=delta_m)
        p_m_norm = trapz(p_m, self.m1s)
        p_m /= p_m_norm
        p_m = xp.nan_to_num(p_m)

        # p(q|m1) grid
        p_q = powerlaw(self.qs_grid, beta, 1, mmin / self.m1s_grid)
        p_q *= self.smoothing(
            self.m1s_grid * self.qs_grid, mmin=mmin, mmax=self.m1s_grid, delta_m=delta_m)
        p_q_norm = trapz(p_q, self.qs, axis=0)
        p_q /= p_q_norm
        p_q = xp.nan_to_num(p_q)

        # p(q) grid
        integrand_q_m = p_q * p_m
        p_q_marg = trapz(integrand_q_m, self.m1s, axis=-1)
        p_q_marg = xp.nan_to_num(p_q_marg)

        # u(q) grid
        u = cumtrapz(p_q_marg, self.qs, initial=0)
        u /= xp.max(u)
        u = xp.nan_to_num(u)

        # Interpolate for u(q)
        res_u = xp.interp(dataset["mass_ratio"], self.qs, u)
        
        '''get v(chi_eff)'''
        # p(chi_eff) grid
        p_chi_eff = truncnorm(self.chi_effs, mu=mu_chi_eff, sigma=sigma_chi_eff,
                              high=1, low=-1)
        
        # v(chi_eff) grid
        v = cumtrapz(p_chi_eff, self.chi_effs, initial=0)
        v /= xp.max(v)
        v = xp.nan_to_num(v)
        
        # Interpolate for v(chi_eff)
        res_v = xp.interp(dataset["chi_eff"], self.chi_effs, v)
        
        return res_u, res_v
