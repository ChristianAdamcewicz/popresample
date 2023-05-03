"""
A place to store custom joint models.
"""
from gwpopulation.cupy_utils import xp, trapz
from gwpopulation.utils import powerlaw, truncnorm
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution, two_component_single
from gwpopulation.models.redshift import PowerLawRedshift

from .model_utils import frank_copula, cumtrapz


class MassRedshiftCopula(SinglePeakSmoothedMassDistribution, PowerLawRedshift):
    """
    Power-Law + Peak mass distribution and power-law redshift correlated with
    a Frank copula density function.
    """
    @property
    def variable_names(self):
        vars = ['alpha', 'beta', 'mmin', 'mmax', 'lam', 'mpp', 'sigpp', 'delta_m',
                'lamb', 'kappa']
        return vars
    
    def __init__(self, mmin=2, mmax=100, z_max=2.3):
        SinglePeakSmoothedMassDistribution.__init__(self, mmin, mmax)
        PowerLawRedshift.__init__(self, z_max)
        
    def __call__(self, dataset, 
                 alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                 lamb, kappa):
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
        lamb: float
            The spectral index for redshift.
        kappa: float
            Covariance between mass and redshift.
        """
        prob = SinglePeakSmoothedMassDistribution.__call__(self, dataset,
                    **{'alpha':alpha,
                       'beta':beta,
                       'mmin':mmin,
                       'mmax':mmax,
                       'lam':lam,
                       'mpp':mpp,
                       'sigpp':sigpp,
                       'delta_m':delta_m})
        prob *= PowerLawRedshift.__call__(self, dataset, **{'lamb':lamb})
        
        u = self.get_u(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m)
        v = self.get_v(dataset, lamb)
        prob *= frank_copula(u, v, kappa)
        return prob
        
    def get_u(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m):
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
        return res_u
        
    def get_v(self, dataset, lamb):
        p_z = self.psi_of_z(self.zs, **{'lamb':lamb})*self.dvc_dz/(1+self.zs)
        v = cumtrapz(p_z, self.zs, initial=0)
        v /= xp.max(v)
        res_v = xp.interp(dataset['redshift'], self.zs, v)
        return res_v


class SPSMD_EffectiveCopula(SinglePeakSmoothedMassDistribution):
    """
    Power-Law + Peak mass distribution and Gaussian effective inspiral spin
    distribution with a Frank copula density function correlating mass ratio
    and effective inspiral spin.
    """
    @property
    def variable_names(self):
        vars = ['alpha', 'beta', 'mmin', 'mmax', 'lam', 'mpp', 'sigpp', 'delta_m',
                'mu_chi_eff', 'log_sigma_chi_eff', 'kappa']
        return vars
    
    def __init__(self, mmin=2, mmax=100):
        super(SPSMD_EffectiveCopula, self).__init__(mmin, mmax)
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
        p_mass = super(SPSMD_EffectiveCopula, self).__call__(dataset,
                                                    **{'alpha':alpha,
                                                       'beta':beta,
                                                       'mmin':mmin,
                                                       'mmax':mmax,
                                                       'lam':lam,
                                                       'mpp':mpp,
                                                       'sigpp':sigpp,
                                                       'delta_m':delta_m})
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
        Computes copula coordinates u(q) and v(chi_eff).
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
