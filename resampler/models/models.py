from ..cupy_utils import xp, trapz, cumtrapz

from gwpopulation.utils import powerlaw, truncnorm, frank_copula
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution, two_component_single

class SPSMD_EffectiveCopula(SinglePeakSmoothedMassDistribution):
    def __call__(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                 mu_chi_eff, log_sigma_chi_eff, kappa_q_chi_eff):
        sigma_chi_eff = 10**log_sigma_chi_eff
        p_mass = super(SPSMD_EffectiveCopula, self).__call__(
            dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m)
        p_spin = truncnorm(dataset["chi_eff"], mu=mu_chi_eff, sigma=sigma_chi_eff,
                           high=1, low=-1)
        u, v = self.copula_coords(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                                  mu_chi_eff, sigma_chi_eff)
        prob = p_mass * p_spin * frank_copula(u, v, kappa_q_chi_eff)
        return prob
    
    def copula_coords(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                      mu_chi_eff, sigma_chi_eff):
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