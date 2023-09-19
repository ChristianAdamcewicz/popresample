"""
A place to store custom joint models.
"""

from gwpopulation.cupy_utils import xp, trapz, cumtrapz
from gwpopulation.utils import truncskewnorm, powerlaw, beta_dist, frank_copula, gaussian_copula, fgm_copula
from gwpopulation.models.mass import two_component_single
from gwpopulation.models.joint import SPSMDEffectiveCopulaBase


class SPSMDEffectiveCopulaNorm(SPSMDEffectiveCopulaBase):
    """
    SPSMDEffectiveCopulaBase model with normalisation.
    """
    def __init__(self, mmin=2, mmax=100, normsamp=10000):
        super(SPSMDEffectiveCopulaNorm, self).__init__(mmin, mmax)
        self.rhos = xp.linspace(0,1,100)
        self.vs = xp.linspace(0,1,100)
        self.normsamp = normsamp
    
    def __call__(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                 xi_chi_eff, omega_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
                 xi_chi_dif, omega_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
                 alpha_rho, beta_rho, amax,
                 lambda_chi_peak=0):
        prob = super(SPSMDEffectiveCopulaNorm, self).__call__(
            dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
            xi_chi_eff, omega_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
            xi_chi_dif, omega_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
            alpha_rho, beta_rho, amax,
            lambda_chi_peak=0)
        prob /= self.phys_norm(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                 xi_chi_eff, omega_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
                 xi_chi_dif, omega_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
                 alpha_rho, beta_rho, amax)
        return prob
    
    def phys_norm(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                 xi_chi_eff, omega_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
                 xi_chi_dif, omega_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
                 alpha_rho, beta_rho, amax):
        chi1_samps, chi2_samps = self.gen_chis(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                             xi_chi_eff, omega_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
                             xi_chi_dif, omega_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
                             alpha_rho, beta_rho, amax)
        physical = (chi1_samps <= 1) & (chi2_samps <= 1)
        n_physical = len(chi1_samps[(physical)])
        norm = n_physical/self.normsamp
        return norm
        
    def gen_chis(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                 xi_chi_eff, omega_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
                 xi_chi_dif, omega_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
                 alpha_rho, beta_rho, amax):
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
        p_q = trapz(p_q * p_m, self.m1s, axis=-1)
        p_q = xp.nan_to_num(p_q)

        # cdf(q)
        p_q = cumtrapz(p_q, self.qs, initial=0)
        p_q /= xp.max(p_q)

        # sample q
        q_interp = xp.random.rand(self.normsamp)
        q_samps = xp.interp(q_interp,
                            p_q, self.qs)

        # pdf(rho)
        p_rho = beta_dist(self.rhos, alpha_rho, beta_rho, amax)
        p_rho = cumtrapz(p_rho, self.rhos, initial=0)
        p_rho /= xp.max(p_rho)

        # sample rho
        rho1_samps = xp.interp(xp.random.rand(self.normsamp),
                               p_rho, self.rhos)
        rho2_samps = xp.interp(xp.random.rand(self.normsamp),
                               p_rho, self.rhos)

        # pdf(chi_eff)
        p_chi_eff = truncskewnorm(self.chi_effs, xi_chi_eff, omega_chi_eff,
                                  chi_eff_max, chi_eff_min, chi_eff_skew)
        p_chi_eff = cumtrapz(p_chi_eff, self.chi_effs, initial=0)
        p_chi_eff /= xp.max(p_chi_eff)

        # sample chi_eff
        chi_eff_interp = self.copula_sample(kappa_q_chi_eff, q_interp)
        chi_eff_samps = xp.interp(chi_eff_interp, p_chi_eff, self.chi_effs)

        # pdf(chi_dif)
        p_chi_dif = truncskewnorm(self.chi_effs, xi_chi_dif, omega_chi_dif,
                                  chi_dif_max, chi_dif_min, chi_dif_skew)
        p_chi_dif = cumtrapz(p_chi_dif, self.chi_effs, initial=0)
        p_chi_dif /= xp.max(p_chi_dif)

        # sample chi_dif
        chi_dif_samps = xp.interp(xp.random.rand(self.normsamp),
                                  p_chi_dif, self.chi_effs)

        # convert
        chi1_samps, chi2_samps = self.conversion(chi_eff_samps, chi_dif_samps,
                                                rho1_samps, rho2_samps, q_samps)
        
        return chi1_samps, chi2_samps
    
    def copula_sample(self, kappa_q_chi_eff, u):
        v_int = xp.random.rand(self.normsamp)
        if kappa_q_chi_eff == 0:
            return v_int

        ug, vg = xp.meshgrid(u, self.vs)

        p = self.copula_function(ug, vg, kappa_q_chi_eff)
        c = cumtrapz(p, self.vs, initial=0, axis=0)

        v_samp = xp.array([])
        for i in range(self.normsamp):
            v_samp = xp.append(v_samp, xp.interp(v_int[i], c[:,i], self.vs))
        return v_samp
    
    def conversion(self, chi_eff, chi_dif, rho1, rho2, q):
        junk1 = (1 + q)*(q*chi_dif + chi_eff)/(1 + q**2)
        junk2 = (1 + q)*(q*chi_eff - chi_dif)/(1 + q**2)

        chi1 = xp.sqrt(junk1**2 + rho1**2)
        chi2 = xp.sqrt(junk2**2 + rho2**2)

        return xp.nan_to_num(chi1), xp.nan_to_num(chi2)

    
class SPSMDEffectiveFrankCopula(SPSMDEffectiveCopulaNorm):
    """
    SPSMDEffectiveCopulaBase model with Frank copula density.
    """
    def copula_function(self, u, v, kappa):
        return frank_copula(u, v, kappa)

    
class SPSMDEffectiveGaussianCopula(SPSMDEffectiveCopulaNorm):
    """
    SPSMDEffectiveCopulaBase model with Gaussian copula density.
    """
    def copula_function(self, u, v, kappa):
        return gaussian_copula(u, v, kappa)
    
    
class SPSMDEffectiveFGMCopula(SPSMDEffectiveCopulaNorm):
    """
    SPSMDEffectiveCopulaBase model with FGM copula density.
    """
    def copula_function(self, u, v, kappa):
        return fgm_copula(u, v, kappa)