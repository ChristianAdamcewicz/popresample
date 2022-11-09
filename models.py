from .cupy_utils import xp, trapz, cumtrapz

from gwpopulation.utils import powerlaw, truncnorm
from gwpopulation.models.mass import two_component_single

import pickle

def frank_copula(u, v, kappa):
    if kappa == 0:
        prob = 1.
    else:
        expkap = xp.exp(kappa)
        expkapuv = expkap**(u + v)
        prob = kappa * expkapuv * (expkap - 1) / (expkap - expkap**u - expkap**v + expkapuv)**2
    return prob

class ExtendedChiCorrelation():
    def __init__(self, lookup_file):
        self.m1s = xp.linspace(2, 100, 1000)
        self.qs = xp.linspace(0.001, 1, 500)
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.chi_effs = xp.linspace(-1, 1, 500)
        
        self.chi_eff_table = []
        with open(lookup_file, "rb") as f:
            while True:
                try:
                    self.chi_eff_table.append(xp.asarray(pickle.load(f)))
                except EOFError:
                    break
        
    def __call__(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                 kappa_cop, table_id):
        # p(chi_eff)        
        p_chi_eff_grid = self.chi_eff_table[table_id]
        p_chi_eff = xp.interp(dataset["chi_eff"], self.chi_effs, p_chi_eff_grid)
        
        # p(m1)
        p_m_grid = two_component_single(
            self.m1s,
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
        )
        p_m_grid *= self.smoothing(self.m1s, mmin=mmin, mmax=100, delta_m=delta_m)
        p_m_norm = trapz(p_m_grid, self.m1s)
        p_m_grid /= p_m_norm
        p_m_grid = xp.nan_to_num(p_m_grid)
        
        # p(q|m1)
        p_q_m_grid = powerlaw(self.qs_grid, beta, 1, mmin / self.m1s_grid)
        p_q_m_grid *= self.smoothing(
            self.m1s_grid * self.qs_grid,
            mmin=mmin,
            mmax=self.m1s_grid,
            delta_m=delta_m,
        )
        p_q_m_norm = trapz(p_q_m_grid, self.qs, axis=0)
        p_q_m_grid /= p_q_m_norm
        p_q_m_grid = xp.nan_to_num(p_q_m_grid)
        
        # p(q)
        p_q_grid = trapz(p_q_m_grid * p_m_grid, self.m1s, axis=-1)
        
        # u(q)
        u_grid = cumtrapz(p_q_grid, self.qs, initial=0)
        u_grid /= xp.max(u_grid)
        u_grid = xp.nan_to_num(u_grid)
        u = xp.interp(dataset["mass_ratio"], self.qs, u_grid)
        
        # v(chi_eff)
        v_grid = cumtrapz(p_chi_eff_grid, self.chi_effs, initial=0)
        v_grid /= xp.max(v_grid)
        v_grid = xp.nan_to_num(v_grid)
        v = xp.interp(dataset["chi_eff"], self.chi_effs, v_grid)
        
        # combine
        copula = frank_copula(u, v, kappa_cop)
        
        return p_chi_eff * copula
        
    @staticmethod
    def smoothing(masses, mmin, mmax, delta_m):
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