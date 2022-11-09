from .cupy_utils import xp
from .utils import chi_effective_prior_from_isotropic_spins

import pickle

def resample_posteriors(posteriors, max_samples=1e300):
    ''''''
    for posterior in posteriors:
        max_samples = min(len(posterior), max_samples)
    data = {key: [] for key in posteriors[0]}
    for posterior in posteriors:
        temp = posterior.sample(max_samples)
        for key in data:
            data[key].append(temp[key])
    for key in data:
        data[key] = xp.array(data[key])
    return data
    
def chi_eff_prep(posteriors):
    ''''''
    for posterior in posteriors:
        posterior["chi_eff"] = (posterior["a_1"] * posterior["cos_tilt_1"] +
                  posterior["mass_ratio"] * posterior["a_2"] * posterior["cos_tilt_2"]) / (
                  1 + posterior["mass_ratio"])

        posterior["prior"] *= 4
        posterior["prior"] *= chi_effective_prior_from_isotropic_spins(posterior["mass_ratio"],
                                                                       1.,
                                                                       posterior["chi_eff"])
    return posteriors

def load_data(file, chi_eff=False):
    with open(file, "rb") as f:
        posteriors = pickle.load(f)
    if chi_eff:
        posteriors = chi_eff_prep(posteriors)
    n_events = len(posteriors)
    data = resample_posteriors(posteriors)
    return data, n_events