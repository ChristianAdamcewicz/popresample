import numpy as np
import pickle

from popresample.resampler import ImportanceSampler
from popresample.preprocessing import load_data

# Define model and selection effects model
from popresample.models.models import SPSMD_EffectiveCopula, SinglePeakSmoothedMassDistribution, PowerLawRedshift
model = [SPSMD_EffectiveCopula(), PowerLawRedshift()]
vt_model = [SinglePeakSmoothedMassDistribution(), PowerLawRedshift()]

# Collect data needed
data, vt_data, results = load_data(data_file="posteriors.pkl",
                                   vt_file="vt_data_cp.pkl",
                                   result_file="gaussian_chieff_result.json")

# Set up new added hyperparam
kappa_min = -25
kappa_max = 25
param_bins = 25
new_param = {"kappa":np.linspace(kappa_min, kappa_max, param_bins),
             "log_prior":np.array(np.log([1/(kappa_max-kappa_min)]*param_bins))}

# Resample
resampler = ImportanceSampler(model=model,
                              vt_model=vt_model,
                              data=data,
                              vt_data=vt_data,
                              results=results,
                              new_param=new_param)
new_results = resampler()

# Save results
with open("correlated_gaussian_chieff_result.pkl", "wb") as f:
    pickle.dump(new_results, f)