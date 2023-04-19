"""
In this example, we take results for a PL+P mass and gaussian chi_eff
population model, and add a correlation between mass ratio and chi_eff
parameterised by kappa.
"""
import pickle

from popresample.resampler import ImportanceSampler
from popresample.preprocessing import load_data, create_new_param

# Define model and vt model
from popresample.models.joint import SPSMD_EffectiveCopula
from popresample.models.mass import SinglePeakSmoothedMassDistribution
from popresample.models.redshift import PowerLawRedshift

model = [SPSMD_EffectiveCopula(), PowerLawRedshift()]
vt_model = [SinglePeakSmoothedMassDistribution(), PowerLawRedshift()]

# Collect data needed
data, vt_data, results = load_data(data_file="posteriors.pkl",
                                   vt_file="vt_data_cp.pkl",
                                   result_file="gaussian_chieff_result.json")

# Set up added hyperparam
new_param = create_new_param("kappa",
                             param_min=-25,
                             param_max=25,
                             n_bins=20)

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