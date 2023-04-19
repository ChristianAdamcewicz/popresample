import numpy as np
from tqdm import tqdm
from bilby.hyper.model import Model

from .preprocessing import resample_posteriors
from .likelihood import Likelihood
from .selection import ResamplingVT
from .utils import trapz_exp, it_sample


class ImportanceSampler():
    def __init__(self,
                 model,
                 vt_model,
                 data,
                 vt_data,
                 results,
                 new_param):
        self.likelihood = Likelihood(
            resample_posteriors(data),
            model,
            ResamplingVT(
                vt_model,
                vt_data,
                len(data)
            ))
        self.results = results
        self.new_param = new_param
        self.new_param_key = self.get_new_param_key(new_param)

    def __call__(self):
        weights = []
        new_param_samples = []
        for i in tqdm(range(len(self.results))):
            hypersample = self.results[i:i+1].to_dict(orient="records")[0]
            log_likelihood = self.log_likelihood_grid(hypersample)
            unnormalised_log_posterior = log_likelihood + self.new_param["log_prior"]
            marginalised_log_likelihood = trapz_exp(unnormalised_log_posterior,
                                                    self.new_param[self.new_param_key])
            weight = np.exp(marginalised_log_likelihood - self.results["log_likelihood"][i])
            weights.append(weight)

            posterior = np.exp(unnormalised_log_posterior - marginalised_log_likelihood)
            new_param_sample = it_sample(posterior, self.new_param[self.new_param_key])
            new_param_samples.append(new_param_sample)
                                
        effective_samples = self.get_effective_samples(weights)
        print(f"effective samples = {effective_samples}")
        
        new_results = self.make_new_result_dict(weights, new_param_samples)
        return new_results

    def log_likelihood_grid(self, hypersample):
        log_likelihood = []
        for new_param in self.new_param[self.new_param_key]:
            hypersample[self.new_param_key] = new_param
            new_log_likelihood = self.likelihood(hypersample)
            log_likelihood.append(new_log_likelihood)
        return np.array(log_likelihood)
    
    def get_effective_samples(self, weights):
        w = np.array(weights)
        n_eff = np.sum(w)**2 / np.sum(w**2)
        return n_eff
    
    def make_new_result_dict(self, weights, new_param_samples):
        new_results = self.results.copy()
        new_results["weight"] = weights
        new_results[self.new_param_key] = new_param_samples
        return new_results

    def get_new_param_key(self, new_param):
        for key in new_param:
            if key not in ["prior", "log_prior"]:
                new_param_key = key
        return new_param_key