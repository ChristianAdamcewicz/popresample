"""
Importance sampler.
"""
import numpy as np
from tqdm import tqdm

from .utils import trapz_exp, it_sample, effective_samples


class ImportanceSampler():
    """
    Class for importance sampling of population inference results.
    """
    def __init__(self,
                 likelihood,
                 results,
                 new_param):
        """
        Parameters
        ----------
        likelihood: object
            Class that computes likelihood.
        results: dict
            Dictionary of results (hyper-posterior samples) from GWPopulation to resample.
        new_param: dict
            Grids for added hyper-parameter with keys for new hyper-parameter and associated
            log hyper-prior.
        """
        self.likelihood = likelihood
        self.results = results
        self.new_param = new_param
        self.new_param_key = self.get_new_param_key(new_param)
        self.new_results = None
        
    def __call__(self):
        if self.new_results is None:
            print("Resampling...")
            return self.resample()
        else:
            print("Using cached result...")
            return self.new_results
    
    def resample(self):
        """
        Resamples population results.
        Returns results dictionary with added weights, new hyper-parameter samples, and updated
        log likelihoods.
        """
        weights = np.array([])
        new_param_samples = np.array([])
        new_log_likelihoods = np.array([])
        
        for i in tqdm(range(len(self.results))):
            hypersample = self.results[i:i+1].to_dict(orient="records")[0]
            target, new_param_posterior = self.get_calculations(hypersample)
            
            weight = np.exp(target - self.results["log_likelihood"][i])
            weights = np.append(weights, weight)

            new_param_sample = it_sample(new_param_posterior, self.new_param[self.new_param_key])
            new_param_samples = np.append(new_param_samples, new_param_sample)
            hypersample[self.new_param_key] = new_param_sample
            
            new_log_likelihood = self.likelihood(hypersample)
            new_log_likelihoods = np.append(new_log_likelihoods, new_log_likelihood)
            
        new_results = self.make_new_result_dict(weights, new_param_samples, new_log_likelihoods)
        print(f"effective samples: {effective_samples(weights)}")
        
        return new_results
    
    def get_calculations(self, hypersample):
        """
        Calculates values used for importance sampling.
        """
        log_likelihood = self.get_log_likelihood_grid(hypersample)
        unnormalised_posterior = log_likelihood + self.new_param["log_prior"]
        target = trapz_exp(unnormalised_posterior, self.new_param[self.new_param_key])
        new_param_posterior = np.exp(unnormalised_posterior - target)
        return target, new_param_posterior
        
    def get_log_likelihood_grid(self, hypersample):
        """
        Computes log likelihood on a grid for the new hyper-sample.
        """
        log_likelihood = []
        for new_param in self.new_param[self.new_param_key]:
            hypersample[self.new_param_key] = new_param
            new_log_likelihood = self.likelihood(hypersample)
            log_likelihood.append(new_log_likelihood)
        return np.array(log_likelihood)
    
    def make_new_result_dict(self, weights, new_param_samples, new_log_likelihoods):
        """
        Makes (and caches) dictionary for new results.
        """
        new_results = self.results.copy()
        new_results["weight"] = weights
        new_results[self.new_param_key] = new_param_samples
        new_results["log_likelihood"] = new_log_likelihoods
        self.new_results = new_results
        return new_results

    def get_new_param_key(self, new_param):
        """
        Finds the name of the new hyper-parameter.
        """
        for key in new_param:
            if key not in ["log_prior"]:
                new_param_key = key
        return new_param_key
