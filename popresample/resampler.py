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
                 new_param=None):
        """
        Parameters
        ----------
        likelihood: gwpopulation.hyperpe.HyperparameterLikelihood
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
        if new_param is not None:
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
        
        pbar = tqdm(range(len(self.results)))
        for i in pbar:
            hypersample = self.results[i:i+1].to_dict(orient="records")[0]
            target, new_param_posterior = self.get_calculations(hypersample)
            
            weight = np.exp(target - self.results["log_likelihood"][i])
            weights = np.append(weights, weight)

            if self.new_param is not None:
                new_param_sample = it_sample(new_param_posterior, self.new_param[self.new_param_key])
                new_param_samples = np.append(new_param_samples, new_param_sample)
                hypersample[self.new_param_key] = new_param_sample
            
                pbar.set_postfix({'log_L(old)':self.results["log_likelihood"][i],
                                  'weight':weight,
                                  self.new_param_key:new_param_sample})
            else:
                pbar.set_postfix({'log_L(old)':self.results["log_likelihood"][i],
                                  'weight':weight})
            
        new_results = self.make_new_result_dict(weights, new_param_samples)
        print(f"effective samples: {effective_samples(weights)}")
        
        return new_results
    
    def get_calculations(self, hypersample):
        """
        Calculates values used for importance sampling.
        """
        if self.new_param is None:
            self.likelihood.parameters = hypersample
            target = self.likelihood.log_likelihood_ratio()
            return target, None
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
            self.likelihood.parameters = hypersample
            new_log_likelihood = self.likelihood.log_likelihood_ratio()
            log_likelihood.append(new_log_likelihood)
        return np.array(log_likelihood)
    
    def make_new_result_dict(self, weights, new_param_samples):
        """
        Makes (and caches) dictionary for new results.
        """
        new_results = self.results.copy()
        new_results["weight"] = weights
        if self.new_param is not None:
            new_results[self.new_param_key] = new_param_samples
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
