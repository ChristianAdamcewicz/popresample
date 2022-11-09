from .cupy_utils import xp
from .selection import ResamplingVT

class Likelihood():
    ''''''
    def __init__(self,
                 data,
                 hyperprior,
                 selection_function,
                 conversion_function,
                 data2=None,
                 hyperprior2=None,
                 mixing_frac_key=None):
        ''''''
        self.data = data
        self.hyperprior = hyperprior
        self.selection_function = selection_function
        self.conversion_function = conversion_function
        self.total_evidence = xp.asarray([val[0] for val in data["ln_evidence"]])
        
        if data2 is not None:
            self.multi_likelihood = True
            self.data2 = data2
            self.hyperprior2 = hyperprior2
            self.mixing_frac_key = mixing_frac_key
            self.total_evidence2 = xp.asarray([val[0] for val in data2["ln_evidence"]])
        else:
            self.multi_likelihood = False

    def __call__(self, hypersample):
        hypersample, added_keys = self.conversion_function(hypersample)
        if self.multi_likelihood:
            per_event = self.multi_log_likelihood_per_event(hypersample)
        else:
            per_event = self.log_likelihood_per_event(hypersample)
        log_likelihood = xp.sum(per_event)
        log_likelihood += self.selection(hypersample)
        if xp.isnan(log_likelihood):
            return -xp.inf
        else:
            return float(xp.nan_to_num(log_likelihood))
        
    def log_likelihood_per_event(self, hypersample):
        self.hyperprior.parameters.update(hypersample)
        weights = self.hyperprior.prob(self.data) / self.data["prior"]
        per_event = xp.mean(weights, axis=-1)
        return xp.log(per_event)
    
    def multi_log_likelihood_per_event(self, hypersample):
        self.hyperprior.parameters.update(hypersample)
        self.hyperprior2.parameters.update(hypersample)
        weights = self.hyperprior.prob(self.data) / self.data["prior"]
        weights2 = self.hyperprior2.prob(self.data2) / self.data2["prior"]
        per_event = xp.mean(weights, axis=-1)
        per_event2 = xp.mean(weights2, axis=-1) * xp.exp(self.total_evidence2 - self.total_evidence)
        mixed = ((1 - hypersample[self.mixing_frac_key]) * per_event + 
                 hypersample[self.mixing_frac_key] * per_event2)
        return xp.log(mixed)
    
    def selection(self, hypersample):
        return -self.selection_function.n_events * xp.log(self.selection_function(hypersample))
