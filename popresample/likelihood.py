"""
Likelihood classes for proposal distributions.
"""
from .cupy_utils import xp
from .preprocessing import resample_posteriors
from bilby.hyper.model import Model


class Likelihood():
    """
    Likelihood class based on bilby.hyper.model.
    """
    def __init__(self,
                 model,
                 data,
                 selection_function,
                 max_samples=1e300):
        """
        Parameters
        ----------
        model: bilby.hyper.model.Model or list
            The population model.
        data: list
            List of posterior samples dictionaries.
        selection_function: class
            Class that computes the selection function.
        """
        if isinstance(model, list):
            model = Model(model)
        elif not isinstance(model, Model):
            model = Model([model])
        self.model = model
        self.data = resample_posteriors(data, max_samples)
        self.selection_function = selection_function

    def __call__(self, hypersample):
        """
        Computes and returns the log likelihood using the given hyper-sample.
        """
        per_event = self.log_likelihood_per_event(hypersample)
        log_likelihood = xp.sum(per_event)
        log_likelihood += self.selection(hypersample)
        if xp.isnan(log_likelihood):
            return -xp.inf
        else:
            return float(xp.nan_to_num(log_likelihood))
    
    def log_likelihood_per_event(self, hypersample):
        """
        Computes the per-event log likelihood using the given hyper-sample.
        """
        self.model.parameters.update(hypersample)
        weights = self.model.prob(self.data) / self.data["prior"]
        per_event = xp.mean(weights, axis=-1)
        return xp.log(per_event)
    
    def selection(self, hypersample):
        """
        Applies selection effects.
        """
        return -self.selection_function.n_events * xp.log(self.selection_function(hypersample))
