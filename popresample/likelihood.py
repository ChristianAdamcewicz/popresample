from .cupy_utils import xp
from bilby.hyper.model import Model


class Likelihood():
    ''''''
    def __init__(self,
                 data,
                 model,
                 selection_function):
        ''''''
        self.data = data
        if isinstance(model, list):
            model = Model(model)
        elif not isinstance(model, Model):
            model = Model([model])
        self.model = model
        self.selection_function = selection_function

    def __call__(self, hypersample):
        per_event = self.log_likelihood_per_event(hypersample)
        log_likelihood = xp.sum(per_event)
        log_likelihood += self.selection(hypersample)
        if xp.isnan(log_likelihood):
            return -xp.inf
        else:
            return float(xp.nan_to_num(log_likelihood))
    
    def log_likelihood_per_event(self, hypersample):
        self.model.parameters.update(hypersample)
        weights = self.model.prob(self.data) / self.data["prior"]
        per_event = xp.mean(weights, axis=-1)
        return xp.log(per_event)
    
    def selection(self, hypersample):
        return -self.selection_function.n_events * xp.log(self.selection_function(hypersample))