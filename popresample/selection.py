"""
Classes for selection effects.
"""
from .cupy_utils import xp
from bilby.hyper.model import Model


class ResamplingVT():
    """
    Lifted from gwpopulation (https://github.com/ColmTalbot/gwpopulation)
    
    Evaluate the sensitive volume using a set of found injections.
    See https://arxiv.org/abs/1904.10879 for details of the formalism.
    """
    def __init__(self, model, data, n_events):
        """
        Parameters
        ----------
        model: callable
            Population model.
        data: dict
            The found injections and relevant meta data.
        n_events: int
            The number of events observed.
        """
        self.data = data
        if isinstance(model, list):
            model = Model(model)
        elif not isinstance(model, Model):
            model = Model([model])
        self.model = model
        self.n_events = n_events
        self.total_injections = data.get("total_generated", len(data["prior"]))

    def __call__(self, parameters):
        """
        Compute the expected number of detections given a set of injections.
        Option to use the uncertainty-marginalized vt_factor from Equation 11
        in https://arxiv.org/abs/1904.10879 by setting `marginalize_uncertainty`
        to True, or use the estimator from Equation 8 (default behavior).
        Recommend not enabling marginalize_uncertainty and setting convergence
        criteria based on uncertainty in total likelihood in HyperparameterLikelihood.
        If using `marginalize_uncertainty` and n_effective < 4 * n_events we
        return np.inf so that the sample is rejected. This condition is also
        enforced if `enforce_convergence` is True.
        Returns either vt_factor or mu and var.
        
        Parameters
        ----------
        parameters: dict
            The population parameters
        """
        mu, var = self.detection_efficiency(parameters)
        if mu**2 <= 4 * self.n_events * var:
            return xp.inf
        n_effective = mu**2 / var
        vt_factor = mu / xp.exp((3 + self.n_events) / 2 / n_effective)
        return vt_factor

    def detection_efficiency(self, parameters):
        self.model.parameters.update(parameters)
        weights = self.model.prob(self.data) / self.data["prior"]
        mu = float(xp.sum(weights) / self.total_injections)
        var = float(
            xp.sum(weights**2) / self.total_injections**2
            - mu**2 / self.total_injections
        )
        return mu, var
