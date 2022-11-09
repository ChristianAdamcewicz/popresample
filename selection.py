from .cupy_utils import xp
from bilby.hyper.model import Model

class _BaseVT(object):
    def __init__(self, model, data):
        self.data = data
        if isinstance(model, list):
            model = Model(model)
        elif not isinstance(model, Model):
            model = Model([model])
        self.model = model

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class ResamplingVT(_BaseVT):
    def __init__(self, model, data, n_events):
        super(ResamplingVT, self).__init__(model=model, data=data)
        self.n_events = n_events
        self.total_injections = data.get("total_generated", len(data["prior"]))
        self.redshift_model = None
        for _model in self.model.models:
            if isinstance(_model, _Redshift):
                self.redshift_model = _model
        if self.redshift_model is None:
            self._surveyed_hypervolume = total_four_volume(
                lamb=0, analysis_time=self.analysis_time
            )

    def __call__(self, parameters):
        mu, var = self.detection_efficiency(parameters)
        if mu**2 <= 4 * self.n_events * var:
            return np.inf
        n_effective = mu**2 / var
        vt_factor = mu / np.exp((3 + self.n_events) / 2 / n_effective)
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
