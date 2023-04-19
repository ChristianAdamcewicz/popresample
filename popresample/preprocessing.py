"""
Functions for processing data and setting up resampler inputs.
"""
from .cupy_utils import xp
import numpy as np
import pickle
from bilby import result


def resample_posteriors(posteriors, max_samples=1e300):
    """
    Lifted from gwpopulation (https://github.com/ColmTalbot/gwpopulation)
    
    Converts list of posterior sample dictionaries to dictionary of arrays
    of parameter samples of shapes (n_events, n_samples_per_event).
    
    Parameters
    ----------
    posteriors: list
        List of dictionaries containing posterior samples.
    max_samples: int
        Maximum number of posterior samples to use per-event.
    
    Returns
    -------
    data: dict
        Dictionary of posterior samples with shapes (n_events, n_samples_per_event).
    """
    for posterior in posteriors:
        max_samples = min(len(posterior), max_samples)
    data = {key: [] for key in posteriors[0]}
    for posterior in posteriors:
        temp = posterior.sample(max_samples)
        for key in data:
            data[key].append(temp[key])
    for key in data:
        data[key] = xp.array(data[key])
    return data


def create_new_param(param_name, param_min, param_max, n_bins=20):
    """
    Creates a dictionary containing a grid for the added parameter and a log prior
    (uniform).
    
    Parameters
    ----------
    param_name: str
        Name of new parameter.
    param_min: float
        Minimum value of new parameter.
    param_max: float
        Maximum value of new parameter.
    
    Returns
    -------
    new_param: dict
        Dictionary for new parameter with keys (param_name, "log_prior").
    """
    new_param = {}
    new_param[param_name] = np.linspace(param_min, param_max, n_bins)
    new_param["log_prior"] = np.array([-np.log(param_max - param_min)]*n_bins)
    return new_param


def load_data(data_file, vt_file, result_file):
    """
    Loads data as packed by gwpopulation.
    
    Parameters
    ----------
    data_file: str
        Name of file containing posterior samples.
    vt_file: str
        Name of file containing vt samples.
    result_file: str
        Name of file containing gwpopulation result file.
    
    Returns
    -------
    data: list
        Unpacked posteriors.
    vt_data: dict
        Unpacked vt data.
    results: dict
        Unpacked results.
    """
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    with open(vt_file, "rb") as f:
        vt_data = pickle.load(f)
    results = result.read_in_result(filename=result_file).posterior
    return data, vt_data, results
