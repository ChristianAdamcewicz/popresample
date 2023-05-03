"""
Functions for processing data and setting up resampler inputs.
"""
import inspect
import numpy as np
import pickle

from bilby import result
from bilby.hyper.model import Model

from .models.model_map import MODEL_MAP


def create_new_param(param_name, param_min, param_max, param_bins=25):
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
    new_param[param_name] = np.linspace(param_min, param_max, param_bins)
    new_param["log_prior"] = np.array([-np.log(param_max - param_min)]*param_bins)
    return new_param


def load_models(models, vt_models):
    """
    Loads population models using model names.
    
    Parameters
    ----------
    models: list (str)
        List of population model mames.
    vt_models: list (str)
        List of vt model mames.
    
    Returns
    -------
    model: bilby.hyper.model.Model
        Loaded model.
    vt_models: bilby.hyper.model.Model
        Loaded vt model.
    """
    if vt_models is None:
        vt_models = models
    model = Model([load_model(key) for key in models])
    vt_model = Model([load_model(key) for key in vt_models])
    return model, vt_model


def load_model(key):
    """
    Loads individual model from model map.
    
    Parameters
    ----------
    key: str
        Name of model to load.
        
    Returns
    -------
    model: func
        Loaded model.
    """
    model = MODEL_MAP[key]
    if inspect.isclass(model):
        model = model()
    return model


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
