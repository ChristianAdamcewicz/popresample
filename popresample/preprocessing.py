"""
Functions for processing data and setting up resampler inputs.
"""
import numpy as np
import pickle

from bilby import result
from bilby.hyper.model import Model

from .models.model_map import MODEL_MAP


def create_new_param(param_name, param_min, param_max, param_bins=20):
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
    model = []
    vt_model = []
    for key in models:
        model.append(MODEL_MAP[key])
    for key in vt_models:
        vt_model.append(MODEL_MAP[key])
    model = Model(model)
    vt_model = Model(vt_model)
    return model, vt_model


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
    evidences = []
    for event in data:
        evidence = event["ln_evidence"][0]
        evidences.append(evidence)
    return data, vt_data, results, evidences
