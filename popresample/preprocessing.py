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


def load_models(args):
    """
    Loads population models using model names.
    
    Parameters
    ----------
    args:
        args.
    
    Returns
    -------
    model: dict of bilby.hyper.model.Model
        Loaded modela.
    vt_model: dict of bilby.hyper.model.Model
        Loaded vt modela.
    """
    model = dict(hyper_prior=Model([load_model(key) for key in args.models]))
    vt_model = dict(model=Model([load_model(key) for key in args.vt_models]))
    if args.n_likelihoods > 1:
        model["hyper_prior2"] = Model([load_model(key) for key in args.models2])
        vt_model["model2"] = Model([load_model(key) for key in args.vt_models2])
        if args.n_likelihoods > 2:
            model["hyper_prior3"] = Model([load_model(key) for key in args.models3])
            vt_model["model3"] = Model([load_model(key) for key in args.vt_models3])
            if args.n_likelihoods > 3:
                model["hyper_prior4"] = Model([load_model(key) for key in args.models4])
                vt_model["model4"] = Model([load_model(key) for key in args.vt_models4])
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


def load_from_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def load_data(args):
    """
    Loads data as packed by gwpopulation.
    
    Parameters
    ----------
    args:
        args.
    
    Returns
    -------
    data: list
        Unpacked posteriors.
    vt_data: dict
        Unpacked vt data.
    results: dict
        Unpacked results.
    """
    for i in range(1, args.n_likelihoods+1):
        data = load_from_pickle(args.data_file[i-1])
        ln_evidences = load_evidences(data)
        vt_data = load_from_pickle(args.vt_file[i-1])
        if i == 1:
            data_dict = dict(posteriors=data)
            ln_evidences_dict = dict(ln_evidences=ln_evidences)
            vt_data_dict = dict(data=vt_data)
        else:
            data_dict[f"posteriors{i}"] = data
            ln_evidences_dict[f"ln_evidences{i}"] = ln_evidences
            vt_data_dict[f"data{i}"] = vt_data
    results = result.read_in_result(filename=args.result_file).posterior
    return data_dict, ln_evidences_dict, vt_data_dict, results


def load_evidences(posteriors):
    evidences = []
    for post in posteriors:
        if "ln_evidence" in post.keys():
            _evidences = post.pop("ln_evidence")
            evidences.append(_evidences.iloc[0])
    if len(evidences) != len(posteriors):
        return None
    else:
        return evidences