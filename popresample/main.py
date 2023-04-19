import sys
import pickle
from bilby_pipe.bilbyargparser import BilbyArgParser

from .preprocessing import load_data, create_new_param
from .likelihood import Likelihood
from .selection import ResamplingVT
from .resampler import ImportanceSampler

from .models.mass import SinglePeakSmoothedMassDistribution
from .models.redshift import PowerLawRedshift
from .models.joint import SPSMD_EffectiveCopula


MODEL_MAP = {
    "SinglePeakSmoothedMassDistribution":SinglePeakSmoothedMassDistribution(),
    "PowerLawRedshift":PowerLawRedshift(),
    "SPSMD_EffectiveCopula":SPSMD_EffectiveCopula()
}


def create_parser():
    parser = BilbyArgParser()
    parser.add("ini", type=str, is_config_file=True, help="Configuration ini file")
    parser.add_argument("--models",
                        type=str,
                        action="append",
                        help="List of models for proposal distribution.")
    parser.add_argument("--vt-models",
                        type=str,
                        action="append",
                        help="List of models for selection effects.")
    parser.add_argument("--data-file",
                        type=str,
                        help="File containing event posteriors.")
    parser.add_argument("--vt-file",
                        type=str,
                        help="File containing vt data.")
    parser.add_argument("--result-file",
                        type=str,
                        help="Population result file to resample.")
    parser.add_argument("--new-param-name",
                        type=str,
                        default="new_param",
                        help="Name of added hyper-parameter.")
    parser.add_argument("--param-min",
                        type=float,
                        help="Minimum value of added hyper-parameter.")
    parser.add_argument("--param-max",
                        type=float,
                        help="Maximum value of added hyper-parameter.")
    parser.add_argument("--param-bins",
                        type=int,
                        default=20,
                        help="Number of grid points to evaluate new hyper-parameter at.")
    parser.add_argument("--output-file",
                        type=str,
                        default="output.pkl",
                        help=".pkl file to store resampled results in.")
    parser.add_argument("--max-samples",
                        type=int,
                        default=1e300,
                        help="Maximum number of samples per posterior to use.")
    return parser

    
def load_models(models, vt_models):
    model = []
    vt_model = []
    for key in models:
        model.append(MODEL_MAP[key])
    for key in vt_models:
        vt_model.append(MODEL_MAP[key])
    return model, vt_model


def run():
    parser = create_parser()
    args, unknown_args = parser.parse_known_args()
    
    model, vt_model = load_models(models=args.models,
                                  vt_models=args.vt_models)
    data, vt_data, results = load_data(data_file=args.data_file,
                                       vt_file=args.vt_file,
                                       result_file=args.result_file)
    new_param = create_new_param(args.new_param_name,
                                 param_min=args.param_min,
                                 param_max=args.param_max,
                                 param_bins=args.param_bins)
    
    selection_function = ResamplingVT(model=vt_model,
                                      data=vt_data,
                                      n_events=len(data))
    likelihood = Likelihood(model=model,
                            data=data,
                            selection_function=selection_function,
                            max_samples=args.max_samples)
    
    resampler = ImportanceSampler(likelihood=likelihood,
                                  results=results,
                                  new_param=new_param)
    new_results = resampler()
    
    with open(args.output_file, "wb") as f:
        pickle.dump(new_results, f)
    