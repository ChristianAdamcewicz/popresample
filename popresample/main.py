import pickle

from bilby_pipe.bilbyargparser import BilbyArgParser

from gwpopulation.hyperpe import (
    HyperparameterLikelihood,
    HyperparameterTwoLikelihood,
    HyperparameterThreeLikelihood,
    HyperparameterFourLikelihood
)
from gwpopulation.vt import (
    ResamplingVT,
    ResamplingVTtwo,
    ResamplingVTthree,
    ResamplingVTfour
)
from gwpopulation.conversions import convert_to_beta_parameters

from .preprocessing import load_models, load_data, create_new_param
from .resampler import ImportanceSampler

LIKELIHOOD_FUNCTION = [
    HyperparameterLikelihood,
    HyperparameterTwoLikelihood,
    HyperparameterThreeLikelihood,
    HyperparameterFourLikelihood
]

VT_FUNCTION = [
    ResamplingVT,
    ResamplingVTtwo,
    ResamplingVTthree,
    ResamplingVTfour
]

def create_parser():
    parser = BilbyArgParser()
    parser.add("ini", type=str, is_config_file=True, help="Configuration ini file.")
    parser.add_argument("--models",
                        type=str,
                        action="append",
                        help="List of models for proposal distribution.")
    parser.add_argument("--models2",
                        type=str,
                        action="append",
                        help="List of models for proposal distribution.")
    parser.add_argument("--models3",
                        type=str,
                        action="append",
                        help="List of models for proposal distribution.")
    parser.add_argument("--models4",
                        type=str,
                        action="append",
                        help="List of models for proposal distribution.")
    parser.add_argument("--vt-models",
                        type=str,
                        action="append",
                        help="List of models for selection effects.")
    parser.add_argument("--vt-models2",
                        type=str,
                        action="append",
                        help="List of models for selection effects.")
    parser.add_argument("--vt-models3",
                        type=str,
                        action="append",
                        help="List of models for selection effects.")
    parser.add_argument("--vt-models4",
                        type=str,
                        action="append",
                        help="List of models for selection effects.")
    parser.add_argument("--data-file",
                        type=str,
                        action="append",
                        help="File containing event posteriors.")
    parser.add_argument("--vt-file",
                        type=str,
                        action="append",
                        help="File containing vt data.")
    parser.add_argument("--result-file",
                        type=str,
                        help="Population result file to resample.")
    parser.add_argument("--n-likelihoods",
                        type=int,
                        default=1,
                        help="Number of likelihoods in mixture model.")
    parser.add_argument("--new-param-name",
                        type=str,
                        help="Name of added hyper-parameter.")
    parser.add_argument("--param-min",
                        type=float,
                        help="Minimum value of added hyper-parameter.")
    parser.add_argument("--param-max",
                        type=float,
                        help="Maximum value of added hyper-parameter.")
    parser.add_argument("--param-bins",
                        type=int,
                        default=25,
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


def run():
    parser = create_parser()
    args, unknown_args = parser.parse_known_args()
    
    model, vt_model = load_models(args)
    data, ln_evidences, vt_data, results = load_data(args)
    if args.new_param_name is None:
        new_param = None
    else:
        new_param = create_new_param(args.new_param_name,
                                     param_min=args.param_min,
                                     param_max=args.param_max,
                                     param_bins=args.param_bins)
    
    likelihood_function = LIKELIHOOD_FUNCTION[args.n_likelihoods-1]
    vt_function = VT_FUNCTION[args.n_likelihoods-1]

    selection_function = vt_function(**vt_model,
                                     **vt_data,
                                     n_events=len(data))
    likelihood = likelihood_function(
                            **data,
                            **model,
                            **ln_evidences,
                            max_samples=args.max_samples,
                            selection_function=selection_function,
                            conversion_function=convert_to_beta_parameters)
    
    resampler = ImportanceSampler(likelihood=likelihood,
                                  results=results,
                                  new_param=new_param)
    new_results = resampler.resample()
    
    with open(args.output_file, "wb") as f:
        pickle.dump(new_results, f)
    