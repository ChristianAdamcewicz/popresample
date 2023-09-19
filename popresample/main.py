import pickle

from bilby_pipe.bilbyargparser import BilbyArgParser

from gwpopulation.hyperpe import HyperparameterLikelihood
from gwpopulation.vt import ResamplingVT
from gwpopulation.conversions import convert_to_beta_parameters

from .preprocessing import load_models, load_data, create_new_param
from .resampler import ImportanceSampler


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
    parser.add_argument("--vt-models",
                        type=str,
                        action="append",
                        help="List of models for selection effects.")
    parser.add_argument("--data-file",
                        type=str,
                        help="File containing event posteriors.")
    parser.add_argument("--data-file2",
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
    
    model, model2, vt_model = load_models(models=args.models,
                                          models2=args.models2,
                                          vt_models=args.vt_models)
    data, data2, vt_data, results = load_data(data_file=args.data_file,
                                              data_file2=args.data_file2,
                                              vt_file=args.vt_file,
                                              result_file=args.result_file)
    if args.new_param_name is None:
        new_param = None
    else:
        new_param = create_new_param(args.new_param_name,
                                     param_min=args.param_min,
                                     param_max=args.param_max,
                                     param_bins=args.param_bins)
    
    selection_function = ResamplingVT(model=vt_model,
                                      data=vt_data,
                                      n_events=len(data))
    likelihood = HyperparameterLikelihood(
                            posteriors1=data,
                            posteriors2=data2,
                            hyper_prior1=model,
                            hyper_prior2=model2,
                            max_samples=args.max_samples,
                            selection_function=selection_function,
                            conversion_function=convert_to_beta_parameters)
    
    resampler = ImportanceSampler(likelihood=likelihood,
                                  results=results,
                                  new_param=new_param)
    new_results = resampler.resample()
    
    with open(args.output_file, "wb") as f:
        pickle.dump(new_results, f)
    