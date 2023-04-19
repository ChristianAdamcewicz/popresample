from .cupy_utils import xp
import pickle
from bilby import result


def resample_posteriors(posteriors, max_samples=1e300):
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


def load_data(data_file, vt_file, result_file):
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    with open(vt_file, "rb") as f:
        vt_data = pickle.load(f)
    results = result.read_in_result(filename=result_file).posterior
    return data, vt_data, results


def load_injection_data(vt_file, ifar_threshold=1, snr_threshold=11):
    import numpy as np
    import h5py

    with h5py.File(vt_file, "r") as ff:
        data = ff["injections"]
        found = np.zeros_like(data["mass1_source"][()], dtype=bool)
        for key in data:
            if "ifar" in key.lower():
                found = found | (data[key][()] > ifar_threshold)
            if "name" in data.keys():
                gwtc1 = (data["name"][()] == b"o1") | (data["name"][()] == b"o2")
                found = found | (gwtc1 & (data["optimal_snr_net"][()] > snr_threshold))
        n_found = sum(found)
        gwpop_data = dict(
            mass_1=xp.asarray(data["mass1_source"][found]),
            mass_ratio=xp.asarray(
                data["mass2_source"][found] / data["mass1_source"][found]
            ),
            redshift=xp.asarray(data["redshift"][found]),
            total_generated=int(data.attrs["total_generated"][()]),
            analysis_time=data.attrs["analysis_time_s"][()] / 365.25 / 24 / 60 / 60,
        )
        for ii in [1, 2]:
            gwpop_data[f"a_{ii}"] = (
                xp.asarray(
                    data.get(f"spin{ii}x", np.zeros(n_found))[found] ** 2
                    + data.get(f"spin{ii}y", np.zeros(n_found))[found] ** 2
                    + data[f"spin{ii}z"][found] ** 2
                )
                ** 0.5
            )
            gwpop_data[f"cos_tilt_{ii}"] = (
                xp.asarray(data[f"spin{ii}z"][found]) / gwpop_data[f"a_{ii}"]
            )
        gwpop_data["prior"] = (
            xp.asarray(data["sampling_pdf"][found])
            * xp.asarray(data["mass1_source"][found])
            * (2 * np.pi * gwpop_data["a_1"] ** 2)
            * (2 * np.pi * gwpop_data["a_2"] ** 2)
        )
    return gwpop_data