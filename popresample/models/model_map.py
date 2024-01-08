"""
Dictionary for mapping population models.
"""
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.spin import (
    iid_spin,
    iid_spin_magnitude_beta,
    independent_spin_magnitude_beta,
    iip_spin_magnitude_beta,
    iis_spin_magnitude_beta,
    inp_spin_magnitude_beta,
    ins_spin_magnitude_beta,
    iid_spin_orientation_gaussian_isotropic,
    independent_spin_orientation_gaussian_isotropic,
    iip_spin_orientation_gaussian_isotropic,
    iis_spin_orientation_gaussian_isotropic,
    inp_spin_orientation_gaussian_isotropic,
    ins_spin_orientation_gaussian_isotropic
)
from gwpopulation.models.redshift import PowerLawRedshift

MODEL_MAP = {
    "SinglePeakSmoothedMassDistribution":SinglePeakSmoothedMassDistribution,
    "iid_spin":iid_spin,
    "PowerLawRedshift":PowerLawRedshift,
    "iid_spin_magnitude":iid_spin_magnitude_beta,
    "ind_spin_magnitude":independent_spin_magnitude_beta,
    "iip_spin_magnitude":iip_spin_magnitude_beta,
    "iis_spin_magnitude":iis_spin_magnitude_beta,
    "inp_spin_magnitude":inp_spin_magnitude_beta,
    "ins_spin_magnitude":ins_spin_magnitude_beta,
    "iid_spin_orientation":iid_spin_orientation_gaussian_isotropic,
    "ind_spin_orientation":independent_spin_orientation_gaussian_isotropic,
    "iip_spin_orientation":iip_spin_orientation_gaussian_isotropic,
    "iis_spin_orientation":iis_spin_orientation_gaussian_isotropic,
    "inp_spin_orientation":inp_spin_orientation_gaussian_isotropic,
    "ins_spin_orientation":ins_spin_orientation_gaussian_isotropic
}
