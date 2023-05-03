"""
Dictionary for mapping population models.
"""
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.spin import iid_spin
from gwpopulation.models.redshift import PowerLawRedshift

from .joint import MassRedshiftCopula, SPSMD_EffectiveCopula


MODEL_MAP = {
    "SinglePeakSmoothedMassDistribution":SinglePeakSmoothedMassDistribution,
    "iid_spin":iid_spin,
    "PowerLawRedshift":PowerLawRedshift,
    "MassRedshiftCopula":MassRedshiftCopula,
    "SPSMD_EffectiveCopula":SPSMD_EffectiveCopula
}
