"""
Dictionary for mapping population models.
"""
from gwpopulation.models.redshift import PowerLawRedshift

from .mass import LegacySinglePeakSmoothedMassDistribution
from .joint import SPSMD_EffectiveCopula


MODEL_MAP = {
    "SinglePeakSmoothedMassDistribution":LegacySinglePeakSmoothedMassDistribution(),
    "PowerLawRedshift":PowerLawRedshift(),
    "SPSMD_EffectiveCopula":SPSMD_EffectiveCopula()
}