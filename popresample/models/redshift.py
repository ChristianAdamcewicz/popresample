"""
A place to store redshift models.
"""
from warnings import warn
from astropy.cosmology import Planck15
import numpy as np
from ..cupy_utils import to_numpy, trapz, xp


class _Redshift(object):
    """
    Lifted from gwpopulation (https://github.com/ColmTalbot/gwpopulation)
    
    Base class for models which include a term like dVc/dz / (1 + z)
    """

    def __init__(self, z_max=2.3):
        self.z_max = z_max
        self.zs_ = np.linspace(1e-3, z_max, 1000)
        self.zs = xp.asarray(self.zs_)
        self.dvc_dz_ = Planck15.differential_comoving_volume(self.zs_).value * 4 * np.pi
        self.dvc_dz = xp.asarray(self.dvc_dz_)
        self.cached_dvc_dz = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _cache_dvc_dz(self, redshifts):
        self.cached_dvc_dz = xp.asarray(
            np.interp(to_numpy(redshifts), self.zs_, self.dvc_dz_)
        )

    def normalisation(self, parameters):
        r"""
        Compute the normalization or differential spacetime volume.
        .. math::
            \mathcal{V} = \int dz \frac{1}{1+z} \frac{dVc}{dz} \psi(z|\Lambda)
        Parameters
        ----------
        parameters: dict
            Dictionary of parameters
        Returns
        -------
        (float, array-like): Total spacetime volume
        """
        psi_of_z = self.psi_of_z(redshift=self.zs, **parameters)
        norm = trapz(psi_of_z * self.dvc_dz / (1 + self.zs), self.zs)
        return norm

    def probability(self, dataset, **parameters):
        normalisation = self.normalisation(parameters=parameters)
        differential_volume = self.differential_spacetime_volume(
            dataset=dataset, **parameters
        )
        return differential_volume / normalisation

    def psi_of_z(self, redshift, **parameters):
        raise NotImplementedError

    def differential_spacetime_volume(self, dataset, **parameters):
        r"""
        Compute the differential spacetime volume.
        .. math::
            d\mathcal{V} = \frac{1}{1+z} \frac{dVc}{dz} \psi(z|\Lambda)
        Parameters
        ----------
        dataset: dict
            Dictionary containing entry "redshift"
        parameters: dict
            Dictionary of parameters
        Returns
        -------
        differential_volume: (float, array-like)
            Differential spacetime volume
        """
        psi_of_z = self.psi_of_z(redshift=dataset["redshift"], **parameters)
        differential_volume = psi_of_z / (1 + dataset["redshift"])
        try:
            differential_volume *= self.cached_dvc_dz
        except (TypeError, ValueError):
            self._cache_dvc_dz(dataset["redshift"])
            differential_volume *= self.cached_dvc_dz
        return differential_volume

    def total_spacetime_volume(self, **parameters):
        """
        Deprecated use normalisation instead.
        {}
        """.format(
            _Redshift.normalisation.__doc__
        )
        warn(
            "The total spacetime volume method is deprecated, "
            "use normalisation instead.",
            DeprecationWarning,
        )
        return self.normalisation(parameters=parameters)


class PowerLawRedshift(_Redshift):
    r"""
    Lifted from gwpopulation (https://github.com/ColmTalbot/gwpopulation)
    
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270
    .. math::
        p(z|\gamma, \kappa, z_p) &\propto \frac{1}{1 + z}\frac{dV_c}{dz} \psi(z|\gamma, \kappa, z_p)
        \psi(z|\gamma, \kappa, z_p) &= (1 + z)^\lambda
    Parameters
    ----------
    lamb: float
        The spectral index.
    """

    def __call__(self, dataset, lamb):
        return self.probability(dataset=dataset, lamb=lamb)

    def psi_of_z(self, redshift, **parameters):
        return (1 + redshift) ** parameters["lamb"]
