from openff.toolkit.typing.engines.smirnoff.parameters import (
    ElectrostaticsHandler,
    IncompatibleParameterError,
    LibraryChargeHandler,
    ParameterAttribute,
    _NonbondedHandler,
    vdWHandler,
)
from openff.toolkit.utils.exceptions import SMIRNOFFSpecError
from packaging.version import Version

from smirnoff_plugins import _interchange_patch


def _alpha_converter(value) -> float:
    """Convert alpha to a float, checking that it lies in [0, 1]."""
    alpha = float(value)

    if not 0.0 <= alpha <= 1.0:
        raise SMIRNOFFSpecError(f"NAGLMBISCharges alpha must be between 0 and 1, found {alpha}.")

    return alpha


class NAGLMBISChargesHandler(_NonbondedHandler):
    """ParameterHandler for applying partially polarised NAGL-MBIS partial charges.

    The charges are computed with the pre-trained gas and water phase models from
    ``naglmbis`` (https://github.com/fjclark/nagl-mbis) and mixed as

        q = (1 - alpha) * q_gas + alpha * q_water

    Library charges take precedence over NAGL-MBIS charges, so e.g. water and ions
    still receive their library charges.

    Parameters
    ----------
    gas_model : str, optional, default="nagl-gas-charge-dipole-esp-wb-default"
        The name of the ``naglmbis`` model used for the gas phase charges.
    water_model : str, optional, default="nagl-water-charge-dipole-esp-wb-default"
        The name of the ``naglmbis`` model used for the water phase charges.
    alpha : float, optional, default=0.5
        The weight of the water phase charges, between 0 (gas phase charges only) and 1
        (water phase charges only).
    version : str, optional
        The version of the NAGLMBISCharges section specification.

    Examples
    --------
    >>> handler = NAGLMBISChargesHandler(alpha=0.3, skip_version_check=True)
    """

    _TAGNAME = "NAGLMBISCharges"
    _DEPENDENCIES = [vdWHandler, ElectrostaticsHandler, LibraryChargeHandler]
    _INFOTYPE = None  # No separate parameter types; just the model names and alpha
    _MIN_SUPPORTED_SECTION_VERSION = Version("0.1")
    _MAX_SUPPORTED_SECTION_VERSION = Version("0.1")
    _SCALETOL = 1e-5

    gas_model = ParameterAttribute(default="nagl-gas-charge-dipole-esp-wb-default", converter=str)
    water_model = ParameterAttribute(default="nagl-water-charge-dipole-esp-wb-default", converter=str)
    alpha = ParameterAttribute(default=0.5, converter=_alpha_converter)

    def check_handler_compatibility(
        self,
        other_handler: "NAGLMBISChargesHandler",
        assume_missing_is_default: bool = True,
    ):
        """
        Checks whether this ParameterHandler encodes compatible physics as another ParameterHandler. This is
        called if a second handler is attempted to be initialized for the same tag.

        Parameters
        ----------
        other_handler
            The handler to compare to.
        assume_missing_is_default

        Raises
        ------
        IncompatibleParameterError if handler_kwargs are incompatible with existing parameters.
        """
        for attribute in ("gas_model", "water_model"):
            if getattr(self, attribute) != getattr(other_handler, attribute):
                raise IncompatibleParameterError(
                    f"Attempted to initialize two NAGLMBISCharges sections with different {attribute} values: "
                    f"{getattr(self, attribute)} is not identical to {getattr(other_handler, attribute)}"
                )

        if abs(self.alpha - other_handler.alpha) > self._SCALETOL:
            raise IncompatibleParameterError(
                "Attempted to initialize two NAGLMBISCharges sections with different alpha values: "
                f"{self.alpha} is not identical to {other_handler.alpha}"
            )


# Interchange has no hook for plugin charge handlers, so teach it about this one.
_interchange_patch.install()
