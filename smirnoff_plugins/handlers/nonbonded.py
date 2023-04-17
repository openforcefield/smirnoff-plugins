import abc

from openff.toolkit.typing.engines.smirnoff.parameters import (
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
    _allow_only,
)
from openff.toolkit.utils.exceptions import IncompatibleParameterError
from openff.units import unit


class _CustomNonbondedHandler(ParameterHandler, abc.ABC):
    """The base class for custom parameter handlers which apply nonbonded parameters."""

    scale12 = ParameterAttribute(default=0.0, converter=float)
    scale13 = ParameterAttribute(default=0.0, converter=float)
    scale14 = ParameterAttribute(default=0.5, converter=float)
    scale15 = ParameterAttribute(default=1.0, converter=float)

    cutoff = ParameterAttribute(default=9.0 * unit.angstroms, unit=unit.angstrom)
    method = ParameterAttribute(
        default="cutoff", converter=_allow_only(["cutoff", "PME"])
    )
    switch_width = ParameterAttribute(default=1.0 * unit.angstroms, unit=unit.angstrom)

    def check_handler_compatibility(self, other_handler: ParameterHandler):
        """Checks whether this ParameterHandler encodes compatible physics as another
        ParameterHandler. This is called if a second handler is attempted to be
        initialized for the same tag.
        Parameters
        ----------
        other_handler : a ParameterHandler object
            The handler to compare to.
        Raises
        ------
        IncompatibleParameterError if handler_kwargs are incompatible with existing
        parameters.
        """

        if self.__class__ != other_handler.__class__:
            return IncompatibleParameterError(
                f"{self.__class__} and {other_handler.__class__} are not compatible."
            )

        float_attrs_to_compare = ["scale12", "scale13", "scale14", "scale15"]
        string_attrs_to_compare = ["method"]
        unit_attrs_to_compare = ["cutoff"]

        self._check_attributes_are_equal(
            other_handler,
            identical_attrs=string_attrs_to_compare,
            tolerance_attrs=float_attrs_to_compare + unit_attrs_to_compare,
            tolerance=self._SCALETOL,
        )


class DampedBuckingham68Handler(_CustomNonbondedHandler):
    """A custom SMIRNOFF handler for damped Buckingham interactions."""

    class DampedBuckingham68Type(ParameterType):
        """A custom SMIRNOFF type for damped Buckingham interactions."""

        _VALENCE_TYPE = "Atom"
        _ELEMENT_NAME = "Atom"

        a = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole)
        b = ParameterAttribute(default=None, unit=unit.nanometer**-1)
        c6 = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole * unit.nanometer**6
        )
        c8 = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole * unit.nanometer**8
        )

    _TAGNAME = "DampedBuckingham68"
    _INFOTYPE = DampedBuckingham68Type

    gamma = ParameterAttribute(default=35.8967, unit=unit.nanometer**-1)


class DoubleExponentialHandler(_CustomNonbondedHandler):
    """A custom SMIRNOFF handler for double exponential interactions."""

    class DoubleExponentialType(ParameterType):
        """A custom SMIRNOFF type for double exponential interactions."""

        _VALENCE_TYPE = "Atom"
        _ELEMENT_NAME = "Atom"

        r_min = ParameterAttribute(default=None, unit=unit.nanometers)
        epsilon = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole)

    _TAGNAME = "DoubleExponential"
    _INFOTYPE = DoubleExponentialType

    # These are defined as dimensionless, we should consider enforcing global parameters
    # as being unit-bearing even if that means using `unit.dimensionless`
    alpha = ParameterAttribute(default=18.7)
    beta = ParameterAttribute(default=3.3)

class DampedExp6810Handler(_CustomNonbondedHandler):
    """
    Damped exponential-6-8-10 potential used in <https://doi.org/10.1021/acs.jctc.0c00837>
    Essentially a Buckingham-6-8-10 potential with mixing rules from
    <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.5.1708>
    """

    class DampedExp6810Type(ParameterType):
        """A custom SMIRNOFF type for 6810 interactions."""

        _VALENCE_TYPE = "Atom"
        _ELEMENT_NAME = "Atom"

        rho = ParameterAttribute(default=None, unit=unit.nanometers)
        beta = ParameterAttribute(default=None, unit=unit.nanometers**-1)
        c6 = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole * unit.nanometer**6)
        c8 = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole * unit.nanometer**8)
        c10 = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole * unit.nanometer**10)

    _TAGNAME = "DampedExp6810"
    _INFOTYPE = DampedExp6810Type

    # These are defined as dimensionless, we should consider enforcing global parameters
    # as being unit-bearing even if that means using `unit.dimensionless`
    forceAtZero = ParameterAttribute(default=49.6144931952, unit=unit.kilojoules_per_mole * unit.nanometer**-1)


class AxilrodTellerHandler(_CustomNonbondedHandler):
    """
    Axilrod-Teller potential
    """

    class AxilrodTellerType(ParameterType):
        """A custom SMIRNOFF type for Axilrod-Teller interactions."""

        _VALENCE_TYPE = "Atom"
        _ELEMENT_NAME = "Atom"

        c9 = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole * unit.nanometer**10)

    _TAGNAME = "AxilrodTeller"
    _INFOTYPE = AxilrodTellerType


class MultipoleHandler(_CustomNonbondedHandler):
    """
    Amoeba multipole handler from openmm
    """

    class MultipoleType(ParameterType):
        """A custom SMIRNOFF type for multipole interactions."""

        _VALENCE_TYPE = "Atom"
        _ELEMENT_NAME = "Atom"

        polarity = ParameterAttribute(default=None, unit=unit.nanometers**3)

    _TAGNAME = "Multipole"
    _INFOTYPE = MultipoleType
