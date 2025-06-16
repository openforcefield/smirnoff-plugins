from openff.toolkit import unit
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ConstraintHandler,
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
)


class UreyBradleyHandler(ParameterHandler):
    """A custom SMIRNOFF handler for Urey-Bradley interactions."""

    class UreyBradleyType(ParameterType):
        """A custom SMIRNOFF type for Urey-Bradley interactions."""

        _ELEMENT_NAME = "UreyBradley"

        k = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole / unit.nanometer**2
        )
        length = ParameterAttribute(default=None, unit=unit.nanometers)

    _TAGNAME = "UreyBradleys"
    _INFOTYPE = UreyBradleyType
    _DEPENDENCIES = [ConstraintHandler]
