"""Type annotations of quantity dimensionality not yet provided by upstreams."""

from typing import Annotated

from openff.interchange._annotations import (
    _dimensionality_validator_factory,
    _DimensionlessQuantity,
    _DistanceQuantity,
    quantity_json_serializer,
    quantity_validator,
)
from openff.toolkit import Quantity
from pydantic import AfterValidator, WrapSerializer, WrapValidator

__all__ = (
    "_InverseDistanceQuantity",
    "_DistanceQuantity",
    "_DimensionlessQuantity",
    "_kJMolNanometerQuantity",
)

(
    _is_inverse_distance,
    _is_kj_mol_nanometer,
) = (
    _dimensionality_validator_factory(unit=_unit)
    for _unit in [
        "nanometer ** -1",
        "kilojoules_per_mole * nanometer ** -1",
    ]
)

_InverseDistanceQuantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    AfterValidator(_is_inverse_distance),
    WrapSerializer(quantity_json_serializer),
]

_kJMolNanometerQuantity = Annotated[
    Quantity,
    WrapValidator(quantity_validator),
    AfterValidator(_is_kj_mol_nanometer),
    WrapSerializer(quantity_json_serializer),
]
