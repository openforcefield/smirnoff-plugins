import numpy
from openff.toolkit.typing.engines.smirnoff.parameters import (
    IndexedParameterAttribute,
    ParameterAttribute,
    VirtualSiteHandler,
    _VirtualSiteType,
    ParameterType,
    _validate_units,
)
from typing import Optional, get_args
from openff.toolkit.utils.exceptions import SMIRNOFFSpecError
from openff.units import unit


class _BasicVirtualSiteType(ParameterType):
    _ELEMENT_NAME = "VirtualSite"

    name = ParameterAttribute(default="EP", converter=str)
    type = ParameterAttribute(converter=str)

    match = ParameterAttribute(converter=str)

    distance = ParameterAttribute(unit=unit.angstrom)
    outOfPlaneAngle = ParameterAttribute(unit=unit.degree)
    inPlaneAngle = ParameterAttribute(unit=unit.degree)

    charge_increment = IndexedParameterAttribute(unit=unit.elementary_charge)

    @property
    def parent_index(self) -> int:
        """Returns the index of the atom matched by the SMIRKS pattern that should
        be considered the 'parent' to the virtual site.
        A value of ``0`` corresponds to the atom matched by the ``:1`` selector in
        the SMIRKS pattern, a value ``2`` the atom matched by ``:2`` and so on.
        """
        return self.type_to_parent_index(self.type)

    @classmethod
    def type_to_parent_index(cls, type_: _VirtualSiteType) -> int:
        """Returns the index of the atom matched by the SMIRKS pattern that should
        be considered the 'parent' to a given type of virtual site.
        A value of ``0`` corresponds to the atom matched by the ``:1`` selector in
        the SMIRKS pattern, a value ``2`` the atom matched by ``:2`` and so on.
        """

        if type_.replace("VirtualSite", "") in get_args(_VirtualSiteType):
            return 0

        raise NotImplementedError()

    @outOfPlaneAngle.converter  # type: ignore[no-redef]
    def outOfPlaneAngle(self, attr, value):
        if value == "None":
            return

        supports_out_of_plane_angle = self._supports_out_of_plane_angle(self.type)

        if not supports_out_of_plane_angle and value is not None:
            raise SMIRNOFFSpecError(
                f"'{self.type}' sites do not support `outOfPlaneAngle`"
            )
        elif supports_out_of_plane_angle:
            return _validate_units(attr, value, unit.degrees)

        return value

    @inPlaneAngle.converter  # type: ignore[no-redef]
    def inPlaneAngle(self, attr, value):
        if value == "None":
            return

        supports_in_plane_angle = self._supports_in_plane_angle(self.type)

        if not supports_in_plane_angle and value is not None:
            raise SMIRNOFFSpecError(
                f"'{self.type}' sites do not support `inPlaneAngle`"
            )
        elif supports_in_plane_angle:
            return _validate_units(attr, value, unit.degrees)

        return value

    def __init__(self, **kwargs):
        self._add_default_init_kwargs(kwargs)
        super().__init__(**kwargs)

    @classmethod
    def _add_default_init_kwargs(cls, kwargs):
        """Adds any missing default values to the ``kwargs`` dictionary, and
        partially validates any provided values that aren't easily validated with
        converters.
        """

        type_ = kwargs.get("type", None)

        if type_ is None:
            raise SMIRNOFFSpecError("the `type` keyword is missing")
        if type_ not in get_args(_VirtualSiteType):
            raise SMIRNOFFSpecError(f"'{type_}' is not a supported virtual site type")

        if "charge_increment" in kwargs:
            expected_num_charge_increments = cls._expected_num_charge_increments(type_)
            num_charge_increments = len(kwargs["charge_increment"])
            if num_charge_increments != expected_num_charge_increments:
                raise SMIRNOFFSpecError(
                    f"'{type_}' virtual sites expect exactly {expected_num_charge_increments} "
                    f"charge increments, but got {kwargs['charge_increment']} "
                    f"(length {num_charge_increments}) instead."
                )

        supports_in_plane_angle = cls._supports_in_plane_angle(type_)
        supports_out_of_plane_angle = cls._supports_out_of_plane_angle(type_)

        if not supports_out_of_plane_angle:
            kwargs["outOfPlaneAngle"] = kwargs.get("outOfPlaneAngle", None)
        if not supports_in_plane_angle:
            kwargs["inPlaneAngle"] = kwargs.get("inPlaneAngle", None)

        match = kwargs.get("match", None)

        if match is None:
            raise SMIRNOFFSpecError("the `match` keyword is missing")

        out_of_plane_angle = kwargs.get("outOfPlaneAngle", 0.0 * unit.degree)
        is_in_plane = (
            None
            if not supports_out_of_plane_angle
            else numpy.isclose(out_of_plane_angle.m_as(unit.degree), 0.0)
        )

        if not cls._supports_match(type_, match, is_in_plane):
            raise SMIRNOFFSpecError(
                f"match='{match}' not supported with type='{type_}'"
                + ("" if is_in_plane is None else f" and is_in_plane={is_in_plane}")
            )

    @classmethod
    def _supports_in_plane_angle(cls, type_: _VirtualSiteType) -> bool:
        return type_ in {"MonovalentLonePair"}

    @classmethod
    def _supports_out_of_plane_angle(cls, type_: _VirtualSiteType) -> bool:
        return type_ in {"MonovalentLonePair", "DivalentLonePair"}

    @classmethod
    def _expected_num_charge_increments(cls, type_: _VirtualSiteType) -> int:
        if type_ == "BondCharge":
            return 2
        elif (type_ == "MonovalentLonePair") or (type_ == "DivalentLonePair"):
            return 3
        elif type_ == "TrivalentLonePair":
            return 4
        raise NotImplementedError()

    @classmethod
    def _supports_match(
        cls, type_: _VirtualSiteType, match: str, is_in_plane: Optional[bool] = None
    ) -> bool:
        is_in_plane = True if is_in_plane is None else is_in_plane

        if match == "once":
            return type_ == "TrivalentLonePair" or (
                type_ == "DivalentLonePair" and is_in_plane
            )
        elif match == "all_permutations":
            return type_ in {"BondCharge", "MonovalentLonePair", "DivalentLonePair"}

        raise NotImplementedError()


class DoubleExponentialVirtualSiteHandler(VirtualSiteHandler):
    """Vsite handler compatible with the Double Exponential handler"""

    class DoubleExponentialVirtualSiteType(_BasicVirtualSiteType):
        r_min = ParameterAttribute(default=None, unit=unit.nanometers)
        epsilon = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole)

    _TAGNAME = "DoubleExponentialVirtualSites"
    _INFOTYPE = DoubleExponentialVirtualSiteType
