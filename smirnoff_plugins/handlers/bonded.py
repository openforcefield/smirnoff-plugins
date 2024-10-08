import abc

from openff.toolkit import unit
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ElectrostaticsHandler,
    IncompatibleParameterError,
    LibraryChargeHandler,
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
    ToolkitAM1BCCHandler,
    VirtualSiteHandler,
    _allow_only,
    vdWHandler,

)


class ImproperTorsionHandler(ParameterHandler):
    """Handle SMIRNOFF ``<ImproperTorsionForce>`` tags

    .. warning :: This API is experimental and subject to change.
    """

    class ImproperTorsionType(ParameterType):
        """A SMIRNOFF torsion type for improper torsions.

        .. warning :: This API is experimental and subject to change.
        """

        _ELEMENT_NAME = "Improper"

        length_eq = ParameterAttribute(default = None, unit=unit.angstrom)
        k = ParameterAttribute(
                default = None, unit=unit.kilocalorie/ unit.mole / unit.angstrom**2
            )

    _TAGNAME = "ImproperTorsionsHarmonic"  # SMIRNOFF tag name to process
    _INFOTYPE = ImproperTorsionType  # info type to store

    potential = ParameterAttribute(
        default="(k/2)*(length-length_eq)**2",
        converter=_allow_only(["(k/2)*(length-length_eq)**2"]),
    )
    default_idivf = ParameterAttribute(default="auto")

    def check_handler_compatibility(self, other_handler: "ImproperTorsionHandler"):
        """
        Checks whether this ParameterHandler encodes compatible physics as another ParameterHandler. This is
        called if a second handler is attempted to be initialized for the same tag.

        Parameters
        ----------
        other_handler
            The handler to compare to.

        Raises
        ------
        IncompatibleParameterError if handler_kwargs are incompatible with existing parameters.
        """
        float_attrs_to_compare = list()
        string_attrs_to_compare = ["potential"]

        if self.default_idivf == "auto":
            string_attrs_to_compare.append("default_idivf")
        else:
            float_attrs_to_compare.append("default_idivf")

        self._check_attributes_are_equal(
            other_handler,
            identical_attrs=tuple(string_attrs_to_compare),
            tolerance_attrs=tuple(float_attrs_to_compare),
        )

    def find_matches(self, entity, unique=False):
        """Find the improper torsions in the topology/molecule matched by a parameter type.

        Parameters
        ----------
        entity
            Topology to search.

        Returns
        ---------
        matches
            ``matches[atom_indices]`` is the ``ParameterType`` object
            matching the 4-tuple of atom indices in ``entity``.

        """
        return self._find_matches(
            entity, transformed_dict_cls=ImproperDict, unique=unique
        )


