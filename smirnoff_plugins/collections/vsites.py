from openff.interchange.smirnoff._virtual_sites import SMIRNOFFVirtualSiteCollection
from openff.toolkit.typing.engines.smirnoff.parameters import (
    VirtualSiteHandler,
)

from openff.interchange.components.potentials import Potential
from openff.interchange.components.toolkit import _validated_list_to_array
from openff.interchange.models import PotentialKey
import abc
from smirnoff_plugins.handlers.vsites import DoubleExponentialVirtualSiteHandler


class _VsitePlugin(SMIRNOFFVirtualSiteCollection, abc.ABC):
    """
    A general vsite plugin class used to make vsite collections compatible with a non-bonded collection
    """

    is_plugin = True
    acts_as = "VirtualSites"

    @classmethod
    def supported_parameters(cls):
        """Return a list of parameter attributes supported by this handler."""
        return [
            "type",
            "name",
            "id",
            "match",
            "smirks",
            "charge_increment",
            "distance",
            "outOfPlaneAngle",
            "inPlaneAngle",
        ].extend(cls.specific_parameters())

    @classmethod
    @abc.abstractmethod
    def specific_parameters(cls) -> list[str]: ...

    @classmethod
    @abc.abstractmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        ...

    def store_potentials(  # type: ignore[override]
        self,
        parameter_handler: VirtualSiteHandler,
        vdw_handler,
        electrostatics_handler,
    ) -> None:
        """Store VirtualSite-specific parameter-like data."""
        if self.potentials:
            self.potentials = dict()
        for virtual_site_key, potential_key in self.key_map.items():
            # TODO: This logic assumes no spaces in the SMIRKS pattern, name or `match` attribute
            smirks, _, _ = potential_key.id.split(" ")
            parameter = parameter_handler.parameters[smirks]

            virtual_site_potential = Potential(
                parameters={
                    "distance": parameter.distance,
                },
            )
            for attr in ["outOfPlaneAngle", "inPlaneAngle"]:
                if hasattr(parameter, attr):
                    virtual_site_potential.parameters.update(
                        {attr: getattr(parameter, attr)},
                    )
            self.potentials[potential_key] = virtual_site_potential

            vdw_key = PotentialKey(id=potential_key.id, associated_handler="vdw")
            vdw_potential = Potential(
                parameters=dict(
                    (parameter_name, getattr(parameter, parameter_name))
                    for parameter_name in self.specific_parameters()
                )
            )
            vdw_handler.key_map[virtual_site_key] = vdw_key
            vdw_handler.potentials[vdw_key] = vdw_potential

            electrostatics_key = PotentialKey(
                id=potential_key.id,
                associated_handler="Electrostatics",
            )
            electrostatics_potential = Potential(
                parameters={
                    "charge_increments": _validated_list_to_array(
                        parameter.charge_increment,
                    ),
                },
            )
            electrostatics_handler.key_map[virtual_site_key] = electrostatics_key
            electrostatics_handler.potentials[electrostatics_key] = (
                electrostatics_potential
            )


class SMIRNOFFDoubleExponentialVirtualSiteCollection(_VsitePlugin):

    @classmethod
    def allowed_parameter_handlers(cls):
        return [DoubleExponentialVirtualSiteHandler]

    @classmethod
    def specific_parameters(cls) -> list[str]:
        return ["r_min", "epsilon"]
