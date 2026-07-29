import math
from typing import Dict, Iterable, Literal, Set, Tuple, Type, TypeVar, Union

from openff.interchange import Interchange
from openmm import CustomCompoundBondForce
from openff.interchange.components.potentials import Potential
from openff.interchange.exceptions import InvalidParameterHandlerError
from openff.interchange.models import VirtualSiteKey
from openff.interchange.smirnoff._base import SMIRNOFFCollection
from openff.interchange.smirnoff._nonbonded import (
    SMIRNOFFvdWCollection,
    _SMIRNOFFNonbondedCollection,
)
from openff.models.types import FloatQuantity
from openff.toolkit import Quantity, Topology, unit
from openff.toolkit.topology import Atom
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterHandler
from openmm import CustomManyParticleForce, openmm

from smirnoff_plugins.handlers.bonded import (
    HarmonicHeightHandler
)

T = TypeVar("T", bound="_BondedPlugin")


class SMIRNOFFImproperTorsionCollection(SMIRNOFFCollection):
    """Handles improper torsions in the SMIRNOFF force field."""

    is_plugin: bool = True
    acts_as: str = "improper-torsion"

    def store_potentials(self, parameter_handler):
        """Store improper torsion parameters from the parameter handler."""
        for potential_key in self.key_map.values():
            smirks = potential_key.id
            parameter = parameter_handler.parameters[smirks]

            self.potentials[potential_key] = Potential(
                parameters={
                    "k": parameter.k,
                    "periodicity": parameter.periodicity,
                    "phase": parameter.phase,
                },
            )

    @classmethod
    def potential_parameters(cls):
        return ("k", "periodicity", "phase")

    @classmethod
    def supported_parameters(cls):
        return "smirks", "id", "k", "periodicity", "phase"

    @classmethod
    def allowed_parameter_handlers(cls):
        return (HarmonicHeightHandler,)

    def modify_openmm_forces(self, interchange, system, *args):
        """Applies the improper torsion potential to an OpenMM system."""
        force = CustomTorsionForce("0.5 * k * (1 + cos(periodicity * theta - phase))")
        force.addPerTorsionParameter("k")
        force.addPerTorsionParameter("periodicity")
        force.addPerTorsionParameter("phase")

        for key, val in self.key_map.items():
            atom_indices = key.atom_indices

            force.addTorsion(
                atom_indices[0], atom_indices[1], atom_indices[2], atom_indices[3],
                [
                    self.potentials[val].parameters["k"].m_as("kilojoule_per_mole"),
                    self.potentials[val].parameters["periodicity"],
                    self.potentials[val].parameters["phase"].m_as("degree"),
                ],
            )

        system.addForce(force)


class HarmonicHeightCollection(SMIRNOFFCollection):

    expression: str = (
        "0.5 * k * (h - h0)^2;"
        "h = abs((Nx*(x2-x1) + Ny*(y2-y1) + Nz*(z2-z1)) / sqrt(Nx^2 + Ny^2 + Nz^2));"
        "Nx = (y3-y1)*(z4-z1) - (z3-z1)*(y4-y1);"
        "Ny = (z3-z1)*(x4-x1) - (x3-x1)*(z4-z1);"
        "Nz = (x3-x1)*(y4-y1) - (y3-y1)*(x4-x1);"
    )


    type: Literal["HarmonicHeight"] = "HarmonicHeight"

    is_plugin: bool = True
    acts_as: str = ""
    periodic_method: str = "cutoff-periodic"
    nonperiodic_method: str = "cutoff-nonperiodic"
    cutoff: unit.Quantity = unit.Quantity(0.9, unit.nanometer)

    def store_potentials(self, parameter_handler):
        """Store height restraint parameters from the parameter handler."""
        self.nonperiodic_method = parameter_handler.nonperiodic_method
        self.periodic_method = parameter_handler.periodic_method
        self.cutoff = parameter_handler.cutoff

        for potential_key in self.key_map.values():
            smirks = potential_key.id
            parameter = parameter_handler.parameters[smirks]

            self.potentials[potential_key] = Potential(
                parameters={"k": parameter.k, "h0": parameter.h0},
            )

    @classmethod
    def potential_parameters(cls):
        return ("k", "h0")

    @classmethod
    def supported_parameters(cls):
        return "smirks", "id", "k", "h0"

    @classmethod
    def allowed_parameter_handlers(cls):
        return (HarmonicHeightHandler,)

    def modify_openmm_forces(
        self,
        interchange,
        system,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ):
        """Applies the harmonic height potential to an OpenMM system."""
        force = CustomCompoundBondForce(4, self.expression)
        force.addPerBondParameter("k")
        force.addPerBondParameter("h0")

        topology = interchange.topology

        for key, val in self.key_map.items():
            atom_indices = key.atom_indices

            force.addBond(
                atom_indices,
                [
                    self.potentials[val].parameters["k"].m_as("kilojoule_per_mole / nanometer**2"),
                    self.potentials[val].parameters["h0"].m_as("nanometer"),
                ],
            )

        system.addForce(force)

    def modify_parameters(
        self,
        original_parameters: Dict[str, unit.Quantity],
    ) -> Dict[str, float]:
        """Converts the parameters to OpenMM-compatible units."""
        _units = {"k": unit.kilojoule_per_mole / unit.nanometer**2, "h0": unit.nanometer}

        return {
            "k": original_parameters["k"].m_as(_units["k"]),
            "h0": original_parameters["h0"].m_as(_units["h0"]),
        }
