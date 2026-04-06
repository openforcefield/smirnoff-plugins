from functools import lru_cache
from typing import Dict, Iterable, Literal, Set, Tuple, Type, Union, cast

from openff.interchange import Interchange
from openff.interchange.components.potentials import Potential
from openff.interchange.components.toolkit import _PERIODICITIES
from openff.interchange.interop.openmm._valence import _is_constrained
from openff.interchange.models import PotentialKey, TopologyKey, VirtualSiteKey
from openff.interchange.smirnoff._base import SMIRNOFFCollection
from openff.toolkit import Quantity, Topology
from openff.toolkit import unit as off_unit
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterHandler
from openmm import openmm
from pydantic import Field

from smirnoff_plugins.handlers.valence import (
    ProperTorsionBendHandler,
    UreyBradleyHandler,
)


@lru_cache
def _cache_urey_bradley_parameter_lookup(
    potential_key: PotentialKey,
    parameter_handler: ParameterHandler,
) -> dict[str, Quantity]:
    parameter = parameter_handler.parameters[potential_key.id]

    return {
        parameter_name: getattr(parameter, parameter_name)
        for parameter_name in ["k", "length"]
    }


@lru_cache
def _cache_proper_torsion_bend_parameter_lookup(
    potential_key: PotentialKey,
    parameter_handler: ParameterHandler,
) -> dict[str, Quantity]:
    smirks = potential_key.id
    n = potential_key.mult
    parameter = parameter_handler.parameters[smirks]

    return {
        "angle0": parameter.angle0,
        "k": parameter.k[n],
        "periodicity": _PERIODICITIES[parameter.periodicity[n]],
        "phase": parameter.phase[n],
    }


class ProperTorsionBendKey(TopologyKey):
    """
    A unique identifier of the atoms associated in a proper torsion bend potential.

    Examples
    --------
    Index into a dictionary with a tuple

    .. code-block:: pycon

        >>> d = {
        ...     ProperTorsionKey(atom_indices=(0, 1, 2, 3)): "torsion1",
        ...     ProperTorsionKey(atom_indices=(0, 1, 2, 3), mult=2): "torsion2",
        ... }
        >>> d[0, 1, 2, 3]
        'torsion1'
        >>> d[(0, 1, 2, 3), 2, None, None]
        'torsion2'
    """

    atom_indices: tuple[int, int, int, int] = Field(
        description="The indices of the atoms occupied by this interaction",
    )

    mult: int | None = Field(
        None,
        description="The index of this duplicate interaction",
    )

    phase: float | None = Field(
        None,
        description="If this key represents as topology component subject to interpolation between "
        "multiple parameters(s), the phase determining the coefficients of the wrapped "
        "potentials.",
    )

    def _tuple(
        self,
    ) -> (
        tuple[int, int, int, int]
        | tuple[
            tuple[int, int, int, int],
            int | None,
            float | None,
        ]
    ):
        if self.mult is None and self.phase is None:
            return cast(tuple[int, int, int, int], self.atom_indices)
        else:
            return (
                cast(
                    tuple[int, int, int, int],
                    self.atom_indices,
                ),
                self.mult,
                self.phase,
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} with atom indices {self.atom_indices}"
            f"{'' if self.mult is None else ', mult ' + str(self.mult)}"
        )


class SMIRNOFFUreyBradleyCollection(SMIRNOFFCollection):
    is_plugin: bool = True

    type: Literal["UreyBradleys"] = "UreyBradleys"

    expression: Literal["k/2*(r-length)**2"] = "k/2*(r-length)**2"

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        """Return an iterable of allowed types of ParameterHandler classes."""
        return (UreyBradleyHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return an iterable of supported parameter attributes."""
        return "smirks", "id", "k", "length"

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return "k", "length"

    @classmethod
    def valence_terms(cls, topology):
        """Return all angles in this topology."""
        return [(angle[0], angle[2]) for angle in topology.angles]

    def store_potentials(self, parameter_handler: UreyBradleyHandler) -> None:
        """Store the potentials from the parameter handler."""
        for potential_key in self.key_map.values():
            self.potentials.update(
                {
                    potential_key: Potential(
                        parameters=_cache_urey_bradley_parameter_lookup(
                            potential_key,
                            parameter_handler,
                        ),
                    ),
                },
            )

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ) -> None:
        # Mainly taken from
        # https://github.com/openforcefield/openff-interchange/blob/83383b8b3af557c167e4a3003495e0e5ffbeff73/openff/interchange/interop/openmm/_valence.py#L50

        harmonic_bond_force = openmm.HarmonicBondForce()
        harmonic_bond_force.setName("UreyBradleyForce")
        system.addForce(harmonic_bond_force)

        has_constraint_handler = "Constraints" in interchange.collections

        for top_key, pot_key in self.key_map.items():
            openff_indices = top_key.atom_indices
            openmm_indices = tuple(particle_map[index] for index in openff_indices)

            if len(openmm_indices) != 2:
                raise ValueError(
                    f"Expected 2 indices for Urey-Bradley potential, got {len(openmm_indices)}: {openmm_indices}",
                )

            if has_constraint_handler and not add_constrained_forces:
                if _is_constrained(
                    constrained_pairs,
                    (openmm_indices[0], openmm_indices[1]),
                ):
                    # This 1-3 length is constrained, so not add a bond force
                    continue

            params = self.potentials[pot_key].parameters
            k = params["k"].m_as(
                off_unit.kilojoule / off_unit.nanometer**2 / off_unit.mol,
            )
            length = params["length"].m_as(off_unit.nanometer)

            harmonic_bond_force.addBond(
                particle1=openmm_indices[0],
                particle2=openmm_indices[1],
                length=length,
                k=k,
            )


class SMIRNOFFProperTorsionBendCollection(SMIRNOFFCollection):
    """Handler storing proper torsion-bend potentials."""

    is_plugin: bool = True

    type: Literal["ProperTorsionBends"] = "ProperTorsionBends"
    expression: Literal[
        "k*(angle(p1,p2,p3)-angle0)*(1+cos(periodicity*dihedral(p1,p2,p3,p4)-phase))"
    ] = "k*(angle(p1,p2,p3)-angle0)*(1+cos(periodicity*dihedral(p1,p2,p3,p4)-phase))"

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [ProperTorsionBendHandler]

    @classmethod
    def supported_parameters(cls):
        """Return a list of supported parameter attribute names."""
        return ["smirks", "id", "angle0", "k", "periodicity", "phase"]

    @classmethod
    def potential_parameters(cls):
        """Return a list of names of parameters included in each potential in this collection."""
        return ["k", "angle0", "periodicity", "phase"]

    def store_matches(
        self,
        parameter_handler: ProperTorsionBendHandler,
        topology: Topology,
    ) -> None:
        """
        Populate self.key_map with key-val pairs of keys and unique potential identifiers.

        """
        if self.key_map:
            self.key_map: dict[ProperTorsionBendKey, PotentialKey] = dict()
        matches = parameter_handler.find_matches(topology)
        for key, val in matches.items():
            parameter: ProperTorsionBendHandler.ProperTorsionBendType = (
                val.parameter_type
            )

            n_terms = len(parameter.phase)

            cosmetic_attributes = {
                cosmetic_attribute: getattr(
                    parameter,
                    f"_{cosmetic_attribute}",
                )
                for cosmetic_attribute in parameter._cosmetic_attribs
            }

            for n in range(n_terms):
                smirks = parameter.smirks

                topology_key = ProperTorsionBendKey(
                    atom_indices=key,
                    mult=n,
                )

                potential_key = PotentialKey(
                    id=smirks,
                    mult=n,
                    associated_handler="ProperTorsionBends",
                    cosmetic_attributes=cosmetic_attributes,
                )

                self.key_map[topology_key] = potential_key

    def store_potentials(self, parameter_handler: ProperTorsionBendHandler) -> None:
        """Store the potentials from the parameter handler."""
        for potential_key in self.key_map.values():
            self.potentials.update(
                {
                    potential_key: Potential(
                        parameters=_cache_proper_torsion_bend_parameter_lookup(
                            potential_key,
                            parameter_handler,
                        ),
                    ),
                },
            )

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[int, int],
    ) -> None:

        proper_torsion_bend_force = openmm.CustomCompoundBondForce(4, self.expression)
        for param_name in self.potential_parameters():
            proper_torsion_bend_force.addPerBondParameter(param_name)

        proper_torsion_bend_force.setName("ProperTorsionBendForce")
        system.addForce(proper_torsion_bend_force)

        proper_torsion_bend_handler = interchange["ProperTorsionBends"]

        for top_key, pot_key in proper_torsion_bend_handler.key_map.items():
            openff_indices = top_key.atom_indices
            openmm_indices = tuple(particle_map[index] for index in openff_indices)

            params = proper_torsion_bend_handler.potentials[pot_key].parameters

            k = params["k"].m_as(off_unit.kilojoule / off_unit.mol)
            angle0 = params["angle0"].m_as(off_unit.radian)
            periodicity = int(params["periodicity"])
            phase = params["phase"].m_as(off_unit.radian)
            proper_torsion_bend_force.addBond(
                (
                    openmm_indices[0],
                    openmm_indices[1],
                    openmm_indices[2],
                    openmm_indices[3],
                ),
                (
                    k,
                    angle0,
                    periodicity,
                    phase,
                ),
            )
