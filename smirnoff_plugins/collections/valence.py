import builtins
from collections.abc import Iterable
from functools import lru_cache
from typing import Literal, Union

from openff.interchange import Interchange
from openff.interchange.components.potentials import Potential
from openff.interchange.interop.openmm._valence import _is_constrained
from openff.interchange.models import PotentialKey, TopologyKey, VirtualSiteKey
from openff.interchange.smirnoff._base import SMIRNOFFCollection
from openff.toolkit import Quantity
from openff.toolkit import unit as off_unit
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterHandler
from openmm import openmm

from smirnoff_plugins.handlers.valence import (
    HarmonicAngleHandler,
    HarmonicHeightHandler,
    LeeKrimmHandler,
    UreyBradleyHandler,
)


@lru_cache
def _cache_urey_bradley_parameter_lookup(
    potential_key: PotentialKey,
    parameter_handler: ParameterHandler,
) -> dict[str, Quantity]:
    parameter = parameter_handler.parameters[potential_key.id]

    return {parameter_name: getattr(parameter, parameter_name) for parameter_name in ["k", "length"]}


class SMIRNOFFUreyBradleyCollection(SMIRNOFFCollection):
    is_plugin: bool = True

    type: Literal["UreyBradley"] = "UreyBradley"

    expression: Literal["k/2*(r-length)**2"] = "k/2*(r-length)**2"

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[builtins.type[ParameterHandler]]:
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
        constrained_pairs: set[tuple[int, ...]],
        particle_map: dict[Union[int, "VirtualSiteKey"], int],
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


class SMIRNOFFHarmonicHeightCollection(SMIRNOFFCollection):
    """Harmonic potential on the pyramid height of the central atom above its three neighbors.

    Particles: p1, p3, p4 are the three neighbors (the base plane, anchored at p1);
    p2 is the central atom (the apex atom whose height above that plane is restrained).
    This atom-index convention (central atom second) matches ``ImproperDict`` and
    ``Topology.impropers``, which is what ``HarmonicHeightHandler.find_matches`` uses to
    canonicalize matches.
    """

    is_plugin: bool = True

    type: Literal["HarmonicHeight"] = "HarmonicHeight"

    expression: str = (
        "0.5 * k * (h - h0)^2; "
        "h = ((x2-x1)*nx + (y2-y1)*ny + (z2-z1)*nz) / normal_mag; "
        "normal_mag = sqrt(nx^2 + ny^2 + nz^2); "
        "nx = (y3-y1)*(z4-z1) - (z3-z1)*(y4-y1); "
        "ny = (z3-z1)*(x4-x1) - (x3-x1)*(z4-z1); "
        "nz = (x3-x1)*(y4-y1) - (y3-y1)*(x4-x1)"
    )

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        """Return an iterable of allowed types of ParameterHandler classes."""
        return (HarmonicHeightHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return an iterable of supported parameter attributes."""
        return "smirks", "id", "k", "h0"

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return "k", "h0"

    @classmethod
    def valence_terms(cls, topology):
        """Return all impropers in this topology."""
        return topology.impropers

    def store_potentials(self, parameter_handler: HarmonicHeightHandler) -> None:
        """Store the potentials from the parameter handler."""
        for potential_key in self.key_map.values():
            param = parameter_handler.parameters[potential_key.id]
            self.potentials[potential_key] = Potential(
                parameters={
                    "k": param.k,
                    "h0": param.h0,
                }
            )

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, VirtualSiteKey], int],
    ) -> None:
        force = openmm.CustomCompoundBondForce(4, self.expression)
        force.addPerBondParameter("k")
        force.addPerBondParameter("h0")
        force.setName("HarmonicHeight")
        system.addForce(force)

        for top_key, pot_key in self.key_map.items():
            indices = [particle_map[i] for i in top_key.atom_indices]
            params = self.potentials[pot_key].parameters
            k = params["k"].m_as("kilojoule / mole / nanometer**2")
            h0 = params["h0"].m_as("nanometer")

            force.addBond(indices, [k, h0])


class SMIRNOFFLeeKrimmCollection(SMIRNOFFCollection):
    """Lee-Krimm potential: V2*((|h|^t)/(1-|h|^s))^2 + V4*((|h|^t)/(1-|h|^s))^4
    where h is the pyramid height of the central atom above its three neighbors.

    Particles: p1, p3, p4 are the three neighbors (the base plane, anchored at p1);
    p2 is the central atom (the apex atom whose height above that plane is used).
    This atom-index convention (central atom second) matches ``ImproperDict`` and
    ``Topology.impropers``, which is what ``LeeKrimmHandler.find_matches`` uses to
    canonicalize matches.
    """

    type: Literal["LeeKrimm"] = "LeeKrimm"
    is_plugin: bool = True

    expression: str = (
        "V2 * ((abs(h)^t) / (1 - abs(h)^s))^2 + V4 * ((abs(h)^t) / (1 - abs(h)^s))^4; "
        "h = ((x2-x1)*nx + (y2-y1)*ny + (z2-z1)*nz) / normal_mag; "
        "normal_mag = sqrt(nx^2 + ny^2 + nz^2); "
        "nx = (y3-y1)*(z4-z1) - (z3-z1)*(y4-y1); "
        "ny = (z3-z1)*(x4-x1) - (x3-x1)*(z4-z1); "
        "nz = (x3-x1)*(y4-y1) - (y3-y1)*(x4-x1)"
    )

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        """Return an iterable of allowed types of ParameterHandler classes."""
        return (LeeKrimmHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return an iterable of supported parameter attributes."""
        return "smirks", "id", "V2", "V4", "t", "s"

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return "V2", "V4", "t", "s"

    @classmethod
    def valence_terms(cls, topology):
        """Return all impropers in this topology."""
        return topology.impropers

    def store_potentials(self, parameter_handler: LeeKrimmHandler) -> None:
        """Store the potentials from the parameter handler."""
        for potential_key in self.key_map.values():
            param = parameter_handler.parameters[potential_key.id]
            self.potentials[potential_key] = Potential(
                parameters={
                    "V2": param.V2,
                    "V4": param.V4,
                    "t": param.t * off_unit.dimensionless,
                    "s": param.s * off_unit.dimensionless,
                }
            )

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, VirtualSiteKey], int],
    ) -> None:
        force = openmm.CustomCompoundBondForce(4, self.expression)
        force.addPerBondParameter("V2")
        force.addPerBondParameter("V4")
        force.addPerBondParameter("t")
        force.addPerBondParameter("s")
        force.setName("LeeKrimm")

        for key, val in self.key_map.items():
            atom_indices = [particle_map[i] for i in key.atom_indices]
            params = self.potentials[val].parameters
            force.addBond(
                atom_indices,
                [
                    params["V2"].m_as("kilojoule / mole"),
                    params["V4"].m_as("kilojoule / mole"),
                    params["t"],
                    params["s"],
                ],
            )

        system.addForce(force)


class SMIRNOFFHarmonicAngleCollection(SMIRNOFFCollection):
    """Harmonic bond-plane angle (Wilson-Decius) for improper torsions.

    For each improper (n1, central, n2, n3) matched by a HarmonicAngle SMIRKS, three
    bond-plane angles are generated, one per choice of "bond" neighbor, measuring the
    angle that bond makes with the plane spanned by the other two neighbor bonds.
    Particles: p1=bond_atom, p2=central_atom, p3=plane_atom2, p4=plane_atom3.
    This atom-index convention (central atom second) matches ``ImproperDict`` and
    ``Topology.impropers``, which is what ``HarmonicAngleHandler.find_matches`` uses to
    canonicalize matches.

    As with ``ImproperTorsionHandler``'s default ``idivf="auto"``, the parameter's ``k``
    is divided by 3 across the three symmetrized terms (see `store_potentials`), so the
    total restraint on a given center is on the same scale as a single bond-plane angle
    term rather than tripling.
    """

    is_plugin: bool = True
    type: Literal["HarmonicAngle"] = "HarmonicAngle"

    expression: str = (
        "0.5 * k * (theta - theta0)^2; "
        "theta = asin(max(-1, min(1, sin_theta))); "
        "sin_theta = dot_product / max(1e-10, cross_norm); "
        "dot_product = crossx*v_oopx_norm + crossy*v_oopy_norm + crossz*v_oopz_norm; "
        "cross_norm = sqrt(crossx^2 + crossy^2 + crossz^2); "
        "crossx = v12y_norm*v13z_norm - v12z_norm*v13y_norm; "
        "crossy = v12z_norm*v13x_norm - v12x_norm*v13z_norm; "
        "crossz = v12x_norm*v13y_norm - v12y_norm*v13x_norm; "
        "v12x_norm = v12x/r12; v12y_norm = v12y/r12; v12z_norm = v12z/r12; "
        "v13x_norm = v13x/r13; v13y_norm = v13y/r13; v13z_norm = v13z/r13; "
        "v_oopx_norm = v_oopx/r_oop; v_oopy_norm = v_oopy/r_oop; v_oopz_norm = v_oopz/r_oop; "
        "r12 = sqrt(v12x^2 + v12y^2 + v12z^2 + 1e-10); "
        "r13 = sqrt(v13x^2 + v13y^2 + v13z^2 + 1e-10); "
        "r_oop = sqrt(v_oopx^2 + v_oopy^2 + v_oopz^2 + 1e-10); "
        "v12x = x3-x2; v12y = y3-y2; v12z = z3-z2; "
        "v13x = x4-x2; v13y = y4-y2; v13z = z4-z2; "
        "v_oopx = x1-x2; v_oopy = y1-y2; v_oopz = z1-z2"
    )

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        """Return an iterable of allowed types of ParameterHandler classes."""
        return (HarmonicAngleHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return an iterable of supported parameter attributes."""
        return "smirks", "id", "k", "theta0"

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return "k", "theta0"

    @classmethod
    def valence_terms(cls, topology):
        """Return all bond-plane angle terms (3 per improper) in this topology."""
        bond_plane_angles = []
        for improper in topology.impropers:
            neighbor1, central, neighbor2, neighbor3 = improper
            bond_plane_angles.append((neighbor1, central, neighbor2, neighbor3))
            bond_plane_angles.append((neighbor2, central, neighbor1, neighbor3))
            bond_plane_angles.append((neighbor3, central, neighbor1, neighbor2))
        return bond_plane_angles

    def store_matches(
        self,
        parameter_handler: HarmonicAngleHandler,
        topology,
    ) -> None:
        """Populate self.key_map, expanding each matched improper into its three
        bond-plane angles (one per choice of which neighbor is the "bond" atom)."""
        if self.key_map:
            self.key_map = dict()

        matches = parameter_handler.find_matches(topology)

        for key, val in matches.items():
            parameter_handler._assert_correct_connectivity(
                val,
                [
                    (0, 1),
                    (1, 2),
                    (1, 3),
                ],
            )

            parameter: HarmonicAngleHandler.HarmonicAngleType = val.parameter_type

            cosmetic_attributes = {
                cosmetic_attribute: getattr(
                    parameter,
                    f"_{cosmetic_attribute}",
                )
                for cosmetic_attribute in parameter._cosmetic_attribs
            }

            potential_key = PotentialKey(
                id=parameter.smirks,
                associated_handler=parameter_handler.TAGNAME,
                cosmetic_attributes=cosmetic_attributes,
            )

            central = key[1]
            neighbors = [key[0], key[2], key[3]]

            for i, j, k in [(0, 1, 2), (1, 0, 2), (2, 0, 1)]:
                topology_key = TopologyKey(
                    atom_indices=(neighbors[i], central, neighbors[j], neighbors[k]),
                )
                self.key_map[topology_key] = potential_key

    def store_potentials(self, parameter_handler: HarmonicAngleHandler) -> None:
        """Store the potentials from the parameter handler.

        Each improper center is symmetrized into 3 bond-plane angle terms (see
        `store_matches`), analogous to how `ImproperTorsionHandler` symmetrizes a
        trivalent center into 3 torsion terms. As with that handler's default
        `idivf="auto"`, `k` is divided by 3 here so the total restraint on a given
        center stays on the same scale as a single bond-plane angle term, rather
        than tripling.
        """
        for potential_key in self.key_map.values():
            param = parameter_handler.parameters[potential_key.id]
            self.potentials[potential_key] = Potential(
                parameters={
                    "k": param.k / 3,
                    "theta0": param.theta0,
                }
            )

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, VirtualSiteKey], int],
    ) -> None:
        force = openmm.CustomCompoundBondForce(4, self.expression)
        force.addPerBondParameter("k")
        force.addPerBondParameter("theta0")
        force.setName("HarmonicAngle")
        system.addForce(force)

        for top_key, pot_key in self.key_map.items():
            indices = [particle_map[i] for i in top_key.atom_indices]
            params = self.potentials[pot_key].parameters
            k = params["k"].m_as("kilojoule / mole / radian**2")
            theta0 = params["theta0"].m_as("radian")
            force.addBond(indices, [k, theta0])
