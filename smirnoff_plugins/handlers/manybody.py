import abc
from typing import Dict, List, Tuple

import numpy
from openff.toolkit.topology import Topology, TopologyVirtualSite
from openff.toolkit.typing.engines.smirnoff import (
    ElectrostaticsHandler,
    LibraryChargeHandler,
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
    ToolkitAM1BCCHandler,
    vdWHandler,
)
from openff.toolkit.typing.engines.smirnoff.parameters import (
    IncompatibleParameterError,
    VirtualSiteHandler,
    _allow_only,
)
from openmm import openmm, unit


class CustomManyBodyHandler(ParameterHandler, abc.ABC):
    """The base class for custom parameter handlers which apply nonbonded parameters."""

    _OPENMMTYPE = openmm.CustomManyParticleForce
    _DEPENDENCIES = [vdWHandler, VirtualSiteHandler]

    cutoff = ParameterAttribute(default=9.0 * unit.angstroms, unit=unit.angstrom)
    method = ParameterAttribute(
        default="CutoffPeriodic", converter=_allow_only(["CutoffPeriodic", "NoCutoff"])
    )
    mode = ParameterAttribute(default="SinglePermutation", converter=_allow_only(["SinglePermutation", "UniqueCentralParticle"]))
    bondCutoff = ParameterAttribute(default=3, converter=int)
    particlesPerSet = ParameterAttribute(default=3, converter=int)

    @classmethod
    @abc.abstractmethod
    def _default_values(cls) -> Tuple[float, ...]:
        """
        Returns a tuple of default values which are first applied to every particle before applying SMARTs matched parameters.
        This is useful for vsites which might need special values.

        Returns
        -------
            A tuple of default per particle parameters for this force type.

        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def _get_potential_function(cls) -> Tuple[str, List[str], List[str]]:
        """Returns the potential energy function applied by this handler, as well
        as the symbols which appear in the function.

        Returns
        -------
            A tuple of the potential energy function and a list of the required per particle
            parameters (e.g. ['epsilon', 'sigma']) and any global parameters.
        """
        raise NotImplementedError()

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

        int_attrs_to_compare = ["bondCutoff", "particlesPerSet"]
        string_attrs_to_compare = ["method", "mode"]
        unit_attrs_to_compare = ["cutoff"]

        self._check_attributes_are_equal(
            other_handler,
            identical_attrs=string_attrs_to_compare + int_attrs_to_compare,
            tolerance_attrs=unit_attrs_to_compare,
            tolerance=self._SCALETOL,
        )

    def create_force(self, system, topology: Topology, **_):
        (
            potential_function,
            potential_parameters,
            global_parameters,
        ) = self._get_potential_function()

        force = self._OPENMMTYPE(potential_function)

        for symbol in potential_parameters:
            force.addPerParticleParameter(symbol)

        for parameter in global_parameters:
            value = getattr(self, parameter)
            force.addGlobalParameter(parameter, value)

        initial_values = self._default_values()

        # Set some starting dummy values
        for _ in topology.topology_particles:
            force.addParticle(tuple(initial_values))

        system.addForce(force)

        # Now get all matches and set the parameters
        matches = self.find_matches(topology)

        for atom_key, atom_match in matches.items():
            force.setParticleParameters(
                atom_key[0], self._process_parameters(atom_match.parameter_type)
            )

        bonds = [
            [atom.topology_particle_index for atom in bond.atoms]
            for bond in topology.topology_bonds
        ]

        force.createExclusionsFromBonds(bonds=bonds, bondCutoff=self.bondCutoff)

        self._check_all_valence_terms_assigned(
            assigned_terms=matches, valence_terms=list(topology.topology_atoms)
        )


class AxilrodTeller(CustomManyBodyHandler):
    """
    Standard Axilrod-Teller potential
    """

    cutoff = ParameterAttribute(default=9.0 * unit.angstroms, unit=unit.angstrom)
    method = ParameterAttribute(
        default="CutoffPeriodic", converter=_allow_only(["CutoffPeriodic", "NoCutoff"])
    )
    mode = ParameterAttribute(default="SinglePermutation", converter=_allow_only(["SinglePermutation", "UniqueCentralParticle"]))
    bondCutoff = ParameterAttribute(default=3, converter=int)
    particlesPerSet = ParameterAttribute(default=3, converter=int)

    class C9Type(ParameterType):

        _VALENCE_TYPE = "Atom"  # ChemicalEnvironment valence type expected for SMARTS
        _ELEMENT_NAME = "Atom"

        c9 = ParameterAttribute(
            default=0.0, unit=unit.kilojoule_per_mole * unit.nanometer**9
        )

    _TAGNAME = "AxilrodTeller"  # SMIRNOFF tag name to process
    _INFOTYPE = C9Type  # info type to store

    @classmethod
    def _default_values(cls) -> Tuple[float, ...]:
        return 0.0,

    @classmethod
    def _get_potential_function(cls) -> Tuple[str, List[str], List[str]]:

        potential_function = (
            "C*(1+3*cos(theta1)*cos(theta2)*cos(theta3))/(r12*r13*r23)^3;"
            "theta1=angle(p1,p2,p3); theta2=angle(p2,p3,p1); theta3=angle(p3,p1,p2);"
            "r12=distance(p1,p2); r13=distance(p1,p3); r23=distance(p2,p3);"
            "C=(C1*C2*C3)^(1.0/3.0)"
        )

        potential_parameters = ["c9"]

        global_parameters = []

        return potential_function, potential_parameters, global_parameters

    def _process_parameters(
        self,
        parameter_type: C9Type,
    ) -> Tuple[float, ...]:

        return (
            parameter_type.c9.value_in_unit(
                unit.kilojoule_per_mole * unit.nanometer**9
            ),
        )

'''
3,


openmm.CustomManyParticleForce.SinglePermutation
'''
