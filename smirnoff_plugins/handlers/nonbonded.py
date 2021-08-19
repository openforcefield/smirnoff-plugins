import abc
from typing import Dict, List, Tuple

import numpy
from openff.toolkit.topology import Topology
from openff.toolkit.typing.engines.smirnoff import (
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
    vdWHandler,
)
from openff.toolkit.typing.engines.smirnoff.parameters import (
    IncompatibleParameterError,
    VirtualSiteHandler,
    _allow_only,
)
from simtk import openmm, unit


class CustomNonbondedHandler(ParameterHandler, abc.ABC):
    """The base class for custom parameter handlers which apply nonbonded parameters."""

    _OPENMMTYPE = openmm.CustomNonbondedForce
    _DEPENDENCIES = [vdWHandler, VirtualSiteHandler]

    scale14 = ParameterAttribute(default=0.5, converter=float)

    cutoff = ParameterAttribute(default=9.0 * unit.angstroms, unit=unit.angstrom)
    method = ParameterAttribute(
        default="cutoff", converter=_allow_only(["cutoff", "PME"])
    )
    switch_width = ParameterAttribute(default=1.0 * unit.angstroms, unit=unit.angstrom)

    @classmethod
    @abc.abstractmethod
    def _get_potential_function(cls) -> Tuple[str, List[str], List[str]]:
        """Returns the the potential energy function applied by this handler, as well
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

        float_attrs_to_compare = ["scale12", "scale13", "scale14", "scale15"]
        string_attrs_to_compare = ["method"]
        unit_attrs_to_compare = ["cutoff"]

        self._check_attributes_are_equal(
            other_handler,
            identical_attrs=string_attrs_to_compare,
            tolerance_attrs=float_attrs_to_compare + unit_attrs_to_compare,
            tolerance=self._SCALETOL,
        )

    @abc.abstractmethod
    def _apply_parameter(
        self,
        force: openmm.CustomNonbondedForce,
        atom_index: int,
        parameter_type: ParameterType,
    ):
        """Apply a parameter to the specified atom."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _pre_computed_terms(self) -> Dict[str, float]:
        """
        Precompute some constant global terms used in the energy function, but are not explicit parameters.

        Returns
        -------
            A dict of the global term and the value which should be used.
        """
        raise NotImplementedError()

    def _apply_nonbonded_settings(
        self, topology: Topology, force: openmm.CustomNonbondedForce
    ):
        """Apply this handlers nonbonded settings (e.g. the cutoff) to a force object.

        Notes
        -----
        * This logic mirrors the logic applied by the OpenFF `vdWHandler`, taken from
          commit eedd8ac
        """

        # If we're using PME, then the only possible openMM Nonbonded type is LJPME
        if self.method == "PME":

            # If we're given a non-periodic box, we always set NoCutoff. Later we'll
            # add support for CutoffNonPeriodic
            if topology.box_vectors is None:
                force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)

            else:
                raise NotImplementedError()

        # If method is cutoff, then we currently support openMM's PME for periodic
        # system and NoCutoff for non-periodic
        elif self.method == "cutoff":

            # If we're given a non-periodic box, we always set NoCutoff. Later we'll
            # add support for CutoffNonPeriodic
            if topology.box_vectors is None:
                force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
            else:
                force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
                force.setUseLongRangeCorrection(True)
                force.setCutoffDistance(self.cutoff)
                if self.switch_width.value_in_unit(unit.angstroms) > 0:
                    force.setUseSwitchingFunction(True)
                    # the separation at which the switch function starts
                    force.setSwitchingDistance(self.cutoff - self.switch_width)

    def create_force(self, system, topology, **_):

        # Check to see if the system already contains a normal non-bonded force with
        # particles which have a non-zero epsilon.
        existing_forces = [
            system.getForce(i)
            for i in range(system.getNumForces())
            if isinstance(system.getForce(i), openmm.NonbondedForce)
        ]

        assert (
            len(existing_forces) < 2
        ), "multiple nonbonded forces are not yet correctly handled."

        existing_parameters = [
            existing_force.getParticleParameters(i)[2].value_in_unit(
                unit.kilojoule_per_mole
            )
            for existing_force in existing_forces
            for i in range(existing_force.getNumParticles())
        ]

        assert numpy.allclose(
            existing_parameters, 0.0
        ), "the system already contains interacting particles."

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

        for parameter, value in self._pre_computed_terms().items():
            force.addGlobalParameter(parameter, value)

        initial_values = tuple(0.0 for _ in range(len(potential_parameters)))

        # Set some starting dummy values
        for _ in topology.topology_particles:
            force.addParticle(tuple(initial_values))

        system.addForce(force)

        # Now get all matches and set the parameters
        matches = self.find_matches(topology)

        for atom_key, atom_match in matches.items():
            self._apply_parameter(force, atom_key[0], atom_match.parameter_type)

        bonds = [
            [atom.topology_particle_index for atom in bond.atoms]
            for bond in topology.topology_bonds
        ]

        force.createExclusionsFromBonds(bonds=bonds, bondCutoff=2)

        # If a nonbonded force already exists then make sure to include its specified
        # inclusions. This is to ensure the V-site exclusions are correctly included.
        current_exclusions = set(
            tuple(sorted(force.getExclusionParticles(i)))
            for i in range(force.getNumExclusions())
        )
        existing_exclusions = set(
            tuple(sorted(existing_force.getExceptionParameters(i)[0:2]))
            for existing_force in existing_forces
            for i in range(existing_force.getNumExceptions())
        )

        for missing_exclusion in existing_exclusions - current_exclusions:
            force.addExclusion(*missing_exclusion)

        # Apply the nonbonded settings.
        self._apply_nonbonded_settings(topology, force)

        self._check_all_valence_terms_assigned(
            assigned_terms=matches, valence_terms=list(topology.topology_atoms)
        )


class DampedBuckingham68(CustomNonbondedHandler):
    """The B68 potential."""

    gamma = ParameterAttribute(default=35.8967, unit=unit.nanometer ** -1)

    class B68Type(ParameterType):

        _VALENCE_TYPE = "Atom"  # ChemicalEnvironment valence type expected for SMARTS
        _ELEMENT_NAME = "Atom"

        a = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole)
        b = ParameterAttribute(default=None, unit=unit.nanometer ** -1)
        c6 = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole * unit.nanometer ** 6
        )
        c8 = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole * unit.nanometer ** 8
        )

    _TAGNAME = "DampedBuckingham68"  # SMIRNOFF tag name to process
    _INFOTYPE = B68Type  # info type to store

    def _pre_computed_terms(self) -> Dict[str, float]:
        d2 = self.gamma ** 2 * 0.5
        d3 = d2 * self.gamma * 0.3333333333
        d4 = d3 * self.gamma * 0.25
        d5 = d4 * self.gamma * 0.2
        d6 = d5 * self.gamma * 0.1666666667
        d7 = d6 * self.gamma * 0.1428571429
        d8 = d7 * self.gamma * 0.125
        return {"d2": d2, "d3": d3, "d4": d4, "d5": d5, "d6": d6, "d7": d7, "d8": d8}

    @classmethod
    def _get_potential_function(cls) -> Tuple[str, List[str], List[str]]:

        potential_function = (
            "buckinghamRepulsion-c6E*c6-c8E*c8;"
            "c6=c61*c62;"
            "c8=c81*c82;"
            "c6E=invR6-expTerm*(invR6+gamma*invR5+d2*invR4+d3*invR3+d4*invR2+d5*invR+d6);"
            "c8E=invR8-expTerm*(invR8+gamma*invR7+d2*invR6+d3*invR5+d4*invR4+d5*invR3+d6*invR2+d7*invR+d8);"
            "buckinghamRepulsion=combinedA*exp(buckinghamExp);"
            "buckinghamExp=-combinedB*r;"
            "combinedA=a1*a2;"
            "combinedB=b1*b2;"
            "invR8=invR7*invR;"
            "invR7=invR6*invR;"
            "invR6=invR5*invR;"
            "invR5=invR4*invR;"
            "invR4=invR3*invR;"
            "invR3=invR2*invR;"
            "invR2=invR*invR;"
            "invR=1.0/r;"
            "expTerm=exp(mdr);"
            "mdr=-gamma*r;"
        )

        potential_parameters = ["a", "b", "c6", "c8"]

        global_parameters = ["gamma"]

        return potential_function, potential_parameters, global_parameters

    def _apply_parameter(
        self,
        force: openmm.CustomNonbondedForce,
        atom_index: int,
        parameter_type: B68Type,
    ):

        force.setParticleParameters(
            atom_index,
            (
                numpy.sqrt(parameter_type.a.value_in_unit(unit.kilojoule_per_mole)),
                numpy.sqrt(parameter_type.b.value_in_unit(unit.nanometer ** -1)),
                numpy.sqrt(
                    parameter_type.c6.value_in_unit(
                        unit.kilojoule_per_mole * unit.nanometer ** 6
                    )
                ),
                numpy.sqrt(
                    parameter_type.c8.value_in_unit(
                        unit.kilojoule_per_mole * unit.nanometer ** 8
                    )
                ),
            ),
        )


class DoubleExponential(CustomNonbondedHandler):
    """
    The double exponential potential as proposed in <https://doi.org/10.1021/acs.jctc.0c01267>
    """

    # these parameters have no units
    alpha = ParameterAttribute(18.7)
    beta = ParameterAttribute(3.3)

    class DEType(ParameterType):

        _VALENCE_TYPE = "Atom"  # ChemicalEnvironment valence type expected for SMARTS
        _ELEMENT_NAME = "Atom"

        r_min = ParameterAttribute(default=None, unit=unit.nanometers)
        epsilon = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole)

    _TAGNAME = "DoubleExponential"  # SMIRNOFF tag name to process
    _INFOTYPE = DEType  # info type to store

    def _apply_parameter(
        self,
        force: openmm.CustomNonbondedForce,
        atom_index: int,
        parameter_type: DEType,
    ):
        # sqrt the epsilon during assignment
        # half r_min during assigment
        force.setParticleParameters(
            atom_index,
            (
                parameter_type.r_min.value_in_unit(unit.nanometers) / 2,
                numpy.sqrt(
                    parameter_type.epsilon.value_in_unit(unit.kilojoule_per_mole)
                ),
            ),
        )

    def _pre_computed_terms(self) -> Dict[str, float]:

        # compute alpha - beta
        alpha_min_beta = self.alpha - self.beta
        # repulsion factor
        repulsion_factor = self.beta * numpy.exp(self.alpha) / alpha_min_beta
        # attraction factor
        attraction_factor = self.alpha * numpy.exp(self.beta) / alpha_min_beta
        return {
            "AlphaMinBeta": alpha_min_beta,
            "RepulsionFactor": repulsion_factor,
            "AttractionFactor": attraction_factor,
        }

    @classmethod
    def _get_potential_function(cls) -> Tuple[str, List[str], List[str]]:

        # do the epsilon square root outside of the evaluation at parameter assignment time
        potential_function = (
            "CombinedEpsilon*RepulsionFactor*RepulsionExp-CombinedEpsilon*AttractionFactor*AttractionExp;"
            "CombinedEpsilon=epsilon1*epsilon2;"
            "RepulsionExp=exp(-alpha*ExpDistance);"
            "AttractionExp=exp(-beta*ExpDistance);"
            "ExpDistance=r/CombinedR;"
            "CombinedR=r_min1+r_min2;"
        )

        potential_parameters = ["r_min", "epsilon"]

        global_parameters = ["alpha", "beta"]

        return potential_function, potential_parameters, global_parameters
