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
        """Returns the the potential energy function applied by this handler, as well
        as the symbols which appear in the function.

        Returns
        -------
            A tuple of the potential energy function and a list of the required per particle
            parameters (e.g. ['epsilon', 'sigma']) and any global parameters.
        """
        raise NotImplementedError()

    @classmethod
    def _get_scaled_potential_function(cls) -> str:
        """Returns a modified version of the potential function which handles the 1-4 scaled interactions.
        Note:
            These are added to the system as a CustomBondForce see <https://github.com/openmm/openmm/issues/1901> for more info.

        Returns
        -------
            A string of the modified potential to be used for 1-4 interactions.
        """
        potential = cls._get_potential_function()[0]
        potential = potential.split(";")
        for i, expression in enumerate(potential):
            if "=" not in expression:
                # This is the final energy so modify
                expression = "(" + expression + ")"
                expression += "*scale14"
                potential[i] = expression
                break
        return ";".join(potential)

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
    def _process_parameters(
        self,
        parameter_type: ParameterType,
    ) -> Tuple[float, ...]:
        """Process the parameters of the parameter type, by applying combination rule pre-processing."""
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

    def create_force(self, system, topology: Topology, **_):

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

        force.createExclusionsFromBonds(bonds=bonds, bondCutoff=2)

        # If a nonbonded force already exists then make sure to include its specified
        # inclusions. This is to ensure the V-site exclusions are correctly included.
        current_exclusions = set(
            tuple(sorted(force.getExclusionParticles(i)))
            for i in range(force.getNumExclusions())
        )
        # Note 14 exceptions have not been added yet and are missed
        existing_exclusions = set(
            tuple(sorted(existing_force.getExceptionParameters(i)[0:2]))
            for existing_force in existing_forces
            for i in range(existing_force.getNumExceptions())
        )

        for missing_exclusion in existing_exclusions - current_exclusions:
            force.addExclusion(*missing_exclusion)

        # Add 1-4 scaled interactions only if the scale is not 1
        if not numpy.isclose(self.scale14, 1.0):
            # build a custom bond force to hold the 1-4s
            scaled_force = openmm.CustomBondForce(self._get_scaled_potential_function())

            for i in range(1, 3):
                for symbol in potential_parameters:
                    scaled_force.addPerBondParameter(symbol + str(i))

            for parameter in global_parameters:
                value = getattr(self, parameter)
                scaled_force.addGlobalParameter(parameter, value)
            # add the scale as a global
            scaled_force.addGlobalParameter("scale14", self.scale14)

            for parameter, value in self._pre_computed_terms().items():
                scaled_force.addGlobalParameter(parameter, value)
            # add the scaled 14 interaction and add an exclusion
            for neighbors in topology.nth_degree_neighbors(n_degrees=3):
                atom_index1, atom_index2 = (
                    atom.topology_particle_index for atom in neighbors
                )
                force.addExclusion(atom_index1, atom_index2)
                scaled_force.addBond(
                    atom_index1,
                    atom_index2,
                    (
                        *self._process_parameters(
                            matches[(atom_index1,)].parameter_type
                        ),
                        *self._process_parameters(
                            matches[(atom_index2,)].parameter_type
                        ),
                    ),
                )
            system.addForce(scaled_force)

        # Apply the nonbonded settings.
        self._apply_nonbonded_settings(topology, force)

        self._check_all_valence_terms_assigned(
            assigned_terms=matches, valence_terms=list(topology.topology_atoms)
        )


class DampedBuckingham68(CustomNonbondedHandler):
    """The B68 potential."""

    gamma = ParameterAttribute(default=35.8967, unit=unit.nanometer**-1)

    class B68Type(ParameterType):

        _VALENCE_TYPE = "Atom"  # ChemicalEnvironment valence type expected for SMARTS
        _ELEMENT_NAME = "Atom"

        a = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole)
        b = ParameterAttribute(default=None, unit=unit.nanometer**-1)
        c6 = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole * unit.nanometer**6
        )
        c8 = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole * unit.nanometer**8
        )

    _TAGNAME = "DampedBuckingham68"  # SMIRNOFF tag name to process
    _INFOTYPE = B68Type  # info type to store

    def _pre_computed_terms(self) -> Dict[str, float]:
        d2 = self.gamma**2 * 0.5
        d3 = d2 * self.gamma * 0.3333333333
        d4 = d3 * self.gamma * 0.25
        d5 = d4 * self.gamma * 0.2
        d6 = d5 * self.gamma * 0.1666666667
        d7 = d6 * self.gamma * 0.1428571429
        d8 = d7 * self.gamma * 0.125
        return {"d2": d2, "d3": d3, "d4": d4, "d5": d5, "d6": d6, "d7": d7, "d8": d8}

    @classmethod
    def _default_values(cls) -> Tuple[float, ...]:
        return 0.0, 0.0, 0.0, 0.0

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

    def _process_parameters(
        self,
        parameter_type: B68Type,
    ) -> Tuple[float, ...]:

        return (
            numpy.sqrt(parameter_type.a.value_in_unit(unit.kilojoule_per_mole)),
            numpy.sqrt(parameter_type.b.value_in_unit(unit.nanometer**-1)),
            numpy.sqrt(
                parameter_type.c6.value_in_unit(
                    unit.kilojoule_per_mole * unit.nanometer**6
                )
            ),
            numpy.sqrt(
                parameter_type.c8.value_in_unit(
                    unit.kilojoule_per_mole * unit.nanometer**8
                )
            ),
        )


class DampedExp6810(CustomNonbondedHandler):
    """Damped exponential-6-8-10 potential used in <https://doi.org/10.1021/acs.jctc.0c00837>"""

    forceAtZero = ParameterAttribute(
        default=49.6144931952, unit=unit.kilojoules_per_mole * unit.nanometer**-1
    )

    class E6810Type(ParameterType):

        _VALENCE_TYPE = "Atom"  # ChemicalEnvironment valence type expected for SMARTS
        _ELEMENT_NAME = "Atom"

        sigma = ParameterAttribute(default=None, unit=unit.nanometer)
        beta = ParameterAttribute(default=None, unit=unit.nanometer**-1)
        c6 = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole * unit.nanometer**6
        )
        c8 = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole * unit.nanometer**8
        )
        c10 = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole * unit.nanometer**10
        )

    _TAGNAME = "DampedExp6810"  # SMIRNOFF tag name to process
    _INFOTYPE = E6810Type  # info type to store

    def _pre_computed_terms(self) -> Dict[str, float]:
        return {}

    @classmethod
    def _default_values(cls) -> Tuple[float, ...]:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    @classmethod
    def _get_potential_function(cls) -> Tuple[str, List[str], List[str]]:

        potential_function = (
            "repulsion - ttdamp6*c6*invR6 - ttdamp8*c8*invR8 - ttdamp10*c10*invR10;"
            "repulsion = forceAtZero*invbeta*exp(-beta*(r-sigma));"
            "ttdamp10 = 1.0 - expbr * ttdamp10Sum;"
            "ttdamp8 = 1.0 - expbr * ttdamp8Sum;"
            "ttdamp6 = 1.0 - expbr * ttdamp6Sum;"
            "ttdamp10Sum = ttdamp8Sum + br9/362880 + br10/3628800;"
            "ttdamp8Sum = ttdamp6Sum + br7/5040 + br8/40320;"
            "ttdamp6Sum = 1.0 + br + br2/2 + br3/6 + br4/24 + br5/120 + br6/720;"
            "expbr = exp(-br);"
            "br10 = br5*br5;"
            "br9 = br5*br4;"
            "br8 = br4*br4;"
            "br7 = br4*br3;"
            "br6 = br3*br3;"
            "br5 = br3*br2;"
            "br4 = br2*br2;"
            "br3 = br2*br;"
            "br2 = br*br;"
            "br = beta*r;"
            "invR10 = invR6*invR4;"
            "invR8 = invR4*invR4;"
            "invR6 = invR4*invR2;"
            "invR4 = invR2*invR2;"
            "invR2 = invR*invR;"
            "invR = 1.0/r;"
            "invbeta = 1.0/beta;"
            "c6 = sqrt(c61*c62);"
            "c8 = sqrt(c81*c82);"
            "c10 = sqrt(c101*c102);"
            "beta = 2.0*beta1*beta2/(beta1+beta2);"
            "sigma = 0.5*(sigma1+sigma2);"
        )

        potential_parameters = ["sigma", "beta", "c6", "c8", "c10"]

        global_parameters = ["forceAtZero"]

        return potential_function, potential_parameters, global_parameters

    def _process_parameters(
        self,
        parameter_type: E6810Type,
    ) -> Tuple[float, ...]:

        return (
            parameter_type.sigma.value_in_unit(unit.nanometer),
            parameter_type.beta.value_in_unit(unit.nanometer**-1),
            parameter_type.c6.value_in_unit(
                unit.kilojoule_per_mole * unit.nanometer**6
            ),
            parameter_type.c8.value_in_unit(
                unit.kilojoule_per_mole * unit.nanometer**8
            ),
            parameter_type.c10.value_in_unit(
                unit.kilojoule_per_mole * unit.nanometer**10
            ),
        )


class DoubleExponential(CustomNonbondedHandler):
    """
    The double exponential potential as proposed in <https://doi.org/10.1021/acs.jctc.0c01267>
    """

    # these parameters have no units
    alpha = ParameterAttribute(default=18.7)
    beta = ParameterAttribute(default=3.3)

    class DEType(ParameterType):

        _VALENCE_TYPE = "Atom"  # ChemicalEnvironment valence type expected for SMARTS
        _ELEMENT_NAME = "Atom"

        r_min = ParameterAttribute(default=None, unit=unit.nanometers)
        epsilon = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole)

    _TAGNAME = "DoubleExponential"  # SMIRNOFF tag name to process
    _INFOTYPE = DEType  # info type to store

    def _process_parameters(
        self,
        parameter_type: DEType,
    ) -> Tuple[float, ...]:
        # sqrt the epsilon during assignment
        # half r_min during assigment
        return (
            parameter_type.r_min.value_in_unit(unit.nanometers) / 2,
            numpy.sqrt(parameter_type.epsilon.value_in_unit(unit.kilojoule_per_mole)),
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
    def _default_values(cls) -> Tuple[float, ...]:
        return 1.0, 0.0

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


class MultipoleHandler(ParameterHandler):
    """Handler for OpenMM's AmoebaMultipoleForce"""

    _TAGNAME = "Multipole"
    _OPENMMTYPE = openmm.AmoebaMultipoleForce
    _DEPENDENCIES = [
        VirtualSiteHandler,
        vdWHandler,
        ElectrostaticsHandler,
        ToolkitAM1BCCHandler,
        LibraryChargeHandler,
    ]

    cutoff = ParameterAttribute(default=9.0 * unit.angstrom, unit=unit.angstrom)
    method = ParameterAttribute(
        default="PME", converter=_allow_only(["NoCutoff", "PME"])
    )
    polarizationType = ParameterAttribute(
        default="Mutual", converter=_allow_only(["Mutual", "Direct", "Extrapolated"])
    )
    ewaldErrorTolerance = ParameterAttribute(default=0.0001, converter=float)
    thole = ParameterAttribute(default=0.39, converter=float)
    targetEpsilon = ParameterAttribute(default=0.00001, converter=float)
    maxIter = ParameterAttribute(default=60, converter=int)

    class MPolType(ParameterType):

        _VALENCE_TYPE = "Atom"  # ChemicalEnvironment valence type expected for SMARTS
        _ELEMENT_NAME = "Atom"

        polarity = ParameterAttribute(default=0.0, unit=unit.angstroms**3)

    _INFOTYPE = MPolType

    def check_handler_compatibility(self, handler_kwargs):
        pass

    def create_force(self, system, topology, **kwargs):

        existing_multipole = [
            system.getForce(i)
            for i in range(system.getNumForces())
            if isinstance(system.getForce(i), self._OPENMMTYPE)
        ]

        assert (
            len(existing_multipole) < 2
        ), "multiple multipole forces are not yet correctly handled."

        if len(existing_multipole) == 0:
            force = self._OPENMMTYPE()
            system.addForce(force)
        else:
            force = existing_multipole[0]

        existing_nonbondeds = [
            system.getForce(i)
            for i in range(system.getNumForces())
            if isinstance(system.getForce(i), openmm.NonbondedForce)
        ]

        assert (
            len(existing_nonbondeds) < 2
        ), "multiple nonbonded forces are not yet correctly handled."

        assert len(existing_nonbondeds) > 0, "can't find existing charges to copy from"

        existing_nonbonded = existing_nonbondeds[0]

        methodMap = {
            "NoCutoff": openmm.AmoebaMultipoleForce.NoCutoff,
            "PME": openmm.AmoebaMultipoleForce.PME,
        }
        force.setNonbondedMethod(methodMap[self.method])
        polarizationTypeMap = {
            "Mutual": openmm.AmoebaMultipoleForce.Mutual,
            "Direct": openmm.AmoebaMultipoleForce.Direct,
            "Extrapolated": openmm.AmoebaMultipoleForce.Extrapolated,
        }
        force.setPolarizationType(polarizationTypeMap[self.polarizationType])
        force.setCutoffDistance(self.cutoff)
        force.setEwaldErrorTolerance(self.ewaldErrorTolerance)
        force.setMutualInducedTargetEpsilon(self.targetEpsilon)
        force.setMutualInducedMaxIterations(self.maxIter)
        force.setExtrapolationCoefficients([-0.154, 0.017, 0.658, 0.474])

        polarities = []
        for topology_molecule in topology.topology_molecules:
            for topology_particle in topology_molecule.particles:
                # TODO: Add assertion that topology_particle isn't a vsite (allowed in theory,
                #  but we should keep the implementation scope small for now)
                assert type(topology_particle) is not TopologyVirtualSite
                force.addMultipole(
                    0.0,
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                )
                polarities.append(0.0)

        # Iterate over all types, allowing later matches to override earlier ones.
        atom_matches = self.find_matches(topology)

        # Set the particle polarizability terms.
        for atom_key, atom_match in atom_matches.items():
            atom_idx = atom_key[0]
            mpoltype = atom_match.parameter_type
            polarities[atom_idx] = mpoltype.polarity

        for topology_molecule in topology.topology_molecules:

            for topology_particle in topology_molecule.particles:

                topology_particle_index = topology_particle.topology_particle_index

                (
                    particle_charge,
                    sigma,
                    epsilon,
                ) = existing_nonbonded.getParticleParameters(topology_particle_index)

                existing_nonbonded.setParticleParameters(
                    topology_particle_index, 0.0, sigma, epsilon
                )

                # setMultipoleParameters(self, index, charge, molecularDipole, molecularQuadrupole, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY, thole, dampingFactor, polarity)
                force.setMultipoleParameters(
                    topology_particle_index,
                    particle_charge,
                    (0.0, 0.0, 0.0) * unit.elementary_charge * unit.angstrom,
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    * unit.elementary_charge
                    * unit.angstrom**2,
                    openmm.AmoebaMultipoleForce.NoAxisType,
                    -1,
                    -1,
                    -1,
                    self.thole,
                    polarities[topology_particle_index] ** (1 / 6),
                    polarities[topology_particle_index],
                )

        #for topology_molecule in topology.topology_molecules:

        #    for topology_particle in topology_molecule.particles:
        for ref_mol in topology.reference_molecules:
            for ref_atom in ref_mol.atoms:
                bonded2 = [
                    a.molecule_particle_index
                    for a in ref_atom.bonded_atoms
                ]

                bonded3 = []
                for first_neighbor in ref_atom.bonded_atoms:
                    for second_neighbor in first_neighbor.bonded_atoms:
                        if second_neighbor.molecule_atom_index in bonded2:
                            continue
                        bonded3.append(second_neighbor.molecule_atom_index)
                #bonded3 = [n2.molecule_particle_index for n2 in n1.bonded_atoms for n1 in ref_mol]

                #bonded3 = [
                #    a.topology_particle_index
                #    for a in ref_mol.atoms
                #    if (
                #        a.topology_particle_index != topology_particle_index
                #        and a.topology_particle_index not in bonded2
                #        and any(
                #            [
                #                topology.is_bonded(a.topology_particle_index, b)
                #                for b in bonded2
                #            ]
                #        )
                #    )
                #]
                bonded4 = []
                for first_neighbor in ref_atom.bonded_atoms:
                    for second_neighbor in first_neighbor.bonded_atoms:
                        for third_neighbor in second_neighbor.bonded_atoms:
                            if third_neighbor.molecule_atom_index in bonded2:
                                continue
                            if third_neighbor.molecule_atom_index in bonded3:
                                continue
                            bonded4.append(third_neighbor.molecule_atom_index)
                # bonded4 = [
                #     a.topology_particle_index
                #     for a in ref_mol.atoms
                #     if (
                #         a.topology_particle_index != topology_particle_index
                #         and a.topology_particle_index not in bonded2
                #         and a.topology_particle_index not in bonded3
                #         and any(
                #             [
                #                 topology.is_bonded(a.topology_particle_index, b)
                #                 for b in bonded3
                #             ]
                #         )
                #     )
                # ]
                for topology_molecule in topology._reference_molecule_to_topology_molecules[
                    ref_mol
                ]:
                    for topology_particle in topology_molecule.particles:

                        topology_particle_index = topology_particle.topology_particle_index
                        ref_mol_particle_index = (
                            topology_particle.atom.molecule_particle_index
                        )
                        force.setCovalentMap(
                            topology_particle_index,
                            openmm.AmoebaMultipoleForce.Covalent12,
                            bonded2,
                        )
                        force.setCovalentMap(
                            topology_particle_index,
                            openmm.AmoebaMultipoleForce.Covalent13,
                            bonded3,
                        )
                        force.setCovalentMap(
                            topology_particle_index,
                            openmm.AmoebaMultipoleForce.Covalent14,
                            bonded4,
                        )

                        # bonded5 = [a.topology_particle_index for a in topology_molecule.particles if (a.topology_particle_index != topology_particle_index and a.topology_particle_index not in bonded2 and a.topology_particle_index not in bonded3 and a.topology_particle_index not in bonded4 and any([topology.is_bonded(a.topology_particle_index, b) for b in bonded4]))]

                        # force.setCovalentMap(topology_particle_index, openmm.AmoebaMultipoleForce.Covalent15, bonded5)

                        force.setCovalentMap(
                            topology_particle_index,
                            openmm.AmoebaMultipoleForce.PolarizationCovalent11,
                            bonded2 + bonded3 + bonded4,
                        )
                # force.setCovalentMap(topology_particle_index, openmm.AmoebaMultipoleForce.PolarizationCovalent12, bonded5)
