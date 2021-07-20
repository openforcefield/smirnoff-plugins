import abc
from typing import List, Tuple

import numpy
from openff.toolkit.topology import Topology
from openff.toolkit.typing.engines.smirnoff import (
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
)
from openff.toolkit.typing.engines.smirnoff.parameters import (
    IncompatibleParameterError,
    VirtualSiteHandler,
    vdWHandler,
    ElectrostaticsHandler,
    ToolkitAM1BCCHandler,
    ChargeIncrementModelHandler,
    LibraryChargeHandler,
    _allow_only,
)
from simtk import openmm, unit


class CustomOBCHandler(ParameterHandler):
    """Handle SMIRNOFF ``<CustomOBC>`` tags

    .. warning :: This API is experimental and subject to change.
    """

    class CustomOBCType(ParameterType):
        """A SMIRNOFF GBSA type.

        .. warning :: This API is experimental and subject to change.
        """

        _VALENCE_TYPE = "Atom"
        _ELEMENT_NAME = "Atom"

        radius = ParameterAttribute(unit=unit.angstrom)
        scale = ParameterAttribute(converter=float)

    _TAGNAME = "CustomOBC"
    _INFOTYPE = CustomOBCType
    #_OPENMMTYPE = openmm.app.internal.customgbforces.CustomAmberGBForceBase
    _OPENMMTYPE = openmm.CustomGBForce
    # It's important that this runs AFTER partial charges are assigned to all particles, since this will need to
    # collect and assign them to the GBSA particles
    _DEPENDENCIES = [
        vdWHandler,
        ElectrostaticsHandler,
        ToolkitAM1BCCHandler,
        ChargeIncrementModelHandler,
        LibraryChargeHandler,
    ]

    # This is where we define all the keywords that may be defined in the section header in the OFFXML
    #gb_model = ParameterAttribute(
    #    default="OBC1", converter=_allow_only(["HCT", "OBC1", "OBC2"])
    #)
    alpha = ParameterAttribute(converter=float)
    beta = ParameterAttribute(converter=float)
    gamma = ParameterAttribute(converter=float)

    solvent_dielectric = ParameterAttribute(default=78.5, converter=float)
    solute_dielectric = ParameterAttribute(default=1, converter=float)
    sa_model = ParameterAttribute(default="ACE", converter=_allow_only(["ACE", None]))

    # TODO: We can't find where this value is actually fed into the CustomGBForce. We should validate that
    #       this is actually being used correctly
    surface_area_penalty = ParameterAttribute(
        default=5.4 * unit.calorie / unit.mole / unit.angstrom ** 2,
        unit=unit.calorie / unit.mole / unit.angstrom ** 2,
    )
    solvent_radius = ParameterAttribute(default=1.4 * unit.angstrom, unit=unit.angstrom)
    # TODO: Check units for Kappa
    kappa = ParameterAttribute(default=0.0 / unit.angstrom, unit=unit.angstrom**-1)

    def _validate_parameters(self):
        """
        Checks internal attributes, raising an exception if they are configured in an invalid way.
        """
        # If we're using HCT via GBSAHCTForce(CustomAmberGBForceBase):, then we need to ensure that:
        #   surface_area_energy is 5.4 cal/mol/A^2
        #   solvent_radius is 1.4 A
        # Justification at https://github.com/openforcefield/openff-toolkit/pull/363
        #if self.gb_model == "HCT":
        #    if (
        #        self.surface_area_penalty
        #        != 5.4 * unit.calorie / unit.mole / unit.angstrom ** 2
        #    ) and (self.sa_model is not None):
        #        raise IncompatibleParameterError(
        #            f"The current implementation of HCT GBSA does not "
        #            f"support surface_area_penalty values other than 5.4 "
        #            f"cal/mol A^2 (data source specified value of "
        #            f"{self.surface_area_penalty})"
        #        )
        #
        #    if (self.solvent_radius != 1.4 * unit.angstrom) and (
        #        self.sa_model is not None
        #    ):
        #        raise IncompatibleParameterError(
        #            f"The current implementation of HCT GBSA does not "
        #            f"support solvent_radius values other than 1.4 "
        #            f"A (data source specified value of "
        #            f"{self.solvent_radius})"
        #        )

        # If we're using OBC1 via GBSAOBC1Force(CustomAmberGBForceBase), then we need to ensure that:
        #   surface_area_energy is 5.4 cal/mol/A^2
        #   solvent_radius is 1.4 A
        # Justification at https://github.com/openforcefield/openff-toolkit/pull/363
        #if self.gb_model == "OBC1":
        #    if (
        #        self.surface_area_penalty
        #        != 5.4 * unit.calorie / unit.mole / unit.angstrom ** 2
        #    ) and (self.sa_model is not None):
        #        raise IncompatibleParameterError(
        #            f"The current implementation of OBC1 GBSA does not "
        #            f"support surface_area_penalty values other than 5.4 "
        #            f"cal/mol A^2 (data source specified value of "
        #            f"{self.surface_area_penalty})"
        #        )
        #
        #    if (self.solvent_radius != 1.4 * unit.angstrom) and (
        #        self.sa_model is not None
        #    ):
        #        raise IncompatibleParameterError(
        #            f"The current implementation of OBC1 GBSA does not "
        #            f"support solvent_radius values other than 1.4 "
        #            f"A (data source specified value of "
        #            f"{self.solvent_radius})"
        #        )

        # If we're using OBC2 via GBSAOBCForce, then we need to ensure that
        #   solvent_radius is 1.4 A
        # Justification at https://github.com/openforcefield/openff-toolkit/pull/363
        #if self.gb_model == "OBC2":
        #
        #    if (self.solvent_radius != 1.4 * unit.angstrom) and (
        #        self.sa_model is not None
        #    ):
        #        raise IncompatibleParameterError(
        #            f"The current implementation of OBC1 GBSA does not "
        #            f"support solvent_radius values other than 1.4 "
        #            f"A (data source specified value of "
        #            f"{self.solvent_radius})"
        #        )

        if (self.solvent_radius != 1.4 * unit.angstrom) and (
                self.sa_model is not None
        ):
            raise IncompatibleParameterError(
                f"The current implementation of OBC1 GBSA does not "
                f"support solvent_radius values other than 1.4 "
                f"A (data source specified value of "
                f"{self.solvent_radius})"
            )

    # Tolerance when comparing float attributes for handler compatibility.
    _SCALETOL = 1e-5

    def check_handler_compatibility(self, other_handler):
        """
        Checks whether this ParameterHandler encodes compatible physics as another ParameterHandler. This is
        called if a second handler is attempted to be initialized for the same tag.

        Parameters
        ----------
        other_handler : a ParameterHandler object
            The handler to compare to.

        Raises
        ------
        IncompatibleParameterError if handler_kwargs are incompatible with existing parameters.
        """
        float_attrs_to_compare = ["alpha", "beta", "gamma", "solvent_dielectric", "solute_dielectric"]
        string_attrs_to_compare = ["sa_model"]
        unit_attrs_to_compare = ["surface_area_penalty", "solvent_radius", "kappa"]

        self._check_attributes_are_equal(
            other_handler,
            identical_attrs=string_attrs_to_compare,
            tolerance_attrs=float_attrs_to_compare + unit_attrs_to_compare,
            tolerance=self._SCALETOL,
        )

    def _createEnergyTerms(self, force, solventDielectric, soluteDielectric, SA, cutoff, kappa, offset):
        """Add the energy terms to the CustomGBForce.
        These are identical for all the GB models.
        """
        #solventDielectric = solventDielectric.value_in_units(unit.)
        kappa = kappa.value_in_unit(unit.nanometer**-1)

        params = "; solventDielectric=%.16g; soluteDielectric=%.16g; kappa=%.16g; offset=%.16g" % (solventDielectric, soluteDielectric, kappa, offset)
        if cutoff is not None:
            params += "; cutoff=%.16g" % cutoff
        if kappa > 0:
            force.addEnergyTerm("-0.5*138.935485*(1/soluteDielectric-exp(-kappa*B)/solventDielectric)*charge^2/B"+params,
                    openmm.CustomGBForce.SingleParticle)
        elif kappa < 0:
            # Do kappa check here to avoid repeating code everywhere
            raise ValueError('kappa/ionic strength must be >= 0')
        else:
            force.addEnergyTerm("-0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*charge^2/B"+params,
                    openmm.CustomGBForce.SingleParticle)
        if SA=='ACE':
            force.addEnergyTerm("28.3919551*(radius+0.14)^2*(radius/B)^6; radius=or+offset"+params, openmm.CustomGBForce.SingleParticle)
        elif SA is not None:
            raise ValueError('Unknown surface area method: '+SA)
        if cutoff is None:
            if kappa > 0:
                force.addEnergyTerm("-138.935485*(1/soluteDielectric-exp(-kappa*f)/solventDielectric)*charge1*charge2/f;"
                                    "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))"+params, openmm.CustomGBForce.ParticlePairNoExclusions)
            else:
                force.addEnergyTerm("-138.935485*(1/soluteDielectric-1/solventDielectric)*charge1*charge2/f;"
                                    "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))"+params, openmm.CustomGBForce.ParticlePairNoExclusions)
        else:
            if kappa > 0:
                force.addEnergyTerm("-138.935485*(1/soluteDielectric-exp(-kappa*f)/solventDielectric)*charge1*charge2*(1/f-"+str(1/cutoff)+");"
                                    "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))"+params, openmm.CustomGBForce.ParticlePairNoExclusions)
            else:
                force.addEnergyTerm("-138.935485*(1/soluteDielectric-1/solventDielectric)*charge1*charge2*(1/f-"+str(1/cutoff)+");"
                                    "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))"+params, openmm.CustomGBForce.ParticlePairNoExclusions)

    def create_force(self, system, topology, **kwargs):
        import simtk

        self._validate_parameters()

        # Grab the existing nonbonded force (which will have particle charges)
        existing = [system.getForce(i) for i in range(system.getNumForces())]
        existing = [f for f in existing if type(f) == openmm.NonbondedForce]
        assert len(existing) == 1

        nonbonded_force = existing[0]

        # No previous GBSAForce should exist, so we're safe just making one here.
        #force_map = {
        #    "HCT": simtk.openmm.app.internal.customgbforces.GBSAHCTForce,
        #    "OBC1": simtk.openmm.app.internal.customgbforces.GBSAOBC1Force,
        #    "OBC2": simtk.openmm.GBSAOBCForce,
            # It's tempting to do use the class below, but the customgbforce
            # version of OBC2 doesn't provide setSolventRadius()
            #'OBC2': simtk.openmm.app.internal.customgbforces.GBSAOBC2Force,
        #}
        #openmm_force_type = force_map[self.gb_model]

        if nonbonded_force.getNonbondedMethod() == openmm.NonbondedForce.NoCutoff:
            amber_cutoff = None
        else:
            amber_cutoff = nonbonded_force.getCutoffDistance().value_in_unit(
                unit.nanometer
            )

        #if self.gb_model == "OBC2":
        gbsa_force = self._OPENMMTYPE()

        #else:
            # We set these values in the constructor if we use the internal AMBER GBSA type wrapper
        #    gbsa_force = openmm_force_type(
        #        solventDielectric=self.solvent_dielectric,
        #        soluteDielectric=self.solute_dielectric,
        #        SA=self.sa_model,
        #        cutoff=amber_cutoff,
        #        kappa=0,
        #    )
            # WARNING: If using a CustomAmberGBForce, the functional form is affected by whether
            # the cutoff kwarg is None *during initialization*. So, if you initialize it with a
            # non-None value, and then try to change it to None, you're likely to get unphysical results.

        # Set the GBSAForce to have the same cutoff as NonbondedForce
        # gbsa_force.setCutoffDistance(nonbonded_force.getCutoffDistance())
        if amber_cutoff is not None:
            gbsa_force.setCutoffDistance(amber_cutoff)

        if nonbonded_force.usesPeriodicBoundaryConditions():
            # WARNING: The lines below aren't equivalent. The NonbondedForce and
            # CustomGBForce NonbondedMethod enums have different meanings.
            # More details:
            # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.NonbondedForce.html
            # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.GBSAOBCForce.html
            # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomGBForce.html

            # gbsa_force.setNonbondedMethod(simtk.openmm.NonbondedForce.CutoffPeriodic)
            gbsa_force.setNonbondedMethod(simtk.openmm.CustomGBForce.CutoffPeriodic)
        else:
            # gbsa_force.setNonbondedMethod(simtk.openmm.NonbondedForce.NoCutoff)
            gbsa_force.setNonbondedMethod(simtk.openmm.CustomGBForce.NoCutoff)

        gbsa_force.addPerParticleParameter("charge")
        gbsa_force.addPerParticleParameter("or") # Offset radius
        gbsa_force.addPerParticleParameter("sr") # Scaled offset radius
        gbsa_force.addComputedValue("I",  "select(step(r+sr2-or1), 0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r), 0);"
                                    "U=r+sr2;"
                                    "L=max(or1, D);"
                                    "D=abs(r-sr2)", openmm.CustomGBForce.ParticlePairNoExclusions)

        #b_eqn = f"1/(1/or-tanh({alpha}*psi+{gamma}*psi^3)/radius);"
        b_eqn = f"1/(1/or-tanh({self.alpha}*psi-{self.beta}*psi^2+{self.gamma}*psi^3)/radius);"
        gbsa_force.addComputedValue("B", b_eqn + "psi=I*or; radius=or+offset; offset=0.009", openmm.CustomGBForce.SingleParticle)

        self._createEnergyTerms(gbsa_force, self.solvent_dielectric, self.solute_dielectric, self.sa_model, amber_cutoff, self.kappa, 0.009)

        # Add all GBSA terms to the system. Note that this will have been done above
        #if self.gb_model == "OBC2":
        #    gbsa_force.setSolventDielectric(self.solvent_dielectric)
        #    gbsa_force.setSoluteDielectric(self.solute_dielectric)
        #    if self.sa_model is None:
        #        gbsa_force.setSurfaceAreaEnergy(0)
        #    else:
        #        gbsa_force.setSurfaceAreaEnergy(self.surface_area_penalty)

        # Iterate over all defined GBSA types, allowing later matches to override earlier ones.
        atom_matches = self.find_matches(topology)

        # Create all particles.

        # !!! WARNING: CustomAmberGBForceBase expects different per-particle parameters
        # depending on whether you use addParticle or setParticleParameters. In
        # setParticleParameters, we have to apply the offset and scale BEFORE setting
        # parameters, whereas in addParticle, the offset is applied automatically, and the particle
        # parameters are not set until an auxillary finalize() method is called. !!!

        # To keep it simple, we DO NOT pre-populate the particles in the GBSA force here.
        # We call addParticle further below instead.
        # These lines are commented out intentionally as an example of what NOT to do.
        # for topology_particle in topology.topology_particles:
        # gbsa_force.addParticle([0.0, 1.0, 0.0])

        params_to_add = [[] for _ in topology.topology_particles]
        for atom_key, atom_match in atom_matches.items():
            atom_idx = atom_key[0]
            gbsatype = atom_match.parameter_type
            charge, _, _2 = nonbonded_force.getParticleParameters(atom_idx)
            params_to_add[atom_idx] = [charge, gbsatype.radius, gbsatype.scale]

        #if self.gb_model == "OBC2":
        #    for particle_param in params_to_add:
        #        gbsa_force.addParticle(*particle_param)
        #else:
        #    for particle_param in params_to_add:
        #        gbsa_force.addParticle(particle_param)
        #    # We have to call finalize() for models that inherit from CustomAmberGBForceBase,
        #    # otherwise the added particles aren't actually passed to the underlying CustomGBForce
        #    gbsa_force.finalize()

        for particle_param in params_to_add:
            gbsa_force.addParticle(particle_param)

        # Check that no atoms (n.b. not particles) are missing force parameters.
        self._check_all_valence_terms_assigned(
            assigned_terms=atom_matches, valence_terms=list(topology.topology_atoms)
        )

        system.addForce(gbsa_force)

