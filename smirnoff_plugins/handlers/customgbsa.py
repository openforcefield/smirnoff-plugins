from openff.toolkit.typing.engines.smirnoff import (
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
)
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ChargeIncrementModelHandler,
    ElectrostaticsHandler,
    LibraryChargeHandler,
    ToolkitAM1BCCHandler,
    _allow_only,
    vdWHandler,
)
from simtk import openmm, unit


class CustomOBCHandler(ParameterHandler):
    """Handle SMIRNOFF ``<CustomOBC>`` tags

    .. warning :: This API is experimental and subject to change.
    """

    class CustomOBCType(ParameterType):
        """A SMIRNOFF GBSA-OBC type.

        .. warning :: This API is experimental and subject to change.
        """

        _VALENCE_TYPE = "Atom"
        _ELEMENT_NAME = "Atom"

        radius = ParameterAttribute(unit=unit.angstrom)
        scale = ParameterAttribute(converter=float)

    _TAGNAME = "CustomOBC"
    _INFOTYPE = CustomOBCType
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
    alpha = ParameterAttribute(converter=float)
    beta = ParameterAttribute(converter=float)
    gamma = ParameterAttribute(converter=float)

    solvent_dielectric = ParameterAttribute(default=78.5, converter=float)
    solute_dielectric = ParameterAttribute(default=1.0, converter=float)
    sa_model = ParameterAttribute(default="ACE", converter=_allow_only(["ACE", None]))

    # TODO: We can't find where this value is actually fed into the CustomGBForce. We should validate that
    #       this is actually being used correctly
    surface_area_penalty = ParameterAttribute(
        default=5.4 * unit.calorie / unit.mole / unit.angstrom ** 2,
        unit=unit.calorie / unit.mole / unit.angstrom ** 2,
    )
    solvent_radius = ParameterAttribute(default=1.4 * unit.angstrom, unit=unit.angstrom)
    # TODO: Check units for Kappa
    kappa = ParameterAttribute(default=0.0 / unit.angstrom, unit=unit.angstrom ** -1)
    offset = ParameterAttribute(default=0.09 * unit.angstrom, unit=unit.angstrom)

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
        float_attrs_to_compare = [
            "alpha",
            "beta",
            "gamma",
            "solvent_dielectric",
            "solute_dielectric",
        ]
        string_attrs_to_compare = ["sa_model"]
        unit_attrs_to_compare = [
            "surface_area_penalty",
            "solvent_radius",
            "kappa",
            "offset",
        ]

        self._check_attributes_are_equal(
            other_handler,
            identical_attrs=string_attrs_to_compare,
            tolerance_attrs=float_attrs_to_compare + unit_attrs_to_compare,
            tolerance=self._SCALETOL,
        )

    @staticmethod
    def _createEnergyTerms(
        force, solventDielectric, soluteDielectric, SA, cutoff, kappa, offset
    ):
        """Add the OBC energy terms to the CustomGBForce. These are identical for all the GB models."""

        kappa_nm = kappa.value_in_unit(unit.nanometer ** -1)
        offset_nm = offset.value_in_unit(unit.nanometer)

        params = (
            "; solventDielectric=%.16g; soluteDielectric=%.16g; kappa=%.16g; offset=%.16g"
            % (solventDielectric, soluteDielectric, kappa_nm, offset_nm)
        )
        if cutoff is not None:
            params += "; cutoff=%.16g" % cutoff
        if kappa_nm > 0:
            # 138.93 may be the coulomb constant in some unit system.
            force.addEnergyTerm(
                "-0.5*138.935485*(1/soluteDielectric-exp(-kappa*B)/solventDielectric)*charge^2/B"
                + params,
                openmm.CustomGBForce.SingleParticle,
            )
        elif kappa_nm < 0:
            # Do kappa check here to avoid repeating code everywhere
            raise ValueError("kappa/ionic strength must be >= 0")
        else:
            force.addEnergyTerm(
                "-0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*charge^2/B"
                + params,
                openmm.CustomGBForce.SingleParticle,
            )
        if SA == "ACE":
            # TODO: Is 0.14 below the solvent probe radius? Is this the only place we'd need to change it?
            # TODO: Is 28.39... just the surface area penalty times 4*pi (units=joules per mol A**2)?
            force.addEnergyTerm(
                "28.3919551*(radius+0.14)^2*(radius/B)^6; radius=or+offset" + params,
                openmm.CustomGBForce.SingleParticle,
            )
        elif SA is not None:
            raise ValueError("Unknown surface area method: " + SA)
        if cutoff is None:
            if kappa_nm > 0:
                force.addEnergyTerm(
                    "-138.935485*(1/soluteDielectric-exp(-kappa*f)/solventDielectric)*charge1*charge2/f;"
                    "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))" + params,
                    openmm.CustomGBForce.ParticlePairNoExclusions,
                )
            else:
                force.addEnergyTerm(
                    "-138.935485*(1/soluteDielectric-1/solventDielectric)*charge1*charge2/f;"
                    "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))" + params,
                    openmm.CustomGBForce.ParticlePairNoExclusions,
                )
        else:
            if kappa_nm > 0:
                force.addEnergyTerm(
                    "-138.935485*(1/soluteDielectric-exp(-kappa*f)/solventDielectric)*charge1*charge2*(1/f-"
                    + str(1 / cutoff)
                    + ");"
                    "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))" + params,
                    openmm.CustomGBForce.ParticlePairNoExclusions,
                )
            else:
                force.addEnergyTerm(
                    "-138.935485*(1/soluteDielectric-1/solventDielectric)*charge1*charge2*(1/f-"
                    + str(1 / cutoff)
                    + ");"
                    "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))" + params,
                    openmm.CustomGBForce.ParticlePairNoExclusions,
                )

    def create_force(self, system, topology, **kwargs):
        import simtk

        self._validate_parameters()

        # Grab the existing nonbonded force (which will have particle charges)
        existing = [system.getForce(i) for i in range(system.getNumForces())]
        existing = [f for f in existing if type(f) == openmm.NonbondedForce]
        assert len(existing) == 1

        nonbonded_force = existing[0]

        if nonbonded_force.getNonbondedMethod() == openmm.NonbondedForce.NoCutoff:
            amber_cutoff = None
        else:
            amber_cutoff = nonbonded_force.getCutoffDistance().value_in_unit(
                unit.nanometer
            )

        gbsa_force = self._OPENMMTYPE()

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

        gbsa_force.addPerParticleParameter("charge")  # Partial charge of atom
        gbsa_force.addPerParticleParameter("or")  # Offset radius
        gbsa_force.addPerParticleParameter("sr")  # Scaled offset radius

        # 3D integral over vdW spheres
        gbsa_force.addComputedValue(
            "I",
            "select(step(r+sr2-or1), 0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r), 0);"
            "U=r+sr2;"
            "L=max(or1, D);"
            "D=abs(r-sr2)",
            openmm.CustomGBForce.ParticlePairNoExclusions,
        )

        # OBC effective radii
        effective_radii = f"1/(1/or-tanh({self.alpha}*psi-{self.beta}*psi^2+{self.gamma}*psi^3)/radius);"
        effective_params = f"psi=I*or; radius=or+offset; offset={self.offset.value_in_unit(unit.nanometer)}"
        gbsa_force.addComputedValue(
            "B",
            effective_radii + effective_params,
            openmm.CustomGBForce.SingleParticle,
        )

        # Create energy terms
        self._createEnergyTerms(
            gbsa_force,
            self.solvent_dielectric,
            self.solute_dielectric,
            self.sa_model,
            amber_cutoff,
            self.kappa,
            self.offset,
        )

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
            params_to_add[atom_idx] = [
                charge,  # charge(atom)
                gbsatype.radius - self.offset,  # radius(atom) - offset
                gbsatype.scale
                * (gbsatype.radius - self.offset),  # scale*(radius(atom) - offset)
            ]

        for particle_param in params_to_add:
            gbsa_force.addParticle(particle_param)

        # Check that no atoms (n.b. not particles) are missing force parameters.
        self._check_all_valence_terms_assigned(
            assigned_terms=atom_matches, valence_terms=list(topology.topology_atoms)
        )

        system.addForce(gbsa_force)
