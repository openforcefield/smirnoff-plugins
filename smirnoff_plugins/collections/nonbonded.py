import math
from typing import Dict, Iterable, Literal, Set, Tuple, Type, TypeVar, Union

from openff.interchange import Interchange
from openff.interchange.components.potentials import Potential
from openff.interchange.exceptions import InvalidParameterHandlerError
from openff.interchange.models import VirtualSiteKey
from openff.interchange.smirnoff._base import SMIRNOFFCollection
from openff.interchange.smirnoff._nonbonded import (
    SMIRNOFFvdWCollection,
    _SMIRNOFFNonbondedCollection,
)
from openff.toolkit import Quantity, Topology, unit
from openff.toolkit.topology import Atom
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterHandler
from openmm import CustomManyParticleForce, openmm
from typing_extensions import Self

from smirnoff_plugins._types import (
    _DimensionlessQuantity,
    _DistanceQuantity,
    _InverseDistanceQuantity,
    _kJMolNanometerQuantity,
)
from smirnoff_plugins.handlers.nonbonded import (
    AxilrodTellerHandler,
    DampedBuckingham68Handler,
    DampedExp6810Handler,
    DoubleExponentialHandler,
    MultipoleHandler,
)

T = TypeVar("T", bound="_NonbondedPlugin")


class _NonbondedPlugin(_SMIRNOFFNonbondedCollection):
    is_plugin: bool = True
    acts_as: str = "vdW"

    periodic_method: str = "cutoff"
    nonperiodic_method: str = "no-cutoff"

    mixing_rule: str = ""
    switch_width: _DistanceQuantity = Quantity("1.0 angstrom")

    @classmethod
    def check_openmm_requirements(cls: Type[T], combine_nonbonded_forces: bool):
        """SMIRNOFF plugins using non-LJ functional forms cannot combine forces."""
        assert combine_nonbonded_forces is False

    @classmethod
    def global_parameters(cls: Type[T]) -> Iterable[str]:
        """Return an iterable of global parameters, i.e. not per-potential parameters."""
        return tuple()

    # This method could be copy-pasted intead of monkey-patched. It's defined in the default
    # vdW class (SMIRNOFFvdWCollection), not the base non-bonded class
    # (_SMIRNOFF_NonbondedCollection) so it's not brought in by _NonbondedPlugin.
    store_potentials = SMIRNOFFvdWCollection.store_potentials  # type: ignore

    @classmethod
    def create(
        cls,
        parameter_handler: ParameterHandler,
        topology: Topology,
    ) -> Self:
        if type(parameter_handler) not in cls.allowed_parameter_handlers():
            raise InvalidParameterHandlerError(
                f"Found parameter handler type {type(parameter_handler)}, which is not "
                f"supported by potential type {type(cls)}",
            )

        # The presence of global parameters on a subclass can modify which args must be passed to
        # the constructor - not sure if there's a cleaner way that shoving everything into a dict
        _args = {
            "scale_13": parameter_handler.scale13,
            "scale_14": parameter_handler.scale14,
            "scale_15": parameter_handler.scale15,
            "cutoff": parameter_handler.cutoff,
            "periodic_method": parameter_handler.periodic_method.lower(),
            "nonperiodic_method": parameter_handler.nonperiodic_method.lower(),
            "switch_width": parameter_handler.switch_width,
        }

        for global_parameter in cls.global_parameters():
            _args[global_parameter] = getattr(parameter_handler, global_parameter)

        handler = cls(**_args)

        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)  # type: ignore

        return handler

    def _recombine_electrostatics_1_4(
        self,
        system: openmm.System,
        scale_14: float = 0.5,
    ):
        """
        Re-combine the CustomBondForce holding 1-4 electrostatics interactions into the NonbondedForce holding the
        pairwise intermolecule electrostatics interactions. Leave intact the CustomNonbondedForce and CustomBondForce
        holding the intermolecular and intramolecular (1-4) vdW interactions.

        Parameters
        ----------
        system
            The system to modify.
        scale_14
            The factor by which to scale the 1-4 electrostatics interactions.

        See for context: https://github.com/openforcefield/openff-interchange/issues/863
        """

        for force_index, force in enumerate(system.getForces()):
            if force.getName() == "Electrostatics 1-4 force":
                electrostatics_14_force_index = force_index
                electrostatics_14_force = force

            elif isinstance(force, openmm.NonbondedForce):
                electrostatics_force = force

        # The main force should keep exceptions between 1-2 and 1-3 neighbors,
        # which must not get intramolecular electrostatics added back. Since the
        # 1-4 vdW interactions are stored in a separate force, we must look up
        # 1-4 pairs by checking their membership in the CustomBondForce
        pairs = [
            tuple(sorted(electrostatics_14_force.getBondParameters(index)[:2]))
            for index in range(electrostatics_14_force.getNumBonds())
        ]

        for exception_index in range(electrostatics_force.getNumExceptions()):
            particle1, particle2, *_ = electrostatics_force.getExceptionParameters(
                exception_index
            )

            if tuple(sorted([particle1, particle2])) not in pairs:
                pass

            charge1 = electrostatics_force.getParticleParameters(particle1)[0]
            charge2 = electrostatics_force.getParticleParameters(particle2)[0]

            # It is still useful to add the exception to the force, but set its charge to zero
            effective_charge = (
                charge1 * charge2 * scale_14
                if tuple(sorted([particle1, particle2])) in pairs
                else 0.0
            )

            electrostatics_force.setExceptionParameters(
                index=exception_index,
                particle1=particle1,
                particle2=particle2,
                chargeProd=effective_charge,
                sigma=0.0,
                epsilon=0.0,
            )

        system.removeForce(electrostatics_14_force_index)


class SMIRNOFFDampedBuckingham68Collection(_NonbondedPlugin):
    """Collection storing damped Buckingham potentials."""

    type: Literal["DampedBuckingham68"] = "DampedBuckingham68"

    expression: str = (
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

    gamma: _InverseDistanceQuantity

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        """Return an interable of allowed types of ParameterHandler classes."""
        return (DampedBuckingham68Handler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return an interable of supported parameter attributes."""
        return "smirks", "id", "a", "b", "c6", "c8"

    @classmethod
    def default_parameter_values(cls) -> Iterable[float]:
        """Per-particle parameter values passed to Force.addParticle()."""
        return 0.0, 0.0, 0.0, 0.0

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return "a", "b", "c6", "c8"

    @classmethod
    def global_parameters(cls) -> Iterable[str]:
        """Return an iterable of global parameters, i.e. not per-potential parameters."""
        return ("gamma",)

    def pre_computed_terms(self) -> dict[str, Quantity]:
        """Return a dictionary of pre-computed terms for use in the expression."""
        d2 = self.gamma.m**2 * 0.5
        d3 = d2 * self.gamma.m * 0.3333333333
        d4 = d3 * self.gamma.m * 0.25
        d5 = d4 * self.gamma.m * 0.2
        d6 = d5 * self.gamma.m * 0.1666666667
        d7 = d6 * self.gamma.m * 0.1428571429
        d8 = d7 * self.gamma.m * 0.125

        return {"d2": d2, "d3": d3, "d4": d4, "d5": d5, "d6": d6, "d7": d7, "d8": d8}

    def modify_parameters(
        self,
        original_parameters: dict[str, Quantity],
    ) -> dict[str, float]:
        """Optionally modify parameters prior to their being stored in a force."""
        _units = {
            "a": unit.kilojoule_per_mole,
            "b": unit.nanometer**-1,
            "c6": unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": unit.kilojoule_per_mole * unit.nanometer**8,
        }

        if "sigma" in original_parameters and "epsilon" in original_parameters:
            if original_parameters["epsilon"].m == 0.0:
                original_parameters = {
                    key: val * _units[key]
                    for key, val in zip(
                        self.potential_parameters(), self.default_parameter_values()
                    )
                }

        return {
            name: math.sqrt(original_parameters[name].m_as(_units[name]))
            for name in self.potential_parameters()
        }

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ):
        self._recombine_electrostatics_1_4(
            system, interchange["Electrostatics"].scale_14
        )


class SMIRNOFFDoubleExponentialCollection(_NonbondedPlugin):
    """Handler storing vdW potentials as produced by a SMIRNOFF force field."""

    type: Literal["DoubleExponential"] = "DoubleExponential"

    expression: str = (
        "CombinedEpsilon*RepulsionFactor*RepulsionExp-CombinedEpsilon*AttractionFactor*AttractionExp;"
        "CombinedEpsilon=epsilon1*epsilon2;"
        "RepulsionExp=exp(-alpha*ExpDistance);"
        "AttractionExp=exp(-beta*ExpDistance);"
        "ExpDistance=r/CombinedR;"
        "CombinedR=r_min1+r_min2;"
    )

    alpha: _DimensionlessQuantity
    beta: _DimensionlessQuantity

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        """Return an iterable of allowed types of ParameterHandler classes."""
        return (DoubleExponentialHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return an iterable of supported parameter attributes."""
        return "smirks", "id", "r_min", "epsilon"

    @classmethod
    def default_parameter_values(cls) -> Iterable[float]:
        """Per-particle parameter values passed to Force.addParticle()."""
        return 1.0, 0.0

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return "r_min", "epsilon"

    @classmethod
    def global_parameters(cls) -> Iterable[str]:
        """Return an iterable of global parameters, i.e. not per-potential parameters."""
        return "alpha", "beta"

    def pre_computed_terms(self) -> dict[str, float]:
        """Return a dictionary of pre-computed terms for use in the expression."""
        alpha_min_beta = self.alpha - self.beta

        return {
            "AlphaMinBeta": alpha_min_beta,
            "RepulsionFactor": self.beta * math.exp(self.alpha) / alpha_min_beta,
            "AttractionFactor": self.alpha * math.exp(self.beta) / alpha_min_beta,
        }

    def modify_parameters(
        self,
        original_parameters: dict[str, Quantity],
    ) -> dict[str, float]:
        """Optionally modify parameters prior to their being stored in a force."""
        # It's important that these keys are in the order of self.potential_parameters(),
        # consider adding a check somewhere that this is the case.
        _units = {"r_min": unit.nanometer, "epsilon": unit.kilojoule_per_mole}
        return {
            "r_min": original_parameters["r_min"].m_as(_units["r_min"]) * 0.5,
            "epsilon": math.sqrt(
                original_parameters["epsilon"].m_as(_units["epsilon"]),
            ),
        }

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ):
        self._recombine_electrostatics_1_4(
            system, interchange["Electrostatics"].scale_14
        )


class SMIRNOFFDampedExp6810Collection(_NonbondedPlugin):
    """
    Damped exponential-6-8-10 potential used in <https://doi.org/10.1021/acs.jctc.0c00837>

    Essentially a Buckingham-6-8-10 potential with mixing rules from
    <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.5.1708>
    and a physically reasonable parameter form from Stone, et al.
    """

    type: Literal["DampedExp6810"] = "DampedExp6810"

    acts_as: str = "vdW"

    expression: str = (
        "repulsion - ttdamp6*c6*invR^6 - ttdamp8*c8*invR^8 - ttdamp10*c10*invR^10;"
        "repulsion = force_at_zero*invbeta*exp(-beta*(r-rho));"
        "ttdamp10 = select(expbr, 1.0 - expbr * ttdamp10Sum, 1);"
        "ttdamp8 = select(expbr, 1.0 - expbr * ttdamp8Sum, 1);"
        "ttdamp6 = select(expbr, 1.0 - expbr * ttdamp6Sum, 1);"
        "ttdamp10Sum = ttdamp8Sum + br^9/362880 + br^10/3628800;"
        "ttdamp8Sum = ttdamp6Sum + br^7/5040 + br^8/40320;"
        "ttdamp6Sum = 1.0 + br + br^2/2 + br^3/6 + br^4/24 + br^5/120 + br^6/720;"
        "expbr = exp(-br);"
        "br = beta*r;"
        "invR = 1.0/r;"
        "c6 = sqrt(c61*c62);"
        "c8 = sqrt(c81*c82);"
        "c10 = sqrt(c101*c102);"
        "invbeta = select(beta_test, 1.0/beta, 0);"
        "beta = select(beta_test, 2.0*beta_test/(beta1+beta2), 0);"
        "beta_test = beta1*beta2;"
        "rho = 0.5*(rho1+rho2);"
    )

    force_at_zero: _kJMolNanometerQuantity = Quantity(
        49.6144931952,
        "kilojoules_per_mole * nanometer**-1",
    )

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        """Return an iterable of allowed types of ParameterHandler classes."""
        return (DampedExp6810Handler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return an iterable of supported parameter attributes."""
        return "smirks", "id", "rho", "beta", "c6", "c8", "c10"

    @classmethod
    def default_parameter_values(cls) -> Iterable[float]:
        """Per-particle parameter values passed to Force.addParticle()."""
        return 0.0, 0.0, 0.0, 0.0, 0.0

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return "rho", "beta", "c6", "c8", "c10"

    @classmethod
    def global_parameters(cls) -> Iterable[str]:
        """Return an iterable of global parameters, i.e. not per-potential parameters."""
        return ("force_at_zero",)

    def pre_computed_terms(self) -> Dict[str, Quantity]:
        return {}

    def modify_parameters(
        self,
        original_parameters: Dict[str, Quantity],
    ) -> Dict[str, float]:
        # It's important that these keys are in the order of self.potential_parameters(),
        # consider adding a check somewhere that this is the case.
        _units = {
            "rho": unit.nanometers,
            "beta": unit.nanometers**-1,
            "c6": unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": unit.kilojoule_per_mole * unit.nanometer**8,
            "c10": unit.kilojoule_per_mole * unit.nanometer**10,
        }

        return {
            "rho": original_parameters["rho"].m_as(_units["rho"]),
            "beta": original_parameters["beta"].m_as(_units["beta"]),
            "c6": original_parameters["c6"].m_as(_units["c6"]),
            "c8": original_parameters["c8"].m_as(_units["c8"]),
            "c10": original_parameters["c10"].m_as(_units["c10"]),
        }


class SMIRNOFFAxilrodTellerCollection(SMIRNOFFCollection):
    """
    Standard Axilrod-Teller potential from <https://aip.scitation.org/doi/10.1063/1.1723844>.
    """

    expression: str = (
        "C*(1+3*cos(theta1)*cos(theta2)*cos(theta3))/(r12*r13*r23)^3;"
        "theta1=angle(p1,p2,p3); theta2=angle(p2,p3,p1); theta3=angle(p3,p1,p2);"
        "r12=distance(p1,p2); r13=distance(p1,p3); r23=distance(p2,p3);"
        "C=(c91*c92*c93)^(1.0/3.0)"
    )

    type: Literal["AxilrodTeller"] = "AxilrodTeller"

    is_plugin: bool = True
    acts_as: str = ""
    periodic_method: str = "cutoff-periodic"
    nonperiodic_method: str = "cutoff-nonperiodic"
    cutoff: _DistanceQuantity = Quantity("0.9 nanometer")

    def store_potentials(self, parameter_handler: AxilrodTellerHandler):
        self.nonperiodic_method = parameter_handler.nonperiodic_method
        self.periodic_method = parameter_handler.periodic_method
        self.cutoff = parameter_handler.cutoff

        for potential_key in self.key_map.values():
            smirks = potential_key.id
            parameter = parameter_handler.parameters[smirks]

            self.potentials[potential_key] = Potential(
                parameters={"c9": parameter.c9},
            )

    @classmethod
    def potential_parameters(cls):
        return ("c9",)

    @classmethod
    def supported_parameters(cls):
        return "smirks", "id", "c9"

    @classmethod
    def allowed_parameter_handlers(cls):
        return (AxilrodTellerHandler,)

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ):
        force: CustomManyParticleForce = CustomManyParticleForce(3, self.expression)
        force.setPermutationMode(CustomManyParticleForce.SinglePermutation)
        force.addPerParticleParameter("c9")

        method_map = {
            "cutoff-periodic": openmm.CustomManyParticleForce.CutoffPeriodic,
            "cutoff-nonperiodic": openmm.CustomManyParticleForce.CutoffNonPeriodic,
            "no-cutoff": openmm.CustomManyParticleForce.NoCutoff,
        }
        if interchange.box is None:
            force.setNonbondedMethod(method_map[self.nonperiodic_method])
        else:
            force.setNonbondedMethod(method_map[self.periodic_method])
        force.setCutoffDistance(self.cutoff.m_as("nanometer"))

        topology = interchange.topology

        for _ in range(topology.n_atoms):
            force.addParticle([0.0])

        for key, val in self.key_map.items():
            force.setParticleParameters(
                key.atom_indices[0],
                [
                    self.potentials[val]
                    .parameters["c9"]
                    .m_as("kilojoule_per_mole * nanometer**9")
                ],
                0,
            )

        existing_nonbondeds = [
            system.getForce(i)
            for i in range(system.getNumForces())
            if isinstance(system.getForce(i), openmm.NonbondedForce)
        ]

        existing_custom_nonbondeds = [
            system.getForce(i)
            for i in range(system.getNumForces())
            if isinstance(system.getForce(i), openmm.CustomNonbondedForce)
        ]

        if len(existing_nonbondeds) > 0:
            nonbonded = existing_nonbondeds[0]
            for idx in range(nonbonded.getNumExceptions()):
                i, j, _, _, _ = nonbonded.getExceptionParameters(idx)
                force.addExclusion(i, j)

        elif len(existing_custom_nonbondeds) > 0:
            nonbonded = existing_custom_nonbondeds[0]
            for idx in range(nonbonded.getNumExclusions()):
                i, j = nonbonded.getExclusionParticles(idx)
                force.addExclusion(i, j)

        system.addForce(force)

    def modify_parameters(
        self,
        original_parameters: Dict[str, Quantity],
    ) -> Dict[str, float]:
        # It's important that these keys are in the order of self.potential_parameters(),
        # consider adding a check somewhere that this is the case.
        _units = {"c9": unit.kilojoule_per_mole * unit.nanometer**9}

        return {"c9": original_parameters["c9"].m_as(_units["c9"])}


class SMIRNOFFMultipoleCollection(SMIRNOFFCollection):
    """
    Collection for OpenMM's AmoebaMultipoleForce

    At the moment this code grabs the partial charges from the Electrostatics collection.
    Support is only provided for the partial charge and induced dipole portion of AmoebaMultipoleForce, all permanent
    dipoles and quadrupoles are set to zero.

    Exclusions in this Force work differently than other Forces, a list of 1-2, 1-3, 1-4, and 1-5 neighbors are added
    to each particle via the setCovalentMap function. Covalent12, Covalent13, Covalent14, and Covalent15 are lists of
    covalently bonded neighbors separated by 1, 2, 3, and 4 bonds respectively. PolarizationCovalent11 is a list
    all atoms in a "group", all atoms in the "group" do not have permanent multipole-induced dipole interactions
    (induced-induced mutual polarization still occurs between all atoms). The scale factors are as follows:

    Covalent12 0.0
    Covalent13 0.0
    Covalent14 0.4
    Covalent15 0.8

    Note that Covalent15 is not set in this code, setting Covalent15 would result in inconsistent exclusions between
    this force and all other forces and cause an OpenMM error.

    Supported options cutoff, (nonbonded) method, polarization type, ewald error tolerance, thole, target epsilon,
    and max iter are directly passed through to the OpenMM force.
    """

    expression: str = ""

    type: Literal["Multipole"] = "Multipole"

    is_plugin: bool = True

    periodic_method: str = "pme"
    nonperiodic_method: str = "no-cutoff"
    polarization_type: str = "extrapolated"
    cutoff: _DistanceQuantity = Quantity("0.9 nanometer")
    ewald_error_tolerance: float = 0.0001
    target_epsilon: float = 0.00001
    max_iter: int = 60
    thole: float = 0.39

    def store_potentials(self, parameter_handler: MultipoleHandler) -> None:
        self.nonperiodic_method = parameter_handler.nonperiodic_method.lower()
        self.periodic_method = parameter_handler.periodic_method.lower()
        self.polarization_type = parameter_handler.polarization_type.lower()
        self.cutoff = parameter_handler.cutoff
        self.ewald_error_tolerance = parameter_handler.ewald_error_tolerance
        self.target_epsilon = parameter_handler.target_epsilon
        self.max_iter = parameter_handler.max_iter
        self.thole = parameter_handler.thole

        for potential_key in self.key_map.values():
            smirks = potential_key.id
            parameter = parameter_handler.parameters[smirks]

            self.potentials[potential_key] = Potential(
                parameters={"polarity": parameter.polarity},
            )

    @classmethod
    def potential_parameters(cls):
        return ("polarity",)

    @classmethod
    def supported_parameters(cls):
        return "smirks", "id", "polarity"

    @classmethod
    def allowed_parameter_handlers(cls):
        return (MultipoleHandler,)

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ):
        # Sanity checks
        existing_multipole = [
            system.getForce(i)
            for i in range(system.getNumForces())
            if isinstance(system.getForce(i), openmm.AmoebaMultipoleForce)
        ]

        assert (
            len(existing_multipole) < 2
        ), "multiple multipole forces are not yet correctly handled."

        if len(existing_multipole) == 0:
            force: openmm.AmoebaMultipoleForce = openmm.AmoebaMultipoleForce()
            system.addForce(force)
        else:
            force = existing_multipole[0]

        existing_nonbonded = [
            system.getForce(i)
            for i in range(system.getNumForces())
            if isinstance(system.getForce(i), openmm.NonbondedForce)
        ]

        # Zero out charges in nonbonded forces to prevent double counting electrostatic interactions
        nonbonded_force: openmm.NonbondedForce
        for nonbonded_force in existing_nonbonded:
            for i in range(nonbonded_force.getNumParticles()):
                params = nonbonded_force.getParticleParameters(i)
                params[0] = 0
                nonbonded_force.setParticleParameters(i, *params)

        existing_custom_bonds = [
            system.getForce(i)
            for i in range(system.getNumForces())
            if isinstance(system.getForce(i), openmm.CustomBondForce)
            and system.getForce(i).getEnergyFunction() == "138.935456*qq/r"
        ]

        # Zero out charges in custom bond forces with a Coulomb's law expression to prevent double counting
        # electrostatic interactions
        custom_bond_force: openmm.CustomBondForce
        for custom_bond_force in existing_custom_bonds:
            for i in range(custom_bond_force.getNumBonds()):
                params = custom_bond_force.getBondParameters(i)
                params[2] = (0.0,)
                custom_bond_force.setBondParameters(i, *params)

        topology: Topology = interchange.topology
        charges = interchange.collections["Electrostatics"].charges  # type: ignore[attr-defined]

        # Set options
        method_map = {
            "no-cutoff": openmm.AmoebaMultipoleForce.NoCutoff,
            "pme": openmm.AmoebaMultipoleForce.PME,
        }
        if interchange.box is None:
            force.setNonbondedMethod(method_map[self.nonperiodic_method])
        else:
            force.setNonbondedMethod(method_map[self.periodic_method])
        polarization_type_map = {
            "mutual": openmm.AmoebaMultipoleForce.Mutual,
            "direct": openmm.AmoebaMultipoleForce.Direct,
            "extrapolated": openmm.AmoebaMultipoleForce.Extrapolated,
        }
        force.setPolarizationType(polarization_type_map[self.polarization_type])
        force.setCutoffDistance(self.cutoff.m_as("nanometer"))
        force.setEwaldErrorTolerance(self.ewald_error_tolerance)
        force.setMutualInducedTargetEpsilon(self.target_epsilon)
        force.setMutualInducedMaxIterations(self.max_iter)
        force.setExtrapolationCoefficients([-0.154, 0.017, 0.658, 0.474])
        force.setForceGroup(1)

        # All forces are required to have a number of particles equal to the number of particles in the system
        for _ in range(topology.n_atoms):
            force.addMultipole(
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                openmm.AmoebaMultipoleForce.NoAxisType,
                -1,
                -1,
                -1,
                self.thole,
                0.0,
                0.0,
            )

        # Copy partial charges from the electrostatics collection
        for key, val in charges.items():
            params = force.getMultipoleParameters(key.atom_indices[0])
            params[0] = val.m_as("elementary_charge")
            force.setMultipoleParameters(key.atom_indices[0], *params)

        # Set the polarity and damping factor
        for key, val in self.key_map.items():
            params = force.getMultipoleParameters(key.atom_indices[0])
            # the amoeba damping factor is polarity ** 1/6
            params[8] = self.potentials[val].parameters["polarity"].m_as(
                "nanometer**3"
            ) ** (1 / 6)
            # this is the actual polarity
            params[9] = self.potentials[val].parameters["polarity"].m_as("nanometer**3")
            force.setMultipoleParameters(key.atom_indices[0], *params)

        # Set exceptions, note that amoeba handles exceptions completely different to every other force, see above
        for unique_mol_index, mol_map in topology.identical_molecule_groups.items():
            unique_mol = topology.molecule(unique_mol_index)
            # bonded2, bonded3, etc is a dict of molecule_atom_index -> list of molecule_atom_indexs 1 (2, 3) bonds away
            # for the unique_mol
            bonded2: dict[int, list[int]] = {}
            bonded3: dict[int, list[int]] = {}
            bonded4: dict[int, list[int]] = {}
            polarization_bonded: dict[int, list[int]] = {}

            atom1: Atom
            atom2: Atom
            for atom1, atom2 in unique_mol.nth_degree_neighbors(1):
                if atom1.molecule_atom_index not in bonded2:
                    bonded2[atom1.molecule_atom_index] = [atom2.molecule_atom_index]
                else:
                    bonded2[atom1.molecule_atom_index].append(atom2.molecule_atom_index)

                if atom2.molecule_atom_index not in bonded2:
                    bonded2[atom2.molecule_atom_index] = [atom1.molecule_atom_index]
                else:
                    bonded2[atom2.molecule_atom_index].append(atom1.molecule_atom_index)

                if atom1.molecule_atom_index not in polarization_bonded:
                    polarization_bonded[atom1.molecule_atom_index] = [
                        atom2.molecule_atom_index
                    ]
                else:
                    if (
                        atom2.molecule_atom_index
                        not in polarization_bonded[atom1.molecule_atom_index]
                    ):
                        polarization_bonded[atom1.molecule_atom_index].append(
                            atom2.molecule_atom_index
                        )

                if atom2.molecule_atom_index not in polarization_bonded:
                    polarization_bonded[atom2.molecule_atom_index] = [
                        atom1.molecule_atom_index
                    ]
                else:
                    if (
                        atom1.molecule_atom_index
                        not in polarization_bonded[atom2.molecule_atom_index]
                    ):
                        polarization_bonded[atom2.molecule_atom_index].append(
                            atom1.molecule_atom_index
                        )

            for atom1, atom2 in unique_mol.nth_degree_neighbors(2):
                if atom1.molecule_atom_index not in bonded3:
                    bonded3[atom1.molecule_atom_index] = [atom2.molecule_atom_index]
                else:
                    bonded3[atom1.molecule_atom_index].append(atom2.molecule_atom_index)

                if atom2.molecule_atom_index not in bonded3:
                    bonded3[atom2.molecule_atom_index] = [atom1.molecule_atom_index]
                else:
                    bonded3[atom2.molecule_atom_index].append(atom1.molecule_atom_index)

                if atom1.molecule_atom_index not in polarization_bonded:
                    polarization_bonded[atom1.molecule_atom_index] = [
                        atom2.molecule_atom_index
                    ]
                else:
                    if (
                        atom2.molecule_atom_index
                        not in polarization_bonded[atom1.molecule_atom_index]
                    ):
                        polarization_bonded[atom1.molecule_atom_index].append(
                            atom2.molecule_atom_index
                        )

                if atom2.molecule_atom_index not in polarization_bonded:
                    polarization_bonded[atom2.molecule_atom_index] = [
                        atom1.molecule_atom_index
                    ]
                else:
                    if (
                        atom1.molecule_atom_index
                        not in polarization_bonded[atom2.molecule_atom_index]
                    ):
                        polarization_bonded[atom2.molecule_atom_index].append(
                            atom1.molecule_atom_index
                        )

            for atom1, atom2 in unique_mol.nth_degree_neighbors(3):
                if atom1.molecule_atom_index not in bonded4:
                    bonded4[atom1.molecule_atom_index] = [atom2.molecule_atom_index]
                else:
                    bonded4[atom1.molecule_atom_index].append(atom2.molecule_atom_index)

                if atom2.molecule_atom_index not in bonded4:
                    bonded4[atom2.molecule_atom_index] = [atom1.molecule_atom_index]
                else:
                    bonded4[atom2.molecule_atom_index].append(atom1.molecule_atom_index)

                if atom1.molecule_atom_index not in polarization_bonded:
                    polarization_bonded[atom1.molecule_atom_index] = [
                        atom2.molecule_atom_index
                    ]
                else:
                    if (
                        atom2.molecule_atom_index
                        not in polarization_bonded[atom1.molecule_atom_index]
                    ):
                        polarization_bonded[atom1.molecule_atom_index].append(
                            atom2.molecule_atom_index
                        )

                if atom2.molecule_atom_index not in polarization_bonded:
                    polarization_bonded[atom2.molecule_atom_index] = [
                        atom1.molecule_atom_index
                    ]
                else:
                    if (
                        atom1.molecule_atom_index
                        not in polarization_bonded[atom2.molecule_atom_index]
                    ):
                        polarization_bonded[atom2.molecule_atom_index].append(
                            atom1.molecule_atom_index
                        )

                if atom1.molecule_atom_index not in polarization_bonded:
                    polarization_bonded[atom1.molecule_atom_index] = [
                        atom2.molecule_atom_index
                    ]
                else:
                    if (
                        atom2.molecule_atom_index
                        not in polarization_bonded[atom1.molecule_atom_index]
                    ):
                        polarization_bonded[atom1.molecule_atom_index].append(
                            atom2.molecule_atom_index
                        )

                if atom2.molecule_atom_index not in polarization_bonded:
                    polarization_bonded[atom2.molecule_atom_index] = [
                        atom1.molecule_atom_index
                    ]
                else:
                    if (
                        atom1.molecule_atom_index
                        not in polarization_bonded[atom2.molecule_atom_index]
                    ):
                        polarization_bonded[atom2.molecule_atom_index].append(
                            atom1.molecule_atom_index
                        )

            for mol_index, atom_map in mol_map:
                base_atom_index = topology.molecule_atom_start_index(
                    topology.molecule(mol_index)
                )

                for unique_atom_index, unique_bonded_list in bonded2.items():
                    atom_index = atom_map[unique_atom_index] + base_atom_index
                    atom_bonded2 = [
                        atom_map[unique_bonded_index] + base_atom_index
                        for unique_bonded_index in unique_bonded_list
                    ]
                    force.setCovalentMap(
                        atom_index, openmm.AmoebaMultipoleForce.Covalent12, atom_bonded2
                    )

                for unique_atom_index, unique_bonded_list in bonded3.items():
                    atom_index = atom_map[unique_atom_index] + base_atom_index
                    atom_bonded3 = [
                        atom_map[unique_bonded_index] + base_atom_index
                        for unique_bonded_index in unique_bonded_list
                    ]
                    force.setCovalentMap(
                        atom_index, openmm.AmoebaMultipoleForce.Covalent13, atom_bonded3
                    )

                for unique_atom_index, unique_bonded_list in bonded4.items():
                    atom_index = atom_map[unique_atom_index] + base_atom_index
                    atom_bonded4 = [
                        atom_map[unique_bonded_index] + base_atom_index
                        for unique_bonded_index in unique_bonded_list
                    ]
                    force.setCovalentMap(
                        atom_index, openmm.AmoebaMultipoleForce.Covalent14, atom_bonded4
                    )

                for (
                    unique_atom_index,
                    unique_bonded_list,
                ) in polarization_bonded.items():
                    atom_index = atom_map[unique_atom_index] + base_atom_index
                    atom_polarization_bonded = [
                        atom_map[unique_bonded_index] + base_atom_index
                        for unique_bonded_index in unique_bonded_list
                    ]
                    force.setCovalentMap(
                        atom_index,
                        openmm.AmoebaMultipoleForce.PolarizationCovalent11,
                        atom_polarization_bonded,
                    )

    def modify_parameters(
        self,
        original_parameters: Dict[str, Quantity],
    ) -> Dict[str, float]:
        # It's important that these keys are in the order of self.potential_parameters(),
        # consider adding a check somewhere that this is the case.
        _units = {"polarity": unit.nanometer**3}

        return {"polarity": original_parameters["polarity"].m_as(_units["polarity"])}
