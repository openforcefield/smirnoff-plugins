import math
from abc import ABC
from typing import Dict, Iterable, Literal, Type, TypeVar, Tuple, Set, Union

from openff.interchange import Interchange
from openff.interchange.exceptions import InvalidParameterHandlerError
from openff.interchange.smirnoff._base import SMIRNOFFCollection, TP
from openff.interchange.smirnoff._nonbonded import (
    SMIRNOFFvdWCollection,
    _SMIRNOFFNonbondedCollection,
)
from openff.models.types import FloatQuantity
from openff.toolkit import Topology
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterHandler
from openff.units import unit
from openmm import openmm

from smirnoff_plugins.handlers.nonbonded import (
    DampedBuckingham68Handler,
    DoubleExponentialHandler, DampedExp6810Handler, AxilrodTellerHandler, MultipoleHandler,
)

T = TypeVar("T", bound="_NonbondedPlugin")


class _NonbondedPlugin(_SMIRNOFFNonbondedCollection):

    is_plugin: bool = True
    acts_as: str = "vdW"

    method: str = "cutoff"
    mixing_rule: str = ""
    switch_width: FloatQuantity["angstrom"] = unit.Quantity(1.0, unit.angstrom)  # noqa

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
    store_potentials = SMIRNOFFvdWCollection.store_potentials

    @classmethod
    def create(  # type: ignore[override]
        cls: Type[T],
        parameter_handler: ParameterHandler,
        topology: Topology,
    ) -> T:
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
            "method": parameter_handler.method.lower(),
            "switch_width": parameter_handler.switch_width,
        }

        for global_parameter in cls.global_parameters():
            _args[global_parameter] = getattr(parameter_handler, global_parameter)

        handler = cls(**_args)

        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler


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

    gamma: FloatQuantity["nanometer ** -1"]  # noqa

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

    def pre_computed_terms(self) -> Dict[str, unit.Quantity]:
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
        original_parameters: Dict[str, unit.Quantity],
    ) -> Dict[str, float]:
        """Optionally modify parameters prior to their being stored in a force."""
        _units = {
            "a": unit.kilojoule_per_mole,
            "b": unit.nanometer**-1,
            "c6": unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": unit.kilojoule_per_mole * unit.nanometer**8,
        }
        return {
            name: math.sqrt(original_parameters[name].m_as(_units[name]))
            for name in self.potential_parameters()
        }


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

    alpha: FloatQuantity["dimensionless"]  # noqa
    beta: FloatQuantity["dimensionless"]  # noqa

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

    def pre_computed_terms(self) -> Dict[str, float]:
        """Return a dictionary of pre-computed terms for use in the expression."""
        alpha_min_beta = self.alpha - self.beta

        return {
            "AlphaMinBeta": alpha_min_beta,
            "RepulsionFactor": self.beta * math.exp(self.alpha) / alpha_min_beta,
            "AttractionFactor": self.alpha * math.exp(self.beta) / alpha_min_beta,
        }

    def modify_parameters(
        self,
        original_parameters: Dict[str, unit.Quantity],
    ) -> Dict[str, float]:
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


class SMIRNOFFDampedExp6810Collection(_NonbondedPlugin):
    """
    Damped exponential-6-8-10 potential used in <https://doi.org/10.1021/acs.jctc.0c00837>

    Essentially a Buckingham-6-8-10 potential with mixing rules from
    <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.5.1708>
    and a physically reasonable parameter form from Stone, et al.
    """

    type: Literal["DampedExp6810"] = "DampedExp6810"

    expression: str = (
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
        "c6 = sqrt(c61*c62);"
        "c8 = sqrt(c81*c82);"
        "c10 = sqrt(c101*c102);"
        "invbeta = select(beta, 1.0/beta, 0);"
        "beta = select(beta_mix, 2.0*beta_mix/(beta1+beta2), 0);"
        "beta_mix = beta1*beta2;"
        "sigma = 0.5*(sigma1+sigma2);"
    )

    forceAtZero: FloatQuantity["kilojoules_per_mole * nanometer**-1"] =\
        unit.Quantity(49.6144931952, unit.kilojoules_per_mole * unit.nanometer**-1)

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        """Return an iterable of allowed types of ParameterHandler classes."""
        return DampedExp6810Handler,

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return an iterable of supported parameter attributes."""
        return "smirks", "id", "rho", "beta", "c6", "c8", "c10"

    @classmethod
    def default_parameter_values(cls) -> Iterable[float]:
        """Per-particle parameter values passed to Force.addParticle()."""
        return 0.0, 1.0, 0.0, 0.0, 0.0

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return "rho", "beta", "c6", "c8", "c10"

    @classmethod
    def global_parameters(cls) -> Iterable[str]:
        """Return an iterable of global parameters, i.e. not per-potential parameters."""
        return "forceAtZero",

    @classmethod
    def create(  # type: ignore[override]
        cls: Type[T],
        parameter_handler: DampedExp6810Handler,
        topology: Topology,
    ) -> T:
        handler = cls(
            scale_13=parameter_handler.scale13,
            scale_14=parameter_handler.scale14,
            scale_15=parameter_handler.scale15,
            cutoff=parameter_handler.cutoff,
            method=parameter_handler.method.lower(),
            switch_width=parameter_handler.switch_width,
            forceAtZero=parameter_handler.forceAtZero,
        )

        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler


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

    def store_potentials(self, parameter_handler: TP):
        pass

    @classmethod
    def potential_parameters(cls):
        return "c9",

    @classmethod
    def supported_parameters(cls):
        return "smirks", "id", "c9"

    @classmethod
    def allowed_parameter_handlers(cls):
        return AxilrodTellerHandler,

    type: Literal["AxilrodTeller"] = "AxilrodTeller"

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ):
        pass


class SMIRNOFFMultipoleCollection(SMIRNOFFCollection):
    """
    Collection for OpenMM's AmoebaMultipoleForce

    At the moment this code grabs the partial charges from the NonbondedForce after all other handlers are loaded.
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

    def store_potentials(self, parameter_handler: TP):
        pass

    @classmethod
    def potential_parameters(cls):
        return "polarity",

    @classmethod
    def supported_parameters(cls):
        return "smirks", "id", "polarity"

    @classmethod
    def allowed_parameter_handlers(cls):
        return MultipoleHandler,

    type: Literal["Multipole"] = "Multipole"

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ):
        pass

    @classmethod
    def create(  # type: ignore[override]
        cls: Type[T],
        parameter_handler: DampedExp6810Handler,
        topology: Topology,
    ) -> T:
        handler = cls(
            scale_13=parameter_handler.scale13,
            scale_14=parameter_handler.scale14,
            scale_15=parameter_handler.scale15,
            cutoff=parameter_handler.cutoff,
            method=parameter_handler.method.lower(),
            switch_width=parameter_handler.switch_width,
        )

        handler.store_matches(parameter_handler=parameter_handler, topology=topology)
        handler.store_potentials(parameter_handler=parameter_handler)

        return handler
