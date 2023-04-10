import math
from typing import Dict, Iterable, Literal, Type, TypeVar

from openff.interchange.exceptions import InvalidParameterHandlerError
from openff.interchange.smirnoff._nonbonded import (
    SMIRNOFFvdWCollection,
    _SMIRNOFFNonbondedCollection,
)
from openff.models.types import FloatQuantity
from openff.toolkit import Topology
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterHandler
from openff.units import unit

from smirnoff_plugins.handlers.nonbonded import (
    DampedBuckingham68Handler,
    DoubleExponentialHandler,
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
    def create(
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
        handler.store_potentials(parameter_handler=parameter_handler)  # type: ignore[misc]

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
