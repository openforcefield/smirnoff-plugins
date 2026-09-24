import functools
import logging

import numpy
from openff.interchange.components.potentials import Potential
from openff.interchange.models import (
    ChargeModelTopologyKey,
    LibraryChargeTopologyKey,
    PotentialKey,
    SingleAtomChargeTopologyKey,
    TopologyKey,
)
from openff.interchange.smirnoff._nonbonded import SMIRNOFFElectrostaticsCollection
from openff.toolkit import Molecule, Quantity
from openff.toolkit.utils.exceptions import MissingPackageError

from smirnoff_plugins.handlers.charges import NAGLMBISChargesHandler

logger = logging.getLogger(__name__)

_HANDLER_NAME = "NAGLMBISChargesHandler"


@functools.lru_cache(None)
def _load_model(model_name: str):
    """Load a pre-trained ``naglmbis`` charge model."""
    from naglmbis.models import load_charge_model

    return load_charge_model(charge_model=model_name)


@functools.lru_cache(None)
def _compute_nagl_mbis_charges(
    mapped_smiles: str,
    gas_model: str,
    water_model: str,
    alpha: float,
) -> numpy.ndarray:
    """Compute partially polarised NAGL-MBIS charges (in units of e), normalised to the formal charge."""
    try:
        from naglmbis.models import ComputePartialPolarised
    except ImportError as error:
        raise MissingPackageError(
            "The force field has a NAGLMBISCharges section, but naglmbis is not installed. "
            "Use the `naglmbis` pixi environment, e.g. `pixi run -e naglmbis ...`.",
        ) from error

    molecule = Molecule.from_mapped_smiles(mapped_smiles, allow_undefined_stereo=True)

    polarised_model = ComputePartialPolarised(
        model_gas=_load_model(gas_model),
        model_water=_load_model(water_model),
        alpha=alpha,
    )
    charges = polarised_model.compute_polarised_charges(molecule.to_rdkit())

    molecule.partial_charges = Quantity(
        charges.detach().numpy().astype(float).reshape(-1),
        "elementary_charge",
    )
    molecule._normalize_partial_charges()

    return molecule.partial_charges.m_as("elementary_charge")


class SMIRNOFFNAGLMBISElectrostaticsCollection(SMIRNOFFElectrostaticsCollection):
    """
    The standard SMIRNOFF electrostatics collection, extended to assign partial charges from
    a ``NAGLMBISCharges`` section.

    Library charges take precedence over NAGL-MBIS charges, which take precedence over the
    other charge methods. The resulting potentials are labelled as coming from the
    ``NAGLChargesHandler`` so that the rest of Interchange (charge lookup, serialization,
    combining) treats them like any other NAGL charges; the NAGL-MBIS provenance is kept in the
    ``extras`` of each topology key.
    """

    @classmethod
    def allowed_parameter_handlers(cls):
        """Return a list of allowed types of ParameterHandler classes."""
        return [*super().allowed_parameter_handlers(), NAGLMBISChargesHandler]

    @classmethod
    def parameter_handler_precedence(cls) -> list[str]:
        """
        Return the order in which parameter handlers take precedence when computing charges.
        """
        return ["LibraryCharges", "NAGLMBISCharges", "NAGLCharges", "ChargeIncrementModel", "ToolkitAM1BCC"]

    @classmethod
    def _find_reference_matches(
        cls,
        parameter_handlers,
        unique_molecule: Molecule,
    ) -> tuple[dict[TopologyKey, PotentialKey], dict[PotentialKey, Potential]]:
        """
        Construct a slot and potential map for a particular reference molecule and set of parameter handlers.
        """
        if "NAGLMBISCharges" not in parameter_handlers:
            return super()._find_reference_matches(parameter_handlers, unique_molecule)

        if "LibraryCharges" in parameter_handlers:
            matches, potentials = cls._find_slot_matches(parameter_handlers["LibraryCharges"], unique_molecule)

            matched_atom_indices = {index for key in matches for index in key.atom_indices}

            if matched_atom_indices == set(range(unique_molecule.n_atoms)):
                return matches, potentials

        return cls._find_nagl_mbis_matches(parameter_handlers["NAGLMBISCharges"], unique_molecule)

    @classmethod
    def _find_nagl_mbis_matches(
        cls,
        parameter_handler: NAGLMBISChargesHandler,
        unique_molecule: Molecule,
    ) -> tuple[dict[TopologyKey, PotentialKey], dict[PotentialKey, Potential]]:
        """Construct a slot and potential map for the NAGL-MBIS charges of a molecule."""
        mapped_smiles = unique_molecule.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)

        partial_charge_method = (
            f"NAGL-MBIS (gas_model={parameter_handler.gas_model}, "
            f"water_model={parameter_handler.water_model}, alpha={parameter_handler.alpha})"
        )

        partial_charges = _compute_nagl_mbis_charges(
            mapped_smiles,
            parameter_handler.gas_model,
            parameter_handler.water_model,
            parameter_handler.alpha,
        )

        matches: dict = {}
        potentials: dict[PotentialKey, Potential] = {}

        for atom_index, partial_charge in enumerate(partial_charges):
            # Label the potential as a NAGLCharges potential, as Interchange only accepts known
            # handler names when looking up charges and deserializing
            potential_key = PotentialKey(
                id=mapped_smiles,
                mult=atom_index,
                associated_handler="NAGLChargesHandler",
            )
            potentials[potential_key] = Potential(
                parameters={"charge": Quantity(float(partial_charge), "elementary_charge")},
            )

            matches[
                SingleAtomChargeTopologyKey(
                    this_atom_index=atom_index,
                    extras={
                        "handler": _HANDLER_NAME,
                        "partial_charge_method": partial_charge_method,
                    },
                )
            ] = potential_key

        return matches, potentials

    @staticmethod
    def _log_charge_provenance(
        unique_molecule: Molecule,
        key: ChargeModelTopologyKey | SingleAtomChargeTopologyKey | LibraryChargeTopologyKey,
    ):
        if type(key) is SingleAtomChargeTopologyKey and key.extras.get("handler") == _HANDLER_NAME:
            logger.debug(
                f"Charge section NAGLMBISCharges, using {key.extras['partial_charge_method']}, applied to "
                f"molecule with Hill formula {unique_molecule.to_hill_formula()}",
            )
        else:
            SMIRNOFFElectrostaticsCollection._log_charge_provenance(unique_molecule, key)
