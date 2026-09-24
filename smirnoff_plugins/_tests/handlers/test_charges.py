import numpy
import openmm
import pytest
from openff.interchange import Interchange
from openff.interchange.exceptions import MissingParameterHandlerError
from openff.interchange.smirnoff._nonbonded import SMIRNOFFElectrostaticsCollection
from openff.toolkit import ForceField, Molecule
from openff.toolkit.typing.engines.smirnoff.parameters import IncompatibleParameterError
from openff.toolkit.utils.exceptions import SMIRNOFFSpecError

from smirnoff_plugins.handlers.charges import NAGLMBISChargesHandler

GAS_MODEL = "nagl-gas-charge-dipole-esp-wb-default"
WATER_MODEL = "nagl-water-charge-dipole-esp-wb-default"

TIP3P_CHARGES = [-0.834, 0.417, 0.417]


def _nagl_mbis_force_field(**kwargs) -> ForceField:
    """Create a force field which uses NAGL-MBIS charges in place of AM1-BCC."""
    force_field = ForceField("openff-2.2.1.offxml", load_plugins=True)
    force_field.deregister_parameter_handler("ToolkitAM1BCC")
    force_field.get_parameter_handler("NAGLMBISCharges", {"version": "0.1", **kwargs})

    return force_field


def _get_charges(interchange: Interchange) -> numpy.ndarray:
    """Get the partial charges on each atom in topology order."""
    charges = {key.atom_indices[0]: val.m for key, val in interchange["Electrostatics"].charges.items()}

    return numpy.array([charges[index] for index in range(interchange.topology.n_atoms)])


def _reference_charges(molecule: Molecule, alpha: float) -> numpy.ndarray:
    """Compute the expected charges directly with naglmbis."""
    from naglmbis.models import ComputePartialPolarised, load_charge_model

    polarised_model = ComputePartialPolarised(
        model_gas=load_charge_model(GAS_MODEL),
        model_water=load_charge_model(WATER_MODEL),
        alpha=alpha,
    )
    charges = polarised_model.compute_polarised_charges(molecule.to_rdkit()).detach().numpy().reshape(-1)

    return charges + (molecule.total_charge.m - charges.sum()) / molecule.n_atoms


def test_handler_defaults():
    handler = _nagl_mbis_force_field()["NAGLMBISCharges"]

    assert handler.alpha == 0.5
    assert handler.gas_model == GAS_MODEL
    assert handler.water_model == WATER_MODEL


def test_handler_round_trip():
    force_field = ForceField(_nagl_mbis_force_field(alpha=0.3).to_string(), load_plugins=True)
    handler = force_field["NAGLMBISCharges"]

    assert handler.alpha == 0.3
    assert isinstance(handler, NAGLMBISChargesHandler)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_handler_invalid_alpha(alpha):
    with pytest.raises(SMIRNOFFSpecError, match="alpha must be between 0 and 1"):
        NAGLMBISChargesHandler(alpha=alpha, skip_version_check=True)


@pytest.mark.parametrize(
    "kwargs",
    [{"alpha": 0.4}, {"gas_model": "nagl-gas-charge-wb"}, {"water_model": "nagl-water-charge-wb"}],
)
def test_handler_compatibility(kwargs):
    handler = NAGLMBISChargesHandler(skip_version_check=True)

    handler.check_handler_compatibility(NAGLMBISChargesHandler(skip_version_check=True))

    with pytest.raises(IncompatibleParameterError):
        handler.check_handler_compatibility(NAGLMBISChargesHandler(skip_version_check=True, **kwargs))


def test_patch_transparent_without_nagl_mbis():
    """Force fields without a NAGLMBISCharges section should use the standard collection."""
    force_field = ForceField("openff-2.2.1.offxml", load_plugins=True)
    force_field.deregister_parameter_handler("ToolkitAM1BCC")

    interchange = Interchange.from_smirnoff(force_field, [Molecule.from_smiles("O")])

    assert type(interchange["Electrostatics"]) is SMIRNOFFElectrostaticsCollection
    assert numpy.allclose(_get_charges(interchange), TIP3P_CHARGES)


def test_missing_electrostatics():
    force_field = _nagl_mbis_force_field()
    force_field.deregister_parameter_handler("Electrostatics")

    with pytest.raises(MissingParameterHandlerError, match="NAGLMBISCharges"):
        Interchange.from_smirnoff(force_field, [Molecule.from_smiles("CCO")])


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_nagl_mbis_charges(alpha):
    pytest.importorskip("naglmbis")

    ethanol = Molecule.from_smiles("CCO")
    water = Molecule.from_smiles("O")

    interchange = Interchange.from_smirnoff(_nagl_mbis_force_field(alpha=alpha), [ethanol, water, ethanol])
    charges = _get_charges(interchange)

    expected = _reference_charges(ethanol, alpha)

    # both copies of ethanol get the NAGL-MBIS charges, water keeps its library charges
    assert numpy.allclose(charges[:9], expected, atol=1e-6)
    assert numpy.allclose(charges[9:12], TIP3P_CHARGES)
    assert numpy.allclose(charges[12:], expected, atol=1e-6)


def test_nagl_mbis_charges_alpha_changes_charges():
    pytest.importorskip("naglmbis")

    ethanol = Molecule.from_smiles("CCO")

    gas = _get_charges(Interchange.from_smirnoff(_nagl_mbis_force_field(alpha=0.0), [ethanol]))
    water = _get_charges(Interchange.from_smirnoff(_nagl_mbis_force_field(alpha=1.0), [ethanol]))
    mixed = _get_charges(Interchange.from_smirnoff(_nagl_mbis_force_field(alpha=0.25), [ethanol]))

    assert not numpy.allclose(gas, water)
    assert numpy.allclose(mixed, 0.75 * gas + 0.25 * water, atol=1e-6)


def test_nagl_mbis_charged_molecule():
    pytest.importorskip("naglmbis")

    acetate = Molecule.from_smiles("CC(=O)[O-]")

    charges = _get_charges(Interchange.from_smirnoff(_nagl_mbis_force_field(), [acetate]))

    assert charges.sum() == pytest.approx(-1.0)
    assert numpy.allclose(charges, _reference_charges(acetate, 0.5), atol=1e-6)


def test_nagl_mbis_to_openmm():
    pytest.importorskip("naglmbis")

    interchange = Interchange.from_smirnoff(
        _nagl_mbis_force_field(),
        [Molecule.from_smiles("CCO"), Molecule.from_smiles("O")],
    )

    system = interchange.to_openmm(combine_nonbonded_forces=True)
    [force] = [force for force in system.getForces() if isinstance(force, openmm.NonbondedForce)]

    openmm_charges = [
        force.getParticleParameters(index)[0].value_in_unit(openmm.unit.elementary_charge)
        for index in range(force.getNumParticles())
    ]

    assert numpy.allclose(openmm_charges, _get_charges(interchange))


def test_nagl_mbis_serialization():
    pytest.importorskip("naglmbis")

    interchange = Interchange.from_smirnoff(_nagl_mbis_force_field(), [Molecule.from_smiles("CCO")])
    round_tripped = Interchange.model_validate_json(interchange.model_dump_json())

    assert numpy.allclose(_get_charges(round_tripped), _get_charges(interchange))
