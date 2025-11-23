from typing import cast

import openmm
import openmm.unit
import pytest
from openff.interchange import Interchange
from openff.interchange.drivers.openmm import _get_openmm_energies
from openff.toolkit import ForceField, Molecule, Topology, unit

from smirnoff_plugins.collections.valence import (
    SMIRNOFFProperTorsionBendCollection,
    SMIRNOFFUreyBradleyCollection,
)


@pytest.fixture(scope="module")
def methane_molecule():
    """Fixture to create a methane molecule with a fixed set of positions."""

    POSITIONS = [
        [0.00511871, -0.0106205, 0.00601428],
        [0.54966796, 0.75543841, -0.59698119],
        [0.7497641, -0.5879439, 0.58528463],
        [-0.58675256, -0.65213582, -0.67609162],
        [-0.71779821, 0.4952618, 0.6817739],
    ] * unit.angstrom

    methane = Molecule.from_mapped_smiles("[C:1]([H:2])([H:3])([H:4])([H:5])")
    methane.add_conformer(POSITIONS)
    return methane


@pytest.fixture(scope="module")
def peroxide_molecule_info():
    """Fixture to create a hydrogen peroxide molecule with a fixed set of positions
    and to supply geometry and expected energy information."""

    POSITIONS = [
        [-1.29431103, -0.18475285, -0.15111464],
        [-0.56095895, 0.40009688, 0.13385895],
        [0.52896993, -0.33976786, 0.17983297],
        [1.32630004, 0.12442383, -0.16257729],
    ] * unit.angstrom

    # Calculated from the coordinates
    ANGLE0 = 120.0 * openmm.unit.degree
    ANGLES = [
        1.8690299459800725 * openmm.unit.radian,
        1.9747422196821367 * openmm.unit.radian,
    ]

    DIHEDRAL_ANGLE = 2.439148499977821 * openmm.unit.radian
    KS = [
        10.0 * openmm.unit.kilocalorie_per_mole,
        5.0 * openmm.unit.kilocalorie_per_mole,
    ]

    peroxide = Molecule.from_mapped_smiles("[H:1][O:2][O:3][H:4]")
    peroxide.add_conformer(POSITIONS)

    # Get the expected proper torsion-bend energy
    expected_ptb_energy = 0.0 * openmm.unit.kilocalorie_per_mole
    for angle in ANGLES:
        for n, k in enumerate(KS, start=1):
            phase = (
                0.0 * openmm.unit.radian
                if n == 1
                else openmm.math.pi * openmm.unit.radian
            )
            expected_ptb_energy += (
                k
                * (angle - ANGLE0)
                / openmm.unit.radian
                * (
                    1
                    + openmm.math.cos((n * DIHEDRAL_ANGLE - phase) / openmm.unit.radian)
                )
            )

    return {
        "molecule": peroxide,
        "expected_ptb_energy": expected_ptb_energy,
    }


@pytest.fixture(scope="module")
def proper_torsion_bend_force_field():
    ff = ForceField("openff_unconstrained-2.2.1.offxml", load_plugins=True)
    proper_torsion_bend_handler = ff.get_parameter_handler("ProperTorsionBends")
    # Parameters for chloroethyne
    proper_torsion_bend_handler.add_parameter(
        {
            "smirks": "[*:1]-[*:2]#[*:3]-[*:4]",
            "angle0": 180.0 * unit.degree,
            "k1": 10.0 * unit.kilocalorie / unit.mole,
            "periodicity1": 1,
            "phase1": 0.0 * unit.degree,
            "k2": 5.0 * unit.kilocalorie / unit.mole,
            "periodicity2": 2,
            "phase2": 180.0 * unit.degree,
        }
    )
    proper_torsion_bend_handler.add_parameter(
        {
            "smirks": "[Cl:1]-[*:2]#[*:3]-[*:4]",
            "angle0": 180.0 * unit.degree,
            "k1": 10.0 * unit.kilocalorie / unit.mole,
            "periodicity1": 1,
            "phase1": 0.0 * unit.degree,
        },
    )
    # Parameters for methanol
    proper_torsion_bend_handler.add_parameter(
        {
            "smirks": "[H:1]-[C:2]-[O:3]-[H:4]",
            "angle0": 180.0 * unit.degree,
            "k1": 10.0 * unit.kilocalorie / unit.mole,
            "periodicity1": 1,
            "phase1": 0.0 * unit.degree,
        },
    )
    proper_torsion_bend_handler.add_parameter(
        {
            "smirks": "[H:1]-[O:2]-[C:3]-[H:4]",
            "angle0": 180.0 * unit.degree,
            "k1": 10.0 * unit.kilocalorie / unit.mole,
            "periodicity1": 1,
            "phase1": 0.0 * unit.degree,
        },
    )
    # Parameters for peroxide
    proper_torsion_bend_handler.add_parameter(
        {
            "smirks": "[H:1]-[O:2]-[O:3]-[H:4]",
            "angle0": 120.0 * unit.degree,
            "k1": 10.0 * unit.kilocalorie / unit.mole,
            "periodicity1": 1,
            "phase1": 0.0 * unit.degree,
            "k2": 5.0 * unit.kilocalorie / unit.mole,
            "periodicity2": 2,
            "phase2": 180.0 * unit.degree,
        },
    )
    return ff


@pytest.mark.parametrize(
    "angle_constraints",
    [False, True],
    ids=["no_constraints", "with_constraints"],
)
def test_urey_bradley_assignment_methane(
    angle_constraints: bool, methane_molecule: Molecule
):
    """Check that the correct Urey-Bradley terms are assigned to methane."""

    # Number of terms and energies (kJ/ mol) expected without constraints (for methane):
    # For equilibriunm distances of ~ 0.181 nM, we expect the energy to be approximately.
    # 6 * 500 * 0.5 * (0.181 - 0.17)**2 = 0.181 kJ/mol. Using the positions below, the
    # expected energy is as given below.
    EXPECTED_NUM_UREY_BRADLEY_TERMS = 6
    EXPECTED_UREY_BRADLEY_ENERGY = 0.19684386  # kJ/mol, close to expected value

    ff = ForceField("openff_unconstrained-2.2.1.offxml", load_plugins=True)

    # Add H-C-H Urey-Bradley term, with arbitrary parameters.
    urey_bradley_handler = ff.get_parameter_handler("UreyBradleys")
    urey_bradley_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#6X4]-[#1:2]",
            "k": 500 * unit.kilojoule_per_mole / unit.nanometer**2,
            # Slightly less than equilibrium distance of ~ 1.8 A
            "length": 0.17 * unit.nanometers,
        }
    )

    if angle_constraints:
        # Add angle constraints to the force field.
        constraint_handler = ff.get_parameter_handler("Constraints")
        constraint_handler.add_parameter(
            {
                "smirks": "[#1:1]-[#6X4]-[#1:2]",
                "distance": 0.18 * unit.nanometers,
            }
        )

    topology = Topology.from_molecules([methane_molecule])
    interchange = Interchange.from_smirnoff(force_field=ff, topology=topology)

    # Check that the Urey-Bradley terms are present in the interchange object.
    collection = cast(
        SMIRNOFFUreyBradleyCollection, interchange.collections["UreyBradleys"]
    )
    urey_bradley_terms_interchange = list(collection.valence_terms(topology))

    assert len(urey_bradley_terms_interchange) == 6

    # Check the OpenMM system for Urey-Bradley terms.
    omm_system = interchange.to_openmm()
    positions = interchange.positions
    assert positions is not None  # Keep mypy happy

    # Get the OpenMM energies. Note that we can't use get_openmm_energies at the moment
    # as it assumes that only custom forces will have duplicate instances (but here we
    # have two instances of the HarmonicBondForce).
    # https://github.com/openforcefield/openff-interchange/blob/036cf8e3b4d09944feaf945635d1bbe521473b17/openff/interchange/drivers/openmm.py#L153
    raw_energies = _get_openmm_energies(
        system=omm_system,
        box_vectors=None,
        positions=positions.to_openmm(),
        round_positions=None,
        platform="Reference",
    )

    forces = omm_system.getForces()
    ub_forces = [force for force in forces if force.getName() == "UreyBradleyForce"]

    assert (
        len(ub_forces) == 1
    ), "Expected exactly one Urey-Bradley force in the OpenMM system, "

    ub_force = ub_forces[0]
    num_ub_bonds = ub_force.getNumBonds()
    ub_force_idx = forces.index(ub_force)
    ub_energy = raw_energies[ub_force_idx].value_in_unit(
        openmm.unit.kilojoules_per_mole
    )

    if angle_constraints:
        # If angle constraints are applied, we should not have any Urey-Bradley terms.
        assert (
            num_ub_bonds == 0
        ), "Expected no Urey-Bradley terms when angle constraints are applied."

        # The bond energies should be non-zero, as we haven't constrained the bonds,
        # but the Urey-Bradley terms should be zero.
        assert (
            pytest.approx(0.0) == ub_energy
        ), f"Expected Urey-Bradley energy to be 0.0 kJ/mol, but got {ub_energy} kJ/mol."

    else:
        # Check we have the expected number of Urey-Bradley terms.
        assert num_ub_bonds == EXPECTED_NUM_UREY_BRADLEY_TERMS, (
            f"Expected {EXPECTED_NUM_UREY_BRADLEY_TERMS} Urey-Bradley terms, "
            f"but got {num_ub_bonds}."
        )

        # Check that the parameters of the Urey-Bradley terms are as expected.
        expected_params = [
            0.17 * openmm.unit.nanometers,
            500 * openmm.unit.kilojoule_per_mole / openmm.unit.nanometer**2,
        ]

        for i in range(num_ub_bonds):
            actual_params = ub_force.getBondParameters(i)
            actual_params_without_idx = actual_params[2:]
            assert actual_params_without_idx == expected_params, (
                f"Bond parameters {i} do not match expected values: "
                f"{actual_params_without_idx} != {expected_params}"
            )

        # Check that the energies are as expected.
        assert pytest.approx(EXPECTED_UREY_BRADLEY_ENERGY) == ub_energy, (
            f"Expected Urey-Bradley energy to be {EXPECTED_UREY_BRADLEY_ENERGY} kJ/mol, "
            f"but got {ub_energy} kJ/mol."
        )


def test_urey_bradley_incorrect_smirks(methane_molecule: Molecule):
    """Check that an error is raised for incorrect SMIRKS patterns which return too many atoms."""

    ff = ForceField("openff_unconstrained-2.2.1.offxml", load_plugins=True)
    urey_bradley_handler = ff.get_parameter_handler("UreyBradleys")
    urey_bradley_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#6X4:2]-[#1:3]",  # Invalid SMIRKS - this specifies three atoms
            "k": 500 * unit.kilojoule_per_mole / unit.nanometer**2,
            # Slightly less than equilibrium distance of ~ 1.8 A
            "length": 0.17 * unit.nanometers,
        }
    )

    topology = Topology.from_molecules([methane_molecule])
    interchange = Interchange.from_smirnoff(force_field=ff, topology=topology)

    with pytest.raises(
        ValueError, match="Expected 2 indices for Urey-Bradley potential"
    ):
        interchange.to_openmm()


@pytest.mark.parametrize(
    "mapped_smiles,expected_smirks_by_indices",
    [
        # Chloroethyne
        (
            "[Cl:1]-[C:2]#[C:3]-[H:4]",
            {
                (0, 1, 2, 3): "[Cl:1]-[*:2]#[*:3]-[*:4]",
                (3, 2, 1, 0): "[*:1]-[*:2]#[*:3]-[*:4]",
            },
        ),
        # Methanol
        (
            "[H:1][O:2][C:3]([H:4])([H:5])[H:6]",
            {
                (0, 1, 2, 3): "[H:1]-[O:2]-[C:3]-[H:4]",
                (0, 1, 2, 4): "[H:1]-[O:2]-[C:3]-[H:4]",
                (0, 1, 2, 5): "[H:1]-[O:2]-[C:3]-[H:4]",
                (3, 2, 1, 0): "[H:1]-[C:2]-[O:3]-[H:4]",
                (4, 2, 1, 0): "[H:1]-[C:2]-[O:3]-[H:4]",
                (5, 2, 1, 0): "[H:1]-[C:2]-[O:3]-[H:4]",
            },
        ),
    ],
)
def test_proper_torsion_bend_assignment(
    mapped_smiles: str,
    expected_smirks_by_indices: dict[tuple[int, int, int, int], str],
    proper_torsion_bend_force_field: ForceField,
):
    """Check that the ProperTorsionBend terms are assigned correctly."""
    topology = Molecule.from_mapped_smiles(mapped_smiles).to_topology()
    interchange = Interchange.from_smirnoff(
        force_field=proper_torsion_bend_force_field, topology=topology
    )

    collection = cast(
        SMIRNOFFProperTorsionBendCollection,
        interchange.collections["ProperTorsionBends"],
    )

    expected_indices = set(expected_smirks_by_indices.keys())
    assigned_indices = set(key.atom_indices for key in collection.key_map.keys())
    assert assigned_indices == expected_indices

    # TODO: Figure out why the index types are not inferred correctly here.
    for key, param in collection.key_map.items():
        expected_smirks = expected_smirks_by_indices[key.atom_indices]  # type: ignore[index]
        assert param.id == expected_smirks, (
            f"For atom indices {key.atom_indices}, expected SMIRKS {expected_smirks}, "
            f"but got {param.id}."
        )


def test_proper_torsion_bend_energy(
    peroxide_molecule_info: dict[str, Molecule | openmm.unit.Quantity],
    proper_torsion_bend_force_field: ForceField,
):
    """Check that the ProperTorsionBend energy is as expected for peroxide."""
    topology = peroxide_molecule_info["molecule"].to_topology()
    interchange = Interchange.from_smirnoff(
        force_field=proper_torsion_bend_force_field, topology=topology
    )

    omm_system = interchange.to_openmm()
    positions = interchange.positions
    assert positions is not None  # Keep mypy happy

    raw_energies = _get_openmm_energies(
        system=omm_system,
        box_vectors=None,
        positions=positions.to_openmm(),
        round_positions=None,
        platform="Reference",
    )

    forces = omm_system.getForces()
    ptb_forces = [
        force for force in forces if force.getName() == "ProperTorsionBendForce"
    ]

    assert (
        len(ptb_forces) == 1
    ), "Expected exactly one ProperTorsionBend force in the OpenMM system."

    ptb_force = ptb_forces[0]
    num_ptb_terms = ptb_force.getNumBonds()
    ptb_force_idx = forces.index(ptb_force)
    ptb_energy = raw_energies[ptb_force_idx]

    # Expect 4 terms, as we have two H-O-O-H torsion-bends (one for each angle in the
    # torsion), each with two periodicities.
    assert (
        num_ptb_terms == 4
    ), f"Expected 4 ProperTorsionBend terms, but got {num_ptb_terms}."

    # Calculate expected energy:

    ptb_energy /= openmm.unit.kilocalorie_per_mole
    expected_ptb_energy = (
        peroxide_molecule_info["expected_ptb_energy"] / openmm.unit.kilocalorie_per_mole
    )

    assert pytest.approx(expected_ptb_energy, rel=1e-5) == ptb_energy, (
        f"Expected ProperTorsionBend energy to be {expected_ptb_energy} kJ/mol, "
        f"but got {ptb_energy} kJ/mol."
    )
