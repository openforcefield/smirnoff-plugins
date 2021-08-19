import pytest
from openff.toolkit.typing.engines.smirnoff import ForceField
from simtk import openmm, unit

from smirnoff_plugins.utilities.openmm import evaluate_water_energy_at_distances


@pytest.mark.parametrize(
    "switch_width, use_switch",
    [
        pytest.param(1 * unit.angstroms, True, id="Switch on"),
        pytest.param(0 * unit.angstroms, False, id="Switch off"),
    ],
)
def test_use_switch_width(
    water_box_topology, buckingham_water_force_field, switch_width, use_switch
):
    """Make sure the switch width is respected when requested"""

    buckingham_handler = buckingham_water_force_field.get_parameter_handler(
        "DampedBuckingham68"
    )
    buckingham_handler.switch_width = switch_width
    system = buckingham_water_force_field.create_openmm_system(water_box_topology)
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, openmm.CustomNonbondedForce):
            custom_force = force
            break
    assert custom_force.getUseSwitchingFunction() is use_switch


def test_switch_width(water_box_topology, buckingham_water_force_field):
    """Make sure the switch width is respected when set."""

    buckingham_handler = buckingham_water_force_field.get_parameter_handler(
        "DampedBuckingham68"
    )
    buckingham_handler.switch_width = 1.0 * unit.angstroms
    buckingham_handler.cutoff = 8.5 * unit.angstroms

    system = buckingham_water_force_field.create_openmm_system(water_box_topology)
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, openmm.CustomNonbondedForce):
            custom_force = force
            break

    # make sure it has been adjusted
    assert custom_force.getSwitchingDistance() == 7.5 * unit.angstroms


def test_double_exp_energies():
    """
    Make sure that energies computed using OpenMM match reference values calculated by hand for two O atoms in water at set distances.
    """
    epsilon = 0.152  # kcal/mol
    r_min = 3.5366  # angstrom
    alpha = 18.7
    beta = 3.3

    # build the force field with no charges
    ff = ForceField(load_plugins=True)
    constraint_handler = ff.get_parameter_handler("Constraints")
    constraint_handler.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0:2]-[#1]", "distance": 0.9572 * unit.angstrom}
    )
    constraint_handler.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0]-[#1:2]", "distance": 1.5139 * unit.angstrom}
    )
    # add a dummy vdW term
    vdw_handler = ff.get_parameter_handler("vdW")
    vdw_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
            "epsilon": 0.0 * unit.kilojoule_per_mole,
            "sigma": 1.0 * unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
            "epsilon": 0.0 * unit.kilojoules_per_mole,
            "sigma": 0.0 * unit.nanometers,
        }
    )
    ff.get_parameter_handler("Electrostatics")
    # add the library charges
    library_charge = ff.get_parameter_handler("LibraryCharges")
    library_charge.add_parameter(
        {"smirks": "[#1]-[#8X2H2+0:1]-[#1]", "charge1": 0 * unit.elementary_charge}
    )
    library_charge.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0]-[#1]", "charge1": 0 * unit.elementary_charge}
    )
    # Add the DE block
    double_exp = ff.get_parameter_handler("DoubleExponential")
    double_exp.alpha = alpha
    double_exp.beta = beta
    double_exp.add_parameter(
        {
            "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
            "r_min": r_min * unit.angstrom,
            "epsilon": epsilon * unit.kilocalorie_per_mole,
        }
    )
    double_exp.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
            "r_min": 1 * unit.angstrom,
            "epsilon": 0 * unit.kilocalorie_per_mole,
        }
    )

    energies = evaluate_water_energy_at_distances(
        force_field=ff, distances=[2, r_min, 4]
    )
    # calculated by hand, at r_min the energy should be epsilon
    ref_values = [457.0334854, -0.635968, -0.4893932627]
    for i, energy in enumerate(energies):
        assert energy == pytest.approx(ref_values[i])


def test_b68_energies():
    """Make sure that energies calculated using OpenMM match reference values calculated by hand for two O atoms in water at set distances"""

    # build the force field with no charges
    gamma = 35.8967
    a = 1600000.0
    b = 42
    c6 = 0.003
    c8 = 0.00003

    ff = ForceField(load_plugins=True)
    constraint_handler = ff.get_parameter_handler("Constraints")
    constraint_handler.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0:2]-[#1]", "distance": 0.9572 * unit.angstrom}
    )
    constraint_handler.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0]-[#1:2]", "distance": 1.5139 * unit.angstrom}
    )
    # add a dummy vdW term
    vdw_handler = ff.get_parameter_handler("vdW")
    vdw_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
            "epsilon": 0.0 * unit.kilojoule_per_mole,
            "sigma": 1.0 * unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
            "epsilon": 0.0 * unit.kilojoules_per_mole,
            "sigma": 0.0 * unit.nanometers,
        }
    )
    ff.get_parameter_handler("Electrostatics")
    # add the library charges
    library_charge = ff.get_parameter_handler("LibraryCharges")
    library_charge.add_parameter(
        {"smirks": "[#1]-[#8X2H2+0:1]-[#1]", "charge1": 0 * unit.elementary_charge}
    )
    library_charge.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0]-[#1]", "charge1": 0 * unit.elementary_charge}
    )
    # add the b68 block
    buckingham_handler = ff.get_parameter_handler("DampedBuckingham68")
    buckingham_handler.gamma = gamma * unit.nanometer ** -1
    buckingham_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
            "a": 0.0 * unit.kilojoule_per_mole,
            "b": 0.0 / unit.nanometer,
            "c6": 0.0 * unit.kilojoule_per_mole * unit.nanometer ** 6,
            "c8": 0.0 * unit.kilojoule_per_mole * unit.nanometer ** 8,
        }
    )
    buckingham_handler.add_parameter(
        {
            "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
            "a": a * unit.kilojoule_per_mole,
            "b": b / unit.nanometer,
            "c6": c6 * unit.kilojoule_per_mole * unit.nanometer ** 6,
            "c8": c8 * unit.kilojoule_per_mole * unit.nanometer ** 8,
        }
    )

    energies = evaluate_water_energy_at_distances(force_field=ff, distances=[2, 3, 4])
    # calculated by hand
    ref_values = [329.305, 1.303183, -0.686559]
    for i, energy in enumerate(energies):
        assert energy == pytest.approx(ref_values[i])
