import pytest
from simtk import openmm, unit


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
