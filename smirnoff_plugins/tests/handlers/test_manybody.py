import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import openmm, unit


def test_axilrod_basic():
    toluene = Molecule.from_mapped_smiles(
        "[H:10][c:3]1[c:2]([c:1]([c:6]([c:5]([c:4]1[H:11])[H:12])[C:7]([H:13])([H:14])[H:15])[H:8])[H:9]"
    )

    top = Topology.from_molecules([toluene])
    ff = ForceField(load_plugins=True)
    ff.get_parameter_handler("ToolkitAM1BCC")
    ath = ff.get_parameter_handler("AxilrodTeller")
    ath.add_parameter(
        {"smirks": "[#1:1]", "c9": "1.0e-6 * kilojoule_per_mole * nanometer**9"}
    )
    ath.add_parameter(
        {"smirks": "[#6:1]", "c9": "2.0e-6 * kilojoule_per_mole * nanometer**9"}
    )

    sys = ff.create_openmm_system(top)

    at_forces = [
        sys.getForce(i)
        for i in range(sys.getNumForces())
        if isinstance(sys.getForce(i), openmm.CustomManyParticleForce)
    ]
    atf = at_forces[0]
    assert len(at_forces) == 1
    assert atf.getNumParticles() == 15
    c_c9 = 2.0e-6 * unit.kilojoule_per_mole * unit.nanometer**9
    h_c9 = 1.0e-6 * unit.kilojoule_per_mole * unit.nanometer**9
    expected_c9 = [c_c9] * 7 + [h_c9] * 8
    for particle_idx in range(atf.getNumParticles()):
        parameters = atf.getParticleParameters(particle_idx)
        assert expected_c9[particle_idx]._value == parameters[0][0]
