from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.topology import Molecule
from openmm import unit

def test_espaloma():
    ff = ForceField('openff-2.0.0.offxml', load_plugins=True)
    ff.deregister_parameter_handler('ToolkitAM1BCC')
    ff.get_parameter_handler('Espaloma')
    molecule = Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    sys = ff.create_openmm_system(molecule.to_topology())

    forces = {
                sys.getForce(index).__class__.__name__: sys.getForce(index)
                for index in range(sys.getNumForces())
            }
    charges = [forces['NonbondedForce'].getParticleParameters(i)[0] for i in range(molecule.n_atoms)]
    assert sum(charges, 0. * unit.elementary_charge) == 0. * unit.elementary_charge
