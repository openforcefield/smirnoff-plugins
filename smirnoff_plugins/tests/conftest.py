import math

import numpy
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField, ParameterList
from simtk import unit


@pytest.fixture()
def water() -> Molecule:
    return Molecule.from_smiles("O")


@pytest.fixture()
def buckingham_water_force_field() -> ForceField:
    """Create a buckingham water model Forcefield object."""

    force_field = ForceField(load_plugins=True)

    # Add in a constraint handler to ensure the correct H-O-H geometry.
    constraint_handler = force_field.get_parameter_handler("Constraints")
    # Keep the H-O bond length fixed at 0.9572 angstroms.
    constraint_handler.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0:2]-[#1]", "distance": 0.9572 * unit.angstrom}
    )
    # Keep the H-O-H angle fixed at 104.52 degrees.
    constraint_handler.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0]-[#1:2]", "distance": 1.5139 * unit.angstrom}
    )

    # Add a default vdW handler which is currently required by the OFF TK.
    vdw_handler = force_field.get_parameter_handler("vdW")
    vdw_handler.add_parameter(
        {
            "smirks": "[*:1]",
            "epsilon": 0.0 * unit.kilojoule_per_mole,
            "sigma": 1.0 * unit.angstrom,
        }
    )

    # Add a charge handler to zero the charges on water. The charges will be
    # applied by the virtual site handler instead.
    force_field.get_parameter_handler("Electrostatics")

    force_field.get_parameter_handler(
        "ChargeIncrementModel",
        {"version": "0.3", "partial_charge_method": "formal_charge"},
    )

    # Add a virtual site handler to add the virtual charge site.
    virtual_site_handler = force_field.get_parameter_handler("VirtualSites")
    virtual_site_handler.add_parameter(
        {
            "smirks": "[#1:2]-[#8X2H2+0:1]-[#1:3]",
            "type": "DivalentLonePair",
            "distance": -0.0106 * unit.nanometers,
            "outOfPlaneAngle": 0.0 * unit.degrees,
            "match": "once",
            "charge_increment1": 1.0552 * 0.5 * unit.elementary_charge,
            "charge_increment2": 0.0 * unit.elementary_charge,
            "charge_increment3": 1.0552 * 0.5 * unit.elementary_charge,
        }
    )
    virtual_site_handler._parameters = ParameterList(virtual_site_handler._parameters)

    # Finally add the custom buckingham charge handler.
    buckingham_handler = force_field.get_parameter_handler("DampedBuckingham68")
    buckingham_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
            "a": 0.0 * unit.kilojoule_per_mole,
            "b": 0.0 / unit.nanometer,
            "c6": 0.0 * unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": 0.0 * unit.kilojoule_per_mole * unit.nanometer**8,
        }
    )
    buckingham_handler.add_parameter(
        {
            "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
            "a": 1600000.0 * unit.kilojoule_per_mole,
            "b": 42.00 / unit.nanometer,
            "c6": 0.003 * unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": 0.00003 * unit.kilojoule_per_mole * unit.nanometer**8,
        }
    )
    return force_field


@pytest.fixture()
def water_box_topology() -> Topology:
    mol = Molecule.from_smiles("O")
    mol.generate_conformers()

    n_molecules = 256

    topology: Topology = Topology.from_molecules([mol] * n_molecules)

    # Create some coordinates (without the v-sites) and estimate box vectors.
    topology.box_vectors = (
        numpy.eye(3) * math.ceil(n_molecules ** (1 / 3) + 2) * 2.5 * unit.angstrom
    )

    return topology


@pytest.fixture()
def ideal_water_force_field() -> ForceField:
    """Returns a force field that will assign constraints, a vdW handler and
    a library charge handler to a three site water molecule with all LJ
    ``epsilon=0.0`` and all ``q=0.0``.
    """
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
            "smirks": "[*:1]",
            "epsilon": 0.0 * unit.kilojoule_per_mole,
            "sigma": 1.0 * unit.angstrom,
        }
    )
    # add the library charges
    library_charge = ff.get_parameter_handler("LibraryCharges")
    library_charge.add_parameter(
        {"smirks": "[#1]-[#8X2H2+0:1]-[#1]", "charge1": 0 * unit.elementary_charge}
    )
    library_charge.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0]-[#1]", "charge1": 0 * unit.elementary_charge}
    )

    return ff
