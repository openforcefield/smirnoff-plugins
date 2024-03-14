"""This script provides an example of how to construct a force field for a four site
water model which uses a custom Buckingham potential to describe the non-bonded vdW
interactions."""

import math

import numpy
import openmm.unit
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField, ParameterList
from openff.units import Quantity, unit

from smirnoff_plugins.utilities.openmm import simulate


def build_force_field() -> ForceField:
    """Construct the force field which contains the custom damped buckingham potential."""

    # Create a force field object which will store and apply the parameters. Here
    # `load_plugins` must be set to true in order to load the custom parameter handlers.
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
            "charge_increment1": 0.0 * unit.elementary_charge,
            "charge_increment2": 1.0552 * 0.5 * unit.elementary_charge,
            "charge_increment3": 1.0552 * 0.5 * unit.elementary_charge,
        }
    )
    virtual_site_handler._parameters = ParameterList(virtual_site_handler._parameters)

    # Finally add the custom buckingham charge handler.
    buckingham_handler = force_field.get_parameter_handler(
        "DampedBuckingham68",
        {
            "version": "0.3",
            "gamma": Quantity(35.8967 / unit.nanometer),
        },
    )
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


def main():
    # Build the custom force field and save it so we can inspect it later.
    force_field = build_force_field()
    force_field.to_file("buckingham-force-field.offxml")

    # Create a topology containing water molecules.
    molecule: Molecule = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")
    molecule.generate_conformers(n_conformers=1)

    n_molecules = 216

    topology: Topology = Topology.from_molecules([molecule] * n_molecules)

    # Create some coordinates (without the v-sites) and estimate box vectors.
    topology.box_vectors = Quantity(
        numpy.eye(3) * math.ceil(n_molecules ** (1 / 3) + 2) * 2.5,
        unit.angstrom,
    )

    positions = openmm.unit.Quantity(
        numpy.vstack(
            [
                (
                    molecule.conformers[0].m_as(unit.angstrom)
                    + numpy.array([[x, y, z]]) * 2.5
                )
                for x in range(math.ceil(n_molecules ** (1 / 3)))
                for y in range(math.ceil(n_molecules ** (1 / 3)))
                for z in range(math.ceil(n_molecules ** (1 / 3)))
            ]
        ),
        openmm.unit.angstrom,
    )

    # Simulate the water box.
    simulate(
        force_field=force_field,
        topology=topology,
        positions=positions,
        box_vectors=None if n_molecules == 1 else topology.box_vectors.to_openmm(),
        n_steps=2000,
        temperature=300.0,
        pressure=None if n_molecules == 1 else 1.0 * openmm.unit.atmosphere,
        platform="Reference" if n_molecules == 1 else "OpenCL",
        output_directory="simulation-output",
    )


if __name__ == "__main__":
    main()
