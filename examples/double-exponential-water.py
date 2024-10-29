"""This script provides an example of how to construct a force field for a four site
water model which uses a custom Buckingham potential to describe the non-bonded vdW
interactions."""

import math

import numpy
import openmm.unit
from openff.toolkit import ForceField, Molecule, Quantity, Topology, unit
from openff.utilities import get_data_file_path

from smirnoff_plugins.utilities.openmm import simulate


def main():
    force_field = ForceField(
        get_data_file_path("_tests/data/de-vs-1.0.1.offxml", "smirnoff_plugins"),
        load_plugins=True,
    )

    # Create a topology containing water molecules.
    molecule = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")
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
        box_vectors=(
            Quantity(2 * numpy.eye(3), "nanometer").to_openmm()
            if n_molecules == 1
            else topology.box_vectors.to_openmm()
        ),
        n_steps=2000,
        temperature=300.0,
        pressure=None if n_molecules == 1 else 1.0 * openmm.unit.atmosphere,
        platform="Reference" if n_molecules == 1 else "OpenCL",
        output_directory="simulation-output",
    )


if __name__ == "__main__":
    main()
