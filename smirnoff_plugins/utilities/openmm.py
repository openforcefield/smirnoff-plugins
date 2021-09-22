import logging
import math
import os
import time
from typing import List, Literal, Optional, Tuple

import numpy
from openff.toolkit.topology import Molecule, Topology, TopologyAtom
from openff.toolkit.typing.engines.smirnoff import ForceField
from simtk import openmm, unit
from simtk.openmm import app

from smirnoff_plugins.utilities import temporary_cd

logger = logging.getLogger(__name__)


def __simulate(
    positions: unit.Quantity,
    box_vectors: Optional[unit.Quantity],
    omm_topology: app.Topology,
    omm_system: openmm.System,
    n_steps: int,
    temperature: unit.Quantity,
    pressure: Optional[unit.Quantity],
    platform: Literal["Reference", "OpenCL", "CUDA", "CPU"] = "Reference",
):
    """

    Parameters
    ----------
    positions
        The starting coordinates of the molecules in the system.
    box_vectors
        The box vectors to use. These will overwrite the topology box vectors.
    omm_topology
        The topology detailing the system to simulate.
    omm_system
        The object which defines the systems hamiltonian.
    n_steps
        The number of steps to simulate for.
    temperature
        The temperature to simulate at.
    pressure
        The pressure to simulate at.
    platform
        The platform to simulate using.
    """

    """A helper function for simulating a system with OpenMM."""

    with open("input.pdb", "w") as file:
        app.PDBFile.writeFile(omm_topology, positions, file)

    with open("system.xml", "w") as file:
        file.write(openmm.XmlSerializer.serialize(omm_system))

    if pressure is not None:
        omm_system.addForce(openmm.MonteCarloBarostat(pressure, temperature, 25))

    integrator = openmm.LangevinIntegrator(
        temperature,  # simulation temperature,
        1.0 / unit.picosecond,  # friction
        2.0 * unit.femtoseconds,  # simulation timestep
    )

    platform = openmm.Platform.getPlatformByName(platform)

    simulation = app.Simulation(omm_topology, omm_system, integrator, platform)

    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(
            box_vectors[0, :], box_vectors[1, :], box_vectors[2, :]
        )

    simulation.context.setPositions(positions)
    simulation.context.computeVirtualSites()

    simulation.minimizeEnergy()

    # Randomize the velocities from a Boltzmann distribution at a given temperature.
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)

    # Configure the information in the output files.
    pdb_reporter = openmm.app.DCDReporter("trajectory.dcd", int(0.05 * n_steps))

    state_data_reporter = openmm.app.StateDataReporter(
        "data.csv",
        int(0.05 * n_steps),
        step=True,
        potentialEnergy=True,
        temperature=True,
        density=True,
    )
    simulation.reporters.append(pdb_reporter)
    simulation.reporters.append(state_data_reporter)

    logger.debug("Starting simulation")
    start = time.process_time()

    # Run the simulation
    simulation.step(n_steps)

    end = time.process_time()
    logger.debug("Elapsed time %.2f seconds" % (end - start))
    logger.debug("Done!")


def simulate(
    force_field: ForceField,
    topology: Topology,
    positions: unit.Quantity,
    box_vectors: Optional[unit.Quantity],
    n_steps: int,
    temperature: unit.Quantity,
    pressure: Optional[unit.Quantity],
    platform: Literal["Reference", "OpenCL", "CUDA", "CPU"] = "Reference",
    output_directory: Optional[str] = None,
):
    """A helper function for simulating a system parameterised with a specific OpenFF
    force field using OpenMM.

    Parameters
    ----------
    force_field
        The force field to apply.
    topology
        The topology detailing the system to simulate.
    positions
        The starting coordinates of the molecules in the system.
    box_vectors
        The box vectors to use. These will overwrite the topology box vectors.
    n_steps
        The number of steps to simulate for.
    temperature
        The temperature to simulate at.
    pressure
        The pressure to simulate at.
    platform
        The platform to simulate using.
    output_directory
        The optional directory to store the simulation outputs in.
    """

    assert pressure is None or (
        pressure is not None and box_vectors is not None
    ), "box vectors must be provided when the pressure is specified."

    topology.box_vectors = box_vectors

    # Create an OpenMM system by applying the parameters to the topology.
    omm_system, topology = force_field.create_openmm_system(
        topology, return_topology=True
    )

    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)

    # Add the virtual sites to the OpenMM topology and positions.
    omm_topology = topology.to_openmm()

    omm_chain = [*omm_topology.chains()][0]
    omm_residue = omm_topology.addResidue("", chain=omm_chain)

    for particle in topology.topology_particles:

        if isinstance(particle, TopologyAtom):
            continue

        omm_topology.addAtom(
            particle.virtual_site.name, app.Element.getByMass(0), omm_residue
        )

    positions = numpy.vstack(
        [positions, numpy.zeros((topology.n_topology_virtual_sites, 3))]
    )

    with temporary_cd(output_directory):

        __simulate(
            positions=positions * unit.angstrom,
            box_vectors=box_vectors,
            omm_topology=omm_topology,
            omm_system=omm_system,
            n_steps=n_steps,
            temperature=temperature,
            pressure=pressure,
            platform=platform,
        )


def water_box(n_molecules: int) -> Tuple[Topology, unit.Quantity]:
    """
    Build a water box with the requested number of water molecules.

    Parameters
    ----------
    n_molecules
        The number of water molecules that should be put into the water box

    Returns
    -------
        The openff.toolkit Topology of the system and the position array wrapped with units.
    """

    # Create a topology containing water molecules.
    molecule = Molecule.from_smiles("O")
    molecule.generate_conformers(n_conformers=1)

    topology = Topology.from_molecules([molecule] * n_molecules)

    topology.box_vectors = (
        numpy.eye(3) * math.ceil(n_molecules ** (1 / 3) + 2) * 2.5 * unit.angstrom
    )

    positions = (
        numpy.vstack(
            [
                (
                    molecule.conformers[0].value_in_unit(unit.angstrom)
                    + numpy.array([[x, y, z]]) * 2.5
                )
                for x in range(math.ceil(n_molecules ** (1 / 3)))
                for y in range(math.ceil(n_molecules ** (1 / 3)))
                for z in range(math.ceil(n_molecules ** (1 / 3)))
            ]
        )[: topology.n_topology_atoms, :]
        * unit.angstrom
    )

    with open("input.pdb", "w") as file:
        app.PDBFile.writeFile(topology.to_openmm(), positions, file)

    return topology, positions


def evaluate_water_energy_at_distances(
    force_field: ForceField, distances: List[float]
) -> List[float]:
    """
    Evaluate the energy of a system of two water molecules at the requested distances using the provided force field.

    Parameters
    ----------
    force_field:
        The openff.toolkit force field object that should be used to parameterise the system.
    distances:
        The list of absolute distances between the oxygen atoms in angstroms.

    Returns
    -------
        A list of energies evaluated at the given distances in kj/mol
    """

    # build the topology
    water = Molecule.from_smiles("O")
    water.generate_conformers(n_conformers=1)
    topology = Topology.from_molecules([water, water])

    # make the openmm system
    omm_system, topology = force_field.create_openmm_system(
        topology, return_topology=True
    )
    # generate positions at the requested distance
    positions = [
        numpy.vstack(
            [
                water.conformers[0].value_in_unit(unit.angstrom),
                water.conformers[0].value_in_unit(unit.angstrom)
                + numpy.array([x, 0, 0]),
            ]
        )
        * unit.angstrom
        for x in distances
    ]
    # Add the virtual sites to the OpenMM topology and positions.
    omm_topology = topology.to_openmm()
    omm_chain = [*omm_topology.chains()][-1]
    omm_residue = omm_topology.addResidue("", chain=omm_chain)

    for particle in topology.topology_particles:

        if isinstance(particle, TopologyAtom):
            continue

        omm_topology.addAtom(
            particle.virtual_site.name, app.Element.getByMass(0), omm_residue
        )

    positions = [
        numpy.vstack([p, numpy.zeros((topology.n_topology_virtual_sites, 3))])
        * unit.angstrom
        for p in positions
    ]

    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin,  # simulation temperature,
        1.0 / unit.picosecond,  # friction
        2.0 * unit.femtoseconds,  # simulation timestep
    )

    platform = openmm.Platform.getPlatformByName("CPU")

    simulation = app.Simulation(omm_topology, omm_system, integrator, platform)

    energies = []
    for i, p in enumerate(positions):
        simulation.context.setPositions(p)
        simulation.context.computeVirtualSites()
        state = simulation.context.getState(getEnergy=True)
        energies.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )

    return energies


def evaluate_energy(
    system: openmm.System, topology: app.Topology, positions: unit.Quantity
) -> float:
    """
    For the given openmm system build a simulation and evaluate the energies.

    Parameters
    ----------
    system:
        The openmm system that should be used to evaluate the energies.
    topology:
        The openmm topology that should be used to build the simulation object.
    positions:
        The positions that should be used when evaluating the energies.


    Returns
    -------
        The energy in kcal/mol,
    """

    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin,  # simulation temperature,
        1.0 / unit.picosecond,  # friction
        2.0 * unit.femtoseconds,  # simulation timestep
    )

    platform = openmm.Platform.getPlatformByName("CPU")

    simulation = app.Simulation(topology, system, integrator, platform)
    # assume the positions are already padded.
    simulation.context.setPositions(positions)
    simulation.context.computeVirtualSites()
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
