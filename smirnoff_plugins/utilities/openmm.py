import logging
import math
import os
import time
from typing import Literal, Optional

import numpy
import openmm
import openmm.app
import openmm.unit
from openff.interchange import Interchange
from openff.interchange.interop.openmm._positions import to_openmm_positions
from openff.toolkit import ForceField, Molecule, Topology, unit
from openff.units.openmm import ensure_quantity
from openff.utilities import temporary_cd

logger = logging.getLogger(__name__)


def __simulate(
    positions: openmm.unit.Quantity,
    box_vectors: Optional[openmm.unit.Quantity],
    omm_topology: openmm.app.Topology,
    omm_system: openmm.System,
    n_steps: int,
    temperature: openmm.unit.Quantity,
    pressure: Optional[openmm.unit.Quantity],
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
        openmm.app.PDBFile.writeFile(omm_topology, positions, file)

    with open("system.xml", "w") as file:
        file.write(openmm.XmlSerializer.serialize(omm_system))

    if pressure is not None:
        omm_system.addForce(openmm.MonteCarloBarostat(pressure, temperature, 25))

    integrator = openmm.LangevinIntegrator(
        temperature,
        1.0 / openmm.unit.picosecond,
        0.5 * openmm.unit.femtoseconds,
    )

    try:
        simulation = openmm.app.Simulation(
            omm_topology,
            omm_system,
            integrator,
            openmm.Platform.getPlatformByName(platform),
        )
    except openmm.OpenMMException:
        logger.debug(
            f"Failed to use platform {platform}, trying again and letting OpenMM select platform."
        )
        simulation = openmm.app.Simulation(
            omm_topology,
            omm_system,
            integrator,
        )

    if box_vectors is not None:
        box_vectors = ensure_quantity(box_vectors, "openmm")
        simulation.context.setPeriodicBoxVectors(
            box_vectors[0], box_vectors[1], box_vectors[2]
        )

    simulation.context.setPositions(positions)
    simulation.context.computeVirtualSites()

    simulation.minimizeEnergy()

    # Randomize the velocities from a Boltzmann distribution at a given temperature.
    simulation.context.setVelocitiesToTemperature(temperature * openmm.unit.kelvin)

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
    positions: openmm.unit.Quantity,
    box_vectors: Optional[openmm.unit.Quantity],
    n_steps: int,
    temperature: openmm.unit.Quantity,
    pressure: Optional[openmm.unit.Quantity],
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

    topology.box_vectors = ensure_quantity(box_vectors, "openff")

    interchange = Interchange.from_smirnoff(
        force_field=force_field,
        topology=topology,
        positions=ensure_quantity(positions, "openff"),
    )

    openmm_system: openmm.System = interchange.to_openmm(combine_nonbonded_forces=False)
    with open("test.xml", "w") as file:
        file.write(openmm.XmlSerializer.serialize(openmm_system))
    openmm_topology: openmm.app.Topology = interchange.to_openmm_topology()
    openmm_positions: openmm.unit.Quantity = ensure_quantity(
        to_openmm_positions(
            interchange,
            include_virtual_sites=True,
        ),
        "openmm",
    )

    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)

    with temporary_cd(output_directory):
        __simulate(
            positions=openmm_positions,
            box_vectors=ensure_quantity(box_vectors, "openmm"),
            omm_topology=openmm_topology,
            omm_system=openmm_system,
            n_steps=n_steps,
            temperature=temperature,
            pressure=pressure,
            platform=platform,
        )


def water_box(n_molecules: int) -> tuple[Topology, openmm.unit.Quantity]:
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
                    molecule.conformers[0].value_in_unit(openmm.unit.angstrom)
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
        openmm.app.PDBFile.writeFile(topology.to_openmm(), positions, file)

    return topology, positions


def evaluate_water_energy_at_distances(
    force_field: ForceField, distances: list[float]
) -> list[float]:
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

    # Interchange doesn't think there's a way to combine PME electrostatics and cutoff vdw
    # (NonbondedForce and CustomNonbondedForce, respectively) without modifying the vdw force
    # to use NoCutoff, which as of today (2022-03-16) is still under discussion. For now, just
    # use an arbitrarily large box to mimic gas phase, even though this is modified below.
    topology.box_vectors = unit.Quantity(numpy.eye(3) * 20, unit.nanometer)

    # make the openmm system
    interchange = Interchange.from_smirnoff(force_field, topology)
    openmm_system = interchange.to_openmm(combine_nonbonded_forces=False)
    # workaround interchange by setting nonbonded method to NoCutoff
    forces = {force.__class__.__name__: force for force in openmm_system.getForces()}
    vdw_force: openmm.CustomNonbondedForce = forces["CustomNonbondedForce"]
    vdw_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    vdw_force.setUseSwitchingFunction(False)
    vdw_force.setUseLongRangeCorrection(False)
    nonbond: openmm.NonbondedForce = forces["NonbondedForce"]
    nonbond.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    openmm_topology = interchange.to_openmm_topology()
    openmm_positions: openmm.unit.Quantity = ensure_quantity(
        to_openmm_positions(
            interchange,
            include_virtual_sites=True,
        ),
        "openmm",
    )

    integrator = openmm.LangevinIntegrator(
        300 * openmm.unit.kelvin,
        1.0 / openmm.unit.picosecond,
        0.01 * openmm.unit.femtoseconds,
    )

    platform = openmm.Platform.getPlatformByName("CPU")

    simulation = openmm.app.Simulation(
        openmm_topology, openmm_system, integrator, platform
    )

    n_positions_per_water = int(openmm_positions.shape[0] / 2)

    energies = []
    for distance in distances:

        translated_positons = numpy.vstack(
            [
                openmm_positions[:3, :].value_in_unit(
                    openmm.unit.angstrom,
                ),
                openmm_positions[:3, :].value_in_unit(
                    openmm.unit.angstrom,
                )
                # only translate the second water in x
                + numpy.array([distance, 0, 0]),
            ]
        )
        if n_positions_per_water > 3:
            # add zeros to pad the positions
            vsites = numpy.zeros((2 * (n_positions_per_water - 3), 3))
            new_positions = openmm.unit.Quantity(
                numpy.vstack([translated_positons, vsites]),
                openmm.unit.angstrom,
            )
        else:
            new_positions = translated_positons * openmm.unit.angstrom

        simulation.context.setPositions(
            new_positions.value_in_unit(openmm.unit.nanometer)
        )
        simulation.context.computeVirtualSites()
        state = simulation.context.getState(getEnergy=True)

        energies.append(
            state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        )

    return energies


def evaluate_energy(
    system: openmm.System,
    topology: openmm.app.Topology,
    positions: openmm.unit.Quantity,
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
        300 * openmm.unit.kelvin,  # simulation temperature,
        1.0 / openmm.unit.picosecond,  # friction
        2.0 * openmm.unit.femtoseconds,  # simulation timestep
    )

    platform = openmm.Platform.getPlatformByName("Reference")

    simulation = openmm.app.Simulation(topology, system, integrator, platform)
    # assume the positions are already padded.
    simulation.context.setPositions(positions)
    simulation.context.computeVirtualSites()
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
