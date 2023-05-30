import openmm
import openmm.unit
import pytest
from openff.interchange import Interchange
from openff.toolkit import Topology
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm

from smirnoff_plugins.utilities.openmm import (
    evaluate_energy,
    evaluate_water_energy_at_distances,
)


def test_vsite_exclusions(buckingham_water_force_field, water_box_topology):
    """Make sure the exclusions/exceptions for vsites match in the Nonbonded and Custom Nonbonded force"""

    system = buckingham_water_force_field.create_interchange(
        water_box_topology
    ).to_openmm(combine_nonbonded_forces=False)
    # check we have the same number of exclusions and exceptions
    nonbonded_force = [
        force
        for force in system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ][0]
    custom_force = [
        force
        for force in system.getForces()
        if isinstance(force, openmm.CustomNonbondedForce)
    ][0]
    assert nonbonded_force.getNumExceptions() == custom_force.getNumExclusions()


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
    system = buckingham_water_force_field.create_interchange(
        water_box_topology
    ).to_openmm(combine_nonbonded_forces=False)

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

    system = buckingham_water_force_field.create_interchange(
        water_box_topology
    ).to_openmm(combine_nonbonded_forces=False)
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, openmm.CustomNonbondedForce):
            custom_force = force
            break

    # make sure it has been adjusted
    assert custom_force.getSwitchingDistance() == 7.5 * openmm.unit.angstroms


def test_double_exp_energies(ideal_water_force_field):
    """
    Make sure that energies computed using OpenMM match reference values calculated by hand for two O atoms in water at set distances.
    """
    epsilon = 0.152  # kcal/mol
    r_min = 3.5366  # angstrom
    alpha = 18.7
    beta = 3.3

    # Add the DE block
    double_exp = ideal_water_force_field.get_parameter_handler("DoubleExponential")
    double_exp.cutoff = 20 * unit.angstrom
    double_exp.switch_width = 0 * unit.angstrom
    double_exp.alpha = alpha
    double_exp.beta = beta
    double_exp.scale14 = 1
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
        force_field=ideal_water_force_field, distances=[2, r_min, 4]
    )

    # calculated by hand (kJ / mol), at r_min the energy should be epsilon
    ref_values = [457.0334854, -0.635968, -0.4893932627]

    for i, energy in enumerate(energies):
        assert energy == pytest.approx(ref_values[i])


def test_b68_energies(ideal_water_force_field):
    """Make sure that energies calculated using OpenMM match reference values calculated by hand for two O atoms in water at set distances"""

    # build the force field with no charges
    gamma = 35.8967
    a = 1600000.0
    b = 42
    c6 = 0.003
    c8 = 0.00003

    # add the b68 block
    buckingham_handler = ideal_water_force_field.get_parameter_handler(
        "DampedBuckingham68"
    )
    buckingham_handler.gamma = gamma * unit.nanometer**-1
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
            "a": a * unit.kilojoule_per_mole,
            "b": b / unit.nanometer,
            "c6": c6 * unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": c8 * unit.kilojoule_per_mole * unit.nanometer**8,
        }
    )

    energies = evaluate_water_energy_at_distances(
        force_field=ideal_water_force_field, distances=[2, 3, 4]
    )
    # calculated by hand (kJ / mol)
    ref_values = [329.30542, 1.303183, -0.686559]
    for i, energy in enumerate(energies):
        assert energy == pytest.approx(ref_values[i], abs=1e-5)


def test_scaled_de_energy():
    """For a molecule with 1-4 interactions make sure the scaling is correctly applied.
    Note that only nonbonded parameters are non zero.
    """

    ff = ForceField(load_plugins=True)
    ff.get_parameter_handler("Electrostatics")

    ff.get_parameter_handler(
        "ChargeIncrementModel",
        {"version": "0.3", "partial_charge_method": "formal_charge"},
    )

    double_exp = ff.get_parameter_handler("DoubleExponential")
    double_exp.alpha = 18.7
    double_exp.beta = 3.3
    double_exp.scale14 = 1
    double_exp.add_parameter(
        {
            "smirks": "[#6X4:1]",
            "r_min": 3.816 * unit.angstrom,
            "epsilon": 0.1094 * unit.kilocalorie_per_mole,
        }
    )
    double_exp.add_parameter(
        {
            "smirks": "[#1:1]-[#6X4]",
            "r_min": 2.974 * unit.angstrom,
            "epsilon": 0.0157 * unit.kilocalorie_per_mole,
        }
    )

    ethane = Molecule.from_smiles("CC")
    ethane.generate_conformers(n_conformers=1)
    off_top = ethane.to_topology()

    # A comically large box is needed to avoid image interactions; a better solution would involve
    # exactly how gas phase calculations are done with vdW method "cutoff"
    off_top.box_vectors = [20, 20, 20] * unit.nanometer

    omm_top = off_top.to_openmm()
    system_no_scale = Interchange.from_smirnoff(ff, off_top).to_openmm(
        combine_nonbonded_forces=False
    )
    energy_no_scale = evaluate_energy(
        system=system_no_scale,
        topology=omm_top,
        positions=ethane.conformers[0].to_openmm(),
    )

    # now scale 1-4 by half
    double_exp.scale14 = 0.5
    system_scaled = Interchange.from_smirnoff(ff, off_top).to_openmm(
        combine_nonbonded_forces=False
    )
    energy_scaled = evaluate_energy(
        system=system_scaled,
        topology=omm_top,
        positions=ethane.conformers[0].to_openmm(),
    )
    assert double_exp.scale14 * energy_no_scale == pytest.approx(
        energy_scaled,
        abs=1e-4,
    )


def test_dampedexp6810_assignment():
    ff = ForceField(load_plugins=True)

    ff.get_parameter_handler(
        "Electrostatics",
        {
            "version": "0.4",
            "periodic_potential": "Ewald3D-ConductingBoundary",
            "nonperiodic_potential": "Coulomb",
            "exception_potential": "Coulomb",
        },
    )
    ff.get_parameter_handler(
        "ChargeIncrementModel",
        {"version": "0.3", "partial_charge_method": "formal_charge"},
    )

    handler = ff.get_parameter_handler("DampedExp6810", {"version": "0.3"})

    handler.add_parameter(
        {
            "smirks": "[#1:1]",
            "rho": 1.5 * unit.angstrom,
            "beta": 3.0 * unit.angstrom**-1,
            "c6": 1.0 * unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": 10.0 * unit.kilojoule_per_mole * unit.nanometer**8,
            "c10": 100.0 * unit.kilojoule_per_mole * unit.nanometer**10,
        }
    )

    handler.add_parameter(
        {
            "smirks": "[#6:1]",
            "rho": 3.0 * unit.angstrom,
            "beta": 3.0 * unit.angstrom**-1,
            "c6": 10.0 * unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": 100.0 * unit.kilojoule_per_mole * unit.nanometer**8,
            "c10": 1000.0 * unit.kilojoule_per_mole * unit.nanometer**10,
        }
    )

    toluene = Molecule.from_mapped_smiles(
        "[H:10][c:3]1[c:2]([c:1]([c:6]([c:5]([c:4]1[H:11])[H:12])[C:7]([H:13])([H:14])[H:15])[H:8])[H:9]"
    )
    toluene.generate_conformers(n_conformers=1)
    off_top = toluene.to_topology()
    off_top.box_vectors = [10, 10, 10] * unit.nanometer

    interchange = Interchange.from_smirnoff(ff, off_top)
    omm_system = interchange.to_openmm(combine_nonbonded_forces=False)

    custom_nonbonded_forces = [
        omm_system.getForce(i)
        for i in range(omm_system.getNumForces())
        if isinstance(omm_system.getForce(i), openmm.CustomNonbondedForce)
    ]

    assert len(custom_nonbonded_forces) == 1

    custom_nonbonded_force: openmm.CustomNonbondedForce = custom_nonbonded_forces[0]

    assert custom_nonbonded_force.getNumParticles() == 15

    h_params = [
        0.15 * unit.nanometer,
        30.0 * unit.nanometer**-1,
        1.0 * unit.kilojoule_per_mole * unit.nanometer**6,
        10.0 * unit.kilojoule_per_mole * unit.nanometer**8,
        100.0 * unit.kilojoule_per_mole * unit.nanometer**10,
    ]

    c_params = [
        0.3 * unit.angstrom,
        30.0 * unit.angstrom**-1,
        10.0 * unit.kilojoule_per_mole * unit.nanometer**6,
        100.0 * unit.kilojoule_per_mole * unit.nanometer**8,
        1000.0 * unit.kilojoule_per_mole * unit.nanometer**10,
    ]

    expected_params = [c_params] * 7 + [h_params] * 8

    for particle_idx in range(custom_nonbonded_force.getNumParticles()):
        parameters = custom_nonbonded_force.getParticleParameters(particle_idx)
        for param, expected_param in zip(parameters, expected_params[particle_idx]):
            assert expected_param.m == param


def test_dampedexp6810_energies():
    ff = ForceField(load_plugins=True)

    handler = ff.get_parameter_handler("DampedExp6810", {"version": "0.3"})

    handler.add_parameter(
        {
            "smirks": "[#10:1]",
            "rho": 2.802400 * unit.angstrom,
            "beta": 4.994320 * unit.angstrom**-1,
            "c6": 3.581767e-04 * unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": 1.097581e-05 * unit.kilojoule_per_mole * unit.nanometer**8,
            "c10": 4.120140e-07 * unit.kilojoule_per_mole * unit.nanometer**10,
        }
    )

    ff.get_parameter_handler("Electrostatics")
    library_charge = ff.get_parameter_handler("LibraryCharges")
    library_charge.add_parameter(
        {"smirks": "[#10:1]", "charge1": 0 * unit.elementary_charge}
    )

    neon = Molecule.from_smiles("[Ne]")
    neon.generate_conformers(n_conformers=1)
    off_top = neon.to_topology()
    off_top.add_molecule(neon)
    off_top.box_vectors = [10, 10, 10] * unit.nanometer

    interchange = Interchange.from_smirnoff(ff, off_top)
    omm_system = interchange.to_openmm(combine_nonbonded_forces=False)

    custom_nonbonded_forces = [
        omm_system.getForce(i)
        for i in range(omm_system.getNumForces())
        if isinstance(omm_system.getForce(i), openmm.CustomNonbondedForce)
    ]

    assert len(custom_nonbonded_forces) == 1

    custom_nonbonded_force: openmm.CustomNonbondedForce = custom_nonbonded_forces[0]

    assert custom_nonbonded_force.getNumParticles() == 2

    custom_nonbonded_force.setUseLongRangeCorrection(False)

    distances = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    energies = [  # calculated by hand
        45.2528743017942,
        2.1502534217325,
        -0.339669883563026,
        -0.105512197509814,
        -0.026137404257782,
        -0.00839846163369,
        -0.0032494303178,
        -0.001435593777657,
    ] * unit.kilojoule_per_mole

    omm_integrator: openmm.LangevinMiddleIntegrator = openmm.LangevinMiddleIntegrator(
        298, 1.0, 0.002
    )
    omm_simulation: openmm.app.Simulation = openmm.app.Simulation(
        off_top.to_openmm(),
        omm_system,
        omm_integrator,
    )
    omm_context: openmm.Context = omm_simulation.context

    for energy, distance in zip(energies, distances):
        omm_context.setPositions(
            to_openmm([[0, 0, 0], [distance, 0, 0]] * unit.angstrom)
        )
        omm_state: openmm.State = omm_context.getState(getEnergy=True)
        assert from_openmm(omm_state.getPotentialEnergy()).m == pytest.approx(energy)


def test_axilrodteller_assignment():
    ff = ForceField(load_plugins=True)

    handler = ff.get_parameter_handler("AxilrodTeller")

    handler.add_parameter(
        {
            "smirks": "[#1:1]",
            "c9": 1.0 * unit.kilojoule_per_mole * unit.angstrom**9,
        }
    )

    handler.add_parameter(
        {
            "smirks": "[#6:1]",
            "c9": 10.0 * unit.kilojoule_per_mole * unit.angstrom**9,
        }
    )

    handler.add_parameter(
        {
            "smirks": "[#8:1]",
            "c9": 5.0 * unit.kilojoule_per_mole * unit.angstrom**9,
        }
    )

    toluene = Molecule.from_mapped_smiles(
        "[H:10][c:3]1[c:2]([c:1]([c:6]([c:5]([c:4]1[H:11])[H:12])[C:7]([H:13])([H:14])[H:15])[H:8])[H:9]"
    )
    toluene.generate_conformers(n_conformers=1)
    off_top = toluene.to_topology()
    off_top.add_molecule(toluene)

    interchange = Interchange.from_smirnoff(ff, off_top)
    omm_system = interchange.to_openmm(combine_nonbonded_forces=False)

    forces = [
        omm_system.getForce(i)
        for i in range(omm_system.getNumForces())
        if isinstance(omm_system.getForce(i), openmm.CustomManyParticleForce)
    ]

    assert len(forces) == 1

    force: openmm.CustomManyParticleForce = forces[0]

    assert force.getNumParticles() == 30

    c_param = 10.0 * unit.kilojoule_per_mole * unit.angstrom**9
    h_param = 1.0 * unit.kilojoule_per_mole * unit.angstrom**9
    expected_params = [c_param] * 7 + [h_param] * 8 + [c_param] * 7 + [h_param] * 8

    for atom_idx in range(force.getNumParticles()):
        expected_param = expected_params[atom_idx]
        actual_param = force.getParticleParameters(atom_idx)[0][0]
        assert pytest.approx(actual_param) == expected_param.m_as(
            "kilojoule_per_mole * nanometer ** 9"
        )


def test_axilrodteller_energies():
    ff = ForceField(load_plugins=True)

    de6810_handler = ff.get_parameter_handler(
        "DampedExp6810", {"version": "0.3", "method": "cutoff"}
    )

    de6810_handler.add_parameter(
        {
            "smirks": "[#10:1]",
            "rho": 0.0 * unit.angstrom,
            "beta": 0.0 * unit.angstrom**-1,
            "c6": 0.0 * unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": 0.0 * unit.kilojoule_per_mole * unit.nanometer**8,
            "c10": 0.0 * unit.kilojoule_per_mole * unit.nanometer**10,
        }
    )

    ff.get_parameter_handler("Electrostatics")
    library_charge = ff.get_parameter_handler("LibraryCharges")
    library_charge.add_parameter(
        {"smirks": "[#10:1]", "charge1": 0 * unit.elementary_charge}
    )

    axilrod_handler = ff.get_parameter_handler(
        "AxilrodTeller",
        {"version": "0.3", "method": "cutoff_periodic", "cutoff": "2 * nanometer"},
    )
    axilrod_handler.add_parameter(
        {"smirks": "[#10:1]", "c9": 0.1 * unit.kilojoule_per_mole * unit.nanometer**9}
    )

    neon = Molecule.from_smiles("[Ne]")
    neon.generate_conformers(n_conformers=1)
    off_top = neon.to_topology()
    off_top.add_molecule(neon)
    off_top.add_molecule(neon)
    off_top.box_vectors = [10, 10, 10] * unit.nanometer

    interchange = Interchange.from_smirnoff(ff, off_top)
    omm_system = interchange.to_openmm(combine_nonbonded_forces=False)

    custom_manyp_forces = [
        omm_system.getForce(i)
        for i in range(omm_system.getNumForces())
        if isinstance(omm_system.getForce(i), openmm.CustomManyParticleForce)
    ]

    assert len(custom_manyp_forces) == 1

    custom_manyp_force: openmm.CustomManyParticleForce = custom_manyp_forces[0]

    assert custom_manyp_force.getNumParticles() == 3

    distances = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
    energies = [
        -2 / 3 * 0.1 / ((distance / 10) ** 6 * (2 * distance / 10) ** 3)
        for distance in distances
    ] * unit.kilojoule_per_mole

    omm_integrator: openmm.LangevinMiddleIntegrator = openmm.LangevinMiddleIntegrator(
        298, 1.0, 0.002
    )
    omm_simulation: openmm.app.Simulation = openmm.app.Simulation(
        off_top.to_openmm(), omm_system, omm_integrator
    )
    omm_context: openmm.Context = omm_simulation.context

    for energy, distance in zip(energies, distances):
        omm_context.setPositions(
            to_openmm(
                [[0, 0, 0], [distance, 0, 0], [2 * distance, 0, 0]] * unit.angstrom
            )
        )
        omm_state: openmm.State = omm_context.getState(getEnergy=True)
        assert from_openmm(omm_state.getPotentialEnergy()).m == pytest.approx(energy)


def test_multipole_assignment():
    ff = ForceField(load_plugins=True)

    ff.get_parameter_handler(
        "Electrostatics",
        {
            "version": "0.4",
            "periodic_potential": "Ewald3D-ConductingBoundary",
            "nonperiodic_potential": "Coulomb",
            "exception_potential": "Coulomb",
        },
    )
    ff.get_parameter_handler(
        "ChargeIncrementModel",
        {"version": "0.3", "partial_charge_method": "formal_charge"},
    )

    multipole_handler = ff.get_parameter_handler("Multipole")

    multipole_handler.add_parameter(
        {
            "smirks": "[#1:1]",
            "polarity": 0.301856 * unit.angstrom**3,
        }
    )

    multipole_handler.add_parameter(
        {
            "smirks": "[#6:1]",
            "polarity": 1.243042 * unit.angstrom**3,
        }
    )

    de6810_handler = ff.get_parameter_handler("DampedExp6810", {"version": "0.3"})

    de6810_handler.add_parameter(
        {
            "smirks": "[#1:1]",
            "rho": 1.5 * unit.angstrom,
            "beta": 3.0 * unit.angstrom**-1,
            "c6": 1.0 * unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": 10.0 * unit.kilojoule_per_mole * unit.nanometer**8,
            "c10": 100.0 * unit.kilojoule_per_mole * unit.nanometer**10,
        }
    )

    de6810_handler.add_parameter(
        {
            "smirks": "[#6:1]",
            "rho": 3.0 * unit.angstrom,
            "beta": 3.0 * unit.angstrom**-1,
            "c6": 10.0 * unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": 100.0 * unit.kilojoule_per_mole * unit.nanometer**8,
            "c10": 1000.0 * unit.kilojoule_per_mole * unit.nanometer**10,
        }
    )

    toluene = Molecule.from_mapped_smiles(
        "[H:10][c:3]1[c:2]([c:1]([c:6]([c:5]([c:4]1[H:11])[H:12])[C:7]([H:13])([H:14])[H:15])[H:8])[H:9]"
    )
    toluene.generate_conformers(n_conformers=1)
    off_top: Topology = toluene.to_topology()
    off_top.add_molecule(toluene)
    off_top.box_vectors = [10, 10, 10] * unit.nanometer

    interchange = Interchange.from_smirnoff(ff, off_top)
    omm_system = interchange.to_openmm(combine_nonbonded_forces=False)

    amoeba_forces = [
        omm_system.getForce(i)
        for i in range(omm_system.getNumForces())
        if isinstance(omm_system.getForce(i), openmm.AmoebaMultipoleForce)
    ]

    assert len(amoeba_forces) == 1

    amoeba_force: openmm.AmoebaMultipoleForce = amoeba_forces[0]

    assert amoeba_force.getNumMultipoles() == 30

    c_polarity = 1.243042 * unit.angstrom**3
    h_polarity = 0.301856 * unit.angstrom**3
    expected_polarities = (
        [c_polarity] * 7 + [h_polarity] * 8 + [c_polarity] * 7 + [h_polarity] * 8
    )

    for particle_idx in range(amoeba_force.getNumMultipoles()):
        multipole_parameters = amoeba_force.getMultipoleParameters(particle_idx)
        expected_polarity = expected_polarities[particle_idx].m_as(unit.nanometer**3)
        assigned_polarity = from_openmm(multipole_parameters[-1]).m_as(
            unit.nanometer**3
        )
        assert assigned_polarity == expected_polarity

        for degree, omm_kw in [
            (1, amoeba_force.Covalent12),
            (2, amoeba_force.Covalent13),
            (3, amoeba_force.Covalent14),
            # (4, amoeba_force.Covalent15),
        ]:
            amoeba_neighs = amoeba_force.getCovalentMap(particle_idx, omm_kw)
            molecule_neighs = []
            for pair in off_top.nth_degree_neighbors(degree):
                if off_top.atom_index(pair[0]) == particle_idx:
                    molecule_neighs.append(off_top.atom_index(pair[1]))
                if off_top.atom_index(pair[1]) == particle_idx:
                    molecule_neighs.append(off_top.atom_index(pair[0]))
            assert set(amoeba_neighs) == set(molecule_neighs)


def test_multipole_energies():
    ff = ForceField(load_plugins=True)

    de6810_handler = ff.get_parameter_handler("DampedExp6810", {"version": "0.3"})

    de6810_handler.add_parameter(
        {
            "smirks": "[*:1]",
            "rho": 0.0 * unit.angstrom,
            "beta": 0.0 * unit.angstrom**-1,
            "c6": 0.0 * unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": 0.0 * unit.kilojoule_per_mole * unit.nanometer**8,
            "c10": 0.0 * unit.kilojoule_per_mole * unit.nanometer**10,
        }
    )

    ff.get_parameter_handler("Electrostatics")
    library_charge = ff.get_parameter_handler("LibraryCharges")
    library_charge.add_parameter(
        {"smirks": "[#1:1]", "charge1": 0.5 * unit.elementary_charge}
    )
    library_charge.add_parameter(
        {"smirks": "[#9:1]", "charge1": -0.5 * unit.elementary_charge}
    )
    library_charge.add_parameter(
        {"smirks": "[#10:1]", "charge1": 0.0 * unit.elementary_charge}
    )

    multipole_handler = ff.get_parameter_handler(
        "Multipole",
        {
            "version": "0.3",
            "method": "PME",
            "polarization_type": "direct",
        },
    )
    multipole_handler.add_parameter({"smirks": "[#1:1]", "polarity": "0 * angstrom**3"})
    multipole_handler.add_parameter({"smirks": "[#9:1]", "polarity": "0 * angstrom**3"})
    multipole_handler.add_parameter(
        {"smirks": "[#10:1]", "polarity": "1 * angstrom**3"}
    )

    hf = Molecule.from_smiles("[F:1][H:2]")
    hf.generate_conformers(n_conformers=1)
    off_top = hf.to_topology()
    neon = Molecule.from_smiles("[Ne]")
    off_top.add_molecule(neon)
    off_top.box_vectors = [10, 10, 10] * unit.nanometer

    interchange = Interchange.from_smirnoff(ff, off_top)
    omm_system = interchange.to_openmm(combine_nonbonded_forces=False)

    multipole_forces = [
        omm_system.getForce(i)
        for i in range(omm_system.getNumForces())
        if isinstance(omm_system.getForce(i), openmm.AmoebaMultipoleForce)
    ]

    assert len(multipole_forces) == 1

    multipole_force: openmm.AmoebaMultipoleForce = multipole_forces[0]

    assert multipole_force.getNumMultipoles() == 3

    distances = [0.3, 0.35, 0.4, 0.5, 0.6]

    omm_integrator: openmm.LangevinMiddleIntegrator = openmm.LangevinMiddleIntegrator(
        298, 1.0, 0.002
    )
    omm_simulation: openmm.app.Simulation = openmm.app.Simulation(
        off_top.to_openmm(), omm_system, omm_integrator
    )
    omm_context: openmm.Context = omm_simulation.context

    for distance in distances:
        omm_context.setPositions(
            to_openmm(
                [
                    [0, 0, 0],
                    [-0.1, 0, 0],
                    [distance, 0, 0],
                ]
                * unit.nanometer
            )
        )
        omm_state: openmm.State = omm_context.getState(getEnergy=True)
        omm_dipoles = multipole_force.getInducedDipoles(omm_context)

        coulomb_constant = 138.935018844  # kj/mol * nm / e**2
        e_field = -0.5 / distance**2 + 0.5 / (distance + 0.1) ** 2
        e_field *= coulomb_constant
        polarizability = 1e-3 / coulomb_constant
        induced_dipole = polarizability * e_field
        predicted_energy = -0.5 * induced_dipole * e_field

        assert omm_dipoles[2][0] == pytest.approx(induced_dipole, rel=1e-3)
        assert from_openmm(omm_state.getPotentialEnergy()).m == pytest.approx(
            predicted_energy, rel=1e-1
        )


def test_multipole_de6810_axilrod_options():
    ff = ForceField(load_plugins=True)
    multipole_handler = ff.get_parameter_handler(
        "Multipole",
        {
            "version": "0.3",
            "method": "PME",
            "polarization_type": "direct",
            "cutoff": "1 * nanometer",
        },
    )
    de6810_handler = ff.get_parameter_handler(
        "DampedExp6810",
        {"version": "0.3", "method": "cutoff", "cutoff": "1 * nanometer"},
    )
    axilrod_handler = ff.get_parameter_handler(
        "AxilrodTeller",
        {"version": "0.3", "method": "cutoff_periodic", "cutoff": "1 * nanometer"},
    )
    library_charge = ff.get_parameter_handler("LibraryCharges")
    ff.get_parameter_handler("Electrostatics")

    library_charge.add_parameter(
        {"smirks": "[*:1]", "charge1": 0 * unit.elementary_charge}
    )
    de6810_handler.add_parameter(
        {
            "smirks": "[*:1]",
            "rho": 0.0 * unit.angstrom,
            "beta": 0.0 * unit.angstrom**-1,
            "c6": 0.0 * unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": 0.0 * unit.kilojoule_per_mole * unit.nanometer**8,
            "c10": 0.0 * unit.kilojoule_per_mole * unit.nanometer**10,
        }
    )
    multipole_handler.add_parameter(
        {"smirks": "[*:1]", "polarity": 0.0 * unit.angstrom**3}
    )
    axilrod_handler.add_parameter(
        {
            "smirks": "[*:1]",
            "c9": 0.0 * unit.kilojoule_per_mole * unit.angstrom**9,
        }
    )

    neon = Molecule.from_smiles("[Ne]")
    neon.generate_conformers(n_conformers=1)
    off_top = neon.to_topology()
    off_top.add_molecule(neon)
    off_top.box_vectors = [10, 10, 10] * unit.nanometer

    interchange = Interchange.from_smirnoff(ff, off_top)
    omm_system = interchange.to_openmm(combine_nonbonded_forces=False)

    custom_manyp_forces = [
        omm_system.getForce(i)
        for i in range(omm_system.getNumForces())
        if isinstance(omm_system.getForce(i), openmm.CustomManyParticleForce)
    ]

    assert len(custom_manyp_forces) == 1

    custom_manyp_force: openmm.CustomManyParticleForce = custom_manyp_forces[0]

    assert custom_manyp_force.getNumParticles() == 2

    multipole_forces = [
        omm_system.getForce(i)
        for i in range(omm_system.getNumForces())
        if isinstance(omm_system.getForce(i), openmm.AmoebaMultipoleForce)
    ]

    assert len(multipole_forces) == 1

    multipole_force: openmm.AmoebaMultipoleForce = multipole_forces[0]

    assert multipole_force.getNumMultipoles() == 2

    custom_nonbonded_forces = [
        omm_system.getForce(i)
        for i in range(omm_system.getNumForces())
        if isinstance(omm_system.getForce(i), openmm.CustomNonbondedForce)
    ]

    assert len(custom_nonbonded_forces) == 1

    custom_nonbonded_force: openmm.CustomNonbondedForce = custom_nonbonded_forces[0]

    assert custom_nonbonded_force.getNumParticles() == 2

    assert from_openmm(custom_nonbonded_force.getCutoffDistance()).m_as(
        "nanometer"
    ) == pytest.approx(1.0)
    assert from_openmm(multipole_force.getCutoffDistance()).m_as(
        "nanometer"
    ) == pytest.approx(1.0)
    assert from_openmm(custom_manyp_force.getCutoffDistance()).m_as(
        "nanometer"
    ) == pytest.approx(1.0)

    assert multipole_force.getNonbondedMethod() == openmm.AmoebaMultipoleForce.PME
    assert multipole_force.getPolarizationType() == openmm.AmoebaMultipoleForce.Direct
    assert (
        custom_manyp_force.getNonbondedMethod()
        == openmm.CustomManyParticleForce.CutoffPeriodic
    )
    assert (
        custom_nonbonded_force.getNonbondedMethod()
        == openmm.CustomNonbondedForce.CutoffPeriodic
    )

    omm_integrator: openmm.LangevinMiddleIntegrator = openmm.LangevinMiddleIntegrator(
        298, 1.0, 0.002
    )
    omm_simulation: openmm.app.Simulation = openmm.app.Simulation(
        off_top.to_openmm(), omm_system, omm_integrator
    )
    omm_context: openmm.Context = omm_simulation.context

    omm_context.setPositions(
        to_openmm(
            [
                [0, 0, 0],
                [5, 0, 0],
            ]
            * unit.angstrom
        )
    )
    omm_state: openmm.State = omm_context.getState(getEnergy=True)
    assert from_openmm(omm_state.getPotentialEnergy()).m == pytest.approx(0.0)
