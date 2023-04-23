import openmm
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule
from openff.units import unit
from openff.units.openmm import to_openmm


def main():
    ff = ForceField(load_plugins=True)

    multipole_handler = ff.get_parameter_handler(
        "Multipole",
        {
            "version": "0.3",
            "method": "PME",
            "polarization_type": "direct",
            "cutoff": "0.9 * nanometer",
        },
    )

    de6810_handler = ff.get_parameter_handler(
        "DampedExp6810",
        {
            "version": "0.3",
            "method": "cutoff",
            "cutoff": "0.9 * nanometer",
            "scale13": "1.0",
            "scale14": "0.5",
        },
    )

    ff.get_parameter_handler("LibraryCharges")
    ff.get_parameter_handler("Electrostatics", {"version": "0.4", "scale14": "0"})
    ff.get_parameter_handler("ToolkitAM1BCC")

    # <Atom smirks="[#1:1]"
    # sigma="2.221305 * angstrom"
    # beta="3.568835 * angstrom**-1"
    # c6="1.065489e-04 * kilojoule_per_mole * nanometer**6"
    # c8="2.591653e-06 * kilojoule_per_mole * nanometer**8"
    # c10="8.454610e-08 * kilojoule_per_mole * nanometer**10"></Atom>
    # <Atom smirks="[#6:1]"
    # sigma="3.426648 * angstrom"
    # beta="3.513660 * angstrom**-1"
    # c6="1.687397e-03 * kilojoule_per_mole * nanometer**6"
    # c8="6.965649e-05 * kilojoule_per_mole * nanometer**8"
    # c10="4.569720e-06 * kilojoule_per_mole * nanometer**10"></Atom>
    # <Atom smirks="[#1:1]" polarity="0.297758 * angstrom**3"></Atom>
    # <Atom smirks="[#6:1]" polarity="0.984072 * angstrom**3"></Atom>

    de6810_handler.add_parameter(
        {
            "smirks": "[#1:1]",
            "rho": 2.221305 * unit.angstrom,
            "beta": 3.568835 * unit.angstrom**-1,
            "c6": 1.065489e-04 * unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": 2.591653e-06 * unit.kilojoule_per_mole * unit.nanometer**8,
            "c10": 8.454610e-08 * unit.kilojoule_per_mole * unit.nanometer**10,
        }
    )

    de6810_handler.add_parameter(
        {
            "smirks": "[#6:1]",
            "rho": 3.426648 * unit.angstrom,
            "beta": 3.513660 * unit.angstrom**-1,
            "c6": 1.687397e-03 * unit.kilojoule_per_mole * unit.nanometer**6,
            "c8": 6.965649e-05 * unit.kilojoule_per_mole * unit.nanometer**8,
            "c10": 4.569720e-06 * unit.kilojoule_per_mole * unit.nanometer**10,
        }
    )

    multipole_handler.add_parameter(
        {"smirks": "[#1:1]", "polarity": 0.297758 * unit.angstrom**3}
    )

    multipole_handler.add_parameter(
        {"smirks": "[#6:1]", "polarity": 0.984072 * unit.angstrom**3}
    )

    benzene = Molecule.from_smiles("c1ccccc1")
    benzene.generate_conformers(n_conformers=1)
    off_top = benzene.to_topology()
    off_top.box_vectors = [10, 10, 10] * unit.nanometer

    interchange = Interchange.from_smirnoff(ff, off_top)
    omm_system = interchange.to_openmm(combine_nonbonded_forces=False)

    omm_integrator: openmm.LangevinMiddleIntegrator = openmm.LangevinMiddleIntegrator(
        to_openmm(298 * unit.kelvin),
        to_openmm(1.0 / unit.picoseconds),
        to_openmm(2 * unit.femtoseconds),
    )
    omm_simulation: openmm.app.Simulation = openmm.app.Simulation(
        off_top.to_openmm(), omm_system, omm_integrator
    )

    omm_context: openmm.Context = omm_simulation.context
    omm_context.setPositions(to_openmm(benzene.conformers[0]))

    omm_simulation.step(100)


if __name__ == "__main__":
    main()
