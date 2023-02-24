import copy
from tempfile import NamedTemporaryFile

import pytest
import simtk.unit as unit
from openff.toolkit.tests.test_forcefield import (
    generate_freesolv_parameters_assignment_cases,
)
from openff.toolkit.tests.utils import requires_openeye_mol2
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField

# HCT - mbondi radii
hct_offxml = """\
<SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
    <Date>2019-08-08</Date>
    <Author>J. Chodera, (MSKCC); J. Wagner (Open Force Field)</Author>
    <!-- This file is intended to replicate the settings and per-particle parameters provided by OpenMM's customgbforces.GBSAHCTForce class -->
    <CustomGBSA version="0.3" gb_model="HCT" solvent_dielectric="78.5" solute_dielectric="1" sa_model="ACE" surface_area_penalty="5.4*calories/mole/angstroms**2" solvent_radius="1.4*angstroms" offset_radius="0.09*angstroms" kappa="0.0 * nanometer**-1">
      <Atom smirks="[*:1]" radius="0.15*nanometer" scale="0.8"/>
      <Atom smirks="[#1:1]" radius="0.12*nanometer" scale="0.85"/>
      <Atom smirks="[#1:1]~[#6]" radius="0.13*nanometer" scale="0.85"/>
      <Atom smirks="[#1:1]~[#7]" radius="0.13*nanometer" scale="0.85"/>
      <Atom smirks="[#1:1]~[#8]" radius="0.08*nanometer" scale="0.85"/>
      <Atom smirks="[#1:1]~[#16]" radius="0.08*nanometer" scale="0.85"/>
      <Atom smirks="[#6:1]" radius="0.17*nanometer" scale="0.72"/>
      <Atom smirks="[#7:1]" radius="0.155*nanometer" scale="0.79"/>
      <Atom smirks="[#8:1]" radius="0.15*nanometer" scale="0.85"/>
      <Atom smirks="[#9:1]" radius="0.15*nanometer" scale="0.88"/>
      <Atom smirks="[#14:1]" radius="0.21*nanometer" scale="0.8"/>
      <Atom smirks="[#15:1]" radius="0.185*nanometer" scale="0.86"/>
      <Atom smirks="[#16:1]" radius="0.18*nanometer" scale="0.96"/>
      <Atom smirks="[#17:1]" radius="0.17*nanometer" scale="0.8"/>
    </CustomGBSA>
</SMIRNOFF>\
"""

# OBC1 - mbondi2 radii
obc1_equiv_offxml = """\
<SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
    <!-- This file is intended to replicate the settings and per-particle parameters provided by OpenMM's customgbforces.GBSAOBC1Force class -->
    <!-- This file should replicate the OBC1 parameter set and energies -->
    <CustomGBSA version="0.3" gb_model="OBC" alpha="0.8" beta="0.0" gamma="2.909125" solvent_dielectric="78.5" solute_dielectric="1" sa_model="ACE" surface_area_penalty="5.4*calories/mole/angstroms**2" solvent_radius="1.4*angstroms" offset_radius="0.09*angstroms" kappa="0.0 * nanometer**-1">
      <Atom smirks="[*:1]" radius="0.15*nanometer" scale="0.8"/>
      <Atom smirks="[#1:1]" radius="0.12*nanometer" scale="0.85"/>
      <Atom smirks="[#1:1]~[#7]" radius="0.13*nanometer" scale="0.85"/>
      <Atom smirks="[#6:1]" radius="0.17*nanometer" scale="0.72"/>
      <Atom smirks="[#7:1]" radius="0.155*nanometer" scale="0.79"/>
      <Atom smirks="[#8:1]" radius="0.15*nanometer" scale="0.85"/>
      <Atom smirks="[#9:1]" radius="0.15*nanometer" scale="0.88"/>
      <Atom smirks="[#14:1]" radius="0.21*nanometer" scale="0.8"/>
      <Atom smirks="[#15:1]" radius="0.185*nanometer" scale="0.86"/>
      <Atom smirks="[#16:1]" radius="0.18*nanometer" scale="0.96"/>
      <Atom smirks="[#17:1]" radius="0.17*nanometer" scale="0.8"/>
    </CustomGBSA>
</SMIRNOFF>\
"""

# OBC2 - mbondi2 radii
obc2_equiv_offxml = """\
<SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
    <!-- This file is intended to replicate the OBC2 GBSA model as provided by OpenMM's GBSAOBCForce class, and per-particle parameters provided by OpenMM's customgbforces.GBSAOBC2Force class -->
    <CustomGBSA version="0.3" gb_model="OBC" alpha="1.0" beta="0.8" gamma="4.85" solvent_dielectric="78.5" solute_dielectric="1" sa_model="ACE" surface_area_penalty="5.4*calories/mole/angstroms**2" solvent_radius="1.4*angstroms" offset_radius="0.09*angstroms" kappa="0.0 * nanometer**-1">
      <Atom smirks="[*:1]" radius="0.15*nanometer" scale="0.8"/>
      <Atom smirks="[#1:1]" radius="0.12*nanometer" scale="0.85"/>
      <Atom smirks="[#1:1]~[#7]" radius="0.13*nanometer" scale="0.85"/>
      <Atom smirks="[#6:1]" radius="0.17*nanometer" scale="0.72"/>
      <Atom smirks="[#7:1]" radius="0.155*nanometer" scale="0.79"/>
      <Atom smirks="[#8:1]" radius="0.15*nanometer" scale="0.85"/>
      <Atom smirks="[#9:1]" radius="0.15*nanometer" scale="0.88"/>
      <Atom smirks="[#14:1]" radius="0.21*nanometer" scale="0.8"/>
      <Atom smirks="[#15:1]" radius="0.185*nanometer" scale="0.86"/>
      <Atom smirks="[#16:1]" radius="0.18*nanometer" scale="0.96"/>
      <Atom smirks="[#17:1]" radius="0.17*nanometer" scale="0.8"/>
    </CustomGBSA>
</SMIRNOFF>\
"""

# GBn - bondi radii
gbn_offxml = """\
<SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
    <!-- This file is intended to replicate the GBn GBSA model as provided by OpenMM's class GBSAGBnForce using CustomGBForce, and per-particle parameters provided by OpenMM's customgbforces.GBSAOBC2Force class -->
    <CustomGBSA version="0.3" gb_model="GBn" alpha="1.09511284" beta="1.907992938" gamma="2.50798245" neck_scale="0.361825" neck_cutoff="0.68" solvent_dielectric="78.5" solute_dielectric="1" sa_model="ACE" surface_area_penalty="5.4*calories/mole/angstroms**2" solvent_radius="1.4*angstroms" offset_radius="0.09*angstroms" kappa="0.0 * nanometer**-1">
      <Atom smirks="[*:1]" radius="0.15*nanometer" scale="0.5"/>
      <Atom smirks="[#1:1]" radius="0.12*nanometer" scale="1.09085413633"/>
      <Atom smirks="[#6:1]" radius="0.17*nanometer" scale="0.48435382330"/>
      <Atom smirks="[#7:1]" radius="0.155*nanometer" scale="0.700147318409"/>
      <Atom smirks="[#8:1]" radius="0.15*nanometer" scale="1.06557401132"/>
      <Atom smirks="[#9:1]" radius="0.15*nanometer" scale="0.5"/>
      <Atom smirks="[#14:1]" radius="0.21*nanometer" scale="0.5"/>
      <Atom smirks="[#15:1]" radius="0.185*nanometer" scale="0.5"/>
      <Atom smirks="[#16:1]" radius="0.18*nanometer" scale="0.602256336067"/>
      <Atom smirks="[#17:1]" radius="0.17*nanometer" scale="0.5"/>
    </CustomGBSA>
</SMIRNOFF>\
"""

# GBn2 - mbondi3 radii
gbn2_offxml = """\
<SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
    <!-- This file is intended to replicate the GBn2 GBSA model as provided by OpenMM's class GBSAGBnForce using CustomGBForce, and per-particle parameters provided by OpenMM's customgbforces.GBSAOBC2Force class -->
    <CustomGBSA version="0.3" gb_model="GBn2" neck_scale="0.826836" neck_cutoff="0.68" solvent_dielectric="78.5" solute_dielectric="1" sa_model="ACE" surface_area_penalty="5.4*calories/mole/angstroms**2" solvent_radius="1.4*angstroms" offset_radius="0.195141*angstroms" kappa="0.0 * nanometer**-1">
      <Atom smirks="[*:1]" radius="0.15*nanometer" scale="0.5" alpha="1.0" beta="0.8" gamma="4.851"/>
      <Atom smirks="[#1:1]" radius="0.12*nanometer" scale="1.425952" alpha="0.788440" beta="0.798699" gamma="0.437334"/>
      <Atom smirks="[#1:1]~[#7]" radius="0.13*nanometer" scale="1.425952" alpha="0.788440" beta="0.798699" gamma="0.437334"/>
      <Atom smirks="[#6:1]" radius="0.17*nanometer" scale="1.058554" alpha="0.733756" beta="0.506378" gamma="0.205844"/>
      <Atom smirks="[#7:1]" radius="0.155*nanometer" scale="0.733599" alpha="0.503364" beta="0.316828" gamma="0.192915"/>
      <Atom smirks="[#8:1]" radius="0.15*nanometer" scale="1.061039" alpha="0.867814" beta="0.876635" gamma="0.387882"/>
      <Atom smirks="[#9:1]" radius="0.15*nanometer" scale="0.5" alpha="1.0" beta="0.8" gamma="4.851"/>
      <Atom smirks="[#14:1]" radius="0.21*nanometer" scale="0.5" alpha="1.0" beta="0.8" gamma="4.851"/>
      <Atom smirks="[#15:1]" radius="0.185*nanometer" scale="0.5" alpha="1.0" beta="0.8" gamma="4.851"/>
      <Atom smirks="[#16:1]" radius="0.18*nanometer" scale="-0.703469" alpha="0.867814" beta="0.876635" gamma="0.387882"/>
      <Atom smirks="[#17:1]" radius="0.17*nanometer" scale="0.5" alpha="1.0" beta="0.8" gamma="4.851"/>
    </CustomGBSA>
</SMIRNOFF>\
"""

# The xml below is not used in the regression test but this the GBn2 parameters for nucleic acids
gbn2_nucleic_offxml = """\
<SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
    <!-- This file is intended to replicate the GBn GBSA model as provided by OpenMM's class GBSAGBnForce using CustomGBForce, and per-particle parameters provided by OpenMM's customgbforces.GBSAOBC2Force class -->
    <GBSANeck version="0.3" gb_model="GBn2" neck_scale="0.826836" neck_cutoff="0.68" solvent_dielectric="78.5" solute_dielectric="1" sa_model="ACE" surface_area_penalty="5.4*calories/mole/angstroms**2" solvent_radius="1.4*angstroms" offset_radius="0.195141*angstroms" kappa="0.0 * nanometer**-1">
      <Atom smirks="[*:1]" radius="0.15*nanometer" scale="0.5" alpha="1.0" beta="0.8" gamma="4.851"/>
      <Atom smirks="[#1:1]" radius="0.12*nanometer" scale="1.696538" alpha="0.537050" beta="0.362861" gamma="0.116704"/>
      <Atom smirks="[#1:1]~[#7]" radius="0.13*nanometer" scale="1.696538" alpha="0.537050" beta="0.362861" gamma="0.116704"/>
      <Atom smirks="[#6:1]" radius="0.17*nanometer" scale="1.268902" alpha="0.331670" beta="0.196842" gamma="0.093422"/>
      <Atom smirks="[#7:1]" radius="0.155*nanometer" scale="1.4259728" alpha="0.686311" beta="0.463189" gamma="0.138722"/>
      <Atom smirks="[#8:1]" radius="0.15*nanometer" scale="0.1840098" alpha="0.606344" beta="0.463006" gamma="0.142262"/>
      <Atom smirks="[#9:1]" radius="0.15*nanometer" scale="0.5" alpha="1.0" beta="0.8" gamma="4.851"/>
      <Atom smirks="[#14:1]" radius="0.21*nanometer" scale="0.5" alpha="1.0" beta="0.8" gamma="4.851"/>
      <Atom smirks="[#15:1]" radius="0.185*nanometer" scale="1.5450597" alpha="0.418365" beta="0.290054" gamma="0.1064245"/>
      <Atom smirks="[#16:1]" radius="0.18*nanometer" scale="0.05" alpha="0.606344" beta="0.463006" gamma="0.142262"/>
      <Atom smirks="[#17:1]" radius="0.17*nanometer" scale="0.5" alpha="1.0" beta="0.8" gamma="4.851"/>
    </GBSANeck>
</SMIRNOFF>\
"""


class TestCustomGBSA:
    @requires_openeye_mol2
    @pytest.mark.parametrize("gbsa_model", ["HCT", "OBC1", "OBC2", "GBn", "GBn2"])
    @pytest.mark.parametrize("is_periodic", (False, True))
    @pytest.mark.parametrize(
        "salt_concentration", [0.0, 0.15] * unit.moles / unit.liter
    )
    @pytest.mark.parametrize(
        ("freesolv_id", "forcefield_version", "allow_undefined_stereo"),
        generate_freesolv_parameters_assignment_cases(),
    )
    def test_freesolv_gbsa_energies(
        self,
        gbsa_model,
        is_periodic,
        salt_concentration,
        freesolv_id,
        forcefield_version,
        allow_undefined_stereo,
    ):
        """
        Regression test on HCT, OBC1, OBC2, GBn and GBn2 Amber-GBSA models. This test ensures that the
        SMIRNOFF-based CustomGBSA models match the equivalent OpenMM implementations.
        """

        import parmed as pmd
        from openff.toolkit.tests.utils import (
            compare_system_energies,
            create_system_from_amber,
            get_context_potential_energy,
            get_freesolv_file_path,
        )
        from simtk import openmm
        from simtk.openmm import Platform

        mol2_file_path, _ = get_freesolv_file_path(freesolv_id, forcefield_version)

        # Load molecules.
        molecule = Molecule.from_file(
            mol2_file_path, allow_undefined_stereo=allow_undefined_stereo
        )

        # Give each atom a unique name, otherwise OpenMM will complain
        for idx, atom in enumerate(molecule.atoms):
            atom.name = f"{atom.element.symbol}{idx}"
        positions = molecule.conformers[0]

        gbsa_offxmls = {
            "HCT": hct_offxml,
            "OBC1": obc1_equiv_offxml,
            "OBC2": obc2_equiv_offxml,
            "GBn": gbn_offxml,
            "GBn2": gbn2_offxml,
        }

        # Load force field
        ff = ForceField(
            "test_forcefields/test_forcefield.offxml",
            gbsa_offxmls[gbsa_model],
            load_plugins=True,
        )

        # Set salt concentration
        temperature = 300.0 * unit.kelvin
        gbsa_handler = ff.get_parameter_handler("CustomGBSA")
        gbsa_handler.salt_concentration = salt_concentration
        gbsa_handler.temperature = temperature

        # Create OpenFF System with the current toolkit.
        off_top = molecule.to_topology()
        if is_periodic:
            off_top.box_vectors = (
                (30.0, 0, 0),
                (0, 30.0, 0),
                (0, 0, 30.0),
            ) * unit.angstrom

        else:
            off_top.box_vectors = None

        off_omm_system = ff.create_openmm_system(
            off_top, charge_from_molecules=[molecule]
        )

        off_nonbonded_force = [
            force
            for force in off_omm_system.getForces()
            if isinstance(force, openmm.NonbondedForce)
        ][0]

        omm_top = off_top.to_openmm()
        pmd_struct = pmd.openmm.load_topology(omm_top, off_omm_system, positions)
        prmtop_file = NamedTemporaryFile(suffix=".prmtop")
        inpcrd_file = NamedTemporaryFile(suffix=".inpcrd")
        gbsa_radii = {
            "HCT": "mbondi",
            "OBC1": "mbondi2",
            "OBC2": "mbondi2",
            "GBn": "bondi",
            "GBn2": "mbondi3",
        }
        pmd.tools.changeRadii(pmd_struct, gbsa_radii[gbsa_model]).execute()
        pmd_struct.save(prmtop_file.name, overwrite=True)
        pmd_struct.save(inpcrd_file.name, overwrite=True)

        # The functional form of the nonbonded force will change depending on whether the cutoff
        # is None during initialization. Therefore, we need to figure that out here.

        # WARNING: The NonbondedMethod enums at openmm.app.forcefield and openmm.CustomGBForce
        # aren't necessarily the same, and could be misinterpreted if the wrong one is used. For
        # create_system_from_amber, we must provide the app.forcefield version.

        if is_periodic:
            amber_nb_method = openmm.app.forcefield.CutoffPeriodic
            amber_cutoff = off_nonbonded_force.getCutoffDistance()
        else:
            amber_nb_method = openmm.app.forcefield.NoCutoff
            amber_cutoff = None

        amber_gbsa_models = {
            "HCT": openmm.app.HCT,
            "OBC1": openmm.app.OBC1,
            "OBC2": openmm.app.OBC2,
            "GBn": openmm.app.GBn,
            "GBn2": openmm.app.GBn2,
        }
        (
            amber_omm_system,
            amber_omm_topology,
            amber_positions,
        ) = create_system_from_amber(
            prmtop_file.name,
            inpcrd_file.name,
            nonbondedMethod=amber_nb_method,
            nonbondedCutoff=amber_cutoff,
            implicitSolvent=amber_gbsa_models[gbsa_model],
            implicitSolventSaltConc=salt_concentration,
            temperature=temperature,
            gbsaModel="ACE",
        )

        # Retrieve the GBSAForce from both the AMBER and OpenForceField systems
        off_gbsa_forces = [
            force
            for force in off_omm_system.getForces()
            if (
                isinstance(force, openmm.GBSAOBCForce)
                or isinstance(force, openmm.openmm.CustomGBForce)
            )
        ]
        assert len(off_gbsa_forces) == 1
        off_gbsa_force = off_gbsa_forces[0]
        amber_gbsa_forces = [
            force
            for force in amber_omm_system.getForces()
            if (
                isinstance(force, openmm.GBSAOBCForce)
                or isinstance(force, openmm.openmm.CustomGBForce)
            )
        ]
        assert len(amber_gbsa_forces) == 1
        amber_gbsa_force = amber_gbsa_forces[0]

        # We get radius and screen values from each model's getStandardParameters method
        gbsa_class = {
            "HCT": openmm.app.internal.customgbforces.GBSAHCTForce,
            "OBC1": openmm.app.internal.customgbforces.GBSAOBC1Force,
            "OBC2": openmm.app.internal.customgbforces.GBSAOBC2Force,
            "GBn": openmm.app.internal.customgbforces.GBSAGBnForce,
            "GBn2": openmm.app.internal.customgbforces.GBSAGBn2Force,
        }
        gb_params = gbsa_class[gbsa_model].getStandardParameters(omm_top)

        # Use GB params from OpenMM GBSA classes to populate parameters
        if gbsa_model in ["HCT", "OBC1", "OBC2"]:
            for idx, (radius, screen) in enumerate(gb_params):
                # Keep the charge, but throw out the old radius and screen values
                q, old_radius, old_screen = amber_gbsa_force.getParticleParameters(idx)

                if isinstance(amber_gbsa_force, openmm.GBSAOBCForce):
                    # Note that in GBSAOBCForce, the per-particle parameters are separate
                    # arguments, while in CustomGBForce they're a single iterable
                    amber_gbsa_force.setParticleParameters(idx, q, radius, screen)

                elif isinstance(amber_gbsa_force, openmm.CustomGBForce):
                    # !!! WARNING: CustomAmberGBForceBase expects different per-particle parameters
                    # depending on whether you use addParticle or setParticleParameters. In
                    # setParticleParameters, we have to apply the offset_radius and scale BEFORE setting
                    # parameters, whereas in addParticle, it is applied afterwards, and the particle
                    # parameters are not set until an auxiliary finalize() method is called. !!!
                    offset_radius = 0.009
                    amber_gbsa_force.setParticleParameters(
                        idx,
                        (q, radius - offset_radius, screen * (radius - offset_radius)),
                    )

        elif gbsa_model == "GBn":
            for idx, (radius, screen) in enumerate(gb_params):
                (
                    q,
                    old_radius,
                    old_screen,
                    radii_index,
                ) = amber_gbsa_force.getParticleParameters(idx)

                offset_radius = 0.009
                amber_gbsa_force.setParticleParameters(
                    idx,
                    (
                        q,
                        radius - offset_radius,
                        screen * (radius - offset_radius),
                        radii_index,
                    ),
                )

        elif gbsa_model == "GBn2":
            for idx, (radius, screen, alpha, beta, gamma) in enumerate(gb_params):
                (
                    q,
                    old_radius,
                    old_screen,
                    old_alpha,
                    old_beta,
                    old_gamma,
                    radii_index,
                ) = amber_gbsa_force.getParticleParameters(idx)

                offset_radius = 0.0195141  # GBn2 offset radius
                amber_gbsa_force.setParticleParameters(
                    idx,
                    (
                        q,
                        radius - offset_radius,
                        screen * (radius - offset_radius),
                        alpha,
                        beta,
                        gamma,
                        radii_index,
                    ),
                )

        # Put the GBSA force into a separate group so we can specifically compare GBSA energies
        amber_gbsa_force.setForceGroup(1)
        off_gbsa_force.setForceGroup(1)

        # Some manual overrides to get the OFF system's NonbondedForce matched up with the AMBER system
        if is_periodic:
            off_nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        else:
            off_nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

        off_nonbonded_force.setReactionFieldDielectric(1.0)

        # Not sure if zeroing the switching width is essential -- This might only make a difference
        # in the energy if we tested on a molecule larger than the 9A cutoff
        # off_nonbonded_force.setSwitchingDistance(0)

        # Create Contexts
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        platform = Platform.getPlatformByName("Reference")
        amber_context = openmm.Context(amber_omm_system, integrator, platform)
        off_context = openmm.Context(
            off_omm_system, copy.deepcopy(integrator), platform
        )

        # Get context energies
        amber_energy = get_context_potential_energy(amber_context, positions)
        off_energy = get_context_potential_energy(off_context, positions)

        # # Very handy for debugging
        # print(openmm.XmlSerializer.serialize(off_gbsa_force))
        # print(openmm.XmlSerializer.serialize(amber_gbsa_force))

        # Ensure that the GBSA energies (which we put into ForceGroup 1) are identical
        # For Platform=OpenCL, we do get "=="-level identical numbers, but for "Reference", we don't.
        assert (
            abs(amber_energy[1] - off_energy[1]) < 2.0e-4 * unit.kilojoule / unit.mole
        )

        # Ensure that all system energies are the same
        compare_system_energies(
            off_omm_system,
            amber_omm_system,
            positions,
            by_force_type=False,
            atol=2.0e-4,
        )
