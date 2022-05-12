from openff.toolkit.typing.engines.smirnoff.parameters import _NonbondedHandler
from openff.toolkit.typing.engines.smirnoff import (
    ElectrostaticsHandler,
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
    vdWHandler,
)
from openff.toolkit.topology import Molecule, TopologyMolecule, TopologyAtom
import warnings

class EspalomaHandler(_NonbondedHandler):
    """Handle SMIRNOFF ``<ToolkitAM1BCC>`` tags
    .. warning :: This API is experimental and subject to change.
    """

    _TAGNAME = "Espaloma"  # SMIRNOFF tag name to process
    _DEPENDENCIES = [vdWHandler, ElectrostaticsHandler]
    _KWARGS = []  # Kwargs to catch when create_force is called

    def check_handler_compatibility(
            self, other_handler, assume_missing_is_default=True
    ):
        """
        Checks whether this ParameterHandler encodes compatible physics as another ParameterHandler. This is
        called if a second handler is attempted to be initialized for the same tag.
        Parameters
        ----------
        other_handler : a ParameterHandler object
            The handler to compare to.
        Raises
        ------
        IncompatibleParameterError if handler_kwargs are incompatible with existing parameters.
        """
        pass

    def create_force(self, system, topology, **kwargs):
        import os
        import torch
        import espaloma as esp
        from openff.toolkit.utils import get_data_file_path

        # grab pretrained model
        if not os.path.exists("espaloma_model.pt"):
            os.system("wget http://data.wangyq.net/espaloma_model.pt")

        force = super().create_force(system, topology, **kwargs)

        for ref_mol in topology.reference_molecules:

            # If charges were already assigned, skip this molecule
            if self.check_charges_assigned(ref_mol, topology):
                continue


            # create an Espaloma Graph object to represent the molecule of interest
            molecule_graph = esp.Graph(Molecule(ref_mol))

            # apply a trained espaloma model to assign parameters
            espaloma_model = torch.load("espaloma_model.pt")
            espaloma_model(molecule_graph.heterograph)
            from simtk import unit
            import numpy as np
            charges = unit.elementary_charge * molecule_graph.nodes["n1"].data[
                "q"
            ].flatten().detach().cpu().numpy().astype(
                np.float64,
            )

            ref_mol.partial_charges = charges

            # Assign charges to relevant atoms
            for topology_molecule in topology._reference_molecule_to_topology_molecules[
                ref_mol
            ]:
                for topology_particle in topology_molecule.atoms:
                    if type(topology_particle) is TopologyAtom:
                        ref_mol_particle_index = (
                            topology_particle.atom.molecule_particle_index
                        )
                    elif type(topology_particle) is TopologyVirtualSite:
                        ref_mol_particle_index = (
                            topology_particle.virtual_site.molecule_particle_index
                        )
                    else:
                        raise ValueError(
                            f"Particles of type {type(topology_particle)} are not supported"
                        )

                    topology_particle_index = topology_particle.topology_particle_index

                    particle_charge = ref_mol._partial_charges[ref_mol_particle_index]

                    # Retrieve nonbonded parameters for reference atom (charge not set yet)
                    _, sigma, epsilon = force.getParticleParameters(
                        topology_particle_index
                    )
                    # Set the nonbonded force with the partial charge
                    force.setParticleParameters(
                        topology_particle_index, particle_charge, sigma, epsilon
                    )
            # Finally, mark that charges were assigned for this reference molecule
            self.mark_charges_assigned(ref_mol, topology)

