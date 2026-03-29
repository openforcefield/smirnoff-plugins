from openff.toolkit import unit
from openff.toolkit.topology.topology import Topology, _TransformedDict
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ConstraintHandler,
    IndexedParameterAttribute,
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
)


class ProperTorsionBendDict(_TransformedDict):
    """For ProperTorsionBend interactions, i,j,k,l <-> l,k,j,i symmetry is not preserved,
    so avoid symmetrising keys as done in ValenceDict."""

    def __repr__(self):
        d = {k: v for k, v in self.items()}
        return f"ProperTorsionBendDict({d!r})"

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("ProperTorsionBendDict(...)")
        else:
            with p.group(2, "ProperTorsionBendDict(", ")"):
                p.breakable("")
                p.pretty({k: v for k, v in self.items()})


class UreyBradleyHandler(ParameterHandler):
    """A custom SMIRNOFF handler for Urey-Bradley interactions."""

    class UreyBradleyType(ParameterType):
        """A custom SMIRNOFF type for Urey-Bradley interactions."""

        _ELEMENT_NAME = "UreyBradley"

        k = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole / unit.nanometer**2
        )
        length = ParameterAttribute(default=None, unit=unit.nanometers)

    _TAGNAME = "UreyBradleys"
    _INFOTYPE = UreyBradleyType
    _DEPENDENCIES = [ConstraintHandler]


class ProperTorsionBendHandler(ParameterHandler):
    """
    A custom SMIRNOFF handler for Proper Torsion-Bend coupling interactions.
    The functional form is as given in eqn. 10 in Nevins, N.; Chen, K.;
    Allinger, N. L. Molecular Mechanics (MM4) Calculations on Alkenes.
    J. Comput. Chem. 1996, 17 (5–6), 669–694.
    https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6%253C669::AID-JCC7%253E3.0.CO;2-S.
    Each line in the OFFXML file specifies the term corresponding to a single bond angle-proper
    torsion combination, hence two lines are needed to specify the whole of eqn. 10. The angle
    is specified by the first 3 labels in the SMIRKS e.g. i,j,k in i,j,k,l. Hence, the order of
    the labels in the SMIRKS matters, and i,j,k,l is not equivalent to l,k,j,i if the type is not
    symmetrical.
    """

    class ProperTorsionBendType(ParameterType):
        """
        A custom SMIRNOFF type for Proper Torsion-Bend coupling interactions.

        """

        _ELEMENT_NAME = "ProperTorsionBend"

        angle0 = ParameterAttribute(unit=unit.degree)
        periodicity = IndexedParameterAttribute(converter=int)
        phase = IndexedParameterAttribute(unit=unit.degree)
        k = IndexedParameterAttribute(default=None, unit=unit.kilocalorie / unit.mole)

    _TAGNAME = "ProperTorsionBends"  # SMIRNOFF tag name to process
    _INFOTYPE = ProperTorsionBendType  # info type to store

    def find_matches(self, entity: Topology, unique: bool = False):
        """Find the improper torsions in the topology/molecule matched by a parameter type.

        Parameters
        ----------
        entity
            Topology to search.

        Returns
        ---------
        matches
            ``matches[atom_indices]`` is the ``ParameterType`` object
            matching the 4-tuple of atom indices in ``entity``.

        """
        return self._find_matches(
            entity, transformed_dict_cls=ProperTorsionBendDict, unique=unique
        )
