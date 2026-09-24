"""Teach Interchange about the ``NAGLMBISCharges`` handler.

Interchange hard codes which parameter handlers can assign partial charges: the
``Electrostatics`` collection is always built by ``openff.interchange.smirnoff._create._electrostatics``
from the ``Electrostatics``, ``NAGLCharges``, ``ChargeIncrementModel``, ``ToolkitAM1BCC`` and
``LibraryCharges`` handlers, before any collection plugins are created. A collection plugin
alone therefore cannot assign charges, so instead this module

* adds ``NAGLMBISCharges`` to ``_create._SUPPORTED_PARAMETER_HANDLERS``, and
* wraps ``_create._electrostatics`` so that, when a force field contains a ``NAGLMBISCharges``
  section, the ``Electrostatics`` collection is built by
  :class:`~smirnoff_plugins.collections.charges.SMIRNOFFNAGLMBISElectrostaticsCollection`.

Force fields without a ``NAGLMBISCharges`` section are passed straight through to Interchange.

The handler plugin can be imported while ``_create`` is only partly initialised (``_create``
loads the handler plugins before defining ``_electrostatics``), so ``_create`` is patched lazily
on the first call to ``Interchange.from_smirnoff``, which ``ForceField.create_interchange`` also
goes through.
"""

import functools

_TAGNAME = "NAGLMBISCharges"

_installed = False
_create_patched = False


def _patch_create():
    """Patch ``openff.interchange.smirnoff._create`` to support ``NAGLMBISCharges``."""
    global _create_patched

    if _create_patched:
        return

    from openff.interchange.exceptions import MissingParameterHandlerError
    from openff.interchange.smirnoff import _create

    from smirnoff_plugins.collections.charges import SMIRNOFFNAGLMBISElectrostaticsCollection

    original_electrostatics = _create._electrostatics

    @functools.wraps(original_electrostatics)
    def _electrostatics(
        interchange,
        force_field,
        topology,
        molecules_with_preset_charges=None,
        allow_nonintegral_charges: bool = False,
    ):
        if _TAGNAME not in force_field.registered_parameter_handlers:
            return original_electrostatics(
                interchange,
                force_field,
                topology,
                molecules_with_preset_charges,
                allow_nonintegral_charges,
            )

        if "Electrostatics" not in force_field.registered_parameter_handlers:
            raise MissingParameterHandlerError(
                f"Force field contains a {_TAGNAME} section, which assigns partial charges, "
                "but no ElectrostaticsHandler was found.",
            )

        interchange.collections.update(
            {
                "Electrostatics": SMIRNOFFNAGLMBISElectrostaticsCollection.create(
                    parameter_handler=[
                        handler
                        for handler in [
                            force_field._parameter_handlers.get(name, None)
                            for name in [
                                "Electrostatics",
                                _TAGNAME,
                                "NAGLCharges",
                                "ChargeIncrementModel",
                                "ToolkitAM1BCC",
                                "LibraryCharges",
                            ]
                        ]
                        if handler is not None
                    ],
                    topology=topology,
                    molecules_with_preset_charges=molecules_with_preset_charges,
                    allow_nonintegral_charges=allow_nonintegral_charges,
                ),
            },
        )

    _create._SUPPORTED_PARAMETER_HANDLERS.add(_TAGNAME)
    _create._electrostatics = _electrostatics

    _create_patched = True


def install():
    """Wrap ``Interchange.from_smirnoff`` so that ``_create`` is patched before it is first used."""
    global _installed

    if _installed:
        return

    from openff.interchange import Interchange

    original_from_smirnoff = Interchange.from_smirnoff.__func__

    @functools.wraps(original_from_smirnoff)
    def from_smirnoff(cls, *args, **kwargs):
        _patch_create()
        return original_from_smirnoff(cls, *args, **kwargs)

    Interchange.from_smirnoff = classmethod(from_smirnoff)

    _installed = True
