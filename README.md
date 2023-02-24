# SMIRNOFF Plugins

[![tests](https://github.com/openforcefield/smirnoff-plugins/workflows/CI/badge.svg?branch=main)](https://github.com/openforcefield/smirnoff-plugins/actions?query=workflow%3ACI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This framework provides parameter handlers that enable using custom force field functional forms in [SMIRNOFF](
https://github.com/openforcefield/openff-toolkit/blob/master/The-SMIRNOFF-force-field-format.md) based force fields
via the OpenFF toolkits built-in plugin system.

Currently, these include:

* `DampedBuckingham68` - a damped version of the 6-8 buckingham potential proposed by [Tang and Toennies](https://aip.scitation.org/doi/10.1063/1.447150).
* `DoubleExponential` - a double exponential van der Waals potential proposed by [Man et al](https://doi.org/10.1021/acs.jctc.0c01267)

## Installation

This framework and its required dependencies can be installed using `conda`:

```
conda env create --name smirnoff-plugins --file devtools/conda-envs/test_env.yaml
python setup.py develop
```

## Getting Started

The custom parameter handlers are made available to the [OpenFF toolkit](https://github.com/openforcefield/openff-toolkit) 
via the plugin system it exposes, and so in most cases users should be able to install this package and have the 
different functional forms be automatically available.

Here we will demonstrate parameterizing a 4-site water model using a custom `Buckingham68` parameter handler and the
parameters presented by [Mohebifar and Rowley](https://aip.scitation.org/doi/10.1063/5.0014469) in their *'An 
efficient and accurate model for water with an improved non-bonded potential'* publication.

To begin with we create a new, empty force field object. We need to specify that the object should load plugins it 
finds, including the custom handler shipped with this package.

```python
from openff.toolkit.typing.engines.smirnoff import ForceField
force_field = ForceField(load_plugins=True)
```

Add the standard bond and angle constraints to ensure the correct rigid geometry of the water model.

```python
constraint_handler = force_field.get_parameter_handler("Constraints")
# Keep the H-O bond length fixed at 0.9572 angstroms.
constraint_handler.add_parameter(
    {"smirks": "[#1:1]-[#8X2H2+0:2]-[#1]", "distance": 0.9572 * unit.angstrom}
)
# Keep the H-O-H angle fixed at 104.52 degrees.
constraint_handler.add_parameter(
    {"smirks": "[#1:1]-[#8X2H2+0]-[#1:2]", "distance": 1.5139 * unit.angstrom}
)
```

Next we will add a `vdW` parameter handler to the force field with all the water interactions zeroed out:

```python
from simtk import unit

vdw_handler = force_field.get_parameter_handler("vdW")
vdw_handler.add_parameter(
    {
        "smirks": "[*:1]",
        "epsilon": 0.0 * unit.kilojoule_per_mole,
        "sigma": 1.0 * unit.angstrom,
    }
)
```

and a set of electrostatics handlers with all the charges zeroed out: 

```python
force_field.get_parameter_handler("Electrostatics")

force_field.get_parameter_handler(
    "ChargeIncrementModel",
    {"version": "0.3", "partial_charge_method": "formal_charge"},
)
```

These extra handlers are currently required due to a quirk of how the OpenFF toolkit builds OpenMM systems which have 
virtual sites present. Without this, the charges applied by the virtual site handler will likely be incorrect.
See [openff-toolkit/issues/885](https://github.com/openforcefield/openff-toolkit/issues/885) for more information.

Add the handler which will place a single virtual site on each oxygen atom in each water molecule. Here we have used the 
virtual site handler to define the charges on the virtual site **and** the hydrogen atoms.

```python
from openff.toolkit.typing.engines.smirnoff import ParameterList

virtual_site_handler = force_field.get_parameter_handler("VirtualSites")
virtual_site_handler.add_parameter(
    {
        "smirks": "[#1:2]-[#8X2H2+0:1]-[#1:3]",
        "type": "DivalentLonePair",
        "distance": -0.0106 * unit.nanometers,
        "outOfPlaneAngle": 0.0 * unit.degrees,
        "match": "once",
        "charge_increment2": 1.0552 * 0.5 * unit.elementary_charge,
        "charge_increment1": 0.0 * unit.elementary_charge,
        "charge_increment3": 1.0552 * 0.5 * unit.elementary_charge,
    }
)
# Currently required due to OpenFF issue #884
virtual_site_handler._parameters = ParameterList(virtual_site_handler._parameters)
```

We are now finally ready to add the custom damped buckingham potential:

```python
buckingham_handler = force_field.get_parameter_handler("DampedBuckingham68")
buckingham_handler.gamma = 35.8967 * unit.nanometer ** -1
buckingham_handler.add_parameter(
    {
        "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
        "a": 0.0 * unit.kilojoule_per_mole,
        "b": 0.0 / unit.nanometer,
        "c6": 0.0 * unit.kilojoule_per_mole * unit.nanometer ** 6,
        "c8": 0.0 * unit.kilojoule_per_mole * unit.nanometer ** 8,
    }
)
buckingham_handler.add_parameter(
    {
        "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
        "a": 1600000.0 * unit.kilojoule_per_mole,
        "b": 42.00 / unit.nanometer,
        "c6": 0.003 * unit.kilojoule_per_mole * unit.nanometer ** 6,
        "c8": 0.00003 * unit.kilojoule_per_mole * unit.nanometer ** 8,
    }
)
```

And that should be everything! The force field can be saved out and inspected manually by calling:

```python
force_field.to_file("buckingham-force-field.offxml")
```

For a more detailed example of how to use this force field to actually simulate a box of water, see the 
[`buckingham-water` example](examples/buckingham-water.py) in the `examples` directory.

## Copyright

Copyright (c) 2021, Open Force Field Consortium
 
