# SMIRNOFF Plugins

[![tests](https://github.com/openforcefield/smirnoff-plugins/workflows/CI/badge.svg?branch=main)](https://github.com/openforcefield/smirnoff-plugins/actions?query=workflow%3ACI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This framework provides parameter handlers that enable using custom force field functional forms in [SMIRNOFF](
https://github.com/openforcefield/openff-toolkit/blob/master/The-SMIRNOFF-force-field-format.md) based force fields
via the OpenFF Toolkit's built-in plugin system.

Currently, these include:

* `DampedBuckingham68` - a damped version of the 6-8 buckingham potential proposed by [Tang and Toennies](https://aip.scitation.org/doi/10.1063/1.447150).
* `DoubleExponential` - a double exponential van der Waals potential proposed by [Man et al](https://doi.org/10.1021/acs.jctc.0c01267)
* `DampedExp6810` - a damped version of the 6-8-10 exponential repulsion proposed by [Tang and Toennies](https://aip.scitation.org/doi/10.1063/1.447150) with mixing rules from [Smith](https://doi.org/10.1103/PhysRevA.5.1708)
* `Multipole` - an induced dipole polarizability potential implemented with the AmoebaMultipoleForce in OpenMM
* `AxilrodTeller` - a three-body dispersion potential proposed by [Axilrod and Teller](https://doi.org/10.1063%2F1.1723844)
* `UreyBradley` - a harmonic [Urey-Bradley](https://journals.aps.org/pr/abstract/10.1103/PhysRev.38.1969) potential.

## Installation

This framework and its required dependencies can be installed using `conda`:

```shell
mamba env create --name smirnoff-plugins --file devtools/conda-envs/test_env.yaml
python setup.py develop
```

## Getting Started

The custom parameter handlers are made available to the [OpenFF Toolkit](https://github.com/openforcefield/openff-toolkit)
via the plugin system it exposes, and so in most cases users should be able to install this package and have the 
different functional forms be automatically available.

Here we will demonstrate parameterizing a 4-site water model using a custom `Buckingham68` parameter handler and the
parameters presented by [Mohebifar and Rowley](https://aip.scitation.org/doi/10.1063/5.0014469) in their *'An 
efficient and accurate model for water with an improved non-bonded potential'* publication.

To begin with we create a new, empty `ForceField` object. We need to specify that the object should load plugins it 
finds, including the custom handler shipped with this package.

```python
from openff.toolkit import ForceField
force_field = ForceField(load_plugins=True)
```

Add the standard bond and angle constraints to ensure the correct rigid geometry of the water model.

```python
from openff.units import unit

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
```

Add the handler which will place a single virtual site on each oxygen atom in each water molecule. Here we have used the 
virtual site handler to define the charges on the virtual site **and** the hydrogen atoms.

```python
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

## Purpose and contributing

This repository is intended to help nucleate scientific research in the force field community which the OpenFF consortium team doesn't have the resources to fully support, but where it can provide assistance to greatly reduce the barrier to others initiating new research.

There is no expectation that researchers using OpenFF infrastructure or standards will contribute to this repo.
Researchers who want to publish SMIRNOFF plugins elsewhere should find it straightforward to replicate the structure of this repo and the MIT license permits fully reuse of the code under its terms.
(We intend to adopt a formal [Contributor License Agreement](https://opensource.org/faq/#contributor-agreements) throughout our ecosystem in the future to provide legal clarity, and we will seek the approval of former contributors if and when that happens.)

This is not a "core package" like the OpenFF Toolkit or Interchange, and so the OpenFF Consortium does not take responsibility for correct, accurate, or performant behavior of the plugins in this repository. While OpenFF will help where it can, the responsibility for these behaviors ultimately lies with the plugin author. 

The OpenFF infrastructure team handles versioning and packaging of this repo, and will perform technical maintenance to ensure that continuous integration testing runs (but not necessarily that it passes).
The OpenFF infrastructure team may remove plugins or turn off tests in this repo if behavior is broken by upstream changes or bugs are reported. In such cases, pull requests fixing behavior are welcome.

All plugin PRs require one approving review from a previous contributor to merge. 
The reviewer pool consists of everyone who has authored a merged PR. 
The PR submitter is responsible for recruiting a reviewer. 
To ensure that reviewers are readily available, we encourage contributors to plan to review one more PR than they author.

Since this package contains multiple independent plugins, semantic versioning is unable to convey the state of each contained plugin. 
Instead, this package will use [Calendar Versioning](https://calver.org/), specifically `YYYY.MM.MICRO` (for example `2023.08.1`.)
This should support the use of this repo in scholarly work, since there will be a particular calendar version that can be cited for reproducibility, and which will correspond to a single state of the plugin code. 
Since different plugins may require different versions of dependencies, the conda package for this repo will be minimally pinned and researchers should plan to communicate versions of major dependencies used when providing instructions on how to reproduce results. 

## Copyright

Copyright (c) 2023, Open Force Field Consortium