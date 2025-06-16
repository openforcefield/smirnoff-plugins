"""
smirnoff-plugins
Custom parameter handlers for extending SMIRNOFF force fields.
"""

import sys

from setuptools import find_packages, setup

import versioneer

short_description = __doc__.split("\n")

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

try:
    with open("README.md") as handle:
        long_description = handle.read()
except OSError:
    long_description = "\n".join(short_description[2:])


setup(
    # Self-descriptive entries which should always be present
    name="smirnoff-plugins",
    author="Simon Boothroyd",
    author_email="simon.boothroyd@colorado.edu",
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    setup_requires=[] + pytest_runner,
    # Make the handler plugins discoverable.
    entry_points={
        "openff.toolkit.plugins.handlers": [
            "UreyBradleyHandler = smirnoff_plugins.handlers.valence:UreyBradleyHandler",
            "DampedBuckingham68Handler = smirnoff_plugins.handlers.nonbonded:DampedBuckingham68Handler",
            "DoubleExponentialHandler = smirnoff_plugins.handlers.nonbonded:DoubleExponentialHandler",
            "DampedExp6810Handler = smirnoff_plugins.handlers.nonbonded:DampedExp6810Handler",
            "AxilrodTellerHandler = smirnoff_plugins.handlers.nonbonded:AxilrodTellerHandler",
            "MultipoleHandler = smirnoff_plugins.handlers.nonbonded:MultipoleHandler",
            "DoubleExponentialVirtualSiteHandler = smirnoff_plugins.handlers.vsites:DoubleExponentialVirtualSiteHandler",
        ],
        "openff.interchange.plugins.collections": [
            "SMIRNOFFUreyBradleyCollection = smirnoff_plugins.collections.valence:SMIRNOFFUreyBradleyCollection",
            "SMIRNOFFDampedBuckingham68Collection = smirnoff_plugins.collections.nonbonded:SMIRNOFFDampedBuckingham68Collection",
            "DoubleExponentialCollection = smirnoff_plugins.collections.nonbonded:SMIRNOFFDoubleExponentialCollection",
            "SMIRNOFFDampedExp6810Collection = smirnoff_plugins.collections.nonbonded:SMIRNOFFDampedExp6810Collection",
            "SMIRNOFFAxilrodTellerCollection = smirnoff_plugins.collections.nonbonded:SMIRNOFFAxilrodTellerCollection",
            "SMIRNOFFMultipoleCollection = smirnoff_plugins.collections.nonbonded:SMIRNOFFMultipoleCollection",
            "SMIRNOFFDoubleExponentialVirtualSiteCollection = smirnoff_plugins.collections.vsites:SMIRNOFFDoubleExponentialVirtualSiteCollection",
        ],
    },
)
