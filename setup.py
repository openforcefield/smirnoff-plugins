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
            "DampedBuckingham68Handler = smirnoff_plugins.handlers.nonbonded:DampedBuckingham68Handler",
            "DoubleExponentialHandler = smirnoff_plugins.handlers.nonbonded:DoubleExponentialHandler",
        ],
        "openff.interchange.plugins.collections": [
            "SMIRNOFFDampedBuckingham68Collection = smirnoff_plugins.collections.nonbonded:SMIRNOFFDampedBuckingham68Collection",
            "DoubleExponentialCollection = smirnoff_plugins.collections.nonbonded:SMIRNOFFDoubleExponentialCollection",
        ],
    },
)
