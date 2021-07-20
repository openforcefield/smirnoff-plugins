from openff.toolkit.typing.engines.smirnoff import ForceField
from smirnoff_plugins.handlers.customgbsa import CustomOBCHandler


obc1_equiv_offxml = '''<SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
    <!-- This file is intended to replicate the settings and per-particle parameters provided by OpenMM's customgbforces.GBSAOBC1Force class -->
    <!-- This file should replicate the OBC1 parameter set and energies -->
    <CustomOBC version="0.3" alpha="0.8" beta="0.0" gamma="2.909125" solvent_dielectric="78.5" solute_dielectric="1" sa_model="ACE" surface_area_penalty="5.4*calories/mole/angstroms**2" solvent_radius="1.4*angstroms">
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
    </CustomOBC>
</SMIRNOFF>
'''

class TestCustomOBC:
    def test_parse_xml(self):
        ff = ForceField(obc1_equiv_offxml)