from openff.toolkit.typing.engines.smirnoff.parameters import (
    _BaseVirtualSiteType,
    ParameterAttribute,
    VirtualSiteHandler,
    _validate_units,
)
from openff.units import unit


class DoubleExponentialVirtualSiteHandler(VirtualSiteHandler):
    """Vsite handler compatible with the Double Exponential handler"""

    class DoubleExponentialVirtualSiteType(_BaseVirtualSiteType):
        r_min = ParameterAttribute(default=None, unit=unit.nanometers)
        epsilon = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole)

    _TAGNAME = "DoubleExponentialVirtualSites"
    _INFOTYPE = DoubleExponentialVirtualSiteType
