# --------------------------------------------------------------------------------------#
# Copyright (c) 2026 MaxwellLink                                                       #
# This file is part of MaxwellLink. Repository: https://github.com/TaoELi/MaxwellLink  #
# If you use this code, always credit and cite arXiv:2512.06173.                       #
# See AGENTS.md and README.md for details.                                             #
# --------------------------------------------------------------------------------------#

"""
Light-induced measurements for the MaxwellLink EM solvers.

Example
-------
>>> from maxwelllink.measurements import MeepTransmissionSpectroscopy
>>> spectrum = MeepTransmissionSpectroscopy(cavity, 2000.0, 2650.0, units="cm-1").run()
>>> spectrum["omega_cminv"], spectrum["transmission"]

Cavities with no transmission port (e.g. a plasmonic ``NPoM``) name closed
detector surfaces instead of transmission/reflection planes, and are probed
by ``MeepEmissionSpectroscopy`` -- through the very same
``cavity.linear_spectrum(...)`` call.
"""

from .dummy_measurement import DummyMeasurement
from .meep_linear import MeepEmissionSpectroscopy, MeepTransmissionSpectroscopy

__all__ = [
    "DummyMeasurement",
    "MeepTransmissionSpectroscopy",
    "MeepEmissionSpectroscopy",
]
