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
>>> from maxwelllink.measurements import MeepLinearSpectroscopy
>>> spectrum = MeepLinearSpectroscopy(cavity, 2000.0, 2650.0, units="cm-1").run()
>>> spectrum["omega_cminv"], spectrum["transmission"]
"""

from .dummy_measurement import DummyMeasurement
from .meep_linear import MeepLinearSpectroscopy

__all__ = ["DummyMeasurement", "MeepLinearSpectroscopy"]
