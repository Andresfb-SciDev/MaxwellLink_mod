# --------------------------------------------------------------------------------------#
# Copyright (c) 2026 MaxwellLink                                                       #
# This file is part of MaxwellLink. Repository: https://github.com/TaoELi/MaxwellLink  #
# If you use this code, always credit and cite arXiv:2512.06173.                       #
# See AGENTS.md and README.md for details.                                             #
# --------------------------------------------------------------------------------------#

"""
Light-induced measurements for the MaxwellLink EM solvers.

Every measurement runs the same kind of experiment -- excite the system with
light pulses, record its response, and combine the records into user-facing
observables -- with the solver-specific steps implemented in one subclass of
``DummyMeasurement`` per EM solver. Linear spectroscopy ships first
(``MeepLinearSpectroscopy``, the two-run flux method for the FDTD cavity
builders); pump-probe, multidimensional spectroscopy, and time-resolved
movies follow the same template.

Example
-------
>>> from maxwelllink.measurements import MeepLinearSpectroscopy
>>> spectrum = MeepLinearSpectroscopy(cavity, 2000.0, 2650.0, units="cm-1").run()
>>> spectrum["omega_cminv"], spectrum["transmission"]

For FDTD cavities, ``cavity.linear_spectrum(...)`` is an equivalent shortcut.
"""

from .dummy_measurement import DummyMeasurement
from .meep_linear import MeepLinearSpectroscopy

__all__ = ["DummyMeasurement", "MeepLinearSpectroscopy"]
