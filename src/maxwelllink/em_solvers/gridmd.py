# --------------------------------------------------------------------------------------#
# Copyright (c) 2026 MaxwellLink                                                        #
# This file is part of MaxwellLink. Repository: https://github.com/TaoELi/MaxwellLink   #
# If you use this code, always credit and cite arXiv:2512.06173.                        #
# See AGENTS.md and README.md for details.                                              #
# --------------------------------------------------------------------------------------#

"""
GridMD implemented in MaxwellLink: Thermodynamical sampling of molecular properties under
large-scale inhomogenenous EM fields.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union, Callable

import numpy as np

from ..molecule import Molecule
from ..sockets import SocketHub
from .laser_driven import LaserDrivenSimulation


class GridMDSimulation(LaserDrivenSimulation):
    r"""
    GridMD of the MaxwellLink molecules.

    This class samples the inhomogenenous electric field on molecules via a time-dependent
    stochastic electric field.

    .. math::

        E(t) = f(t)

    A thermostat should be attached to the molecules as well for removing the artificial excitation due to the electric field time evolution.

    All quantities are in atomic units.
    """

    def __init__(
        self,
        dt_au: float,
        molecules: Optional[Iterable[Molecule]] = None,
        hub: Optional[SocketHub] = None,
        # GridMD specific options
        grid_diffusion_au: float = 1e-4,
        dimension: int = 1,
        efield_map: Optional[Union[float, Callable[[float], float]]] = None,
        # end with GridMD specific options
        coupling_axis: str = "xyz",
        record_history: bool = True,
    ):

        super().__init__(
            dt_au=dt_au,
            molecules=molecules,
            drive=None,
            coupling_axis=coupling_axis,
            hub=hub,
            record_history=record_history,
        )

        self.grid_diffusion_au = grid_diffusion_au
        self.dimension = dimension
        self.efield_map = efield_map

        # store coordinates of the grid point
        self.R0 = np.array([0.0, 0.0, 0.0])  # initial coordinate
        self.R = np.copy(self.R0)  # instantaneous coordinate
        self.R_history = []  # history trajectory of the COM grid point

        # determine the output label in self.run()
        self._tag = "GridMD"

    # ------------------------------------------------------------------
    # Core helpers [reloaded]
    # ------------------------------------------------------------------

    def _calc_effective_efield(self, time_au: float) -> np.ndarray:
        """
        Calculate the effective electric field vector for GridMD.

        Parameters
        ----------
        time_au : float
            Current time in atomic units.

        Returns
        -------
        numpy.ndarray of float, shape (3,)
            Effective electric field vector in atomic units.
        """

        # LaserDrivenSimulation almost has everything for GridMD except an explicit
        # biased Langevin motion of the E-field, representing the center-of-mass motion
        # of molecules under large-scale inhomogenous EM fields.
        efield_vec = np.ones(3, dtype=float)

        # We need to propagate E_vec at each time step now ...

        # finally filtered by the axes (so users can decide to turn off interactions at specific dimensions)
        efield_vec *= self.axis
        return efield_vec
