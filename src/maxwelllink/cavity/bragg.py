# --------------------------------------------------------------------------------------#
# Copyright (c) 2026 MaxwellLink                                                       #
# This file is part of MaxwellLink. Repository: https://github.com/TaoELi/MaxwellLink  #
# If you use this code, always credit and cite arXiv:2512.06173.                       #
# See AGENTS.md and README.md for details.                                             #
# --------------------------------------------------------------------------------------#

import warnings

import numpy as np
import meep as mp

from .dummy_cavity import DummyCavity


class BraggResonator(DummyCavity):
    """
    A quarter-wave Bragg (DBR) cavity along x, in 1, 2, or 3 dimensions.

    A ``BraggResonator`` is a planar cavity whose two mirrors are quarter-wave
    dielectric stacks: alternating layers of high (``n_hi``) and low (``n_lo``)
    refractive index, each one quarter of the design wavelength thick inside
    its medium. 
    
    Increasing ``n_pairs`` increases the mirror reflectivity and the quality factor. 

    Notes
    -----
    - With ``transverse_boundary="periodic"`` the cell is Bloch-periodic
      (``k_point = (0, 0, 0)``), and Meep may then use complex fields. 

    Examples
    --------
    >>> from maxwelllink.cavity import BraggResonator
    >>> cav = BraggResonator(omega=2320.0, units="cm-1", n_pairs=10,
    ...                      n_hi=2.0, n_lo=1.0, dimensions=1)
    >>> print(cav.summary())
    """

    def __init__(
        self,
        omega: float,
        units: str = "cm-1",
        n_pairs: int = 3,
        n_hi: float = 2.0,
        n_lo: float = 1.0,
        n_defect: float = 1.0,
        defect_order: int = 1,
        dimensions: int = 1,
        transverse_size_nm: float = None,
        transverse_boundary: str = "pml",
        resolution: float = None,
        pml_nm: float = None,
    ):
        """
        Initialize the parameters of a quarter-wave Bragg (DBR) cavity.

        Parameters
        ----------
        omega : float
            Target cavity resonance in ``units``.
        units : str, default: "cm-1"
            Units of ``omega``: "cm-1", "eV", "au", "nm", or "um".
        n_pairs : int, default: 3
            Number of quarter-wave layer pairs per mirror (the Q dial).
        n_hi : float, default: 2.0
            High refractive index of the mirror stack (``n_hi > n_lo``).
        n_lo : float, default: 1.0
            Low refractive index of the mirror stack.
        n_defect : float, default: 1.0
            Refractive index of the defect gap between the mirrors.
        defect_order : int, default: 1
            The gap has an optical length of ``defect_order`` half wavelengths.
        dimensions : int, default: 1
            1, 2, or 3; the layer stack always runs along x.
        transverse_size_nm : float or None, optional
            Transverse extent (nm) of the allowed region in 2D/3D. Default:
            5 cavity wavelengths. Must be omitted in 1D.
        transverse_boundary : str, default: "pml"
            ``"periodic"`` for an infinite planar cavity (Bloch-periodic
            boundaries) or ``"pml"`` for absorbing transverse boundaries.
        resolution : float or None, optional
            Meep resolution. Default: at least 20 pixels per wavelength in the
            densest medium and 8 pixels across the thinnest layer.
        pml_nm : float or None, optional
            PML thickness in nm. Default: one cavity wavelength.
        """

        # -------------- input checks --------------
        if n_hi <= n_lo:
            raise ValueError("n_hi must be larger than n_lo for a Bragg mirror.")
        if min(n_hi, n_lo, n_defect) <= 0:
            raise ValueError("Refractive indexes must be positive.")
        if int(n_pairs) < 1:
            raise ValueError("n_pairs must be at least 1.")
        if int(defect_order) < 1:
            raise ValueError("defect_order must be a positive integer.")
        if transverse_boundary not in ("periodic", "pml"):
            raise ValueError("transverse_boundary must be 'periodic' or 'pml'.")
        if int(dimensions) == 1 and transverse_size_nm is not None:
            warnings.warn("transverse_size_nm has no meaning in a 1D cavity.")
        if int(dimensions) == mp.CYLINDRICAL:
            raise ValueError(
                "BraggResonator supports only dimensions 1, 2, or 3."
            )

        # default attributes (units, grid, hotspot, ...), overridden below
        super().__init__(omega=omega, units=units, dimensions=dimensions)
        lam = self.nm_to_meep(self.wavelength_nm)  # cavity wavelength in um

        self.n_pairs = int(n_pairs)
        self.n_hi = float(n_hi)
        self.n_lo = float(n_lo)
        self.n_defect = float(n_defect)
        self.defect_order = int(defect_order)
        self.transverse_boundary = transverse_boundary

        # -------------- layer stack along x (Meep units: um) --------------
        # quarter-wave mirror layers (n * t = lambda / 4) around a defect gap
        # of optical length defect_order half wavelengths
        t_hi = 0.25 * lam / self.n_hi
        t_lo = 0.25 * lam / self.n_lo
        t_gap = 0.5 * lam * self.defect_order / self.n_defect
        # default PML thickness: one design wavelength
        self.pml_thickness = self.nm_to_meep(pml_nm) if pml_nm is not None else lam
        pml = self.pml_thickness

        indexes = np.array(
            [self.n_lo, self.n_hi] * self.n_pairs
            + [self.n_defect]
            + [self.n_hi, self.n_lo] * self.n_pairs
        )
        thicknesses = np.array(
            [t_lo, t_hi] * self.n_pairs + [t_gap] + [t_hi, t_lo] * self.n_pairs
        )
        # extend the outermost (low-index) layers through the PML
        thicknesses[0] += pml
        thicknesses[-1] += pml
        # center the stack so that the defect gap center sits at the origin
        length = float(np.sum(thicknesses))
        centers = np.cumsum(thicknesses) - 0.5 * thicknesses - 0.5 * length

        self.layer_indexes = indexes
        self.layer_thicknesses = thicknesses
        self.layer_centers = centers
        # one block per layer, spanning the full transverse extent
        self.geometry = [
            mp.Block(
                size=mp.Vector3(float(t), mp.inf, mp.inf),
                center=mp.Vector3(float(c), 0.0, 0.0),
                material=mp.Medium(index=float(n)),
            )
            for t, c, n in zip(thicknesses, centers, indexes)
        ]

        # -------------- cell size and boundaries --------------
        boundary_layers = [mp.PML(thickness=pml, direction=mp.X)]
        k_point = None
        allowed_bounds = {"x": (-0.5 * t_gap, 0.5 * t_gap)}
        cell = mp.Vector3(length, 0.0, 0.0)
        if self.dimensions > 1:
            # transverse extent of the allowed region (default: five wavelength)
            if transverse_size_nm is not None:
                t_size = self.nm_to_meep(transverse_size_nm)
            else:
                t_size = 5.0 * lam
            if transverse_boundary == "periodic":
                cell_t = t_size
                k_point = mp.Vector3()  # Bloch-periodic transverse boundaries
            else:  # "pml": pad the cell and absorb in the transverse directions
                cell_t = t_size + 2.0 * pml
                boundary_layers.append(mp.PML(thickness=pml, direction=mp.Y))
                if self.dimensions == 3:
                    boundary_layers.append(mp.PML(thickness=pml, direction=mp.Z))
            allowed_bounds["y"] = (-0.5 * t_size, 0.5 * t_size)
            cell = mp.Vector3(length, cell_t, 0.0)
            if self.dimensions == 3:
                allowed_bounds["z"] = (-0.5 * t_size, 0.5 * t_size)
                cell = mp.Vector3(length, cell_t, cell_t)

        self.cell_size = cell
        self.boundary_layers = boundary_layers
        self.k_point = k_point
        self.allowed_bounds = allowed_bounds

        # -------------- grid resolution --------------
        # default: at least 20 px per wavelength in the densest medium and
        # 8 px across the thinnest layer
        t_min = min(t_hi, t_lo, t_gap)
        n_max = max(self.n_hi, self.n_lo, self.n_defect)
        if resolution is not None:
            self.resolution = float(resolution)
        else:
            self.resolution = float(np.ceil(max(20.0 * n_max / lam, 8.0 / t_min)))

        # -------------- analytic estimates --------------
        # mirror reflectance from the effective admittance of the quarter-wave
        # (hi/lo)^N stack terminated by the semi-infinite outer n_lo layer
        # (thin-film optics; see e.g. Macleod, Thin-Film Optical Filters)
        admittance = self.n_lo * (self.n_hi / self.n_lo) ** (2 * self.n_pairs)
        reflectance = ((self.n_defect - admittance) / (self.n_defect + admittance)) ** 2
        finesse = np.pi * np.sqrt(reflectance) / (1.0 - reflectance)
        # each mirror adds a penetration depth of lambda / (4 (n_hi - n_lo)),
        # so the effective gap holds m_eff (not defect_order) half wavelengths;
        # then FSR = omega / m_eff, kappa = FSR / finesse, and Q = omega / kappa
        m_eff = self.defect_order + 1.0 / (self.n_hi - self.n_lo)
        omega_cminv = 1.0e7 / self.wavelength_nm
        quality_factor = m_eff * finesse
        self.predicted = {
            "omega_cminv": omega_cminv,
            "wavelength_nm": self.wavelength_nm,
            "mirror_reflectance": float(reflectance),
            "quality_factor": float(quality_factor),
            "kappa_cminv": float(omega_cminv / quality_factor),
            "fsr_cminv": float(omega_cminv / m_eff),
        }
        self._warn_if_coarse(n_max=n_max, t_min=t_min)

    # -------------- light-induced measurements --------------

    def optical_setup(self):
        """
        Optical setup of the Bragg cavity: the generic x-axis planes of
        ``DummyCavity.optical_setup``, with the reference structure replaced
        by a homogeneous ``n_lo`` medium (for the default ``n_lo = 1``:
        vacuum).
        """

        setup = super().optical_setup()
        setup["reference_geometry"] = [
            mp.Block(
                size=mp.Vector3(mp.inf, mp.inf, mp.inf),
                material=mp.Medium(index=self.n_lo),
            )
        ]
        return setup
