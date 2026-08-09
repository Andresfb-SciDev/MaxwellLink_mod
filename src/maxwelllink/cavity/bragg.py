# --------------------------------------------------------------------------------------#
# Copyright (c) 2026 MaxwellLink                                                       #
# This file is part of MaxwellLink. Repository: https://github.com/TaoELi/MaxwellLink  #
# If you use this code, always credit and cite arXiv:2512.06173.                       #
# See AGENTS.md and README.md for details.                                             #
# --------------------------------------------------------------------------------------#

"""
A quarter-wave Bragg (DBR) cavity builder for Meep.
"""

import warnings

import numpy as np
import meep as mp

from .dummy_cavity import DummyCavity, CYLINDRICAL

# an Er point source exactly on the axis of a cylindrical cell is numerically
# broken (https://github.com/NanoComp/meep/issues/2704), so near-axis dipoles
# are shifted off the axis by this many grid points (the Meep tutorial value)
OFF_AXIS_SHIFT_PX = 1.5


class BraggResonator(DummyCavity):
    """
    A quarter-wave Bragg (DBR) cavity in 1, 2, or 3 dimensions (stack along
    x), or in cylindrical coordinates (stack along z).

    The two mirrors are quarter-wave dielectric stacks: alternating layers of
    high (``n_hi``) and low (``n_lo``) refractive index, each one quarter of
    the design wavelength thick inside its medium.

    Increasing ``n_pairs`` increases the mirror reflectivity and the quality
    factor.

    Notes
    -----
    - With ``transverse_boundary="periodic"`` the cell is Bloch-periodic
      (``k_point = (0, 0, 0)``), and Meep may then use complex fields.
    - With ``dimensions=mxl.CYLINDRICAL`` the mirrors are disks stacked along
      z in the (r, z) half plane, ``transverse_size_nm`` is the cavity
      radius (absorbing outer boundary), and ``m = 0`` is the default sector 
      (for z-polarized molecules). Pass ``m=+1`` or ``m=-1`` to ``make_simulation`` 
      for an on-axis Cartesian x/y-polarized molecule.

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
            1, 2, or 3 (layer stack along x), or ``mxl.CYLINDRICAL``
            (layer stack along z in the (r, z) half plane; ``m = 0`` by
            default).
        transverse_size_nm : float or None, optional
            Transverse extent (nm) of the allowed region in 2D/3D, or the
            cavity radius in cylindrical cells. Default: 5 cavity
            wavelengths. Must be omitted in 1D.
        transverse_boundary : str, default: "pml"
            ``"periodic"`` for an infinite planar cavity (Bloch-periodic
            boundaries) or ``"pml"`` for absorbing transverse boundaries
            (the only option for cylindrical cells).
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
        if int(dimensions) == CYLINDRICAL and transverse_boundary == "periodic":
            raise ValueError(
                "A cylindrical cell has an absorbing side boundary; use "
                "transverse_boundary='pml'."
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

        # -------------- the quarter-wave layer stack (Meep units: um) --------------
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
        # one block per layer, spanning the full transverse extent; the stack
        # runs along x in Cartesian cells and along z in cylindrical ones
        cylindrical = self.dimensions == CYLINDRICAL
        if cylindrical:
            self.geometry = [
                mp.Block(
                    size=mp.Vector3(mp.inf, mp.inf, float(t)),
                    center=mp.Vector3(0.0, 0.0, float(c)),
                    material=mp.Medium(index=float(n)),
                )
                for t, c, n in zip(thicknesses, centers, indexes)
            ]
        else:
            self.geometry = [
                mp.Block(
                    size=mp.Vector3(float(t), mp.inf, mp.inf),
                    center=mp.Vector3(float(c), 0.0, 0.0),
                    material=mp.Medium(index=float(n)),
                )
                for t, c, n in zip(thicknesses, centers, indexes)
            ]

        # -------------- cell size and boundaries --------------
        if cylindrical:
            # the (r, z) half plane: mirrors are disks stacked along z, and
            # r spans [0, R] with the axis at r = 0 and PML at the outer edge
            r_size = (
                self.nm_to_meep(transverse_size_nm)
                if transverse_size_nm is not None
                else 5.0 * lam
            )
            self.cell_size = mp.Vector3(r_size + pml, 0.0, length)
            self.boundary_layers = [
                mp.PML(thickness=pml, direction=mp.Z),
                mp.PML(thickness=pml, direction=mp.R, side=mp.High),
            ]
            self.k_point = None
            # Use the azimuthally symmetric sector unless make_simulation()
            # receives an explicit m value.
            self.m = 0
            self.allowed_bounds = {
                "x": (0.0, r_size),  # x plays the role of r
                "z": (-0.5 * t_gap, 0.5 * t_gap),
            }
        else:
            self.boundary_layers = [mp.PML(thickness=pml, direction=mp.X)]
            self.k_point = None
            self.allowed_bounds = {"x": (-0.5 * t_gap, 0.5 * t_gap)}
            self.cell_size = mp.Vector3(length, 0.0, 0.0)
            if self.dimensions > 1:
                # transverse extent of the allowed region (default: five
                # wavelengths)
                t_size = (
                    self.nm_to_meep(transverse_size_nm)
                    if transverse_size_nm is not None
                    else 5.0 * lam
                )
                if transverse_boundary == "periodic":
                    cell_t = t_size
                    self.k_point = mp.Vector3()  # Bloch-periodic boundaries
                else:  # "pml": pad the cell and absorb in the transverse directions
                    cell_t = t_size + 2.0 * pml
                    self.boundary_layers.append(mp.PML(thickness=pml, direction=mp.Y))
                    if self.dimensions == 3:
                        self.boundary_layers.append(
                            mp.PML(thickness=pml, direction=mp.Z)
                        )
                self.allowed_bounds["y"] = (-0.5 * t_size, 0.5 * t_size)
                self.cell_size = mp.Vector3(length, cell_t, 0.0)
                if self.dimensions == 3:
                    self.allowed_bounds["z"] = (-0.5 * t_size, 0.5 * t_size)
                    self.cell_size = mp.Vector3(length, cell_t, cell_t)

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
        # textbook thin-film estimates (Macleod, Thin-Film Optical Filters)
        admittance = self.n_lo * (self.n_hi / self.n_lo) ** (2 * self.n_pairs)
        reflectance = ((self.n_defect - admittance) / (self.n_defect + admittance)) ** 2
        finesse = np.pi * np.sqrt(reflectance) / (1.0 - reflectance)
        # mirror penetration makes the effective gap hold m_eff (not
        # defect_order) half wavelengths; then Q = m_eff * finesse
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
        Optical setup of the Bragg cavity: the generic transmission planes of
        ``DummyCavity.optical_setup``.

        The reference structure is replaced by a homogeneous ``n_lo`` medium
        (for the default ``n_lo = 1``: vacuum).
        """

        setup = super().optical_setup()
        setup["reference_geometry"] = [
            mp.Block(
                size=mp.Vector3(mp.inf, mp.inf, mp.inf),
                material=mp.Medium(index=self.n_lo),
            )
        ]
        return setup

    def emission_setup(self, offset_nm=(0.0, 0.0, 0.0), component=None):
        """
        Local-dipole (Purcell) probe of the Bragg cavity: a dipole in the
        defect gap, polarized parallel to the mirror planes, read out through
        one plane outside each mirror.

        The reference is the homogeneous defect medium (``n_defect``),
        so the LDOS ratio is exact. Same keys as ``DummyCavity.emission_setup``.

        Notes
        -----
        Cylindrical cells default to an azimuthally symmetric (m = 0) ring of
        radial dipole.

        For the m = +-1 near-axis dipole, pass ``component=mp.Er`` together
        with ``m=1``.

        Parameters
        ----------
        offset_nm : sequence of three floats, default: (0, 0, 0)
            Displacement (nm) of the dipole from the defect center.
        component : Meep field component or None, optional
            Dipole orientation. Default: ``mp.Ez`` (parallel to the mirrors)
            in Cartesian cells, ``mp.Er`` in cylindrical ones.
        """

        if self.dimensions == CYLINDRICAL:
            # a closed box just inside the PML: the two mirror-side disks
            # (axial) plus the outer side wall (lateral)
            box = self._emission_box_regions()
            center = self.hotspot_center + self._offset_to_meep(offset_nm)
            if component is None:
                component = mp.Er
                if center.x == 0.0:
                    # the default m = 0 radial dipole is a ring one design
                    # wavelength off axis, clamped inside small-radius cells
                    lam = self.nm_to_meep(self.wavelength_nm)
                    center += mp.Vector3(
                        min(lam, 0.5 * self.allowed_bounds["x"][1]), 0.0, 0.0
                    )
            elif component == mp.Er and center.x == 0.0:
                # the near-axis transverse dipole of the m = +-1 sectors,
                # shifted off the singular axis (run it with m=1;
                # make_simulation rejects it at m=0)
                center += mp.Vector3(OFF_AXIS_SHIFT_PX / self.resolution, 0.0, 0.0)
            return {
                "excitation": {"center": center, "size": mp.Vector3()},
                "component": component,
                "detectors": {
                    "radiated": box,
                    "axial": box[:2],
                    "lateral": box[2:],
                },
                "reference_geometry": [
                    mp.Block(
                        size=mp.Vector3(mp.inf, mp.inf, mp.inf),
                        material=mp.Medium(index=self.n_defect),
                    )
                ],
                "reference_surface": box,
                # watch the ringdown at the lid, away from the dipole
                "decay_monitor": box[0].center,
            }

        # Cartesian cells: one flux plane outside each mirror, along x
        pml = self.pml_thickness
        x_left = -0.5 * self.cell_size.x + pml  # inner edge of the left PML
        x_right = 0.5 * self.cell_size.x - pml  # inner edge of the right PML
        # plane spacing: three grid points, capped for coarse grids (the same
        # convention as the transmission planes)
        spacing = min(3.0 / self.resolution, (x_right - x_left) / 8.0)
        transverse = mp.Vector3(0.0, self.cell_size.y, self.cell_size.z)
        planes = [
            mp.FluxRegion(  # outward normals: the left plane counts down
                center=mp.Vector3(x_right - spacing),
                size=transverse,
                direction=mp.X,
                weight=+1.0,
            ),
            mp.FluxRegion(
                center=mp.Vector3(x_left + spacing),
                size=transverse,
                direction=mp.X,
                weight=-1.0,
            ),
        ]
        return {
            "excitation": {
                "center": self.hotspot_center + self._offset_to_meep(offset_nm),
                "size": mp.Vector3(),
            },
            "component": component if component is not None else mp.Ez,
            "detectors": {"radiated": planes},
            "reference_geometry": [
                mp.Block(
                    size=mp.Vector3(mp.inf, mp.inf, mp.inf),
                    material=mp.Medium(index=self.n_defect),
                )
            ],
            "reference_surface": planes,
            # watch the ringdown at the radiated plane, away from the dipole
            "decay_monitor": mp.Vector3(x_right - spacing),
        }

    # -------------- simulation assembly --------------

    def make_simulation(
        self,
        molecules=None,
        hub=None,
        sources=None,
        extra_geometry=(),
        **meep_kwargs,
    ):
        """
        Build the simulation as ``DummyCavity.make_simulation`` does, after
        checking that a cylindrical run is consistent with its azimuthal
        sector ``m`` (see ``_check_cylindrical_sector``).

        Cartesian cells pass straight through.
        """

        if self.dimensions == CYLINDRICAL:
            m = meep_kwargs.get("m", self.m if self.m is not None else 0)
            self._check_cylindrical_sector(m, sources, extra_geometry)
        return super().make_simulation(
            molecules=molecules,
            hub=hub,
            sources=sources,
            extra_geometry=extra_geometry,
            **meep_kwargs,
        )

    def _check_cylindrical_sector(self, m, sources, extra_geometry):
        """
        Reject configurations inconsistent with the azimuthal sector m.

        An on-axis transverse (``Er``/``Ep``) point dipole exists only at
        m = +-1, and an on-axis ``Ez`` one only at m = 0.

        m != 0 forces complex fields, so molecular regions must be built with
        ``real_field_only=False``.
        """

        for source in sources or ():
            if source.size.norm() != 0.0:
                continue  # extended sources (e.g. the transmission plane)
            near_axis = source.center.x < (OFF_AXIS_SHIFT_PX + 1.0) / self.resolution
            if source.component in (mp.Er, mp.Ep) and near_axis and m == 0:
                raise ValueError(
                    "A near-axis transverse dipole belongs to the m = +-1 "
                    "sectors; run it with m=1, e.g. "
                    "purcell(..., component=mp.Er, m=1)."
                )
            if source.component == mp.Ez and source.center.x == 0.0 and m != 0:
                raise ValueError(
                    "An on-axis z dipole is azimuthally symmetric; run it at m=0."
                )
        if m != 0:
            for shape in list(extra_geometry):
                material = getattr(shape, "material", None)
                for sus in getattr(material, "E_susceptibilities", None) or []:
                    if getattr(sus, "real_field_only", False):
                        raise ValueError(
                            "m != 0 runs use complex fields; rebuild the "
                            "molecular region with "
                            "place_region(..., real_field_only=False)."
                        )
