# --------------------------------------------------------------------------------------#
# Copyright (c) 2026 MaxwellLink                                                       #
# This file is part of MaxwellLink. Repository: https://github.com/TaoELi/MaxwellLink  #
# If you use this code, always credit and cite arXiv:2512.06173.                       #
# See AGENTS.md and README.md for details.                                             #
# --------------------------------------------------------------------------------------#

"""
Publication-ready plotting helpers shared across MaxwellLink.
"""

from contextlib import contextmanager

import numpy as np

# house colors for publication-ready plots, adapted from group's previous columnplot style
# (https://github.com/TaoELi/columnplots).
PLOT_COLORS = {
    "red": "#EA4E34",
    "yellow": "#ECA300",
    "navy_blue": "#006CA3",
    "cyan": "#3ABCD2",
    "sky_blue": "#009BD6",
    "brown": "#7B2B15",
    "red_economist": "#E3000F",
    "black": "k",
    "dark_green": "#285F17",
    "magenta": "#907DAC",
    "lightblue_background": "#D9E5EC",
    "lightgray_background": "#F6F6F4",
}

# publication style: 12 pt Arial-like sans-serif fonts
PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 12,
}

# colors cycled over the named detector planes of an optical setup
_DETECTOR_COLORS = ("cyan", "dark_green", "magenta", "brown")

# the two ways MaxwellLink molecules enter a cavity, as drawn:
# (cavity attribute, color, alpha, legend label)
_MOLECULES = (
    ("placed_molecules", "magenta", 0.45, "molecule"),  # molecule-level route
    ("placed_regions", "brown", 0.30, "region"),  # grid-level route
)

# -------------- style primitives (for any MaxwellLink figure) --------------


@contextmanager
def use_style():
    """
    Apply the house fonts (``PLOT_STYLE``) within a ``with`` block.
    """

    import matplotlib.pyplot as plt

    previous = {key: plt.rcParams[key] for key in PLOT_STYLE}
    plt.rcParams.update(PLOT_STYLE)
    try:
        yield
    finally:
        plt.rcParams.update(previous)


def polish_axes(ax, xlabel=None, ylabel=None, despine=False):
    """
    Apply the publication finishing touches to an axes.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to polish.
    xlabel, ylabel : str or None, optional
        Axis labels to set (existing labels are kept when None).
    despine : bool, default: True
        Whether to remove the top and right spines.
    """

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=12)
    if despine:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(
            fontsize=12,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            columnspacing=1.2,
            handlelength=1.5,
        )
    ax.figure.canvas.draw()


# -------------- cavity annotation helpers (draw onto an existing axes) -----


def draw_optical_planes(cavity, ax, in_nm=True, setup=None):
    """
    Draw the excitation and detectors of a cavity measurement setup.

    The drawing is structural: a zero-size excitation is a point source
    (star), a finite one a plane or sheet (line/segment); a detector that is
    a ``{"center", "size"}`` dict is a plane (line), and one that is a list
    of ``mp.FluxRegion`` faces is outlined face by face.

    Parameters
    ----------
    cavity : DummyCavity subclass
        The cavity whose setup is drawn.
    ax : matplotlib Axes
        The axes to draw into.
    in_nm : bool, default: True
        Whether the x-axis of ``ax`` is in nm (the 1D profile view) or in
        Meep units (the ``plot2D`` plan view).
    setup : dict or None, optional
        A pre-fetched setup dict (e.g. ``cavity.emission_setup()``).
        Default: ``cavity.optical_setup()``.
    """

    import meep as mp

    if setup is None:
        try:
            setup = cavity.optical_setup()
        except NotImplementedError:
            return
    # Cartesian planes vary along x (vertical lines); cylindrical planes
    # vary along z, the vertical axis of the r-z plan view (horizontal lines)
    cylindrical = cavity.dimensions == mp.CYLINDRICAL
    axis = "z" if cylindrical else "x"
    draw_line = ax.axhline if cylindrical else ax.axvline
    # vertical axis of the plan view: y in 2D, z in 3D and cylindrical cells
    vertical = "y" if cavity.dimensions == 2 else "z"
    scale = cavity.length_units_nm if in_nm else 1.0
    source = setup["excitation"]
    source_color = PLOT_COLORS["sky_blue" if in_nm else "yellow"]

    if source["size"].norm() == 0.0 and not in_nm:
        # a point source in the plan view: a star marker
        ax.plot(
            source["center"].x,
            getattr(source["center"], vertical),
            marker="*",
            color=source_color,
            markersize=10,
            linestyle="none",
            label="excitation",
        )
    elif in_nm:
        # the 1D profile view: every excitation is a line at its x position
        draw_line(
            scale * getattr(source["center"], axis),
            color=source_color,
            linestyle="-.",
            linewidth=1.5,
            label="excitation",
        )
    else:
        # a finite source in the plan view: the segment it spans (a
        # transmission plane, or the grazing sheet of a scattering probe)
        _draw_segment(
            ax,
            source["center"],
            source["size"],
            vertical,
            source_color,
            "--",
            "excitation",
        )

    for (name, value), color in zip(setup["detectors"].items(), _DETECTOR_COLORS):
        if isinstance(value, dict):  # a plane with a center and a size
            draw_line(
                scale * getattr(value["center"], axis),
                color=PLOT_COLORS[color],
                linestyle=":",
                linewidth=1.5,
                label=name,
            )
        else:  # a list of flux-surface faces
            for i, region in enumerate(value):
                if in_nm:
                    draw_line(
                        scale * getattr(region.center, axis),
                        color=PLOT_COLORS[color],
                        linestyle=":",
                        linewidth=1.5,
                        label=name if i == 0 else None,
                    )
                else:
                    _draw_segment(
                        ax,
                        region.center,
                        region.size,
                        vertical,
                        PLOT_COLORS[color],
                        ":",
                        name if i == 0 else None,
                    )


def _draw_segment(ax, center, size, vertical, color, linestyle, label):
    """Draw one plane/face as the segment it spans in the plan view."""

    x0 = center.x - 0.5 * size.x
    x1 = center.x + 0.5 * size.x
    v0 = getattr(center, vertical) - 0.5 * getattr(size, vertical)
    v1 = getattr(center, vertical) + 0.5 * getattr(size, vertical)
    ax.plot(
        [x0, x1], [v0, v1], color=color, linestyle=linestyle, linewidth=1.5, label=label
    )


def draw_molecules(cavity, ax, vertical=None):
    """
    Shade the MaxwellLink molecules placed in the cavity: those of the
    molecule-level route (``placed_molecules``, from ``place_molecule``) and
    the molecular media of the grid-level route (``placed_regions``, from
    ``place_region``).

    Parameters
    ----------
    cavity : DummyCavity subclass
        The cavity whose ``placed_molecules``/``placed_regions`` are drawn.
    ax : matplotlib Axes
        The axes to draw into.
    vertical : str or None, optional
        None for the 1D profile view (x-spans in nm); the vertical axis
        label ("y" or "z") for a plan view (rectangles in Meep units).
    """

    from matplotlib.patches import Rectangle

    def _clipped(cavity, size, axis):
        """Extent of a molecule along an axis, clipped to the cell."""
        return min(getattr(size, axis), getattr(cavity.cell_size, axis))

    for attr, color, alpha, label in _MOLECULES:
        for i, molecule in enumerate(getattr(cavity, attr)):
            sx = _clipped(cavity, molecule["size"], "x")
            if vertical is None:
                x0 = cavity.meep_to_nm(molecule["center"].x - 0.5 * sx)
                x1 = cavity.meep_to_nm(molecule["center"].x + 0.5 * sx)
                ax.axvspan(
                    x0,
                    x1,
                    color=PLOT_COLORS[color],
                    alpha=alpha,
                    label=label if i == 0 else None,
                )
            else:
                sv = _clipped(cavity, molecule["size"], vertical)
                ax.add_patch(
                    Rectangle(
                        (
                            molecule["center"].x - 0.5 * sx,
                            getattr(molecule["center"], vertical) - 0.5 * sv,
                        ),
                        sx,
                        sv,
                        facecolor=PLOT_COLORS[color],
                        edgecolor=PLOT_COLORS[color],
                        alpha=alpha,
                        linewidth=1.0,
                        label=label if i == 0 else None,
                    )
                )


# -------------- assembled cavity views --------------


def _fetch_setup(cavity, setup):
    """Resolve the ``setup=`` argument of the ``plot_cavity`` family: the
    optical (far-field) or the emission (local-dipole) setup dict."""

    if setup == "optical":
        try:
            return cavity.optical_setup()
        except NotImplementedError:
            return None
    if setup == "emission":
        return cavity.emission_setup()
    raise ValueError(f"setup must be 'optical' or 'emission', not {setup!r}.")


def plot_cavity_1d(cavity, ax=None, setup="optical"):
    """
    Draw the 1D view of a cavity with a refractive-index profile n(x)

    Parameters
    ----------
    cavity : DummyCavity subclass
        The cavity to draw (its x-axis, in nm).
    ax : matplotlib Axes or None, optional
        Axes to draw into. A new figure is created when None.
    setup : str, default: "optical"
        Which measurement setup to draw: the far-field probe ("optical") or
        the local-dipole one ("emission").

    Returns
    -------
    matplotlib Axes
        The axes containing the plot.
    """

    import meep as mp
    import matplotlib.pyplot as plt

    with use_style():
        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 3.2), constrained_layout=True)
        # sample the dielectric profile that Meep actually discretizes
        sim = mp.Simulation(**cavity.sim_kwargs())
        sim.init_sim()
        eps = sim.get_array(
            center=mp.Vector3(),
            size=mp.Vector3(cavity.cell_size.x),
            component=mp.Dielectric,
        )
        half_nm = cavity.meep_to_nm(0.5 * cavity.cell_size.x)
        x_nm = np.linspace(-half_nm, half_nm, len(eps))

        if cavity.pml_thickness is not None:
            pml_nm = cavity.meep_to_nm(cavity.pml_thickness)
            ax.axvspan(
                -half_nm,
                -half_nm + pml_nm,
                color=PLOT_COLORS["lightblue_background"],
                label="PML",
                zorder=0,
            )
            ax.axvspan(
                half_nm - pml_nm,
                half_nm,
                color=PLOT_COLORS["lightblue_background"],
                zorder=0,
            )
        lo_nm, hi_nm = cavity.allowed_bounds_nm["x"]
        ax.axvspan(
            lo_nm,
            hi_nm,
            color=PLOT_COLORS["yellow"],
            alpha=0.15,
            label="allowed region",
            zorder=0,
        )
        draw_molecules(cavity, ax)
        ax.axvline(
            cavity.meep_to_nm(cavity.hotspot_center.x),
            color=PLOT_COLORS["red"],
            linestyle="--",
            linewidth=1.5,
            label="hotspot",
        )
        draw_optical_planes(cavity, ax, in_nm=True, setup=_fetch_setup(cavity, setup))
        ax.plot(x_nm, np.sqrt(eps), color=PLOT_COLORS["navy_blue"], linewidth=1.8)
        polish_axes(ax, xlabel="x (nm)", ylabel="refractive index n")
    return ax


def plot_cavity_2d(cavity, ax=None, setup="optical", **kwargs):
    """
    Draw the plan view of a cavity via ``mp.Simulation.plot2D`` (2D, 3D, and cylindrical cells).
    In 3D the default view is the x-z plane through the cell center (override with ``output_plane=``).

    Parameters
    ----------
    cavity : DummyCavity subclass
        The cavity to draw.
    ax : matplotlib Axes or None, optional
        Axes to draw into. A new figure is created when None.
    setup : str, default: "optical"
        Which measurement setup to draw: the far-field probe ("optical") or
        the local-dipole one ("emission").
    **kwargs
        Forwarded to ``mp.Simulation.plot2D``.

    Returns
    -------
    matplotlib Axes
        The axes containing the plot.
    """

    import meep as mp

    with use_style():
        if cavity.dimensions == 3 and "output_plane" not in kwargs:
            kwargs["output_plane"] = mp.Volume(
                size=mp.Vector3(cavity.cell_size.x, 0, cavity.cell_size.z)
            )
        sim = mp.Simulation(**cavity.sim_kwargs())
        ax = sim.plot2D(ax=ax, **kwargs)
        draw_optical_planes(cavity, ax, in_nm=False, setup=_fetch_setup(cavity, setup))
        # vertical axis of the plotted plane: y in 2D, z in the 3D default
        # view and in the cylindrical r-z plane
        draw_molecules(cavity, ax, vertical="y" if cavity.dimensions == 2 else "z")
        polish_axes(ax, despine=False)
    return ax


def plot_cavity(cavity, ax=None, setup="optical", **kwargs):
    """
    Visualize a cavity in MEEP cavity module (src/maxwelllink/cavity/).

    Parameters
    ----------
    cavity : DummyCavity subclass
        The cavity to draw.
    ax : matplotlib Axes or None, optional
        Axes to draw into. A new figure is created when None.
    setup : str, default: "optical"
        Which measurement setup to draw: the far-field probe ("optical") or
        the local-dipole one ("emission").
    **kwargs
        Forwarded to ``mp.Simulation.plot2D`` in the plan view.

    Returns
    -------
    matplotlib Axes
        The axes containing the plot.
    """

    import meep as mp

    if cavity.dimensions > 1 or cavity.dimensions == mp.CYLINDRICAL:
        return plot_cavity_2d(cavity, ax=ax, setup=setup, **kwargs)
    return plot_cavity_1d(cavity, ax=ax, setup=setup)
