"""Paper-friendly font sizes and subplot-label helpers for the diagnostic plots.

Reviewers and co-authors consistently ask for two things on every paper
draft: bigger fonts across titles/axes/legends, and ``a.)``-style subplot
labels.  Encoding those choices in one shared module — rather than in
each ``cases/<event>.yaml`` — keeps the styling consistent across cases
(the same figure type should look the same regardless of which storm it
shows) and makes future tweaks a one-line edit.

Two size profiles are provided.  ``STANDARD`` covers the common
1x1 / 1x2 / 1x3 / Nx1 layouts; ``DENSE`` covers grids with three or more
columns AND three or more rows (e.g. ``4x3`` ageostrophic, ``4x4``
free-end-states) where the standard sizes would crowd the tick labels.
"""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# --------------------------------------------------------------------------
# Font-size constants
# --------------------------------------------------------------------------

# Standard layouts (1x1, 1x2, 1x3, Nx1 up to ~3 panels).
TITLE_SIZE = 16
SUPTITLE_SIZE = 20
SECTION_HEADER_SIZE = 17
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_SIZE = 11
SUBPLOT_LABEL_SIZE = 14
COLORBAR_LABEL_SIZE = 13
COLORBAR_TICK_SIZE = 11

# Dense layouts (e.g. 4x3, 4x4) where the standard sizes would crowd ticks.
TITLE_SIZE_DENSE = 15
SUPTITLE_SIZE_DENSE = 19
SECTION_HEADER_SIZE_DENSE = 16
AXIS_LABEL_SIZE_DENSE = 12
TICK_LABEL_SIZE_DENSE = 10
LEGEND_SIZE_DENSE = 10
SUBPLOT_LABEL_SIZE_DENSE = 13
COLORBAR_LABEL_SIZE_DENSE = 12
COLORBAR_TICK_SIZE_DENSE = 10


# --------------------------------------------------------------------------
# Subplot labels (a.), b.), c.) ...)
# --------------------------------------------------------------------------


def _default_labels(n: int) -> list[str]:
    """Return ``["a.)", "b.)", ..., "z.)"]`` up to ``n`` items."""
    if n > 26:
        raise ValueError(
            f"add_subplot_labels: default labels only cover a-z ({n} requested); "
            "pass explicit `labels=[...]` for larger grids."
        )
    return [f"{chr(ord('a') + i)}.)" for i in range(n)]


def _flatten_axes(axes: Axes | Sequence[Axes] | np.ndarray) -> list[Axes]:
    """Flatten a possibly-2D ``axes`` container into a list of ``Axes``."""
    if isinstance(axes, Axes):
        return [axes]
    arr = np.asarray(axes, dtype=object).ravel()
    return [ax for ax in arr.tolist() if isinstance(ax, Axes)]


def add_subplot_labels(
    axes: Axes | Sequence[Axes] | np.ndarray,
    labels: Sequence[str] | None = None,
    *,
    placement: str = "outside",
    fontsize: float = SUBPLOT_LABEL_SIZE,
    fontweight: str = "bold",
    x: float | None = None,
    y: float | None = None,
) -> None:
    """Annotate each subplot with a label like ``a.)``, ``b.)``, ``c.)``.

    Parameters
    ----------
    axes
        A single ``Axes``, a 1D/2D ``np.ndarray`` of ``Axes`` (as returned by
        ``plt.subplots``), or any flat sequence of ``Axes``.
    labels
        Optional explicit label strings.  When ``None`` (default), uses
        ``["a.)", "b.)", ...]``.
    placement
        ``"outside"`` (default) places labels above-left of each subplot, just
        outside the axes; ``"inside"`` places them inside the top-left corner
        of the axes with a small white background patch so they stay legible
        over busy plots (maps, log-log spectra, etc.).
    fontsize, fontweight
        Standard matplotlib text styling.
    x, y
        Optional override of the default placement coordinates in
        ``ax.transAxes``.  Useful for the rare panel that needs a nudge.
    """
    axs = _flatten_axes(axes)
    if not axs:
        return
    lbls = list(labels) if labels is not None else _default_labels(len(axs))
    if len(lbls) < len(axs):
        raise ValueError(
            f"add_subplot_labels: got {len(lbls)} labels for {len(axs)} axes"
        )

    if placement == "outside":
        x_pos = -0.02 if x is None else x
        y_pos = 1.06 if y is None else y
        bbox = None
        va, ha = "bottom", "right"
    elif placement == "inside":
        x_pos = 0.02 if x is None else x
        y_pos = 0.97 if y is None else y
        bbox = dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2.0)
        va, ha = "top", "left"
    else:
        raise ValueError(
            f"add_subplot_labels: unknown placement '{placement}' "
            "(use 'outside' or 'inside')"
        )

    for ax, lbl in zip(axs, lbls):
        ax.text(
            x_pos,
            y_pos,
            lbl,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
            va=va,
            ha=ha,
            bbox=bbox,
        )


# --------------------------------------------------------------------------
# Convenience: bulk-apply fontsizes to an existing axis
# --------------------------------------------------------------------------


def style_axes(
    ax: Axes,
    *,
    title_size: float = TITLE_SIZE,
    label_size: float = AXIS_LABEL_SIZE,
    tick_size: float = TICK_LABEL_SIZE,
    legend_size: float | None = LEGEND_SIZE,
) -> None:
    """Apply the paper font sizes to ``ax`` after the artists are drawn.

    Useful when an axis label is set by a helper that does not take an
    explicit ``fontsize`` kwarg (e.g. ``format_local_time_axis``).  Legend
    fontsize is only applied if a legend exists.
    """
    ax.title.set_fontsize(title_size)
    ax.xaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontsize(label_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    if legend_size is not None:
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(legend_size)


def style_colorbar(
    cbar: "plt.cm.ScalarMappable",
    *,
    label_size: float = COLORBAR_LABEL_SIZE,
    tick_size: float = COLORBAR_TICK_SIZE,
) -> None:
    """Apply the paper font sizes to a colorbar's label and tick labels."""
    if cbar.ax.yaxis.label.get_text() or cbar.ax.xaxis.label.get_text():
        cbar.ax.yaxis.label.set_fontsize(label_size)
        cbar.ax.xaxis.label.set_fontsize(label_size)
    cbar.ax.tick_params(labelsize=tick_size)


# --------------------------------------------------------------------------
# Geographic coordinate formatters
# --------------------------------------------------------------------------
#
# Publication-ready lat/lon strings for axis labels, panel titles, and
# bbox annotations.  Pure formatters — no matplotlib dependency — but
# they live here because their purpose is figure text consistency.
# Integer-valued inputs format as ``38°N``; fractional inputs format as
# ``38.5°N`` (1 decimal).  Convention: positive lon -> ``E``, negative
# -> ``W``; positive lat -> ``N``, negative -> ``S``; zero takes no
# hemisphere suffix.


def _fmt_deg(magnitude: float) -> str:
    """Render an unsigned degree value as ``38°`` or ``38.5°``."""
    return (
        f"{int(magnitude)}°" if float(magnitude).is_integer() else f"{magnitude:.1f}°"
    )


def format_lon(x: float) -> str:
    """Format a longitude in degrees as e.g. ``140°W`` / ``110°E`` / ``0°``."""
    s = _fmt_deg(abs(x))
    if x > 0:
        return s + "E"
    if x < 0:
        return s + "W"
    return s


def format_lat(x: float) -> str:
    """Format a latitude in degrees as e.g. ``38°N`` / ``25°S`` / ``0°``."""
    s = _fmt_deg(abs(x))
    if x > 0:
        return s + "N"
    if x < 0:
        return s + "S"
    return s


def format_bbox(bbox: Sequence[float]) -> str:
    """Format a cartopy-style extent ``[lon_min, lon_max, lat_min, lat_max]``.

    Returns e.g. ``"140°W–110°W, 38°N–55°N"`` — uses cardinal-direction
    notation rather than negative numbers because that reads better in
    a publication figure.  Dateline-crossing bboxes are not supported
    (they are rejected upstream in the compute scripts).
    """
    lon_min, lon_max, lat_min, lat_max = (float(v) for v in bbox)
    return (
        f"{format_lon(lon_min)}–{format_lon(lon_max)}, "
        f"{format_lat(lat_min)}–{format_lat(lat_max)}"
    )


# --------------------------------------------------------------------------
# Computation-domain box overlay
# --------------------------------------------------------------------------
#
# Diagnostics that reduce a field over a fixed lat/lon box (PCA, bbox-mean
# impact, in-domain track counts, ...) read more clearly when downstream
# maps mark *where* that box sits.  A single shared drawer keeps the cue
# identical across every diagnostic figure.


def draw_domain_box(
    ax: Axes,
    bbox: Sequence[float],
    *,
    transform: Any = None,
    edgecolor: str = "black",
    linewidth: float = 1.4,
    linestyle: str = "--",
    halo: bool = True,
    halo_color: str = "white",
    zorder: float = 6.0,
    clip_on: bool = False,
) -> mpatches.Rectangle:
    """Outline a fixed lon/lat computation domain on a map axis.

    Draws a dashed rectangle around ``bbox`` (cartopy extent convention
    ``[lon_min, lon_max, lat_min, lat_max]``) to show where a fixed-domain
    diagnostic was computed.  A white halo under the dashed edge keeps the
    box legible over busy fields — diverging anomaly colormaps, coastlines,
    track spaghetti.

    Two usages share this drawer:

    * a *sub-domain* box on a wider map (e.g. the impact bbox / TC-track
      domain on a synoptic precursor panel), where the box sits inside the
      view, and
    * a *frame* box on a map already clipped to the domain (e.g. the
      free-end-states bbox, the TC-tracking domain), where the box traces
      the panel edge — hence ``clip_on`` defaults to ``False`` so it is not
      hidden behind the axes spine.

    Parameters
    ----------
    ax
        The cartopy ``GeoAxes`` (or plain ``Axes``) to draw on.
    bbox
        ``[lon_min, lon_max, lat_min, lat_max]`` in degrees.
    transform
        Coordinate system of ``bbox``.  Defaults to
        ``cartopy.crs.PlateCarree()`` — the right choice for a lon/lat box
        on a ``GeoAxes``; pass ``ax.transData`` for a non-geographic axis.
    edgecolor, linewidth, linestyle, zorder
        Standard matplotlib styling for the box edge.
    halo, halo_color
        When ``halo`` is true, stroke a wider ``halo_color`` outline under
        the edge so the dashed line reads over either tail of a diverging
        colormap.
    clip_on
        When false (default), the box is drawn even where it coincides with
        the axes boundary, so a box at the exact map extent stays visible.

    Returns
    -------
    matplotlib.patches.Rectangle
        The patch added to ``ax`` (returned so callers can restyle it or
        build a legend handle).
    """
    lon_min, lon_max, lat_min, lat_max = (float(v) for v in bbox)
    if transform is None:
        # Local import keeps this module cartopy-free for the non-map
        # diagnostics (timeseries, spectra) that only want the font sizes.
        import cartopy.crs as ccrs

        transform = ccrs.PlateCarree()
    rect = mpatches.Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        transform=transform,
        zorder=zorder,
        clip_on=clip_on,
    )
    if halo:
        rect.set_path_effects(
            [path_effects.withStroke(linewidth=linewidth + 1.2, foreground=halo_color)]
        )
    ax.add_patch(rect)
    return rect


# --------------------------------------------------------------------------
# Lat/lon graticule
# --------------------------------------------------------------------------
#
# Every map figure in this pipeline draws the same reference furniture — a
# dashed, semi-transparent lat/lon graticule with labelled outer edges — so a
# reader can situate a regional panel without counting coastline features.
# The width/alpha/style are fixed here rather than exposed, for the same reason
# `draw_quiver` fixes its styling: one graticule look across the paper.
#
# Seven figures still carry this inline (`_map_panel` in aggregate_synoptic_pca,
# the four track figures, aggregate_free_end_states, tc_landfall_split, and
# plot_ensemble_member's video layout).  They are visually equivalent already;
# migrate each when something else touches that file, rather than editing
# committed paper figures for a cosmetic consolidation.

GRATICULE_LW = 0.4
GRATICULE_ALPHA = 0.5
GRATICULE_STYLE = "--"


def draw_graticule(
    ax: Any,
    *,
    show_xlabels: bool = True,
    show_ylabels: bool = True,
    color: str = "gray",
    fontsize: float = TICK_LABEL_SIZE_DENSE,
    max_ticks: int | None = None,
) -> Any:
    """Draw the shared lat/lon graticule, labelling only the requested edges.

    Top and right labels are always off.  In a multi-panel grid, pass
    ``show_xlabels`` only for the bottom row and ``show_ylabels`` only for the
    left column: interior panels share their neighbours' axes, so repeating the
    labels on every panel costs legibility and buys nothing.  ``max_ticks``
    caps the label count per axis — cartopy's automatic locator picks a step
    for a full-page map, which on a panel a twelfth of a page runs the labels
    together (``96°W93°W90°W...``).  Returns the ``Gridliner``.

    **Caller obligation:** on an axes carrying labels, set the panel title with
    an explicit ``y`` (``ax.set_title(..., y=1.0)``).  Cartopy's ``Gridliner``
    takes over the title's automatic vertical placement, and on **matplotlib
    >= 3.11** it resolves the position to ``inf`` with top labels disabled,
    which silently drops the title *and* poisons
    ``savefig(bbox_inches="tight")`` with a NaN bounding box, cropping the
    figure.  Measured either side of it on the same cartopy 0.25.0: matplotlib
    3.11.1 gives ``(0.5, inf)``, matplotlib 3.10.9 gives ``(0.5, 1.0)``.  So an
    explicit ``y=1.0`` restores the title where the bug bites and is exactly
    what the automatic placement already chooses where it does not — safe to
    pass unconditionally, and the reason to keep passing it is that an
    environment rebuild onto a newer matplotlib would otherwise break figures
    silently.
    """
    import cartopy.crs as ccrs  # local: keep this module cartopy-free

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=show_xlabels or show_ylabels,
        linewidth=GRATICULE_LW,
        alpha=GRATICULE_ALPHA,
        linestyle=GRATICULE_STYLE,
        color=color,
    )
    if max_ticks is not None:
        import matplotlib.ticker as mticker

        gl.xlocator = mticker.MaxNLocator(max_ticks)
        gl.ylocator = mticker.MaxNLocator(max_ticks)
    gl.xlabel_style = {"size": fontsize}
    gl.ylabel_style = {"size": fontsize}
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = bool(show_xlabels)
    gl.left_labels = bool(show_ylabels)
    return gl


# --------------------------------------------------------------------------
# Raw-flow quiver overlay
# --------------------------------------------------------------------------
#
# Raw-flow steering arrows (a subsampled quiver) overlaid on a map panel.
# Subtle by design so they read as context, not the headline: a thin
# dark-gray shaft, semi-transparent, on a coarse grid, with one
# reference-scale key per figure.  Gray (not blue) so it does not clash with
# a diverging anomaly field underneath.  No scipy needed (unlike cartopy
# ``streamplot``) -- the grid is subsampled directly.
#
# Lifted verbatim out of ``aggregate_synoptic_pca.py`` (where it draws the
# paper's synoptic-precursor arrows) so the member-grid overlay in
# ``plot_ensemble_member.py`` draws the *same* arrows rather than a second
# convention that only looks similar.  Both call these.

QUIVER_COLOR = "#303030"
QUIVER_ALPHA = 0.7
QUIVER_WIDTH = 0.004  # shaft width as a fraction of the panel width
QUIVER_HEADWIDTH = 3.0
QUIVER_HEADLENGTH = 3.5
QUIVER_TARGET_ACROSS = 22  # ~arrows along the panel's longer dimension
QUIVER_REF_LEN_FRAC = 0.07  # a reference-speed arrow spans this fraction of width


def quiver_step(lat: np.ndarray, lon: np.ndarray) -> int:
    """Subsample stride so the quiver shows ~``QUIVER_TARGET_ACROSS`` arrows."""
    return max(1, int(round(max(lat.size, lon.size) / QUIVER_TARGET_ACROSS)))


def nice_speed(x: float) -> float:
    """Round a wind speed to a tidy reference value for the quiver key."""
    cand = np.array([0.5, 1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100], dtype=float)
    return float(cand[int(np.argmin(np.abs(cand - x)))]) if x > 0 else 1.0


def draw_quiver(
    ax: Any,
    u: np.ndarray,
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    scale: float,
    step: int,
) -> Any:
    """Overlay a subsampled raw-flow quiver on a map panel.

    ``u``/``v`` are on the same ``(lat, lon)`` grid as the field already drawn.
    The grid is subsampled by ``step`` so the arrows stay sparse and subtle
    (context, not the headline), and ``scale`` (shared across the figure's
    panels, ``scale_units="width"``) keeps every panel's arrows comparable.
    Returns the quiver handle so the caller can attach one reference key via
    :func:`draw_quiver_key`.  No scipy dependency (unlike cartopy
    ``streamplot``).

    No styling knobs on purpose: a second caller that recoloured or restroked
    the arrows would be a second convention wearing this one's name.  A white
    halo was tried for the ``cmr.sapphire`` member grids, whose dark end is the
    storm core, and read worse — at panel size the stroke turns every arrow
    into a fat dash and buries the field it sits on.
    """
    import cartopy.crs as ccrs  # local: keep this module cartopy-free

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.size < 2 or lon.size < 2:
        return None
    ys = slice(None, None, step)
    xs = slice(None, None, step)
    lon_g, lat_g = np.meshgrid(lon[xs], lat[ys])
    return ax.quiver(
        lon_g,
        lat_g,
        u[ys][:, xs],
        v[ys][:, xs],
        transform=ccrs.PlateCarree(),
        color=QUIVER_COLOR,
        alpha=QUIVER_ALPHA,
        scale=scale,
        scale_units="width",
        width=QUIVER_WIDTH,
        headwidth=QUIVER_HEADWIDTH,
        headlength=QUIVER_HEADLENGTH,
        zorder=3,
    )


def draw_quiver_key(
    ax: Any,
    handle: Any,
    magnitude: float,
    *,
    x: float = 0.16,
    y: float = 0.07,
    fontsize: float = TICK_LABEL_SIZE_DENSE,
) -> Any:
    """Attach the single reference-scale key for a figure's quiver.

    One key per *figure*, not per panel: with a shared ``scale`` every panel's
    arrows already mean the same thing, so a second key would only repeat it.
    Draw it on the first panel that produced arrows.  ``x``/``y`` are in axes
    coordinates, for the rare panel whose bottom-left corner is busy.
    """
    return ax.quiverkey(
        handle,
        x,
        y,
        magnitude,
        f"{magnitude:g} m/s",
        labelpos="E",
        coordinates="axes",
        color=QUIVER_COLOR,
        fontproperties={"size": fontsize},
    )
