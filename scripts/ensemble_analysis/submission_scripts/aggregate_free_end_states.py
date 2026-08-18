"""Aggregate free-end-state plots across conditioning modes.

Reads the two per-mode ``free_end_states_<mode>.npz`` files written by
``compute_free_end_states.py`` (one for ``start``, one for ``end``) and
renders a single multi-row figure per variable: an ERA5 reference panel
plus member-minus-ERA5 difference panels at the free-end frame, with
members ranked by their signed box-mean departure from ERA5 (the
coolest/lowest tail above the warmest/highest tail), followed by a
companion time-series panel showing per-member bbox domain-mean scalars
over every lead frame.  ``end`` mode goes on top because
end-conditioning is the novel capability the paper emphasizes; a single
section header above each mode block names the mode rather than
duplicating that label across row labels and panel titles.

Layout (one PNG per variable, both modes present; the two spatial rows are
the signed-bias tails -- coolest/warmest for temperature, lowest/highest
otherwise)::

      [ "End conditioning" header ]
      Row 1  END   coolest:   [ ERA5(t=0)  | (M1-ERA5) | (M2-ERA5) | (M3-ERA5) ]
      Row 2  END   warmest:   [ ERA5(t=0)  | (M1-ERA5) | (M2-ERA5) | (M3-ERA5) ]
      Row 3  END   time series:                [ <------ 4-col-span ----> ]
      [ "Start conditioning" header ]
      Row 4  START coolest:   [ ERA5(t=11) | (M1-ERA5) | (M2-ERA5) | (M3-ERA5) ]
      Row 5  START warmest:   [ ERA5(t=11) | (M1-ERA5) | (M2-ERA5) | (M3-ERA5) ]
      Row 6  START time series:                [ <------ 4-col-span ----> ]

ERA5 panels use the variable's natural cmap (from ``cases/_palettes.yaml``)
with a vmin/vmax derived from the data — the 1st/99th percentile shared
across modes, matching ``aggregate_synoptic_pca``'s raw-panel convention so
the two figures show the same field on the same scale.  The
member-difference panels use a diverging cmap with ``±max_abs(member-ERA5)``
shared across the low tail, the high tail, and modes so the cool/low and
warm/high rows read honestly against each other.  Two colorbars sit on
opposite figure edges — ERA5 (absolute, variable cmap) on the LEFT,
member-minus-ERA5 (diverging) on the right — so neither edge stacks two bars.

The time-series panel draws every member as faint background spaghetti,
overlays the K low-tail members in sequential blues and the K high-tail in
sequential reds (matching the spatial-panel titles directly above), and
prints the ensemble mean and ERA5 on top.  Vertical markers show the
conditioning frame(s) and the free-end frame so the reader can see how
the spread fans out from the conditioned side.

The lead-time x-axis is rendered as the local wall-clock time at the
event location via ``local_time_axis.format_local_time_axis`` (Sandy in
EDT, PNW in PDT, etc.).  The variable label embeds MathText units via
``var_metadata.var_label`` and sits horizontally above the time-series
panel (rather than rotated on the left axis) so it doesn't collide with
the in-axes subplot label.  The legend is a single figure-level box at
the bottom: per-mode legends duplicated the same rank-color semantics
across panels, so we promote the legend once and let member ids be
cross-referenced from the spatial-panel titles directly above each
color.

Output::

    <output-dir>/free_end_states.png

Usage::

    python aggregate_free_end_states.py \\
        --output-dir /path/to/diagnostics/free_end_states/<variable>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import cmasher  # noqa: F401, E402  — registers "cmr.*" colormap names
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent.parent / "_shared"
sys.path.insert(0, str(SHARED_DIR))

from _dispatch_lib import load_palettes  # noqa: E402
from local_time_axis import format_local_time_axis  # noqa: E402
from plot_style import (  # noqa: E402
    AXIS_LABEL_SIZE_DENSE,
    COLORBAR_LABEL_SIZE_DENSE,
    COLORBAR_TICK_SIZE_DENSE,
    LEGEND_SIZE_DENSE,
    SECTION_HEADER_SIZE_DENSE,
    SUPTITLE_SIZE_DENSE,
    TICK_LABEL_SIZE_DENSE,
    TITLE_SIZE_DENSE,
    add_subplot_labels,
    draw_domain_box,
    format_bbox,
)
from var_metadata import (  # noqa: E402
    is_geopotential,
    to_display_units,
    units_tex,
    var_label,
)
from var_metadata import long_name as var_long_name  # noqa: E402
from var_metadata import units as var_units  # noqa: E402

MODE_ORDER = ("end", "start")  # rendered top-to-bottom; "end" first per paper emphasis.
MODE_LABEL = {"end": "End conditioning", "start": "Start conditioning"}

# Diverging colormap for the member-minus-ERA5 difference panels.  Symmetric
# around zero so the reader can read sign and magnitude at a glance; RdBu_r
# matches the convention used by other diff-style figures (positive = red).
DIFF_CMAP = "RdBu_r"

# Highlight palette for the time-series spaghetti.  Three sequential blues for
# the low (coolest/lowest) tail (rank 1 = darkest) and three sequential reds
# for the high (warmest/highest) tail; shared across modes so the same color
# always means the same rank.  Blue=low / red=high also matches the diverging
# difference colormap, so the rank colors read with temperature intuition.
# Colors are picked from matplotlib's "Blues_r" / "Reds_r" but spelled out so
# they stay stable across matplotlib versions.
LOW_HIGHLIGHTS = ["#08306b", "#2171b5", "#6baed6"]  # dark -> light
HIGH_HIGHLIGHTS = ["#67000d", "#cb181d", "#fb6a4a"]  # dark -> light

# Spaghetti style for the un-highlighted background members.
SPAGHETTI_COLOR = "0.55"
SPAGHETTI_ALPHA = 0.08
SPAGHETTI_LW = 0.4


def _scalar(arr: np.ndarray) -> object:
    return arr.item() if isinstance(arr, np.ndarray) and arr.shape == () else arr


def _tail_nouns(var_name: str) -> tuple[str, str]:
    """``(low_tail, high_tail)`` display nouns for the signed-bias rows.

    Members are ranked by signed box-mean departure from ERA5, so the two
    spatial rows are the cold/low tail and the warm/high tail.  Temperature
    fields read naturally as coolest/warmest; heights, pressures, and
    everything else as lowest/highest (units ``"K"`` flags a temperature).
    """
    if var_units(var_name) == "K":
        return ("Coolest", "Warmest")
    return ("Lowest", "Highest")


# npz keys holding the variable's physical field, or a linear field-unit
# quantity derived from it (a domain-mean scalar, a signed bias, or an RMSE).
# Converted from storage to display units once at load (see _to_display_units),
# so every downstream color limit, difference, panel, title, and colorbar is
# consistent.
_FIELD_UNIT_KEYS = (
    "era5_field",
    "low_fields",
    "high_fields",
    "member_bias",
    "low_bias",
    "high_bias",
    "low_rmse",
    "high_rmse",
    "member_scalar",
    "era5_scalar",
    "ensemble_mean_scalar",
)


def _to_display_units(diag: Dict[str, np.ndarray]) -> None:
    """In place, convert this mode's field-unit arrays to display units."""
    var_name = str(_scalar(diag["var_name"]))
    if not is_geopotential(var_name):
        return
    for key in _FIELD_UNIT_KEYS:
        if key in diag:
            diag[key] = to_display_units(var_name, np.asarray(diag[key]))


def _load_mode(output_dir: Path, mode: str) -> Optional[Dict[str, np.ndarray]]:
    path = output_dir / f"free_end_states_{mode}.npz"
    if not path.exists():
        print(f"WARNING: missing {path}, skipping mode '{mode}'")
        return None
    diag = dict(np.load(path, allow_pickle=True))
    _to_display_units(diag)
    return diag


def _variable_style(var_name: str) -> Dict[str, object]:
    """Look up the cmap + coastline color for ``var_name`` from the palette.

    The ERA5-panel color *limits* are derived from the data (see
    :func:`_shared_clim`), not the palette, so only the styling keys are
    read here.
    """
    palettes = load_palettes()
    var_defaults = palettes.get("variable_defaults", {}) or {}
    spec = var_defaults.get(var_name, {}) or {}
    return {
        "cmap": spec.get("cmap", "viridis"),
        "coastline_color": spec.get("coastline_color", "white"),
    }


def _shared_clim(per_mode: Dict[str, Dict[str, np.ndarray]]) -> tuple[float, float]:
    """Data-derived (vmin, vmax) for the ERA5 reference panels.

    Matches ``aggregate_synoptic_pca._shared_minmax`` (1st / 99th
    percentile) rather than the shared palette.  Now that the free-end
    panels share ``regions.synoptic`` as their field of view, the ERA5
    reference panel shows the *same* field over the *same* domain as the
    synoptic-PCA raw-ERA5 panel, so deriving the scale from the data keeps
    the two figures consistent.  It also sidesteps the palette being stored
    in raw units (``cases/_palettes.yaml`` is shared with raw-unit
    consumers like ``plot_ensemble_member.py``, so z500 lives there as
    m²/s²) while ``_to_display_units`` has already converted the field to
    display units (m): applying the raw palette limits to a converted field
    was what clamped the z500 ERA5 panels to one end of the colormap (the
    "plain light blue" panels).

    A percentile (not min/max) keeps a single outlier pixel from
    compressing the colorbar; taking the joint extrema across modes puts
    the end- and start-conditioned ERA5 panels on one shared scale.
    """
    los: list[float] = []
    his: list[float] = []
    for diag in per_mode.values():
        arr = np.asarray(diag["era5_field"], dtype=np.float32)
        if arr.size:
            los.append(float(np.nanpercentile(arr, 1.0)))
            his.append(float(np.nanpercentile(arr, 99.0)))
    if not los:
        return 0.0, 1.0
    return min(los), max(his)


def _excluded_mask(diag: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Return the (ny, nx) excluded-region mask if the compute saved one.

    ``compute_free_end_states.py`` only writes ``excluded_mask_box`` when
    ``--mask != 'none'``, so older npz files (or unmasked runs) hit the
    ``None`` path and every downstream caller just skips the masked-pixel
    logic.  We also return ``None`` for an all-False mask: ``compute``
    can't produce one (it errors out if the bbox ∩ metric mask is empty),
    but defending against it keeps the aggregator robust to hand-built
    test fixtures and matches the "no excluded region → no overlay"
    visual contract.
    """
    mask = diag.get("excluded_mask_box")
    if mask is None:
        return None
    mask_arr = np.asarray(mask)
    if mask_arr.size == 0 or not mask_arr.any():
        return None
    return mask_arr.astype(bool)


def _shared_diff_clim(per_mode: Dict[str, Dict[str, np.ndarray]]) -> float:
    """Symmetric ``±max_abs`` cap for the member-minus-ERA5 diff panels.

    Shared across the low tail, the high tail, and modes so the cool/low
    and warm/high rows sit on one symmetric scale and read honestly
    against each other.  When an excluded-region mask is present
    (``--mask land`` or ``sea``), the excluded pixels are ignored when
    picking the cap so an ocean outlier in a land-masked t2m figure can't
    blow out the diverging scale and flatten the on-land signal.
    """
    max_abs = 0.0
    for diag in per_mode.values():
        era5 = np.asarray(diag["era5_field"])
        excluded = _excluded_mask(diag)
        for key in ("low_fields", "high_fields"):
            diff = np.asarray(diag[key]) - era5[None, :, :]
            if excluded is not None:
                diff = np.where(excluded[None, :, :], np.nan, diff)
            max_abs = max(max_abs, float(np.nanmax(np.abs(diff))))
    return max_abs if max_abs > 0 else 1.0


def _draw_panel(
    ax: plt.Axes,
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    coastline_color: str,
    excluded_mask: Optional[np.ndarray] = None,
    show_xlabels: bool = False,
    show_ylabels: bool = False,
) -> object:
    """Pcolormesh ``field`` on a PlateCarree axis with coastlines + gridlines.

    When ``excluded_mask`` (same (ny, nx) shape as ``field``) is given,
    pixels marked True are NaN-ed out of the pcolormesh (transparent
    via ``cmap.set_bad('none')``) and overlaid with diagonal hatching
    — a standard cartographic convention for "not used in the
    statistic".  This makes the metric domain visible at a glance
    without crowding panel titles with "over land" qualifiers.

    Dashed gray gridlines are drawn to match ``aggregate_synoptic_pca``;
    lat/lon tick labels are only rendered on the figure's edge panels
    (``show_ylabels`` on the left column, ``show_xlabels`` on the bottom
    spatial row) to keep the dense grid uncluttered.
    """
    # Copy so we don't mutate the registered cmap globally (other figs
    # in the same process would inherit set_bad otherwise).
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("none")
    field_to_plot = (
        np.where(excluded_mask, np.nan, field) if excluded_mask is not None else field
    )
    mesh = ax.pcolormesh(
        lon,
        lat,
        field_to_plot,
        transform=ccrs.PlateCarree(),
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    if excluded_mask is not None:
        # contourf renders hatching on the True region with no fill
        # colour, leaving the underlying cartopy ocean/land features
        # visible.  Levels [0.5, 1.5] selects pixels where the bool
        # mask is True (cast to {0., 1.}).
        ax.contourf(
            lon,
            lat,
            excluded_mask.astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["///"],
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
    ax.coastlines(resolution="50m", color=coastline_color, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, edgecolor=coastline_color, linewidth=0.3)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=show_xlabels or show_ylabels,
        linewidth=0.3,
        color="gray",
        alpha=0.4,
        linestyle="--",
    )
    gl.xlabel_style = {"size": TICK_LABEL_SIZE_DENSE}
    gl.ylabel_style = {"size": TICK_LABEL_SIZE_DENSE}
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = bool(show_xlabels)
    gl.left_labels = bool(show_ylabels)
    ax.set_title(title, fontsize=TITLE_SIZE_DENSE)
    return mesh


def _draw_timeseries(
    ax: plt.Axes,
    diag: Dict[str, np.ndarray],
    *,
    var_name: str,
    mode: str,
    start_time_iso: str,
    timezone_name: str,
) -> None:
    """Per-member bbox domain-mean spaghetti for one (variable, mode).

    Background: every member as faint gray.  Foreground: the low
    (coolest/lowest) tail in sequential blues, the high (warmest/highest)
    tail in sequential reds, then the ensemble mean (thick black solid)
    and ERA5 (thick black dashed).
    Vertical markers show the conditioning frame(s) and the free-end
    frame so the reader can see the spread fan out from the conditioned
    side.  The x-axis is replaced with local wall-clock ticks via
    ``format_local_time_axis``; the legend is built once at figure level
    by :func:`_build_timeseries_legend_handles` so it can sit outside
    the panel rather than crowding the lines.  No subplot title is set
    here — the per-mode section header in :func:`make_figure` already
    identifies the mode block this panel belongs to.
    """
    hours = np.asarray(diag["lead_hours_all"])  # [n_leads]
    member_scalar = np.asarray(diag["member_scalar"])  # [N, n_leads]
    era5_scalar = np.asarray(diag["era5_scalar"])  # [n_leads]
    ens_mean_scalar = np.asarray(diag["ensemble_mean_scalar"])  # [n_leads]
    low_idx = np.asarray(diag["low_idx"]).astype(int)
    high_idx = np.asarray(diag["high_idx"]).astype(int)

    # Background spaghetti — all members in faint gray.
    for i in range(member_scalar.shape[0]):
        ax.plot(
            hours,
            member_scalar[i],
            color=SPAGHETTI_COLOR,
            alpha=SPAGHETTI_ALPHA,
            linewidth=SPAGHETTI_LW,
            zorder=1,
        )

    # Highlighted low (coolest/lowest) tail — sequential blues, rank 1 darkest.
    for k, idx in enumerate(low_idx):
        if k >= len(LOW_HIGHLIGHTS):
            break
        ax.plot(
            hours,
            member_scalar[idx],
            color=LOW_HIGHLIGHTS[k],
            linewidth=1.6,
            zorder=3,
        )

    # Highlighted high (warmest/highest) tail — sequential reds, rank 1 darkest.
    for k, idx in enumerate(high_idx):
        if k >= len(HIGH_HIGHLIGHTS):
            break
        ax.plot(
            hours,
            member_scalar[idx],
            color=HIGH_HIGHLIGHTS[k],
            linewidth=1.6,
            zorder=3,
        )

    # Ensemble mean (solid black) + ERA5 (dashed black) on top.
    ax.plot(hours, ens_mean_scalar, color="black", linewidth=2.4, zorder=5)
    ax.plot(hours, era5_scalar, color="black", linewidth=2.4, linestyle="--", zorder=5)

    # MathText-aware label (e.g. "m$^{2}$/s$^{2}$") via the shared
    # var_metadata helper, placed horizontally above the panel rather
    # than as a rotated y-axis label.  Horizontal placement avoids the
    # multi-line rotated label colliding with the in-axes subplot
    # labels (q.) / r.) at top-left, and reads more naturally as a
    # panel header now that the per-mode subplot title is gone.  Local
    # wall-clock x-axis via the shared local_time_axis helper keeps every
    # aggregator's lead-axis consistent (Sandy in EDT, PNW in PDT, etc.).
    bbox_text = format_bbox(np.asarray(diag["bbox"]))
    mask_kind = str(_scalar(diag.get("mask_kind", np.array("none"))))
    region_text = {
        "land": f"land within {bbox_text}",
        "sea": f"ocean within {bbox_text}",
    }.get(mask_kind, bbox_text)
    ax.set_title(
        f"{var_label(var_name)} — mean over {region_text}",
        fontsize=AXIS_LABEL_SIZE_DENSE,
    )
    format_local_time_axis(ax, hours, start_time_iso, timezone_name)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE_DENSE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE_DENSE)
    ax.set_xlim(float(hours[0]), float(hours[-1]))
    ax.grid(True, alpha=0.3)


def _build_timeseries_legend_handles(
    low_noun: str, high_noun: str
) -> tuple[list, list]:
    """Static legend proxies that describe the time-series styling.

    Member indices vary per mode (the coolest member in END is not the
    same member as the coolest in START), so the legend identifies ranks
    by color rather than member number — the actual member ids appear in
    the spatial-panel titles directly above each rank-colored line.
    """
    from matplotlib.lines import Line2D

    handles: list[Line2D] = []
    labels: list[str] = []
    for k, color in enumerate(LOW_HIGHLIGHTS):
        handles.append(Line2D([0], [0], color=color, linewidth=1.6))
        labels.append(f"{low_noun} #{k + 1}")
    for k, color in enumerate(HIGH_HIGHLIGHTS):
        handles.append(Line2D([0], [0], color=color, linewidth=1.6))
        labels.append(f"{high_noun} #{k + 1}")
    handles.append(Line2D([0], [0], color="black", linewidth=2.4))
    labels.append("Ensemble mean")
    handles.append(Line2D([0], [0], color="black", linewidth=2.4, linestyle="--"))
    labels.append("ERA5")
    handles.append(
        Line2D(
            [0],
            [0],
            color=SPAGHETTI_COLOR,
            alpha=SPAGHETTI_ALPHA * 6,  # bump alpha so the swatch is legible
            linewidth=SPAGHETTI_LW * 2,
        )
    )
    labels.append("Other members")
    return handles, labels


def make_figure(
    per_mode: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    any_mode = next(iter(per_mode.values()))
    var_name = str(_scalar(any_mode["var_name"]))
    case_name = str(_scalar(any_mode["case_name"]))
    pretty_var = var_long_name(var_name)
    units = units_tex(var_name)
    unit_suffix = f" {units}" if units else ""
    # Variable-aware nouns for the two signed-bias rows: coolest/warmest for
    # temperature, lowest/highest otherwise.
    low_noun, high_noun = _tail_nouns(var_name)
    style = _variable_style(var_name)
    era5_vmin, era5_vmax = _shared_clim(per_mode)
    diff_max = _shared_diff_clim(per_mode)
    era5_cmap = style["cmap"]
    coastline_color = style["coastline_color"]
    # If every present mode used the same non-"none" mask, surface that
    # qualifier on the diff colorbar label so the reader knows the
    # symmetric scale + the bias numbers in the panel titles are
    # land/ocean-only.  Mixed masks across modes (rare in practice) fall
    # back to no qualifier — the hatched overlay already makes the
    # metric domain visible per panel.
    mask_kinds = {
        str(_scalar(d.get("mask_kind", np.array("none")))) for d in per_mode.values()
    }
    uniform_mask = mask_kinds.pop() if len(mask_kinds) == 1 else "none"
    metric_region_word = {"land": "land", "sea": "ocean"}.get(uniform_mask)

    modes = [m for m in MODE_ORDER if m in per_mode]
    n_cols = 4
    # Per mode: 2 spatial rows (low + high tail) + 1 time-series row that
    # spans every column.  Row heights are tuned against the panels' aspect:
    # cartopy preserves the map aspect ratio inside the cell, so a row much
    # taller than the natural map height wastes vertical space.  The panels
    # now show the wider ``regions.synoptic`` view (landscape, aspect ~1.2–1.7
    # wide), so each ~3.4-inch-wide column map is only ~2.2 inches tall; 2.5
    # inches leaves room for the 2-line panel titles without a big gap below
    # each map (the old 3.0 was tuned for the near-square impact-box maps).
    rows_per_mode = 3
    spatial_row_h = 2.5
    ts_row_h = 2.8
    # Spacer row inserted between mode blocks.  Without it the next
    # mode's section header lands on top of the previous time series'
    # x-tick labels, since within-mode hspace alone is not enough to
    # clear the tick text plus the bolded header plus the 2-line panel
    # titles of the next block.
    spacer_row_h = 0.6
    # Reserve vertical space at the bottom for a single shared figure-level
    # legend describing the time-series styling (low/high tail ranks,
    # mean, ERA5, marker types).
    legend_strip_h = 0.9
    # Space reserved at the top for the suptitle.  The suptitle and the
    # END section header below it both occupy this strip, so it needs to
    # be tall enough to keep them visually separated after the title-font
    # bumps in plot_style.py.
    suptitle_strip_h = 1.0
    n_spacers = max(len(modes) - 1, 0)
    n_rows = rows_per_mode * len(modes) + n_spacers
    fig_h = (
        (2 * spatial_row_h + ts_row_h) * len(modes)
        + spacer_row_h * n_spacers
        + legend_strip_h
        + suptitle_strip_h
    )
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, fig_h))
    height_ratios: list[float] = []
    for i in range(len(modes)):
        height_ratios.extend([spatial_row_h, spatial_row_h, ts_row_h])
        if i < len(modes) - 1:
            height_ratios.append(spacer_row_h)
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        height_ratios=height_ratios,
        hspace=0.45,
        wspace=0.10,
    )

    # Pre-build axes per mode so we can collect them in row-major order for
    # add_subplot_labels at the end and so we can re-query their positions
    # to anchor the per-mode section headers after tight_layout.
    map_axes_all: list[plt.Axes] = []
    ts_axes_all: list[plt.Axes] = []
    # All axes of the top (low/coolest) row for each mode.  Used to anchor
    # the per-mode section header against the tallest of the row's titles
    # — the ERA5 column-0 title is 1-line, but the member-panel titles
    # are 2-line, so anchoring against the whole row avoids overlap.
    mode_top_rows: list[list[plt.Axes]] = []
    era5_mesh = None
    diff_mesh = None
    for mode_i, mode in enumerate(modes):
        diag = per_mode[mode]
        lat = np.asarray(diag["lat"])
        lon = np.asarray(diag["lon"])
        bbox = np.asarray(diag["bbox"])
        # The plotted fields now cover the wider ``regions.synoptic`` view
        # (see compute_free_end_states.py); ``bbox`` is the smaller impact
        # box the bias + timeseries reduce over, drawn as a dashed rectangle
        # inside the view.  Older npz files predate ``view_bbox`` and were
        # cropped to ``bbox`` itself, so fall back to it (the box then hugs
        # the frame, the previous behaviour).
        view_bbox = np.asarray(diag.get("view_bbox", diag["bbox"]))
        free_end_h = float(_scalar(diag["free_end_hours"]))
        era5 = np.asarray(diag["era5_field"])
        excluded_mask = _excluded_mask(diag)

        base = (
            rows_per_mode + 1
        ) * mode_i  # +1 accounts for the spacer row between modes.
        map_row_low = [
            fig.add_subplot(gs[base, c], projection=proj) for c in range(n_cols)
        ]
        map_row_high = [
            fig.add_subplot(gs[base + 1, c], projection=proj) for c in range(n_cols)
        ]
        ts_ax = fig.add_subplot(gs[base + 2, :])

        for ax in map_row_low:
            ax.set_extent(view_bbox.tolist(), crs=proj)
        for ax in map_row_high:
            ax.set_extent(view_bbox.tolist(), crs=proj)
        # Outline the impact box inside the wider view — a clear cue that the
        # bias ranking and the bbox-mean timeseries below reduce over this
        # box, while the panels show the surrounding synoptic context (the
        # same convention as the synoptic-PCA precursor panels).
        for ax in (*map_row_low, *map_row_high):
            draw_domain_box(ax, bbox.tolist())

        # Column 0: ERA5 reference panel in both the low and high rows.
        # Member panels in cols 1..3 show member - ERA5 differences (see below)
        # against a diverging cmap, so the ERA5 column anchors the absolute
        # scale that the diffs are computed against.  lat/lon labels ride the
        # figure edges only: the left column (column 0, the ERA5 panels) gets
        # y labels, and the bottom spatial row of each mode block (the high
        # tail) gets x labels.
        for row_axes, noun, is_bottom in (
            (map_row_low, low_noun, False),
            (map_row_high, high_noun, True),
        ):
            era5_mesh = _draw_panel(
                row_axes[0],
                era5,
                lat,
                lon,
                title=f"ERA5 @ +{free_end_h:g}h ({noun.lower()} row)",
                cmap=era5_cmap,
                vmin=era5_vmin,
                vmax=era5_vmax,
                coastline_color=coastline_color,
                excluded_mask=excluded_mask,
                show_xlabels=is_bottom,
                show_ylabels=True,
            )

        # Low (coolest/lowest) member diffs in columns 1..3 of the top row.
        # Diff with symmetric ±diff_max scale shared across the low tail, the
        # high tail, and modes so the cool and warm rows read on one scale.
        # The headline annotation is the signed bias vs ERA5 (RMSE is kept in
        # the npz as a secondary field).
        low_fields = np.asarray(diag["low_fields"])
        low_idx = np.asarray(diag["low_idx"])
        low_bias = np.asarray(diag["low_bias"])
        for k, (mem, b) in enumerate(zip(low_idx, low_bias)):
            if k + 1 >= n_cols:
                break
            diff_mesh = _draw_panel(
                map_row_low[k + 1],
                low_fields[k] - era5,
                lat,
                lon,
                title=(
                    f"{low_noun} #{k + 1}  member {int(mem)}\n"
                    f"{b:+.3g}{unit_suffix} vs ERA5"
                ),
                cmap=DIFF_CMAP,
                vmin=-diff_max,
                vmax=diff_max,
                coastline_color=coastline_color,
                excluded_mask=excluded_mask,
            )

        # High (warmest/highest) member diffs in columns 1..3 of the bottom row.
        high_fields = np.asarray(diag["high_fields"])
        high_idx = np.asarray(diag["high_idx"])
        high_bias = np.asarray(diag["high_bias"])
        for k, (mem, b) in enumerate(zip(high_idx, high_bias)):
            if k + 1 >= n_cols:
                break
            diff_mesh = _draw_panel(
                map_row_high[k + 1],
                high_fields[k] - era5,
                lat,
                lon,
                title=(
                    f"{high_noun} #{k + 1}  member {int(mem)}\n"
                    f"{b:+.3g}{unit_suffix} vs ERA5"
                ),
                cmap=DIFF_CMAP,
                vmin=-diff_max,
                vmax=diff_max,
                coastline_color=coastline_color,
                excluded_mask=excluded_mask,
                show_xlabels=True,
            )

        # Companion time-series panel.  start_time / timezone are scalars
        # saved by compute_free_end_states.py; the helper falls back to
        # UTC ticks if timezone is empty.
        _draw_timeseries(
            ts_ax,
            diag,
            var_name=var_name,
            mode=mode,
            start_time_iso=str(_scalar(diag["start_time"])),
            timezone_name=str(_scalar(diag["timezone"])),
        )

        map_axes_all.extend(map_row_low)
        map_axes_all.extend(map_row_high)
        ts_axes_all.append(ts_ax)
        mode_top_rows.append(map_row_low)

    # Inside placement keeps the labels visible over the maps without
    # colliding with the multi-line per-panel titles.  Time-series labels
    # also go inside because the bbox-mean ylabel is a wrapped 2-line
    # rotated label that pushes well above the axes — outside placement
    # collides with its top line (see the g/r overlap in the old layout).
    add_subplot_labels(map_axes_all, placement="inside")
    add_subplot_labels(
        ts_axes_all,
        labels=[
            f"{chr(ord('a') + len(map_axes_all) + i)}.)"
            for i in range(len(ts_axes_all))
        ],
        placement="inside",
    )

    fig.suptitle(
        f"{case_name}\n{pretty_var}\nFree-end states vs. ERA5",
        fontsize=SUPTITLE_SIZE_DENSE,
        y=0.99,
    )
    # Reserve the bottom strip of the figure for a single shared legend
    # describing the time-series styling.  Per-mode legends were dropped
    # because they crowd the bbox-mean lines and duplicate the same
    # rank-color semantics across modes — member numbers differ between
    # modes, so the legend names ranks by color and lets the reader
    # cross-reference member ids in the spatial-panel titles.
    legend_strip_frac = legend_strip_h / fig_h
    suptitle_strip_frac = suptitle_strip_h / fig_h
    # Reserve room on both sides for the colorbars: the ERA5 absolute
    # colorbar sits on the LEFT next to the column-0 ERA5 panels (the
    # only panels that use that scale), and the member-minus-ERA5
    # difference colorbar sits on the right next to the 12 diff panels.
    # Splitting them this way avoids stacking two colorbars on one side,
    # which read as visually heavy in the prior single-edge layout.
    fig.tight_layout(rect=[0.07, legend_strip_frac, 0.93, 1.0 - suptitle_strip_frac])

    cbar_y0 = legend_strip_frac + 0.02
    cbar_h = (1.0 - suptitle_strip_frac) - cbar_y0 - 0.02
    era5_cax = fig.add_axes([0.04, cbar_y0, 0.012, cbar_h])
    diff_cax = fig.add_axes([0.945, cbar_y0, 0.012, cbar_h])
    # ERA5 colorbar: ticks/label on the LEFT so they read outside-in from
    # the figure edge, mirroring the standard right-side colorbar layout
    # but flipped for the left-side placement.
    era5_cbar = fig.colorbar(era5_mesh, cax=era5_cax)
    era5_cbar.ax.yaxis.set_ticks_position("left")
    era5_cbar.ax.yaxis.set_label_position("left")
    era5_cbar.set_label(pretty_var, fontsize=COLORBAR_LABEL_SIZE_DENSE)
    era5_cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)
    diff_cbar = fig.colorbar(diff_mesh, cax=diff_cax)
    diff_label = f"Member − ERA5{f' ({units})' if units else ''}"
    if metric_region_word is not None:
        # Suffix names the metric domain in plain language; the hatched
        # overlay on each panel already shows *which* pixels were
        # excluded — the label answers "what's in the RMSE number".
        diff_label = (
            f"{diff_label}, over {metric_region_word} only (hatched = excluded)"
        )
    diff_cbar.set_label(diff_label, fontsize=COLORBAR_LABEL_SIZE_DENSE)
    diff_cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)

    # Section header for each mode block, positioned just above the title
    # text of that block's top row.  This replaces both the rotated row
    # labels on the leftmost panel (which doubled the "(N members)" text
    # nobody needs to read) and the time-series subplot title (which
    # repeated the mode name a third time).  We anchor against the actual
    # post-draw title bbox rather than the axes y1 because the 2-line
    # panel titles extend further above the axes than a fixed offset can
    # reliably clear once the title font is bumped.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_inv = fig.transFigure.inverted()
    for top_row, mode in zip(mode_top_rows, modes):
        # Anchor against the highest title across the whole top row, not
        # just the leftmost panel: the ERA5 column-0 title is 1-line but
        # the member-panel titles are 2-line, so anchoring against col 0
        # would undercount and the header would overlap the member titles.
        title_y1_max = 0.0
        for ax in top_row:
            title_bbox = ax.title.get_window_extent(renderer)
            title_bbox_fig = title_bbox.transformed(fig_inv)
            title_y1_max = max(title_y1_max, title_bbox_fig.y1)
        header_y = min(title_y1_max + 0.008, 1.0 - 0.035)
        fig.text(
            0.5,
            header_y,
            MODE_LABEL[mode],
            fontsize=SECTION_HEADER_SIZE_DENSE,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    handles, labels = _build_timeseries_legend_handles(low_noun, high_noun)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        fontsize=LEGEND_SIZE_DENSE,
        framealpha=0.9,
        title="Time Series Legend",
        title_fontsize=LEGEND_SIZE_DENSE,
    )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing free_end_states_{start,end}.npz; the "
        "aggregated PNG is written here.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"ERROR: --output-dir does not exist: {output_dir}")

    per_mode: Dict[str, Dict[str, np.ndarray]] = {}
    for mode in MODE_ORDER:
        d = _load_mode(output_dir, mode)
        if d is not None:
            per_mode[mode] = d

    if not per_mode:
        raise SystemExit(
            f"ERROR: no per-mode npz files found in {output_dir} "
            f"(expected one of: free_end_states_start.npz, free_end_states_end.npz)"
        )

    print(f"[aggregate_free_end_states] modes: {list(per_mode)}")
    out_path = output_dir / "free_end_states.png"
    make_figure(per_mode, out_path)
    print(f"[aggregate_free_end_states] wrote {out_path}")


if __name__ == "__main__":
    main()
