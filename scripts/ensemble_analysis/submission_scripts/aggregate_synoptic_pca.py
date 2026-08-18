"""Aggregate synoptic-PCA figures across conditioning modes.

Reads the per-mode ``synoptic_pca_<mode>.npz`` files (written by
``compute_synoptic_pca.py``) and produces, for each conditioning mode and
each leading EOF, a single combined **precursor -> impact** figure plus a
couple of supplements:

* ``synoptic_pca_combined_<mode>_eof<i>.png`` -- the headline figure.
  Top block: the synoptic *precursor* (z500) for the ensemble members
  closest to fixed PC percentiles of EOF ``i``, across selected lead
  frames.  Rows are ``ERA5 truth`` (raw) then ``ens mean - ERA5`` (bias)
  then ``member - ens mean`` (diversity) for the low / median / high
  percentile members, with the two anomaly-type rows sharing one diverging
  colorbar so bias and diversity amplitudes compare directly.  Bottom
  panel: how that precursor diversity manifests in the *impact*, with
  **every member coloured by its PC percentile** (low alpha) so a real
  EOF->impact link reads as a dominant cool->warm gradient --
    - scalar impact (heatwave): bbox-mean impact variable (t2m) vs lead
      time; ens mean (solid) + ERA5 (dashed) on top.
    - track impact (TC): each member's storm track on a map, with a dot at
      each track's free end, the ERA5 observed track dashed, and a landfall
      marker.
  For TC cases the precursor panels also carry two overlays: subtle raw-flow
  steering arrows (a subsampled u500/v500 quiver with one reference key) and
  the progressive storm track "so far" up to each column's frame (bold, one
  shared colour across the percentile rows; ERA5 dashed).
* ``synoptic_pca_eof_tracks_<mode>_eof<a>-<b>.png`` -- opt-in via
  ``--pair-eofs``: a minimal loading + track-sorting figure, one row per
  requested EOF (loading map beside the percentile-coloured track map,
  with ERA5's percentile along the mode annotated).  For SI figures that
  need only a mode's loading geometry and track sorting, not the full
  per-EOF precursor-map anatomy.  Track impact cases only.
* ``synoptic_pca_video_<mode>_eof<i>.mp4`` -- TC cases: a 3-panel animation
  of the low / median / high percentile members' MSL field over the lead
  window, with each member's track drawn progressively and the ERA5 track
  dashed, so the storm's motion can be watched directly.  Needs a working
  ffmpeg; the figures are unaffected if it is missing.
* ``synoptic_pca_eofs_<var>.png`` -- supplement: the leading EOF loading
  patterns per mode, in physical units.
* ``synoptic_pca_pc_scatter_<mode>.png`` -- supplement: the PC1-PC2
  scatter of every member, with the percentile members and ERA5 marked,
  situating the headline samples in PC space.
* ``synoptic_pca_precursor_impact_<mode>.png`` -- combined precursor ->
  scalar-impact figure (2x2): warmest- and coolest-decile precursor
  composites and their warm - cool difference (the model-free precursor
  pattern of an extreme impact) plus a predicted-vs-ensemble-member scatter with
  in-sample + cross-validated R^2, an F-test p-value, and ERA5 marked.  The
  composites carry the *pattern*, the scatter its *magnitude*.  Scalar impact
  only.
* ``synoptic_pca_domain_ladder_<mode>.png`` -- domain-sensitivity ladder:
  the regression composite + cross-validated R^2 over a nested set of impact
  boxes (the headline box scaled about its centre), a de-attenuation check
  that R^2 sharpens with box size while the pattern stays stable.  Scalar
  impact only.
* ``synoptic_pca_diversity.png`` -- supplement quantifying the precursor
  diversity, start vs end: physical free-end spread, effective number of
  modes (participation ratio + spectral perplexity), cumulative EVR, and
  the eigenvalue spectrum with North (1982) separation error bars.

There is no clustering figure: the diagnostic is the continuous
PCA / percentile-member view.

Usage:
    python aggregate_synoptic_pca.py \\
        --output-dir /path/to/diagnostics/synoptic_pca
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent.parent / "_shared"
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))  # ensemble_analysis/ for track_qc

from _dispatch_lib import (  # noqa: E402
    MODE_COLORS_TAB,
    MODE_LABELS_SHORT,
    tc_tracks_parquet_path,
)
from track_qc import compute_track_qc  # noqa: E402
from local_time_axis import format_local_time_axis, local_datetimes  # noqa: E402
from pca_stats import (  # noqa: E402
    _kfold_r2,
    _ols_fit,
    _r2_score,
    _regression_f_pvalue,
    _spearman,
)
from plot_style import (  # noqa: E402
    AXIS_LABEL_SIZE_DENSE,
    COLORBAR_LABEL_SIZE_DENSE,
    COLORBAR_TICK_SIZE_DENSE,
    LEGEND_SIZE_DENSE,
    QUIVER_REF_LEN_FRAC,
    SECTION_HEADER_SIZE_DENSE,
    SUBPLOT_LABEL_SIZE_DENSE,
    SUPTITLE_SIZE_DENSE,
    TICK_LABEL_SIZE_DENSE,
    TITLE_SIZE_DENSE,
    add_subplot_labels,
    draw_domain_box,
    draw_quiver,
    draw_quiver_key,
    nice_speed,
    quiver_step,
)
from var_metadata import (  # noqa: E402
    axis_label,
    is_geopotential,
    long_name,
    symbol,
    to_display_units,
)
from var_metadata import units_tex as var_units  # noqa: E402

# Modes ordered so the end-conditioned mode (the paper's headline) sorts
# first.  start follows.
MODE_ORDER: Tuple[str, ...] = ("end", "start")

# Selected frame indices for the precursor-map columns: IC, two interior
# snapshots, the final analysis time, and the landfall-equivalent frame.
TRAJECTORY_FRAMES: Tuple[int, ...] = (0, 3, 6, 9, 11)

# Diverging cmap for anomaly views; sequential cmap for raw fields.
ANOM_CMAP = "RdBu_r"
RAW_CMAP = "viridis"

# Ordered cool->warm palette for the percentile members so they read as a
# low->high gradient along the EOF axis (5 -> coldest, 95 -> warmest).  A
# fixed palette is used for the default 5 percentiles rather than sampling a
# continuous map, because diverging maps wash out near the median -- a
# light-gray p50 line/label is illegible on white and over the gray
# spaghetti.  The median is a distinct dark gray instead.  Other percentile
# counts fall back to sampling ``PCT_CMAP``.
PCT_CMAP = "coolwarm"
_PCT_COLORS_5 = ("#2166ac", "#67a9cf", "#444444", "#ef8a62", "#b2182b")

# Gray spaghetti styling (still used by the PC-scatter background).
_SPAGHETTI_COLOR = "#888888"
_SPAGHETTI_ALPHA = 0.16
_SPAGHETTI_LW = 0.5
_HIGHLIGHT_LW = 1.9

# Impact panels colour *every* member by its PC percentile (low alpha) so a
# real EOF->impact relationship emerges as a dominant colour gradient, while
# noise averages out -- more robust than highlighting a handful of single
# percentile members.  (Group-mean trajectories are the documented fallback
# if this reads as a mess.)
_PCT_SPAGHETTI_ALPHA = 0.40
_PCT_SPAGHETTI_LW = 0.7

# Raw-flow steering arrows (subsampled quiver) overlaid on the precursor panels
# (ERA5-truth + each percentile-member row) come from plot_style:
# `draw_quiver` / `draw_quiver_key` / `quiver_step` / `nice_speed` and the
# `QUIVER_*` constants, shared with plot_ensemble_member.py's member-grid
# overlay so both figures draw the same arrows.

# Progressive storm-track overlay on the precursor panels.  Bold and a single
# shared colour across the percentile rows so the track reads as "the storm"
# everywhere (the row label already carries the percentile); ERA5 stays black
# dashed.  A white halo keeps it legible over any field.
_TRACK_MEMBER_COLOR = "#ff1f8f"
_TRACK_LW = 2.3
_TRACK_HALO_LW = 4.0
_TRACK_MARKER_SIZE = 6.0

# Per-EOF percentile-member animation (MSL by default).  fps is low so the
# storm's slow translation stays watchable; the view zooms to the TC domain.
_VIDEO_CMAP = "viridis"
_VIDEO_FPS = 2
_VIDEO_DPI = 110


def _shared_minmax(
    fields: Sequence[np.ndarray], pct: float = 1.0
) -> Tuple[float, float]:
    """(vmin, vmax) across ``fields`` at the ``pct`` / ``100 - pct`` tails.

    Used for raw sequential color ranges so a single outlier pixel doesn't
    compress the colorbar.  ``pct=1.0`` keeps the 1st and 99th percentile.
    """
    los, his = [], []
    for f in fields:
        arr = np.asarray(f, dtype=np.float32)
        if arr.size:
            los.append(np.percentile(arr, pct))
            his.append(np.percentile(arr, 100.0 - pct))
    if not los:
        return 0.0, 1.0
    return float(min(los)), float(max(his))


def _shared_color_range(
    fields: Sequence[np.ndarray], percentile: float = 99.0
) -> float:
    """Symmetric color range from the ``percentile`` absolute value across fields.

    Using a percentile (instead of the global max) keeps an outlier pixel
    from compressing a diverging colorbar.
    """
    abs_vals = []
    for f in fields:
        a = np.abs(np.asarray(f, dtype=np.float32))
        if a.size:
            abs_vals.append(np.percentile(a, percentile))
    if not abs_vals:
        return 1.0
    v = float(max(abs_vals))
    return v if v > 0 else 1.0


# ---------------------------------------------------------------------------
# I/O + small helpers
# ---------------------------------------------------------------------------


def _scalar(arr) -> object:
    return arr.item() if isinstance(arr, np.ndarray) and arr.shape == () else arr


def _units_suffix(var_name: str) -> str:
    """`` (units)`` for a colorbar label, or ``""`` when units are unknown."""
    u = var_units(var_name)
    return f" ({u})" if u else ""


# npz keys holding per-variable physical fields, shaped [..., n_vars, lat, lon]
# (the variable axis is always -3).  Geopotential slices are converted from
# storage to display units (m^2/s^2 -> m) once at load so every shared color
# range, anomaly, panel, and colorbar downstream stays consistent.
_FIELD_UNIT_KEYS = (
    "era5_trajectory_latlon",
    "ensemble_mean_trajectory_latlon",
    "pc_components_latlon",
    "percentile_member_trajectory_latlon",
    "impact_decile_warm_latlon",
    "impact_decile_cool_latlon",
)


def _to_display_units(diag: dict) -> None:
    """In place, convert geopotential field slices to display units."""
    variables = [str(v) for v in np.asarray(diag["variables"])]
    geo_idx = [j for j, v in enumerate(variables) if is_geopotential(v)]
    if not geo_idx:
        return
    for key in _FIELD_UNIT_KEYS:
        if key not in diag:
            continue
        arr = np.asarray(diag[key]).copy()
        # Optional field keys (the decile composites) are empty on a non-scalar
        # impact case or an npz predating them -- no variable axis to slice.
        if arr.size == 0 or arr.ndim < 3:
            continue
        for j in geo_idx:
            sl = [slice(None)] * arr.ndim
            sl[-3] = j
            arr[tuple(sl)] = to_display_units(variables[j], arr[tuple(sl)])
        diag[key] = arr


def _load_mode(output_dir: Path, mode: str) -> dict | None:
    path = output_dir / f"synoptic_pca_{mode}.npz"
    if not path.exists():
        print(f"WARNING: missing {path}, skipping mode '{mode}'")
        return None
    diag = dict(np.load(path, allow_pickle=True))
    _to_display_units(diag)
    return diag


def _mode_label(mode: str) -> str:
    """Long-form conditioning label, e.g. 'end-conditioned'."""
    short = MODE_LABELS_SHORT.get(mode, mode)
    return f"{short}-conditioned"


def _free_end_role(mode: str) -> str:
    """What the free-end (PCA) frame represents, for the figure title.

    The PCA runs on the *free-end* frame.  Under end-conditioning the pinned
    frame is the outcome, so the free end is the **precursor**; under
    start-conditioning the pinned frame is the precursor, so the free end is
    the **outcome** (the landfall-time field).  Labeling it per-mode keeps a
    start-conditioned figure from implying the EOF is a precursor pattern when
    it is actually the concurrent outcome field.
    """
    return {"end": "precursor", "start": "outcome"}.get(mode, "free-end")


def _pct_member_colors(n_pct: int) -> List:
    """Ordered cool->warm colors for the ``n_pct`` percentile members."""
    if n_pct == len(_PCT_COLORS_5):
        return list(_PCT_COLORS_5)
    cmap = plt.get_cmap(PCT_CMAP)
    if n_pct == 1:
        return [cmap(0.5)]
    return [cmap(i / (n_pct - 1)) for i in range(n_pct)]


def _member_percentile(scores_eof: np.ndarray) -> np.ndarray:
    """Percentile rank (0-100) of each member's PC score along one EOF.

    Rank (not raw score) so the colour gradient spreads uniformly over the
    members regardless of the score distribution's shape.
    """
    n = int(scores_eof.size)
    if n <= 1:
        return np.full(n, 50.0, dtype=np.float64)
    ranks = np.argsort(np.argsort(scores_eof))
    return 100.0 * ranks / (n - 1)


# ---------------------------------------------------------------------------
# Precursor -> impact regression (combined-EOF, for plot_precursor_impact)
# ---------------------------------------------------------------------------
#
# No single variance-maximizing EOF need carry the impact signal: the
# impact-relevant precursor pattern generally projects onto several EOFs at
# once.  These helpers collapse the leading PCs onto the single direction
# that best predicts the scalar impact -- the principled way to combine the
# modes -- and quantify how much of the across-member impact spread it
# explains.  Pure NumPy (no scipy/sklearn), to match _spearman's style and
# stay importable in plot-only contexts.


def _field_impact_regression_weights(
    predictors: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Per-PC weights for the field-on-impact regression map.

    Returns ``c[K]`` with ``c_k = cov(PC_k, target) / var(target)``.  The
    regression map at grid point ``x`` is then ``sum_k c_k * EOF_k(x)`` -- the
    precursor-field anomaly that accompanies a +1-unit change in the impact
    (e.g. metres of z500 per kelvin of land-mean t2m).  This is the
    field-on-impact direction (units field/impact), distinct from the
    impact-on-PC fit (:func:`_ols_fit`) that builds the predicted-impact index.
    Both are driven by ``cov(PC_k, target)``, so they tell one story.
    """
    target = np.asarray(target, dtype=np.float64)
    var_t = float(target.var())
    if var_t <= 0:
        return np.zeros(predictors.shape[1], dtype=np.float64)
    t_dev = target - target.mean()
    p_dev = predictors - predictors.mean(axis=0, keepdims=True)
    cov = (p_dev * t_dev[:, None]).mean(axis=0)
    return (cov / var_t).astype(np.float64)


def _map_pct_subset(n_pct: int) -> List[int]:
    """Indices of the low / median / high percentiles to render as maps.

    The full set of percentiles is shown as lines in the impact panel; the
    spatial maps show just the extremes and the median to save real estate.
    """
    if n_pct <= 3:
        return list(range(n_pct))
    return [0, n_pct // 2, n_pct - 1]


def _map_panel(
    ax,
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    extent: Tuple[float, float, float, float],
    title: str = "",
    show_xlabels: bool = False,
    show_ylabels: bool = False,
    excluded_mask: np.ndarray | None = None,
) -> "matplotlib.collections.QuadMesh":
    """Render one PlateCarree panel with coastlines + gridlines.

    When ``excluded_mask`` (a bool array on the same ``(lat, lon)`` grid as
    ``field``) is given, its True pixels are overlaid with diagonal hatching --
    the standard "not used in the statistic" cartographic convention.  Used to
    hatch the ocean (or land) pixels a drawn computation box excludes from its
    land-/sea-masked mean, so the dashed box is not read as a plain area mean.
    The field itself is left visible everywhere (only the hatch is added),
    since the precursor field shown is not the masked impact variable.
    """
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    # pcolormesh needs lat/lon ascending; lat in the npz is typically
    # descending (90 -> -90), so sort lat ascending for the call.
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        field = field[..., ::-1, :]
        if excluded_mask is not None:
            excluded_mask = excluded_mask[..., ::-1, :]
    lon_sort_idx = np.argsort(lon)
    lon_sorted = lon[lon_sort_idx]
    field_sorted = field[..., :, lon_sort_idx]

    mesh = ax.pcolormesh(
        lon_sorted,
        lat,
        field_sorted,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )
    if excluded_mask is not None:
        excluded_sorted = np.asarray(excluded_mask)[..., :, lon_sort_idx]
        if excluded_sorted.any():
            # contourf hatches the True region with no fill colour, leaving the
            # field + coastlines visible underneath.  Levels [0.5, 1.5] select
            # the True pixels (bool cast to {0., 1.}).
            ax.contourf(
                lon_sorted,
                lat,
                excluded_sorted.astype(float),
                levels=[0.5, 1.5],
                colors="none",
                hatches=["///"],
                transform=ccrs.PlateCarree(),
                zorder=2,
            )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
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

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE_DENSE)
    return mesh


def _draw_progressive_track(
    ax,
    track: np.ndarray,
    upto_frame: int,
    color,
    linestyle: str,
) -> None:
    """Draw a storm track up to (and including) ``upto_frame``.

    ``track`` is ``[n_leads, 2]`` (lat, lon), NaN-padded off the storm's
    lifetime.  Plots the valid fixes with index <= ``upto_frame`` and marks the
    latest one, so each timestep column shows the path "so far" with the current
    position highlighted.  A white halo keeps it legible over any field.
    """
    seg = np.asarray(track[: upto_frame + 1], dtype=np.float64)
    if seg.size == 0:
        return
    lat = seg[:, 0]
    lon = seg[:, 1]
    valid = np.isfinite(lat) & np.isfinite(lon)
    if not valid.any():
        return
    halo = [pe.withStroke(linewidth=_TRACK_HALO_LW, foreground="white")]
    ax.plot(
        lon[valid],
        lat[valid],
        color=color,
        linewidth=_TRACK_LW,
        linestyle=linestyle,
        transform=ccrs.PlateCarree(),
        zorder=6,
        path_effects=halo,
    )
    last = int(np.flatnonzero(valid)[-1])
    ax.plot(
        lon[last],
        lat[last],
        marker="o",
        color=color,
        markersize=_TRACK_MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=0.7,
        transform=ccrs.PlateCarree(),
        zorder=7,
    )


# ---------------------------------------------------------------------------
# Impact panels (bottom of the combined figure)
# ---------------------------------------------------------------------------


def _impact_timeseries_panel(
    ax, diag: dict, eof_idx: int, member_pct: np.ndarray, cmap, norm
) -> None:
    """Scalar-impact line panel: bbox-mean impact variable vs lead time.

    Every member is coloured by its PC percentile (``member_pct``) at low
    alpha; the ensemble mean (solid black) and ERA5 (dashed) ride on top.
    """
    lead_h = np.asarray(diag["lead_hours"])
    n_leads = len(lead_h)
    member = np.asarray(diag["member_impact_scalar"])  # [N, n_leads]
    era5 = np.asarray(diag["era5_impact_scalar"])  # [n_leads]
    ens_mean = np.asarray(diag["ensemble_mean_impact_scalar"])  # [n_leads]
    impact_variable = str(_scalar(diag["impact_variable"]))
    if is_geopotential(impact_variable):
        member = to_display_units(impact_variable, member)
        era5 = to_display_units(impact_variable, era5)
        ens_mean = to_display_units(impact_variable, ens_mean)

    member_colors = cmap(norm(member_pct))  # [N, 4]
    for i in range(member.shape[0]):
        ax.plot(
            lead_h,
            member[i],
            color=member_colors[i],
            alpha=_PCT_SPAGHETTI_ALPHA,
            linewidth=_PCT_SPAGHETTI_LW,
            zorder=1,
        )
    ax.plot(lead_h, ens_mean, color="black", linewidth=2.2, zorder=5, label="Ens. mean")
    ax.plot(
        lead_h,
        era5,
        color="black",
        linewidth=2.2,
        linestyle="--",
        zorder=5,
        label="ERA5",
    )

    free_end_frame = int(_scalar(diag["free_end_frame"]))
    if free_end_frame < n_leads:
        ax.axvline(
            lead_h[free_end_frame],
            color="black",
            linestyle=":",
            linewidth=0.9,
            alpha=0.55,
            zorder=2,
        )

    # Quantify the EOF -> impact link the colour gradient shows: the Spearman
    # rank correlation between each member's PC percentile (its colour) and its
    # area-mean impact at the free-end frame -- the frame where the EOF /
    # percentile is defined (and ensemble spread is maximal), matching the
    # "truth PC pct" annotation on the precursor maps above.  Rank-based, so it
    # is consistent with the percentile colouring and robust to the heatwave
    # tail's nonlinearity.
    rho = (
        _spearman(member_pct, member[:, free_end_frame])
        if free_end_frame < member.shape[1]
        else float("nan")
    )
    rho_txt = f"{rho:+.2f}" if np.isfinite(rho) else "n/a"
    print(
        f"[aggregate_synoptic_pca] EOF{eof_idx + 1} impact Spearman rho "
        f"(PC pct vs {impact_variable} @ free-end frame {free_end_frame}) = {rho_txt}"
    )
    ax.text(
        0.02,
        0.97,
        f"Spearman ρ (PC{eof_idx + 1} pct vs {impact_variable}\nat free end) = {rho_txt}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=TICK_LABEL_SIZE_DENSE,
        color="black",
        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="0.7",
            boxstyle="round,pad=0.3",
        ),
        zorder=6,
    )
    for cf in np.asarray(
        diag.get("conditioning_frames", np.array([])), dtype=np.int64
    ).tolist():
        if 0 <= int(cf) < n_leads:
            ax.axvline(
                lead_h[int(cf)],
                color="tab:gray",
                linestyle=":",
                linewidth=0.9,
                alpha=0.55,
                zorder=2,
            )

    start_time_iso = str(
        _scalar(diag.get("start_time", np.array("1970-01-01T00:00:00Z")))
    )
    timezone_name = str(_scalar(diag.get("timezone", np.array("UTC"))))
    format_local_time_axis(ax, lead_h, start_time_iso, timezone_name)
    mask_kind = str(_scalar(diag.get("impact_mask_kind", np.array("none"))))
    mask_txt = {"land": " (land mean)", "sea": " (ocean mean)"}.get(
        mask_kind, " (area mean)"
    )
    # axis_label already includes units (e.g. "$T_{\mathrm{2m}}$ (K)"), so
    # do not append _units_suffix here -- that would double the units.
    ax.set_ylabel(
        f"{axis_label(impact_variable)}{mask_txt}",
        fontsize=AXIS_LABEL_SIZE_DENSE,
    )
    ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE_DENSE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE_DENSE)
    ax.set_xlim(float(lead_h[0]), float(lead_h[-1]))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE_DENSE, loc="best", framealpha=0.9, ncol=2)


def _impact_track_panel(
    ax,
    diag: dict,
    eof_idx: int,
    member_pct: np.ndarray,
    cmap,
    norm,
    extent: Tuple[float, float, float, float],
    mode: str,
) -> None:
    """Track-impact map panel.

    Every member's storm track is coloured by its PC percentile
    (``member_pct``) at low alpha, with a dot marking each track's free
    end, the observed ERA5 track (dashed), and the landfall marker on top.
    """
    member_track = np.asarray(diag["member_track_latlon"])  # [N, n_leads, 2] (lat, lon)
    era5_track = np.asarray(diag["era5_track_latlon"])  # [n_leads, 2]
    landfall = int(_scalar(diag.get("landfall_frame", np.int32(-1))))

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray")
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": TICK_LABEL_SIZE_DENSE}
    gl.ylabel_style = {"size": TICK_LABEL_SIZE_DENSE}

    # Every member coloured by its PC percentile, with a dot marking each
    # track's free end -- the conditioning-free, diverging endpoint.  This
    # mirrors the tc_tracks static-map convention (aggregate_tc_tracks.py
    # ._draw_track_map_panel): under end-conditioning the free end is the
    # track *start*; under start-conditioning it is the *last* fix.  Tracks
    # are NaN-padded to the lead axis, so the dot sits on the first / last
    # *valid* fix.
    member_colors = cmap(norm(member_pct))  # [N, 4]
    free_end_pick = {"end": "first", "start": "last"}.get(mode)
    for i in range(member_track.shape[0]):
        lat_i = member_track[i, :, 0]
        lon_i = member_track[i, :, 1]
        ax.plot(
            lon_i,
            lat_i,
            color=member_colors[i],
            alpha=_PCT_SPAGHETTI_ALPHA,
            linewidth=_PCT_SPAGHETTI_LW,
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        if free_end_pick is None:
            continue
        valid = np.flatnonzero(np.isfinite(lat_i) & np.isfinite(lon_i))
        if valid.size == 0:
            continue
        end_i = valid[0] if free_end_pick == "first" else valid[-1]
        ax.plot(
            lon_i[end_i],
            lat_i[end_i],
            "o",
            color=member_colors[i],
            markersize=4.5,
            alpha=0.75,
            markeredgecolor="black",
            markeredgewidth=0.3,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
    # ERA5 observed track.
    ax.plot(
        era5_track[:, 1],
        era5_track[:, 0],
        color="black",
        linestyle="--",
        linewidth=2.2,
        transform=ccrs.PlateCarree(),
        zorder=5,
        label="ERA5",
    )
    if 0 <= landfall < era5_track.shape[0] and np.isfinite(era5_track[landfall, 0]):
        ax.scatter(
            era5_track[landfall, 1],
            era5_track[landfall, 0],
            marker="*",
            s=160,
            color="black",
            edgecolor="white",
            linewidth=0.6,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
    ax.legend(fontsize=LEGEND_SIZE_DENSE, loc="best", framealpha=0.9, ncol=3)


# ---------------------------------------------------------------------------
# Combined precursor -> impact figure (one per mode, EOF)
# ---------------------------------------------------------------------------


def _eof_loading_panel(
    ax,
    diag: dict,
    eof_idx: int,
    var_idx: int,
    extent: Tuple[float, float, float, float],
    track_box: Tuple[float, float, float, float] | None = None,
) -> "matplotlib.collections.QuadMesh":
    """Render the EOF loading pattern that *defines* this figure's PC axis.

    Same field as the standalone ``plot_eofs`` supplement
    (``pc_components_latlon[eof_idx, var_idx]``, already in display units),
    drawn beside the track impact so the headline figure stands alone: the
    precursor *mode* sits next to the track spread it sorts.  Its loading
    amplitude is a different (much smaller) scale than the member anomalies
    above, so the caller gives it a dedicated colorbar rather than reusing
    the anomaly one.

    When ``track_box`` is given, that lon/lat box is outlined on the panel so
    the (smaller) sub-region the adjacent track map zooms into is explicit
    against the wider loading domain.
    """
    eofs = np.asarray(diag["pc_components_latlon"])  # [n_eof, n_vars, lat, lon]
    domain_lat = np.asarray(diag["domain_lat"])
    domain_lon = np.asarray(diag["domain_lon"])
    field = eofs[eof_idx, var_idx]
    vrange = _shared_color_range([field], percentile=99.0)
    mesh = _map_panel(
        ax,
        field,
        domain_lat,
        domain_lon,
        vmin=-vrange,
        vmax=vrange,
        cmap=ANOM_CMAP,
        extent=extent,
        title=f"EOF{eof_idx + 1} loading",
        show_xlabels=True,
        show_ylabels=True,
    )
    if track_box is not None:
        draw_domain_box(ax, track_box)
    return mesh


def plot_combined(
    diag: dict,
    var_idx: int,
    var_name: str,
    mode: str,
    eof_idx: int,
    out_path: Path,
) -> None:
    """One combined precursor-maps + impact-panel figure for (mode, EOF)."""
    if "percentile_member_trajectory_latlon" not in diag:
        return
    n_eof_show = int(_scalar(diag["n_eof_show"]))
    if eof_idx >= n_eof_show:
        return

    extent = (
        float(_scalar(diag["lon_min"])),
        float(_scalar(diag["lon_max"])),
        float(_scalar(diag["lat_min"])),
        float(_scalar(diag["lat_max"])),
    )
    case_name = str(_scalar(diag["case_name"]))
    lead_h = np.asarray(diag["lead_hours"])
    n_leads = len(lead_h)
    sel_frames = tuple(f for f in TRAJECTORY_FRAMES if 0 <= f < n_leads)
    n_cols = len(sel_frames)
    if n_cols == 0:
        return

    pct_values = np.asarray(diag["percentile_values"]).astype(int)
    n_pct = pct_values.size
    pct_colors = _pct_member_colors(n_pct)
    map_subset = _map_pct_subset(n_pct)
    member_idx = np.asarray(diag["percentile_member_idx"]).astype(
        int
    )  # [n_eof_show, n_pct]

    pct_traj = np.asarray(diag["percentile_member_trajectory_latlon"])
    # [n_eof_show, n_pct, n_leads, n_vars, lat, lon]
    era5_traj = np.asarray(
        diag["era5_trajectory_latlon"]
    )  # [n_leads, n_vars, lat, lon]
    ens_mean_traj = np.asarray(diag["ensemble_mean_trajectory_latlon"])
    domain_lat = np.asarray(diag["domain_lat"])
    domain_lon = np.asarray(diag["domain_lon"])

    evr = np.asarray(diag["explained_variance_ratio"]).astype(float)
    era5_pc_pct = np.asarray(diag["era5_pc_percentile"]).astype(float)
    pct_var = 100.0 * evr[eof_idx] if eof_idx < evr.size else 0.0
    truth_pct = (
        float(era5_pc_pct[eof_idx]) if eof_idx < era5_pc_pct.size else float("nan")
    )

    # The truth-PC percentile is a *free-end* quantity: compute_synoptic_pca
    # projects the ERA5 free-end state onto the ensemble's free-end PC basis, so
    # it ranks where truth sits in PC space at the free end -- not at the IC.
    # Annotate it on the column that actually shows the free-end frame instead
    # of always the first timestep.  Under end-conditioning the free end is
    # frame 0 (the first column, so this is unchanged); under start-conditioning
    # it is the last frame, so the label rides the outcome column where the
    # percentile is meaningful.  Fall back to the nearest rendered frame if the
    # exact free-end frame is not among the selected columns (only possible for
    # non-standard window lengths).
    free_end_frame = int(_scalar(diag["free_end_frame"]))
    truth_pct_col = (
        sel_frames.index(free_end_frame)
        if free_end_frame in sel_frames
        else int(np.argmin([abs(int(f) - free_end_frame) for f in sel_frames]))
    )

    # Selected-frame slabs for the chosen precursor variable.
    raw_era5 = era5_traj[list(sel_frames), var_idx]  # [n_cols, lat, lon]
    ens_mean_sel = ens_mean_traj[list(sel_frames), var_idx]  # [n_cols, lat, lon]
    members_sel = pct_traj[eof_idx][
        :, list(sel_frames), var_idx
    ]  # [n_pct, n_cols, lat, lon]

    bias = ens_mean_sel - raw_era5  # [n_cols, lat, lon]
    member_anom = members_sel - ens_mean_sel[np.newaxis]  # [n_pct, n_cols, lat, lon]
    raw_vmin, raw_vmax = _shared_minmax([raw_era5], pct=1.0)
    anom_vrange = _shared_color_range([bias, member_anom[map_subset]], percentile=99.0)

    impact_kind = str(_scalar(diag.get("impact_kind", np.array("none"))))

    # Scalar-impact cases reduce the impact variable over a sub-domain
    # (impact_bbox) of the wider precursor PCA domain; outline that box on
    # the precursor maps so the reader sees where the impact metric is taken
    # relative to the synoptic pattern.  TC cases instead mark tc_domain on
    # the embedded loading panel in the impact row below (see track_box).
    impact_bbox = np.asarray(diag.get("impact_bbox", np.zeros(0)))
    precursor_box = (
        tuple(float(v) for v in impact_bbox)
        if impact_kind == "scalar" and impact_bbox.size == 4
        else None
    )
    # Ocean (or land) pixels inside that box that the land/sea-masked impact
    # mean drops -- hatched on the precursor panels so the dashed box reads as
    # "land mean within the box", not a plain area mean.  Empty for unmasked
    # (area-mean) scalar cases and for TC cases.
    impact_mask_kind = str(_scalar(diag.get("impact_mask_kind", np.array("none"))))
    impact_excluded = np.asarray(diag.get("impact_excluded_mask_box", np.zeros(0)))
    precursor_excluded = (
        impact_excluded
        if (impact_kind == "scalar" and impact_excluded.size and impact_excluded.any())
        else None
    )

    # Precursor-panel overlays (TC cases): the progressive storm track "so far"
    # and the raw-flow steering arrows that explain why each member's storm
    # is steered where it is.  Both degrade gracefully -- a scalar case or an npz
    # written before these fields existed simply omits them.
    member_track = np.asarray(diag.get("member_track_latlon", np.zeros(0)))
    era5_track = np.asarray(diag.get("era5_track_latlon", np.zeros(0)))
    tracks_available = bool(
        impact_kind == "track" and era5_track.size and member_track.size
    )
    wind_vars = [str(v) for v in np.asarray(diag.get("wind_variables", np.zeros(0)))]
    pct_wind = np.asarray(diag.get("percentile_member_wind_latlon", np.zeros(0)))
    era5_wind = np.asarray(diag.get("era5_wind_latlon", np.zeros(0)))
    winds_available = bool(len(wind_vars) >= 2 and pct_wind.size and era5_wind.size)

    # Row layout: truth, bias, then one row per shown percentile, a spacer,
    # then the impact panel.  An extra narrow column holds the colorbars.
    # The track impact is a (roughly square) equal-aspect map, so it gets a
    # taller row to grow it toward the panel width instead of sitting small
    # and centered; the scalar impact is a line plot that fills any box.
    n_map_rows = 2 + len(map_subset)
    impact_ratio = 2.0 if impact_kind == "track" else 2.4
    height_ratios = [1.0] * n_map_rows + [impact_ratio]
    total_rows = len(height_ratios)
    impact_row = total_rows - 1

    fig_w = 2.45 * n_cols + 1.1
    fig_h = 1.7 * sum(height_ratios) + 1.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        total_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.06],
        height_ratios=height_ratios,
        left=0.075,
        right=0.9,
        # Leave headroom above the maps for the suptitle, the subordinate
        # timezone line, and the two-line local-time column headers.
        top=0.88,
        bottom=0.07,
        wspace=0.08,
        hspace=0.28,
    )

    # Column headers carry the absolute local wall-clock time (date +
    # HH:MM, the same format format_local_time_axis puts on its tick
    # labels) rather than relative lead hours.  The timezone is named once
    # in the suptitle below instead of being repeated on every column.
    start_time_iso = str(
        _scalar(diag.get("start_time", np.array("1970-01-01T00:00:00Z")))
    )
    timezone_name = str(_scalar(diag.get("timezone", np.array("UTC"))))
    frame_local_dt = dict(
        zip(
            sel_frames,
            local_datetimes(
                start_time_iso, [lead_h[f] for f in sel_frames], timezone_name
            ),
        )
    )

    def _frame_title(frame: int) -> str:
        return frame_local_dt[frame].strftime("%b %d\n%H:%M")

    # Build the map rows.  Row 0: ERA5 truth (raw sequential).  Row 1:
    # ens mean - ERA5 (bias).  Rows 2+: member - ens mean per percentile.
    raw_mesh = None
    anom_mesh = None
    row_labels = ["ERA5 truth", "Ens. mean\n− ERA5"]
    row_fields = [raw_era5, bias]
    row_is_raw = [True, False]
    row_colors = ["black", "black"]
    # Parallel per-row overlays, aligned with ``row_fields``: the raw-wind field
    # ([n_leads, n_wind, nlat, nlon] or None) and the storm track
    # ((track[n_leads, 2], color, linestyle) or None).  Truth row -> ERA5 winds +
    # observed track; bias row -> neither (a difference field has no raw flow);
    # each percentile-member row -> that member's own winds and track.
    row_wind = [era5_wind if winds_available else None, None]
    row_track_src = [
        (era5_track, "black", "--") if tracks_available else None,
        None,
    ]
    for si in map_subset:
        m_idx = int(member_idx[eof_idx, si])
        row_labels.append(f"PC p{int(pct_values[si])}\n(mem {m_idx})\n− ens. mean")
        row_fields.append(member_anom[si])
        row_is_raw.append(False)
        row_colors.append(pct_colors[si])
        row_wind.append(pct_wind[eof_idx, si] if winds_available else None)
        # One shared bold colour for every member track (the row label already
        # carries the percentile), so the track reads as "the storm" throughout.
        row_track_src.append(
            (member_track[m_idx], _TRACK_MEMBER_COLOR, "-")
            if tracks_available
            else None
        )

    # One quiver scale + reference key shared across the figure's panels so the
    # arrows are directly comparable row-to-row.  Derive the reference speed from
    # the 90th-percentile flow magnitude over the panels that carry winds.
    quiver_scale = 1.0
    quiver_stride = 1
    quiver_key_mag = 0.0
    quiver_handle = None
    quiver_key_done = False
    if winds_available:
        speeds = [
            np.hypot(w[f, 0], w[f, 1]).ravel()
            for w in row_wind
            if w is not None
            for f in sel_frames
            if f < w.shape[0] and w.shape[1] >= 2
        ]
        if speeds:
            ref_speed = float(np.percentile(np.concatenate(speeds), 90)) or 1.0
            quiver_scale = ref_speed / QUIVER_REF_LEN_FRAC
            quiver_stride = quiver_step(domain_lat, domain_lon)
            quiver_key_mag = nice_speed(ref_speed)

    band_first_axes: List = []
    for r in range(n_map_rows):
        is_bottom_map = r == n_map_rows - 1
        for c, frame in enumerate(sel_frames):
            ax = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
            if row_is_raw[r]:
                raw_mesh = _map_panel(
                    ax,
                    row_fields[r][c],
                    domain_lat,
                    domain_lon,
                    vmin=raw_vmin,
                    vmax=raw_vmax,
                    cmap=RAW_CMAP,
                    extent=extent,
                    title=_frame_title(frame) if r == 0 else "",
                    show_xlabels=is_bottom_map,
                    show_ylabels=(c == 0),
                    excluded_mask=precursor_excluded,
                )
            else:
                anom_mesh = _map_panel(
                    ax,
                    row_fields[r][c],
                    domain_lat,
                    domain_lon,
                    vmin=-anom_vrange,
                    vmax=anom_vrange,
                    cmap=ANOM_CMAP,
                    extent=extent,
                    title="",
                    show_xlabels=is_bottom_map,
                    show_ylabels=(c == 0),
                    excluded_mask=precursor_excluded,
                )
            if precursor_box is not None:
                draw_domain_box(ax, precursor_box)
            # Raw-flow steering arrows for the rows that carry winds (ERA5 truth +
            # each percentile member; never the bias row).
            if row_wind[r] is not None:
                w = row_wind[r]
                if frame < w.shape[0] and w.shape[1] >= 2:
                    quiver_handle = draw_quiver(
                        ax,
                        w[frame, 0],
                        w[frame, 1],
                        domain_lat,
                        domain_lon,
                        scale=quiver_scale,
                        step=quiver_stride,
                    )
                    # One reference-scale key for the whole figure, on the first
                    # panel that drew arrows.
                    if (
                        quiver_handle is not None
                        and not quiver_key_done
                        and quiver_key_mag > 0
                    ):
                        draw_quiver_key(ax, quiver_handle, quiver_key_mag)
                        quiver_key_done = True
            # Progressive storm track "so far" up to this column's frame.
            if row_track_src[r] is not None:
                trk, t_color, t_ls = row_track_src[r]
                if frame < trk.shape[0]:
                    _draw_progressive_track(ax, trk, frame, t_color, t_ls)
            if c == 0:
                band_first_axes.append(ax)
                ax.text(
                    -0.32,
                    0.5,
                    row_labels[r],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=AXIS_LABEL_SIZE_DENSE,
                    color=row_colors[r],
                    fontweight="bold",
                )
            if r == 0 and c == truth_pct_col and np.isfinite(truth_pct):
                ax.text(
                    0.97,
                    0.96,
                    f"truth PC{eof_idx + 1} pct: {truth_pct:.0f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=TICK_LABEL_SIZE_DENSE,
                    color="black",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.75,
                        edgecolor="none",
                        boxstyle="round,pad=0.2",
                    ),
                )

    # Colorbars in the narrow right column: sequential for the truth row,
    # diverging for the bias + member rows.
    if raw_mesh is not None:
        cax_raw = fig.add_subplot(gs[0, n_cols])
        cb = fig.colorbar(raw_mesh, cax=cax_raw)
        cb.set_label(
            f"{var_name}{_units_suffix(var_name)} (ERA5)",
            fontsize=COLORBAR_LABEL_SIZE_DENSE,
        )
        cb.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)
    if anom_mesh is not None:
        cax_anom = fig.add_subplot(gs[1:n_map_rows, n_cols])
        cb = fig.colorbar(anom_mesh, cax=cax_anom)
        cb.set_label(
            f"{var_name} anomaly{_units_suffix(var_name)}",
            fontsize=COLORBAR_LABEL_SIZE_DENSE,
        )
        cb.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)

    # Impact panel spanning the map columns.  Every member is coloured by
    # its percentile along *this figure's* PC, so a real EOF->impact link
    # shows up as a dominant colour gradient (cool=low PC, warm=high PC).
    pct_cmap = plt.get_cmap(PCT_CMAP)
    pct_norm = plt.Normalize(vmin=0.0, vmax=100.0)
    member_pct = None
    # Leftmost axis of the impact row (band) that carries that band's per-row
    # subplot label; left None for the "no impact" placeholder.
    impact_label_ax = None
    # Set when the percentile colorbar has already been drawn (track + EOF
    # inset case rides it off the track edge so the pair can be centred);
    # otherwise the shared far-right-column colorbar below handles it.
    impact_cbar_done = False
    if impact_kind in ("track", "scalar"):
        scores = np.asarray(diag["pc_scores"])
        member_pct = _member_percentile(scores[:, eof_idx])
    if impact_kind == "track":
        tc_domain = np.asarray(diag.get("tc_domain", np.zeros(0)))
        impact_extent = (
            (
                float(tc_domain[0]),
                float(tc_domain[1]),
                float(tc_domain[2]),
                float(tc_domain[3]),
            )
            if tc_domain.size == 4
            else extent
        )
        # Outline the (smaller) track domain on the wider loading panel below so
        # the region the track map zooms into is explicit.  Only when tc_domain
        # is a real box -- otherwise impact_extent is the full loading extent and
        # the box would just trace the panel edge.
        track_box = impact_extent if tc_domain.size == 4 else None
        # The equal-aspect track map fills the row height and centres
        # horizontally, leaving wide L/R margins.  Reclaim the left margin
        # for the EOF loading pattern that defines this figure's PC axis, so
        # the headline figure is self-contained: precursor *mode* (left)
        # beside the track spread it sorts (right).  Falls back to the
        # original full-width track panel if the EOF field is absent.
        if "pc_components_latlon" in diag:
            # The precursor maps above are visually widened on the left by
            # their rotated row-labels; the impact row has none, so anchoring
            # the EOF+track pair to the map columns leaves it indented (more
            # whitespace on the left than the right).  Lay the pair out in its
            # own gridspec that reclaims most of that empty left band -- spans
            # the same vertical extent as the impact row, but starts at
            # ``CENTER_LEFT`` so the pair reads as horizontally centred.
            imp_pos = gs[impact_row, 0:n_cols].get_position(fig)
            # Reclaim most of the empty left row-label band and pull the right
            # edge in to balance the track's own percentile colorbar, so the
            # EOF map + track read as horizontally centred rather than
            # indented under the labelled maps above.  Offsets are relative to
            # the map-column span (so they track the gridspec margins) and
            # were tuned by rendering.
            sub = fig.add_gridspec(
                1,
                2,
                left=imp_pos.x0 - 0.034,
                right=imp_pos.x1 - 0.042,
                top=imp_pos.y1,
                bottom=imp_pos.y0,
                width_ratios=[1.1, 1.0],
                wspace=0.30,
            )
            # The synoptic EOF map is wide and short; in the tall impact row
            # it is width-bound, so give it (almost) the full left-cell width
            # with a slim vertical colorbar hung off its right edge -- no
            # label/tick overlap, and consistent with the diverging colorbars
            # above.  A spacer/map/spacer split vertically centres the group;
            # the map row's height is tuned to the wide synoptic aspect so the
            # map nearly fills its box and the bar matches the map's height.
            eof_cell = sub[0, 0].subgridspec(
                3,
                2,
                height_ratios=[0.05, 1.0, 0.05],
                width_ratios=[1.0, 0.045],
                hspace=0.0,
                wspace=0.04,
            )
            ax_eof = fig.add_subplot(eof_cell[1, 0], projection=ccrs.PlateCarree())
            # The EOF loading map is the leftmost axis of the impact band, so
            # it carries that band's per-row subplot label.
            impact_label_ax = ax_eof
            cax_eof = fig.add_subplot(eof_cell[1, 1])
            eof_mesh = _eof_loading_panel(
                ax_eof, diag, eof_idx, var_idx, extent, track_box=track_box
            )
            cb_eof = fig.colorbar(eof_mesh, cax=cax_eof, orientation="vertical")
            cb_eof.set_label(
                f"{var_name} EOF loading{_units_suffix(var_name)}",
                fontsize=COLORBAR_LABEL_SIZE_DENSE,
            )
            cb_eof.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)
            ax_imp = fig.add_subplot(sub[0, 1], projection=ccrs.PlateCarree())
            # Hang the percentile colorbar just off the track's right edge so
            # it rides with the pair (rather than living in the far-right
            # figure column), letting the EOF+track group centre as a unit.
            cax_imp = ax_imp.inset_axes([1.04, 0.13, 0.035, 0.74])
            sm = plt.cm.ScalarMappable(norm=pct_norm, cmap=pct_cmap)
            cb_imp = fig.colorbar(sm, cax=cax_imp)
            cb_imp.set_label(
                f"PC{eof_idx + 1} percentile", fontsize=COLORBAR_LABEL_SIZE_DENSE
            )
            cb_imp.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)
            impact_cbar_done = True
        else:
            ax_imp = fig.add_subplot(
                gs[impact_row, 0:n_cols], projection=ccrs.PlateCarree()
            )
            impact_label_ax = ax_imp
        _impact_track_panel(
            ax_imp, diag, eof_idx, member_pct, pct_cmap, pct_norm, impact_extent, mode
        )
    elif impact_kind == "scalar":
        ax_imp = fig.add_subplot(gs[impact_row, 0:n_cols])
        impact_label_ax = ax_imp
        _impact_timeseries_panel(ax_imp, diag, eof_idx, member_pct, pct_cmap, pct_norm)
    else:
        ax_imp = fig.add_subplot(gs[impact_row, 0:n_cols])
        ax_imp.axis("off")
        ax_imp.text(
            0.5,
            0.5,
            "impact panel unavailable\n(run the tc_tracks "
            "diagnostic, or configure a scalar impact)",
            ha="center",
            va="center",
            fontsize=AXIS_LABEL_SIZE_DENSE,
            color="gray",
            transform=ax_imp.transAxes,
        )

    # Percentile colorbar for the impact panel (narrow right column, impact
    # row).  Skipped when the track branch already hung it off the track edge.
    if member_pct is not None and not impact_cbar_done:
        cax_imp = fig.add_subplot(gs[impact_row, n_cols])
        sm = plt.cm.ScalarMappable(norm=pct_norm, cmap=pct_cmap)
        cb = fig.colorbar(sm, cax=cax_imp)
        cb.set_label(f"PC{eof_idx + 1} percentile", fontsize=COLORBAR_LABEL_SIZE_DENSE)
        cb.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)

    # Name the timezone once (mirrors local_time_axis._axis_label) so the
    # per-column wall-clock headers can stay tz-free.  It rides just below
    # the suptitle as a small, subordinate gray line rather than a
    # same-size second title line.
    tz_zone = (timezone_name or "UTC").replace("_", " ")
    tz_abbrev = (
        next(iter(frame_local_dt.values())).strftime("%Z") if frame_local_dt else ""
    )
    tz_label = (
        f"Local time ({tz_zone}; {tz_abbrev})"
        if tz_abbrev and tz_abbrev != tz_zone
        else f"Local time ({tz_zone})"
    )
    # Name the precursor-panel overlays once, on the same subordinate line, so a
    # reader knows the arrows are raw flow and the line+dot is the track.
    overlay_bits = []
    if winds_available:
        overlay_bits.append(f"arrows: raw {'/'.join(wind_vars[:2])} flow")
    if tracks_available:
        overlay_bits.append("line + dot: storm track so far")
    if precursor_excluded is not None:
        excl_word = {"land": "ocean", "sea": "land"}.get(impact_mask_kind, "masked")
        overlay_bits.append(f"hatch: {excl_word} (excluded from the impact mean)")
    if overlay_bits:
        tz_label += "      ·      " + "   ·   ".join(overlay_bits)
    fig.suptitle(
        f"{case_name} · {_mode_label(mode)} · "
        f"EOF{eof_idx + 1} ({pct_var:.0f}% var) · {var_name} {_free_end_role(mode)}",
        fontsize=SUPTITLE_SIZE_DENSE,
        color="black",
        y=0.99,
    )
    fig.text(
        0.5,
        0.95,
        tz_label,
        ha="center",
        va="center",
        fontsize=AXIS_LABEL_SIZE_DENSE,
        color="0.35",
    )
    # Per-row band labels: one ``a.)``/``b.)``/... per conceptual row (ERA5
    # truth, bias, each percentile member, then the impact panel) rather than
    # per map cell, which would scatter labels across the grid.  The leftmost
    # axis of each row carries the label via the shared helper.
    band_label_axes = band_first_axes + (
        [impact_label_ax] if impact_label_ax is not None else []
    )
    add_subplot_labels(
        band_label_axes, placement="inside", fontsize=SUBPLOT_LABEL_SIZE_DENSE
    )
    # Row labels sit at a fixed negative axes-fraction offset, and cartopy's
    # equal-aspect shrink makes the rendered axes width domain-dependent, so
    # wide-domain cases (e.g. ian, pnw_heatwave) push the labels past the
    # canvas edge while squarer ones (sandy) just clear it.  bbox_inches=
    # "tight" expands the canvas to include them (the labels are not clipped
    # to the axes), matching what plot_eofs already does.
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[aggregate_synoptic_pca] wrote {out_path}")


def plot_eof_track_pair(
    diag: dict,
    var_idx: int,
    var_name: str,
    mode: str,
    eof_indices: Sequence[int],
    out_path: Path,
) -> None:
    """Minimal loading + track-sorting figure for a set of EOFs.

    One row per requested EOF: the EOF loading over the full PCA domain
    (dashed box = the track domain) beside the member-track map coloured by
    that EOF's PC percentile, with ERA5's percentile along the mode
    annotated.  This is the minimum content needed to read a mode's loading
    geometry (broad/environmental vs. storm-collocated) and its track
    sorting (which axis it organizes, and where ERA5 sits along it); the
    per-EOF combined figures carry the full precursor-map anatomy.  Track
    impact cases only.
    """
    if "pc_components_latlon" not in diag:
        print(
            "[aggregate_synoptic_pca] npz has no pc_components_latlon; "
            f"skipping {out_path.name}"
        )
        return
    impact_kind = str(_scalar(diag.get("impact_kind", np.array("none"))))
    if impact_kind != "track":
        print(
            "[aggregate_synoptic_pca] --pair-eofs needs a track-impact case "
            f"(impact_kind={impact_kind!r}); skipping {out_path.name}"
        )
        return
    n_eof_show = int(_scalar(diag["n_eof_show"]))
    eof_list = [int(ei) for ei in eof_indices if 0 <= int(ei) < n_eof_show]
    if not eof_list:
        print(
            "[aggregate_synoptic_pca] none of the requested EOFs are baked "
            f"into the npz (n_eof_show={n_eof_show}); skipping {out_path.name}"
        )
        return

    extent = (
        float(_scalar(diag["lon_min"])),
        float(_scalar(diag["lon_max"])),
        float(_scalar(diag["lat_min"])),
        float(_scalar(diag["lat_max"])),
    )
    case_name = str(_scalar(diag["case_name"]))
    evr = np.asarray(diag["explained_variance_ratio"]).astype(float)
    era5_pc_pct = np.asarray(diag["era5_pc_percentile"]).astype(float)
    scores = np.asarray(diag["pc_scores"])

    tc_domain = np.asarray(diag.get("tc_domain", np.zeros(0)))
    impact_extent = (
        (
            float(tc_domain[0]),
            float(tc_domain[1]),
            float(tc_domain[2]),
            float(tc_domain[3]),
        )
        if tc_domain.size == 4
        else extent
    )
    track_box = impact_extent if tc_domain.size == 4 else None

    pct_cmap = plt.get_cmap(PCT_CMAP)
    pct_norm = plt.Normalize(vmin=0.0, vmax=100.0)

    n_rows = len(eof_list)
    fig_w = 11.0
    fig_h = 3.4 * n_rows + 1.1
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows,
        2,
        width_ratios=[1.35, 1.0],
        left=0.06,
        right=0.92,
        top=1.0 - 0.9 / fig_h,
        bottom=0.35 / fig_h,
        wspace=0.24,
        hspace=0.3,
    )

    label_axes: List = []
    for j, ei in enumerate(eof_list):
        # Loading panel (full PCA domain) with its own colorbar: loading
        # amplitudes differ per EOF, so the bars are per-row.
        load_cell = gs[j, 0].subgridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.05)
        ax_eof = fig.add_subplot(load_cell[0, 0], projection=ccrs.PlateCarree())
        cax_eof = fig.add_subplot(load_cell[0, 1])
        eof_mesh = _eof_loading_panel(
            ax_eof, diag, ei, var_idx, extent, track_box=track_box
        )
        pct_var = 100.0 * evr[ei] if ei < evr.size else 0.0
        ax_eof.set_title(
            f"EOF{ei + 1} loading ({pct_var:.0f}% var)",
            fontsize=TITLE_SIZE_DENSE,
        )
        cb_eof = fig.colorbar(eof_mesh, cax=cax_eof, orientation="vertical")
        cb_eof.set_label(
            f"{var_name} EOF loading{_units_suffix(var_name)}",
            fontsize=COLORBAR_LABEL_SIZE_DENSE,
        )
        cb_eof.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)

        # Track panel coloured by this EOF's PC percentile, with ERA5's
        # free-end percentile along the mode annotated (matches the "truth
        # PC<i> pct" box on the per-EOF combined figure).
        ax_imp = fig.add_subplot(gs[j, 1], projection=ccrs.PlateCarree())
        member_pct = _member_percentile(scores[:, ei])
        _impact_track_panel(
            ax_imp, diag, ei, member_pct, pct_cmap, pct_norm, impact_extent, mode
        )
        truth_pct = float(era5_pc_pct[ei]) if ei < era5_pc_pct.size else float("nan")
        if np.isfinite(truth_pct):
            ax_imp.text(
                0.03,
                0.03,
                f"ERA5 PC{ei + 1} pct: {truth_pct:.0f}",
                transform=ax_imp.transAxes,
                va="bottom",
                ha="left",
                fontsize=TICK_LABEL_SIZE_DENSE,
                color="black",
                bbox=dict(
                    facecolor="white",
                    alpha=0.75,
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                ),
            )
        cax_imp = ax_imp.inset_axes([1.04, 0.13, 0.035, 0.74])
        sm = plt.cm.ScalarMappable(norm=pct_norm, cmap=pct_cmap)
        cb_imp = fig.colorbar(sm, cax=cax_imp)
        cb_imp.set_label(f"PC{ei + 1} percentile", fontsize=COLORBAR_LABEL_SIZE_DENSE)
        cb_imp.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)
        label_axes.extend([ax_eof, ax_imp])

    eof_title = " + ".join(
        (
            f"EOF{ei + 1} ({100.0 * evr[ei]:.0f}% var)"
            if ei < evr.size
            else f"EOF{ei + 1}"
        )
        for ei in eof_list
    )
    fig.suptitle(
        f"{case_name} · {_mode_label(mode)} · "
        f"{eof_title} · {var_name} {_free_end_role(mode)}",
        fontsize=SUPTITLE_SIZE_DENSE,
        color="black",
        y=0.99,
    )
    add_subplot_labels(
        label_axes, placement="inside", fontsize=SUBPLOT_LABEL_SIZE_DENSE
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[aggregate_synoptic_pca] wrote {out_path}")


# ---------------------------------------------------------------------------
# Per-EOF percentile-member animation (watch the storm move)
# ---------------------------------------------------------------------------


def _save_animation(anim, out_base: Path, fps: int) -> Path:
    """Save ``anim`` as an mp4 via ffmpeg.

    ffmpeg is a hard requirement for the videos.  If it is missing or broken the
    caller logs the failure and skips the video (the figures are unaffected), so
    the environment problem surfaces instead of being masked by a lower-quality
    format.
    """
    out = out_base.with_suffix(".mp4")
    anim.save(str(out), writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    return out


def render_member_videos(
    diag: dict,
    mode: str,
    eof_idx: int,
    out_base: Path,
) -> None:
    """Animate the p5/p50/p95 members' storm field over the lead window.

    One 3-panel video per (mode, EOF): each panel shows a percentile member's
    video field (MSL, where the hurricane low is clearest) evolving frame by
    frame, with that member's track drawn progressively, the current position
    marked, and the observed ERA5 track dashed for reference.  TC cases only;
    silently skipped when the video field or tracks are absent (e.g. scalar
    cases or an npz written before the field existed).
    """
    video = np.asarray(diag.get("percentile_member_video_latlon", np.zeros(0)))
    member_track = np.asarray(diag.get("member_track_latlon", np.zeros(0)))
    if video.size == 0 or member_track.size == 0:
        return
    n_eof_show = int(_scalar(diag["n_eof_show"]))
    if eof_idx >= n_eof_show or eof_idx >= video.shape[0]:
        return
    era5_track = np.asarray(diag.get("era5_track_latlon", np.zeros(0)))
    video_var = str(_scalar(diag.get("video_variable", np.array(""))))

    pct_values = np.asarray(diag["percentile_values"]).astype(int)
    map_subset = _map_pct_subset(pct_values.size)
    member_idx = np.asarray(diag["percentile_member_idx"]).astype(int)
    sub_members = [int(member_idx[eof_idx, si]) for si in map_subset]

    domain_lat = np.asarray(diag["domain_lat"])
    domain_lon = np.asarray(diag["domain_lon"])
    lead_h = np.asarray(diag["lead_hours"])
    n_leads = len(lead_h)
    case_name = str(_scalar(diag["case_name"]))

    # Fields for the shown members, in display units, on a shared colour scale.
    vids = video[eof_idx][map_subset].astype(np.float32)  # [n_show, n_leads, ny, nx]
    if is_geopotential(video_var):
        vids = to_display_units(video_var, vids)
    vmin, vmax = _shared_minmax([vids], pct=1.0)

    # Zoom to the TC domain when available; else the precursor box.
    tc_domain = np.asarray(diag.get("tc_domain", np.zeros(0)))
    if tc_domain.size == 4:
        view = tuple(float(v) for v in tc_domain)
    else:
        view = (
            float(_scalar(diag["lon_min"])),
            float(_scalar(diag["lon_max"])),
            float(_scalar(diag["lat_min"])),
            float(_scalar(diag["lat_max"])),
        )
    img_extent = [
        float(domain_lon.min()),
        float(domain_lon.max()),
        float(domain_lat.min()),
        float(domain_lat.max()),
    ]
    origin = "upper" if domain_lat[0] > domain_lat[-1] else "lower"

    start_time_iso = str(
        _scalar(diag.get("start_time", np.array("1970-01-01T00:00:00Z")))
    )
    tz_name = str(_scalar(diag.get("timezone", np.array("UTC"))))

    n_show = len(map_subset)
    fig, axes = plt.subplots(
        1,
        n_show,
        figsize=(4.3 * n_show, 4.4),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = np.atleast_1d(axes)
    halo = [pe.withStroke(linewidth=_TRACK_HALO_LW, foreground="white")]
    images, track_lines, dots = [], [], []
    for k, ax in enumerate(axes):
        ax.set_extent(view, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray")
        images.append(
            ax.imshow(
                vids[k, 0],
                origin=origin,
                extent=img_extent,
                transform=ccrs.PlateCarree(),
                cmap=_VIDEO_CMAP,
                vmin=vmin,
                vmax=vmax,
                zorder=0,
            )
        )
        if era5_track.size:
            ax.plot(
                era5_track[:, 1],
                era5_track[:, 0],
                color="black",
                linestyle="--",
                linewidth=1.6,
                transform=ccrs.PlateCarree(),
                zorder=4,
                path_effects=halo,
                label="ERA5",
            )
        (ln,) = ax.plot(
            [],
            [],
            color=_TRACK_MEMBER_COLOR,
            linewidth=_TRACK_LW,
            transform=ccrs.PlateCarree(),
            zorder=5,
            path_effects=halo,
            label="member",
        )
        (dot,) = ax.plot(
            [],
            [],
            marker="o",
            color=_TRACK_MEMBER_COLOR,
            markersize=_TRACK_MARKER_SIZE + 1.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        track_lines.append(ln)
        dots.append(dot)
        ax.set_title(
            f"p{int(pct_values[map_subset[k]])}  (mem {sub_members[k]})",
            fontsize=TITLE_SIZE_DENSE,
        )
    axes[0].legend(loc="lower left", fontsize=LEGEND_SIZE_DENSE, framealpha=0.9)
    cbar = fig.colorbar(
        images[0], ax=list(axes), orientation="vertical", fraction=0.025, pad=0.02
    )
    cbar.set_label(axis_label(video_var), fontsize=COLORBAR_LABEL_SIZE_DENSE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)

    title_base = (
        f"{case_name} · {_mode_label(mode)} · EOF{eof_idx + 1} · "
        f"{long_name(video_var)} evolution"
    )
    sup = fig.suptitle(
        title_base,
        fontsize=SUPTITLE_SIZE_DENSE,
        color="black",
    )

    def _update(frame: int):
        for k in range(n_show):
            images[k].set_data(vids[k, frame])
            trk = np.asarray(member_track[sub_members[k]], dtype=np.float64)
            seg = trk[: frame + 1]
            la, lo = seg[:, 0], seg[:, 1]
            valid = np.isfinite(la) & np.isfinite(lo)
            track_lines[k].set_data(lo[valid], la[valid])
            if valid.any():
                last = int(np.flatnonzero(valid)[-1])
                dots[k].set_data([lo[last]], [la[last]])
            else:
                dots[k].set_data([], [])
        stamp = local_datetimes(start_time_iso, [lead_h[frame]], tz_name)[0].strftime(
            "%b %d %H:%M %Z"
        )
        sup.set_text(f"{title_base}  ·  {stamp}")
        return [*images, *track_lines, *dots]

    anim = animation.FuncAnimation(fig, _update, frames=n_leads, blit=False)
    try:
        saved = _save_animation(anim, out_base, _VIDEO_FPS)
        print(f"[aggregate_synoptic_pca] wrote {saved}")
    except Exception as exc:  # noqa: BLE001 - one video must not sink the figures
        print(
            f"[aggregate_synoptic_pca] video skipped ({out_base.name}): {exc}\n"
            f"    the videos need a working ffmpeg; verify with `ffmpeg -version`."
        )
    plt.close(fig)


# ---------------------------------------------------------------------------
# EOF patterns supplement
# ---------------------------------------------------------------------------


def plot_eofs(
    per_mode: Dict[str, dict],
    var_idx: int,
    var_name: str,
    out_path: Path,
) -> None:
    """Leading EOF loading patterns per mode, in physical units."""
    modes = [m for m in MODE_ORDER if m in per_mode]
    if not modes:
        return

    any_diag = per_mode[modes[0]]
    extent = (
        float(_scalar(any_diag["lon_min"])),
        float(_scalar(any_diag["lon_max"])),
        float(_scalar(any_diag["lat_min"])),
        float(_scalar(any_diag["lat_max"])),
    )
    case_name = str(_scalar(any_diag["case_name"]))

    # Outline the impact sub-domain (the region the downstream impact metric is
    # taken over) on every loading panel, so the reader can see where each EOF's
    # lobes fall relative to it.  Scalar cases use the impact bbox; TC cases use
    # the track domain.  Mirrors the precursor-map box in plot_combined.
    impact_kind = str(_scalar(any_diag.get("impact_kind", np.array("none"))))
    impact_bbox = np.asarray(any_diag.get("impact_bbox", np.zeros(0)))
    tc_domain = np.asarray(any_diag.get("tc_domain", np.zeros(0)))
    domain_box: Tuple[float, ...] | None = None
    if impact_kind == "scalar" and impact_bbox.size == 4:
        domain_box = tuple(float(v) for v in impact_bbox)
    elif impact_kind == "track" and tc_domain.size == 4:
        domain_box = tuple(float(v) for v in tc_domain)

    eof_counts = [int(_scalar(per_mode[m]["n_eof_show"])) for m in modes]
    n_cols = max(eof_counts)

    fields_for_range: List[np.ndarray] = []
    for m in modes:
        eofs = np.asarray(
            per_mode[m]["pc_components_latlon"]
        )  # [n_eof, n_vars, lat, lon]
        fields_for_range.append(eofs[:, var_idx])
    vrange = _shared_color_range(fields_for_range, percentile=99.0)

    fig_w = 3.4 * n_cols
    fig_h = 3.6 * len(modes) + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        len(modes),
        n_cols,
        left=0.06,
        right=0.94,
        top=0.85,
        bottom=0.16,
        wspace=0.10,
        hspace=0.20,
    )

    mesh = None
    any_excluded = False
    eof_axes: List = []
    for i, mode in enumerate(modes):
        d = per_mode[mode]
        n_eof = int(_scalar(d["n_eof_show"]))
        evr = np.asarray(d["explained_variance_ratio"]).astype(float)
        eofs = np.asarray(d["pc_components_latlon"])  # [n_eof, n_vars, lat, lon]
        domain_lat = np.asarray(d["domain_lat"])
        domain_lon = np.asarray(d["domain_lon"])
        # Hatch the part of the impact box dropped by the land/sea mask, so the
        # outlined sub-domain matches what the downstream mean actually uses.
        d_excluded = np.asarray(d.get("impact_excluded_mask_box", np.zeros(0)))
        mode_excluded = (
            d_excluded
            if (impact_kind == "scalar" and d_excluded.size and d_excluded.any())
            else None
        )
        any_excluded = any_excluded or mode_excluded is not None

        leftmost_ax = None
        for c in range(n_cols):
            ax = fig.add_subplot(gs[i, c], projection=ccrs.PlateCarree())
            if c == 0:
                leftmost_ax = ax
            if c < n_eof:
                pct_var = 100.0 * evr[c] if c < evr.size else 0.0
                mesh = _map_panel(
                    ax,
                    eofs[c, var_idx],
                    domain_lat,
                    domain_lon,
                    vmin=-vrange,
                    vmax=vrange,
                    cmap=ANOM_CMAP,
                    extent=extent,
                    title=f"EOF {c + 1}  ({pct_var:.1f}% var)",
                    show_xlabels=(i == len(modes) - 1),
                    show_ylabels=(c == 0),
                    excluded_mask=mode_excluded,
                )
                if domain_box is not None:
                    draw_domain_box(ax, domain_box)
                eof_axes.append(ax)
            else:
                ax.set_axis_off()

        if leftmost_ax is not None:
            leftmost_ax.text(
                -0.22,
                0.5,
                f"{_mode_label(mode)}",
                transform=leftmost_ax.transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=SECTION_HEADER_SIZE_DENSE,
                color="black",
                fontweight="bold",
            )

    add_subplot_labels(eof_axes, placement="inside", fontsize=SUBPLOT_LABEL_SIZE_DENSE)
    fig.suptitle(
        f"{case_name} · leading EOF patterns · {var_name}",
        fontsize=SUPTITLE_SIZE_DENSE,
    )
    if any_excluded:
        excl_word = {"land": "ocean", "sea": "land"}.get(
            str(_scalar(any_diag.get("impact_mask_kind", np.array("none")))), "masked"
        )
        fig.text(
            0.5,
            0.89,
            f"dashed box: impact domain · hatch: {excl_word} "
            f"(excluded from the impact mean)",
            ha="center",
            va="center",
            fontsize=AXIS_LABEL_SIZE_DENSE,
            color="0.35",
        )
    if mesh is not None:
        cbar_ax = fig.add_axes([0.18, 0.07, 0.66, 0.018])
        cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(
            f"{var_name} EOF loading{_units_suffix(var_name)}",
            fontsize=COLORBAR_LABEL_SIZE_DENSE,
        )
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[aggregate_synoptic_pca] wrote {out_path}")


# ---------------------------------------------------------------------------
# PC1-PC2 scatter supplement
# ---------------------------------------------------------------------------


def plot_pc_scatter(diag: dict, mode: str, out_path: Path) -> None:
    """PC1-PC2 scatter with the percentile members + ERA5 marked."""
    scores = np.asarray(diag["pc_scores"])  # [N, max_d]
    if scores.shape[1] < 2:
        return
    n_eof_show = int(_scalar(diag["n_eof_show"]))
    pct_values = np.asarray(diag["percentile_values"]).astype(int)
    n_pct = pct_values.size
    pct_colors = _pct_member_colors(n_pct)
    member_idx = np.asarray(diag["percentile_member_idx"]).astype(
        int
    )  # [n_eof_show, n_pct]
    evr = np.asarray(diag["explained_variance_ratio"]).astype(float)
    era5_pc = np.asarray(diag["era5_pc_score"]).astype(float)
    case_name = str(_scalar(diag["case_name"]))

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(
        scores[:, 0],
        scores[:, 1],
        s=8,
        color=_SPAGHETTI_COLOR,
        alpha=0.4,
        zorder=1,
        label="members",
    )

    # PC1 percentile members along the x mode.
    for pi in range(n_pct):
        m = int(member_idx[0, pi])
        ax.scatter(
            scores[m, 0],
            scores[m, 1],
            s=90,
            color=pct_colors[pi],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
            label=f"EOF1 p{int(pct_values[pi])}",
        )
    # PC2 percentile members along the y mode (open markers to distinguish).
    if n_eof_show >= 2:
        for pi in range(n_pct):
            m = int(member_idx[1, pi])
            ax.scatter(
                scores[m, 0],
                scores[m, 1],
                s=120,
                facecolors="none",
                edgecolors=pct_colors[pi],
                linewidth=1.6,
                marker="s",
                zorder=2,
            )

    if era5_pc.size >= 2:
        ax.scatter(
            era5_pc[0],
            era5_pc[1],
            marker="*",
            s=260,
            color="black",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label="ERA5",
        )

    pc1_var = 100.0 * evr[0] if evr.size > 0 else 0.0
    pc2_var = 100.0 * evr[1] if evr.size > 1 else 0.0
    ax.set_xlabel(f"PC1 ({pc1_var:.0f}% var)", fontsize=AXIS_LABEL_SIZE_DENSE)
    ax.set_ylabel(f"PC2 ({pc2_var:.0f}% var)", fontsize=AXIS_LABEL_SIZE_DENSE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE_DENSE)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=LEGEND_SIZE_DENSE, loc="best", framealpha=0.9, ncol=2)
    if n_eof_show >= 2:
        ax.set_title(
            "filled = EOF1 percentiles · open square = EOF2",
            fontsize=TICK_LABEL_SIZE_DENSE,
        )
    fig.suptitle(
        f"{case_name} · {_mode_label(mode)} · PC space",
        fontsize=TITLE_SIZE_DENSE,
        color="black",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[aggregate_synoptic_pca] wrote {out_path}")


# ---------------------------------------------------------------------------
# Combined-EOF precursor -> impact summary (regression map + skill scatter)
# ---------------------------------------------------------------------------


def plot_precursor_impact(
    diag: dict,
    var_idx: int,
    var_name: str,
    mode: str,
    out_path: Path,
) -> None:
    """Combined precursor -> scalar-impact figure (one per mode), 2x2.

    Folds the precursor *pattern* and its *predictive magnitude* into one figure:

      (0,0) warmest-decile precursor anomaly vs the ensemble mean
      (0,1) coolest-decile precursor anomaly vs the ensemble mean
      (1,0) warm - cool composite -- the precursor pattern of an extreme impact,
            model-free (no assumed precursor -> impact form)
      (1,1) predicted-vs-ensemble-member land-mean impact: the leading-PC regression's
            skill (in-sample + 10-fold CV R^2, F-test p), with the 1:1 line and
            ERA5 marked (its precursor projected through the same fit).

    The three z500 panels share one colorbar; the warm-cool difference is ~2x
    each decile's deviation (the deciles are near mirror images), so on a single
    scale the deciles read at about half the difference's saturation -- honest
    rather than separately normalized.  The composites carry the *pattern*, the
    scatter its *magnitude*: a small CV R^2 with a clean composite is a real but
    weak link, not a null one (hence the explicit p-value).  The standalone
    field-on-impact regression map (z500 per +1 K) is dropped as redundant with
    the model-free warm - cool composite.  No-op for non-scalar impact or an npz
    lacking the decile / scalar fields.
    """
    impact_kind = str(_scalar(diag.get("impact_kind", np.array("none"))))
    if impact_kind != "scalar":
        return
    warm = np.asarray(diag.get("impact_decile_warm_latlon", np.zeros(0)))
    cool = np.asarray(diag.get("impact_decile_cool_latlon", np.zeros(0)))
    member_scalar = np.asarray(diag.get("member_impact_scalar", np.zeros(0)))
    scores = np.asarray(diag.get("pc_scores", np.zeros(0)))
    if (
        warm.size == 0
        or cool.size == 0
        or member_scalar.size == 0
        or member_scalar.ndim != 2
        or scores.size == 0
    ):
        return
    free_end_frame = int(_scalar(diag["free_end_frame"]))
    if not 0 <= free_end_frame < member_scalar.shape[1]:
        return
    ens_mean_traj = np.asarray(diag["ensemble_mean_trajectory_latlon"])
    if free_end_frame >= ens_mean_traj.shape[0] or var_idx >= warm.shape[0]:
        return
    n_eof_show = int(_scalar(diag["n_eof_show"]))
    n_pc = int(min(n_eof_show, scores.shape[1]))
    if n_pc < 1:
        return

    impact_variable = str(_scalar(diag["impact_variable"]))
    impact_u = var_units(impact_variable)
    field_u = var_units(var_name)

    # --- Decile composites (the model-free precursor pattern) ---
    ens_fe = ens_mean_traj[free_end_frame, var_idx]  # [lat, lon]
    warm_anom = warm[var_idx] - ens_fe
    cool_anom = cool[var_idx] - ens_fe
    frac = float(_scalar(diag.get("impact_decile_frac", np.float32(0.1))))
    warm_mean = float(_scalar(diag.get("impact_decile_warm_mean", np.float32("nan"))))
    cool_mean = float(_scalar(diag.get("impact_decile_cool_mean", np.float32("nan"))))

    # --- Leading-PC regression skill (the predictive magnitude) ---
    y = np.asarray(member_scalar[:, free_end_frame], dtype=np.float64)
    era5_scalar = np.asarray(
        diag.get("era5_impact_scalar", np.zeros(0)), dtype=np.float64
    )
    if is_geopotential(impact_variable):
        y = to_display_units(impact_variable, y)
        era5_scalar = to_display_units(impact_variable, era5_scalar)
        warm_mean = float(to_display_units(impact_variable, warm_mean))
        cool_mean = float(to_display_units(impact_variable, cool_mean))
    pcs = np.asarray(scores[:, :n_pc], dtype=np.float64)
    beta, intercept = _ols_fit(pcs, y)
    yhat = intercept + pcs @ beta
    r2_in = _r2_score(y, yhat)
    r2_cv = _kfold_r2(pcs, y, n_folds=10)
    p_value = _regression_f_pvalue(r2_in, int(y.size), n_pc)
    # ERA5: project its saved precursor through the same fit (predicted), and
    # read its observed land-mean impact at the free-end frame (observed).
    era5_pc = np.asarray(diag.get("era5_pc_score", np.zeros(0)), dtype=np.float64)
    era5_yhat = (
        intercept + float(era5_pc[:n_pc] @ beta)
        if era5_pc.size >= n_pc
        else float("nan")
    )
    era5_y = (
        float(era5_scalar[free_end_frame])
        if free_end_frame < era5_scalar.size
        else float("nan")
    )
    r2_in_txt = f"{r2_in:.2f}" if np.isfinite(r2_in) else "n/a"
    r2_cv_txt = f"{r2_cv:.2f}" if np.isfinite(r2_cv) else "n/a"
    if not np.isfinite(p_value):
        p_txt = "p = n/a"
    elif p_value < 1e-3:
        p_txt = "p < 0.001"
    else:
        p_txt = f"p = {p_value:.3f}"
    print(
        f"[aggregate_synoptic_pca] {mode} precursor->impact ({n_pc} PCs): "
        f"R2 in-sample={r2_in_txt} ({p_txt}), R2 CV(10-fold)={r2_cv_txt}"
    )
    # Regression composite: the z500 field regressed on the impact (sum_k c_k
    # EOF_k) -- the spatial pattern the precursor index measures, in field units
    # per impact unit.  This is the spatial referent for the scatter's x-axis.
    eofs = np.asarray(diag["pc_components_latlon"])  # [n_eof, n_vars, lat, lon]
    reg_composite = np.tensordot(
        _field_impact_regression_weights(pcs, y), eofs[:n_pc, var_idx], axes=(0, 0)
    )  # [lat, lon]

    domain_lat = np.asarray(diag["domain_lat"])
    domain_lon = np.asarray(diag["domain_lon"])
    extent = (
        float(_scalar(diag["lon_min"])),
        float(_scalar(diag["lon_max"])),
        float(_scalar(diag["lat_min"])),
        float(_scalar(diag["lat_max"])),
    )
    case_name = str(_scalar(diag["case_name"]))
    impact_excluded = np.asarray(diag.get("impact_excluded_mask_box", np.zeros(0)))
    excluded = (
        impact_excluded if impact_excluded.size and impact_excluded.any() else None
    )
    impact_bbox = np.asarray(diag.get("impact_bbox", np.zeros(0)))
    box = tuple(float(v) for v in impact_bbox) if impact_bbox.size == 4 else None
    # Decile composites (m) share one range/colorbar; the regression composite
    # is field-per-impact (m/K), a different quantity, so it gets its own.
    vr_dec = _shared_color_range([warm_anom, cool_anom], percentile=99.0)
    vr_reg = _shared_color_range([reg_composite], percentile=99.0)

    # Decile-composite peak amplitude: max |anomaly| over the warm/cool
    # composites, already in display units (geopotential -> height, m).  The
    # scripted source for the paper's "z500 decile composite peaks at +-N m"
    # number -- quote this peak, not the shared colorbar bound vr_dec, which is
    # the 99th percentile and saturates the extremes.
    comp_peak = float(
        np.nanmax(np.abs(np.concatenate([np.ravel(warm_anom), np.ravel(cool_anom)])))
    )
    comp_unit = var_units(var_name)
    print(
        f"[aggregate_synoptic_pca] {mode} {var_name} decile-composite peak "
        f"|anomaly| = {comp_peak:.2f}{(' ' + comp_unit) if comp_unit else ''} "
        f"(colorbar 99th-pct range = {vr_dec:.2f})"
    )

    fig = plt.figure(figsize=(13.0, 10.5))
    gs = fig.add_gridspec(
        2,
        2,
        hspace=0.32,
        wspace=0.10,
        left=0.06,
        right=0.97,
        top=0.84,
        bottom=0.10,
    )
    ax_warm = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_cool = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax_reg = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax_sc = fig.add_subplot(gs[1, 1])

    # --- Top row: warmest/coolest decile composites (model-free; shared bar) ---
    decile_panels = [
        (
            ax_warm,
            warm_anom,
            "Warmest decile composite − ens. mean",
            warm_mean,
            False,
            True,
        ),
        (
            ax_cool,
            cool_anom,
            "Coolest decile composite − ens. mean",
            cool_mean,
            True,
            False,
        ),
    ]
    dec_mesh = None
    for ax_p, field, title, dec_mean, showx, showy in decile_panels:
        dec_mesh = _map_panel(
            ax_p,
            field,
            domain_lat,
            domain_lon,
            vmin=-vr_dec,
            vmax=vr_dec,
            cmap=ANOM_CMAP,
            extent=extent,
            title=title,
            show_xlabels=showx,
            show_ylabels=showy,
            excluded_mask=excluded,
        )
        if box is not None:
            draw_domain_box(ax_p, box)
        ax_p.text(
            0.97,
            0.04,
            f"{symbol(impact_variable)} = {dec_mean:.1f}"
            f"{(' ' + impact_u) if impact_u else ''}",
            transform=ax_p.transAxes,
            va="bottom",
            ha="right",
            fontsize=TICK_LABEL_SIZE_DENSE,
            bbox=dict(
                facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"
            ),
            zorder=6,
        )
    cb_dec = fig.colorbar(
        dec_mesh,
        ax=[ax_warm, ax_cool],
        orientation="horizontal",
        fraction=0.046,
        pad=0.07,
    )
    cb_dec.set_label(
        f"Decile-composite {symbol(var_name)} anomaly{_units_suffix(var_name)}",
        fontsize=COLORBAR_LABEL_SIZE_DENSE,
    )
    cb_dec.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)

    # --- Bottom-left: precursor-index pattern (z500 regressed on the impact) ---
    # The spatial referent for the scatter's x-axis: the z500 anomaly the index
    # ties to +1 unit of impact.  Should resemble the warmest-decile ridge (a
    # linear vs. tail cross-check); its own colorbar since the units differ.
    reg_mesh = _map_panel(
        ax_reg,
        reg_composite,
        domain_lat,
        domain_lon,
        vmin=-vr_reg,
        vmax=vr_reg,
        cmap=ANOM_CMAP,
        extent=extent,
        title=f"{symbol(var_name)} regressed on land-mean "
        f"{symbol(impact_variable)} ({n_pc} EOFs)",
        show_xlabels=True,
        show_ylabels=True,
        excluded_mask=excluded,
    )
    if box is not None:
        draw_domain_box(ax_reg, box)
    cb_reg = fig.colorbar(
        reg_mesh, ax=ax_reg, orientation="horizontal", fraction=0.046, pad=0.10
    )
    reg_unit = f" ({field_u} {impact_u}$^{{-1}}$)" if field_u and impact_u else ""
    cb_reg.set_label(
        f"{symbol(var_name)} regressed on {symbol(impact_variable)}{reg_unit}",
        fontsize=COLORBAR_LABEL_SIZE_DENSE,
    )
    cb_reg.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)

    # --- Bottom-right: predicted vs ensemble-member land-mean impact ---
    # Highlight the members that make up the warm/cool composites above -- the
    # top/bottom decile by ensemble-member t2m (the same ranking compute_synoptic_pca
    # composited).  They are the top/bottom bands in y by construction; their
    # spread in x shows how well the precursor index separates the extremes
    # (weak at low R^2) and ties the scatter to the composite panels.  Red =
    # warm, blue = cool, matching the RdBu composite colormap.
    n_mem = int(y.size)
    n_dec = max(1, int(round(frac * n_mem)))
    order = np.argsort(y)
    cool_sel = order[:n_dec]
    warm_sel = order[-n_dec:]
    mid_mask = np.ones(n_mem, dtype=bool)
    mid_mask[cool_sel] = False
    mid_mask[warm_sel] = False
    ax_sc.scatter(
        yhat[mid_mask],
        y[mid_mask],
        s=9,
        color=_SPAGHETTI_COLOR,
        alpha=0.30,
        edgecolors="none",
        zorder=1,
        label="other members",
    )
    ax_sc.scatter(
        yhat[cool_sel],
        y[cool_sel],
        s=14,
        color=_PCT_COLORS_5[0],
        alpha=0.55,
        edgecolors="none",
        zorder=2,
        label="coolest decile",
    )
    ax_sc.scatter(
        yhat[warm_sel],
        y[warm_sel],
        s=14,
        color=_PCT_COLORS_5[-1],
        alpha=0.55,
        edgecolors="none",
        zorder=2,
        label="warmest decile",
    )
    finite_pts = [yhat, y]
    if np.isfinite(era5_yhat):
        finite_pts.append(np.array([era5_yhat]))
    if np.isfinite(era5_y):
        finite_pts.append(np.array([era5_y]))
    allv = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in finite_pts])
    lo, hi = float(allv.min()), float(allv.max())
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    span = np.array([lo - pad, hi + pad])
    ax_sc.plot(
        span, span, color="black", linewidth=1.2, alpha=0.8, zorder=3, label="1:1"
    )
    if np.isfinite(era5_yhat) and np.isfinite(era5_y):
        ax_sc.scatter(
            era5_yhat,
            era5_y,
            marker="*",
            s=260,
            color="black",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label="ERA5",
        )
    ax_sc.set_xlim(float(span[0]), float(span[1]))
    ax_sc.set_ylim(float(span[0]), float(span[1]))
    ax_sc.set_aspect("equal", adjustable="box")
    iu_suffix = f" ({impact_u})" if impact_u else ""
    ax_sc.set_xlabel(
        f"Precursor-index prediction of {symbol(impact_variable)}{iu_suffix}",
        fontsize=AXIS_LABEL_SIZE_DENSE,
    )
    ax_sc.set_ylabel(
        f"Ensemble-member land-mean {symbol(impact_variable)}{iu_suffix}",
        fontsize=AXIS_LABEL_SIZE_DENSE,
    )
    ax_sc.tick_params(axis="both", labelsize=TICK_LABEL_SIZE_DENSE)
    ax_sc.grid(True, alpha=0.25)
    # Stats ride the top-right (the right band is empty for the same reason the
    # legend sits in the left one) so they clear the d.) subplot label, which
    # the shared helper pins to the top-left corner.
    ax_sc.text(
        0.97,
        0.97,
        f"{n_pc} PCs · N = {int(y.size)}\n"
        f"$R^2$ in-sample = {r2_in_txt}  ({p_txt})\n"
        f"$R^2$ CV (10-fold) = {r2_cv_txt}",
        transform=ax_sc.transAxes,
        va="top",
        ha="right",
        fontsize=TICK_LABEL_SIZE_DENSE,
        bbox=dict(
            facecolor="white", alpha=0.85, edgecolor="0.7", boxstyle="round,pad=0.3"
        ),
        zorder=6,
    )
    # Legend rides the empty left band (predictions are compressed toward the
    # mean, so the cloud is a central vertical strip -- the left/right bands are
    # empty), keeping it off the points.
    ax_sc.legend(fontsize=LEGEND_SIZE_DENSE, loc="center left", framealpha=0.9)

    # a.) b.) c.) d.) over the four panels; inside placement keeps the labels
    # off the per-panel titles (the three maps each carry one) and legible over
    # the busy anomaly fields / scatter cloud.
    add_subplot_labels(
        [ax_warm, ax_cool, ax_reg, ax_sc],
        placement="inside",
        fontsize=SUBPLOT_LABEL_SIZE_DENSE,
    )

    fig.suptitle(
        f"{case_name}\n"
        f"{_mode_label(mode).capitalize()} · {_free_end_role(mode)} → "
        f"{long_name(impact_variable)}\n"
        f"warmest & coolest decile composites + {n_pc}-PC regression",
        fontsize=SUPTITLE_SIZE_DENSE,
        color="black",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"[aggregate_synoptic_pca] wrote {out_path}  "
        f"(R2 in={r2_in_txt} {p_txt}, CV={r2_cv_txt}, {n_pc} PCs)"
    )


def plot_domain_ladder(
    diag: dict,
    var_idx: int,
    var_name: str,
    mode: str,
    out_path: Path,
) -> None:
    """Domain-sensitivity ladder: precursor -> impact skill + pattern vs box size.

    De-attenuation check on the impact-averaging domain choice.  For the nested
    set of impact boxes (the headline box scaled about its centre, computed in
    ``compute_synoptic_pca``), regress the free-end impact scalar on the leading
    PCs and show:

      * top    -- the field-on-impact regression composite for each box (the
        same field-on-impact construction dropped from the headline figure but
        kept here as the per-domain stability probe), that box dashed, so the
        precursor *pattern* can be checked for stability across box sizes.  Each
        carries its spatial correlation with the widest box's composite.
      * bottom -- in-sample and cross-validated R^2 vs the number of masked
        pixels in the box (the mean's effective sample size).  R^2 that rises
        then flattens as the box grows is the signature of noise reduction
        (de-attenuation), not a domain tuned to the relationship; a composite
        that drifts across the top row flags a fragile, domain-dependent link.

    No-op for non-scalar impact or an npz written before the ladder fields.
    """
    impact_kind = str(_scalar(diag.get("impact_kind", np.array("none"))))
    if impact_kind != "scalar":
        return
    member_ladder = np.asarray(diag.get("member_impact_ladder", np.zeros(0)))
    scores = np.asarray(diag.get("pc_scores", np.zeros(0)))
    if member_ladder.size == 0 or member_ladder.ndim != 2 or scores.size == 0:
        return
    ladder_bboxes = np.asarray(diag.get("ladder_bboxes", np.zeros(0)))
    ladder_n_pix = np.asarray(diag.get("ladder_n_pix", np.zeros(0))).astype(int)
    ladder_scales = np.asarray(diag.get("ladder_scales", np.zeros(0))).astype(float)
    n_ladder = int(member_ladder.shape[1])
    if n_ladder < 2 or ladder_bboxes.shape[0] != n_ladder:
        return

    n_eof_show = int(_scalar(diag["n_eof_show"]))
    n_pc = int(min(n_eof_show, scores.shape[1]))
    pcs = np.asarray(scores[:, :n_pc], dtype=np.float64)
    eofs = np.asarray(diag["pc_components_latlon"])
    impact_variable = str(_scalar(diag["impact_variable"]))
    ladder_target = np.asarray(member_ladder, dtype=np.float64)
    if is_geopotential(impact_variable):
        ladder_target = to_display_units(impact_variable, ladder_target)

    composites = []
    r2_in = np.empty(n_ladder, dtype=np.float64)
    r2_cv = np.empty(n_ladder, dtype=np.float64)
    for d in range(n_ladder):
        y_d = ladder_target[:, d]
        beta, intercept = _ols_fit(pcs, y_d)
        r2_in[d] = _r2_score(y_d, intercept + pcs @ beta)
        r2_cv[d] = _kfold_r2(pcs, y_d, n_folds=10)
        weights = _field_impact_regression_weights(pcs, y_d)
        composites.append(np.tensordot(weights, eofs[:n_pc, var_idx], axes=(0, 0)))
    composites = np.stack(composites, axis=0)  # [n_ladder, lat, lon]
    # Pattern stability: spatial correlation of each composite with the widest
    # (largest-area) box's composite -- a fixed pattern across box sizes
    # supports "the relationship is real, the small box just measured it
    # through noise"; a drifting one flags fragility.
    ref = (
        int(np.argmax(ladder_n_pix)) if ladder_n_pix.size == n_ladder else n_ladder - 1
    )
    pattern_corr = [
        float(np.corrcoef(composites[d].ravel(), composites[ref].ravel())[0, 1])
        for d in range(n_ladder)
    ]

    domain_lat = np.asarray(diag["domain_lat"])
    domain_lon = np.asarray(diag["domain_lon"])
    extent = (
        float(_scalar(diag["lon_min"])),
        float(_scalar(diag["lon_max"])),
        float(_scalar(diag["lat_min"])),
        float(_scalar(diag["lat_max"])),
    )
    case_name = str(_scalar(diag["case_name"]))
    impact_excluded = np.asarray(diag.get("impact_excluded_mask_box", np.zeros(0)))
    excluded = (
        impact_excluded if impact_excluded.size and impact_excluded.any() else None
    )
    field_u = var_units(var_name)
    impact_u = var_units(impact_variable)
    vr = _shared_color_range(list(composites), percentile=99.0)

    fig = plt.figure(figsize=(3.4 * n_ladder + 0.6, 8.4))
    gs = fig.add_gridspec(
        2,
        n_ladder,
        height_ratios=[1.25, 1.0],
        hspace=0.34,
        wspace=0.08,
        left=0.06,
        right=0.97,
        top=0.88,
        bottom=0.08,
    )
    map_axes = []
    mesh = None
    for d in range(n_ladder):
        ax = fig.add_subplot(gs[0, d], projection=ccrs.PlateCarree())
        mesh = _map_panel(
            ax,
            composites[d],
            domain_lat,
            domain_lon,
            vmin=-vr,
            vmax=vr,
            cmap=ANOM_CMAP,
            extent=extent,
            title=f"{ladder_scales[d]:.1f}× · n={int(ladder_n_pix[d])}",
            show_xlabels=True,
            show_ylabels=(d == 0),
            excluded_mask=excluded,
        )
        draw_domain_box(ax, tuple(float(v) for v in ladder_bboxes[d]))
        ax.text(
            0.97,
            0.04,
            f"pattern r = {pattern_corr[d]:+.2f}",
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=TICK_LABEL_SIZE_DENSE,
            bbox=dict(
                facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"
            ),
            zorder=6,
        )
        map_axes.append(ax)
    if mesh is not None:
        cb = fig.colorbar(
            mesh,
            ax=map_axes,
            orientation="horizontal",
            fraction=0.04,
            pad=0.07,
        )
        unit_lbl = f" ({field_u}/{impact_u})" if field_u and impact_u else ""
        cb.set_label(
            f"{symbol(var_name)} regression on {symbol(impact_variable)}{unit_lbl}",
            fontsize=COLORBAR_LABEL_SIZE_DENSE,
        )
        cb.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)

    # R^2 vs box size (effective sample size of the impact mean).
    ax_r = fig.add_subplot(gs[1, :])
    order = np.argsort(ladder_n_pix)
    x = ladder_n_pix[order]
    ax_r.plot(
        x,
        r2_in[order],
        "o-",
        color="tab:blue",
        label="$R^2$ in-sample",
        zorder=3,
    )
    ax_r.plot(
        x,
        r2_cv[order],
        "s--",
        color="tab:red",
        label="$R^2$ CV (10-fold)",
        zorder=3,
    )
    # Mark the headline box (scale 1.0) so the reader sees where the configured
    # domain sits on the noise-reduction curve.
    headline = np.flatnonzero(np.isclose(ladder_scales, 1.0))
    if headline.size:
        ax_r.axvline(
            float(ladder_n_pix[headline[0]]),
            color="black",
            linestyle=":",
            linewidth=1.0,
            alpha=0.6,
            zorder=2,
            label="headline box (1.0×)",
        )
    ax_r.set_xlabel(
        f"masked pixels in impact box (effective sample size of the {impact_variable} mean)",
        fontsize=AXIS_LABEL_SIZE_DENSE,
    )
    ax_r.set_ylabel(
        f"$R^2$ (precursor → {impact_variable})", fontsize=AXIS_LABEL_SIZE_DENSE
    )
    ax_r.tick_params(axis="both", labelsize=TICK_LABEL_SIZE_DENSE)
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=LEGEND_SIZE_DENSE, loc="best", framealpha=0.9)

    fig.suptitle(
        f"{case_name}\n"
        f"{_mode_label(mode).capitalize()} · {_free_end_role(mode)} → "
        f"{impact_variable} · domain-sensitivity ladder",
        fontsize=SUPTITLE_SIZE_DENSE,
        color="black",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"[aggregate_synoptic_pca] wrote {out_path}  "
        f"(R2_cv by box: {[round(float(v), 2) for v in r2_cv]})"
    )


# ---------------------------------------------------------------------------
# Diversity-metrics supplement: how much / how many real modes (start vs end)
# ---------------------------------------------------------------------------


def _participation_ratio(eigenvalues: np.ndarray) -> float:
    """Effective # of modes: (Σλ)² / Σλ²  (1 if one mode dominates, d if flat)."""
    e = np.asarray(eigenvalues, dtype=np.float64)
    denom = float((e**2).sum())
    return float((e.sum() ** 2) / denom) if denom > 0 else 0.0


def _spectral_perplexity(eigenvalues: np.ndarray) -> float:
    """Entropy-based effective # of modes: exp(-Σ pᵢ ln pᵢ), pᵢ = λᵢ/Σλ."""
    e = np.asarray(eigenvalues, dtype=np.float64)
    s = float(e.sum())
    if s <= 0:
        return 0.0
    p = e[e > 0] / s
    return float(np.exp(-(p * np.log(p)).sum()))


def _north_delta_lambda(eigenvalues: np.ndarray, n_members: int) -> np.ndarray:
    """North et al. (1982) eigenvalue sampling error δλᵢ ≈ λᵢ·√(2/N)."""
    return np.asarray(eigenvalues, dtype=np.float64) * np.sqrt(2.0 / float(n_members))


def _n_separable_modes(eigenvalues: np.ndarray, n_members: int) -> int:
    """Leading modes (contiguous from the top) that clear North's separation:
    ``λᵢ − λ_{i+1} ≥ δλᵢ``.  A proxy for "how many modes are statistically real."
    """
    e = np.asarray(eigenvalues, dtype=np.float64)
    dl = _north_delta_lambda(e, n_members)
    k = 0
    for i in range(e.size - 1):
        if (e[i] - e[i + 1]) >= dl[i]:
            k = i + 1
        else:
            break
    return k


def plot_diversity_metrics(per_mode: Dict[str, dict], out_path: Path) -> None:
    """Quantitative precursor-diversity backbone, start vs end overlaid.

    2x2: (a) physical free-end spread per variable, (b) effective # of modes
    (participation ratio + spectral perplexity), (c) cumulative EVR, and
    (d) the eigenvalue spectrum with North (1982) separation error bars +
    a count of statistically separable leading modes.  Answers "how much
    diversity, how many independent modes, and are they real."
    """
    modes = [m for m in MODE_ORDER if m in per_mode]
    if not modes:
        return
    any_diag = per_mode[modes[0]]
    case_name = str(_scalar(any_diag["case_name"]))
    variables = [str(v) for v in np.asarray(any_diag["variables"])]
    n_var = len(variables)
    n_spec = 15  # eigenvalues to show in the spectrum / cumulative panels

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))
    width = 0.8 / max(len(modes), 1)

    # (a) Physical free-end spread per variable (display units²).
    ax = axes[0, 0]
    x = np.arange(n_var)
    for j, m in enumerate(modes):
        pv = np.asarray(
            per_mode[m]["physical_variance_mean_per_var"], dtype=np.float64
        ).copy()
        for vi, v in enumerate(variables):
            # variance is a squared quantity, so scale by the display-unit
            # conversion factor squared (1 for non-geopotential variables).
            f = float(to_display_units(v, np.array(1.0))) if is_geopotential(v) else 1.0
            pv[vi] *= f * f
        ax.bar(
            x + j * width, pv, width, color=MODE_COLORS_TAB.get(m), label=_mode_label(m)
        )
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(
        [f"{v}\n({var_units(v)}²)" if var_units(v) else v for v in variables],
        fontsize=TICK_LABEL_SIZE_DENSE,
    )
    ax.set_ylabel("free-end variance (per-pixel mean)", fontsize=AXIS_LABEL_SIZE_DENSE)
    ax.set_title("Physical spread", fontsize=TITLE_SIZE_DENSE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE_DENSE)
    ax.legend(fontsize=LEGEND_SIZE_DENSE, framealpha=0.9)

    # (b) Effective number of modes.
    ax = axes[0, 1]
    labels = ["participation\nratio", "spectral\nperplexity"]
    xb = np.arange(len(labels))
    for j, m in enumerate(modes):
        eig = np.asarray(per_mode[m]["eigenvalues"], dtype=np.float64)
        vals = [_participation_ratio(eig), _spectral_perplexity(eig)]
        ax.bar(xb + j * width, vals, width, color=MODE_COLORS_TAB.get(m))
    ax.set_xticks(xb + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_SIZE_DENSE)
    ax.set_ylabel("effective # of modes", fontsize=AXIS_LABEL_SIZE_DENSE)
    ax.set_title("Effective dimensionality", fontsize=TITLE_SIZE_DENSE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE_DENSE)

    # (c) Cumulative explained variance.
    ax = axes[1, 0]
    for m in modes:
        evr = np.asarray(per_mode[m]["explained_variance_ratio"], dtype=np.float64)
        k = min(n_spec, evr.size)
        ax.plot(
            np.arange(1, k + 1),
            100.0 * np.cumsum(evr[:k]),
            marker="o",
            markersize=3,
            color=MODE_COLORS_TAB.get(m),
            label=_mode_label(m),
        )
    ax.set_xlabel("mode index", fontsize=AXIS_LABEL_SIZE_DENSE)
    ax.set_ylabel("cumulative EVR (%)", fontsize=AXIS_LABEL_SIZE_DENSE)
    ax.set_title("Cumulative explained variance", fontsize=TITLE_SIZE_DENSE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE_DENSE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE_DENSE, framealpha=0.9)

    # (d) Eigenvalue spectrum + North (1982) separation.
    ax = axes[1, 1]
    for m in modes:
        eig = np.asarray(per_mode[m]["eigenvalues"], dtype=np.float64)
        n_mem = int(_scalar(per_mode[m]["n_members"]))
        k = min(n_spec, eig.size)
        idx = np.arange(1, k + 1)
        ax.errorbar(
            idx,
            eig[:k],
            yerr=_north_delta_lambda(eig[:k], n_mem),
            marker="o",
            markersize=3,
            capsize=2,
            linewidth=1.0,
            color=MODE_COLORS_TAB.get(m),
            label=f"{_mode_label(m)} ({_n_separable_modes(eig, n_mem)} sep.)",
        )
    ax.set_yscale("log")
    ax.set_xlabel("mode index", fontsize=AXIS_LABEL_SIZE_DENSE)
    ax.set_ylabel("eigenvalue (S²)", fontsize=AXIS_LABEL_SIZE_DENSE)
    ax.set_title("Spectrum + North (1982) separation", fontsize=TITLE_SIZE_DENSE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE_DENSE)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=LEGEND_SIZE_DENSE, framealpha=0.9)

    add_subplot_labels(axes, fontsize=SUBPLOT_LABEL_SIZE_DENSE)
    fig.suptitle(
        f"{case_name} · precursor diversity (start vs end) · {', '.join(variables)}",
        fontsize=SUPTITLE_SIZE_DENSE,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[aggregate_synoptic_pca] wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# Free-end track/intensity targets, in the column order track_qc's
# free_end_targets provides them (lat, lon, distance, MSL), with display
# labels + units for the regression figure.
_TRACK_TARGETS = (
    ("latitude", 0, "free-end latitude", "°N"),
    ("longitude", 1, "free-end longitude", "°E"),
    ("distance", 2, "free-end distance from ERA5", "km"),
    ("MSL", 3, "free-end central MSL", "hPa"),
)

# Leading PCs the track regression uses.  Fixed at 8 to match the track-stats
# report's --n-pc default (and the locked round-2 numbers), independent of the
# case's n_eof_show (sandy 10 / ian 6): pc_scores always carries max_d columns.
_TRACK_REGRESSION_N_PC = 8


def plot_track_regression(diag: dict, mode: str, out_path: Path) -> None:
    """Predicted-vs-ensemble-member free-end track/intensity regression (TC cases).

    The TC twin of :func:`plot_precursor_impact`'s predicted-vs-ensemble-member panel.
    Resolves each member's storm of interest anchor-first from the raw all-paths
    tc_tracks parquet through ``track_qc`` -- the round-5 unified QC, the same
    definition ``synoptic_pca_track_stats`` and the paper use -- rather than the
    superseded path-0 targets baked into the frozen npz.  For each free-end
    target (latitude, longitude, great-circle distance from the ERA5 fix,
    central MSL) the leading-PC OLS prediction is scattered against the ensemble-member
    value over the *analyzable* members, with in-sample + 10-fold CV R^2 and an
    ERA5 marker.  Returns early for non-TC cases or when the parquet / track
    domain is unavailable.
    """
    impact_kind = str(_scalar(diag.get("impact_kind", np.array("none"))))
    if impact_kind != "track":
        return
    scores = np.asarray(diag.get("pc_scores", np.zeros(0)), dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0:
        return
    n_pc = int(min(_TRACK_REGRESSION_N_PC, scores.shape[1]))
    if n_pc < 1:
        return
    pcs = scores[:, :n_pc]

    # Round-5 unified QC: resolve the storm of interest anchor-first from the
    # raw parquet, a sibling of this figure's synoptic_pca dir at
    # <diagnostics>/tc_tracks/<tracker>/tc_tracks_<mode>.parquet.  free_end_targets
    # NaNs the non-analyzable members, so the regression runs over exactly the
    # analyzable population the report quotes.
    track_box = np.asarray(diag.get("tc_domain", np.zeros(0)), dtype=float).reshape(-1)
    if track_box.size != 4:
        return
    tracker = str(_scalar(diag.get("tc_tracker", np.array("tempest"))))
    parquet = tc_tracks_parquet_path(str(out_path.parent.parent), mode, tracker)
    if not Path(parquet).exists():
        print(
            f"[aggregate_synoptic_pca] track regression: {parquet} missing; "
            f"skipping mode '{mode}'."
        )
        return
    free_end_frame = int(_scalar(diag["free_end_frame"]))
    n_leads = int(np.asarray(diag["lead_hours"]).size)
    landfall_frame = int(_scalar(diag.get("landfall_frame", np.int32(-1))))
    anchor_frame = (
        0
        if mode == "start"
        else (landfall_frame if 0 <= landfall_frame < n_leads else n_leads - 1)
    )
    qc = compute_track_qc(
        parquet,
        track_box=(
            float(track_box[0]),
            float(track_box[1]),
            float(track_box[2]),
            float(track_box[3]),
        ),
        anchor_frame=anchor_frame,
        free_end_frame=free_end_frame,
        n_members=int(scores.shape[0]),
    )
    targets = qc.free_end_targets()  # analyzable-only (excluded members NaN)
    fe_anch = np.column_stack(
        [
            targets["fe_lat_deg"],
            targets["fe_lon_deg"],
            targets["fe_dist_era5_km"],
            targets["fe_msl_hpa"],
        ]
    ).astype(np.float64)
    era5_fe = np.array(
        [qc.era5_free_end["lat"], qc.era5_free_end["lon"], qc.era5_free_end["msl_hpa"]],
        dtype=np.float64,
    )
    era5_pc = np.asarray(diag.get("era5_pc_score", np.zeros(0)), dtype=np.float64)
    n_dropped = int(qc.counts()["excluded"])
    case_name = str(_scalar(diag["case_name"]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    flat_axes = list(axes.ravel())
    summary: List[str] = []
    for ax, (short, col, ylabel, unit) in zip(flat_axes, _TRACK_TARGETS):
        y = fe_anch[:, col]
        ok = np.isfinite(y) & np.isfinite(pcs).all(axis=1)
        n = int(ok.sum())
        # ERA5 observed: its own free-end fix (0 km for the distance target).
        era5_obs = (
            0.0
            if short == "distance"
            else (float(era5_fe[col]) if col < era5_fe.size else float("nan"))
        )
        if n <= n_pc + 1:
            ax.text(
                0.5,
                0.5,
                f"{ylabel}\n(insufficient analyzable members)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=TICK_LABEL_SIZE_DENSE,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        xx, yy = pcs[ok], y[ok]
        beta, intercept = _ols_fit(xx, yy)
        yhat = intercept + xx @ beta
        r2_in = _r2_score(yy, yhat)
        r2_cv = _kfold_r2(xx, yy, n_folds=10)
        p_value = _regression_f_pvalue(r2_in, n, n_pc)
        era5_yhat = (
            intercept + float(era5_pc[:n_pc] @ beta)
            if era5_pc.size >= n_pc
            else float("nan")
        )
        summary.append(
            f"{short} CV={r2_cv:.2f}" if np.isfinite(r2_cv) else f"{short} CV=n/a"
        )

        ax.scatter(
            yhat,
            yy,
            s=10,
            color=_SPAGHETTI_COLOR,
            alpha=0.35,
            edgecolors="none",
            zorder=1,
            label="analyzable members",
        )
        pts = [yhat, yy]
        if np.isfinite(era5_yhat):
            pts.append(np.array([era5_yhat]))
        if np.isfinite(era5_obs):
            pts.append(np.array([era5_obs]))
        allv = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in pts])
        lo, hi = float(allv.min()), float(allv.max())
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        span = np.array([lo - pad, hi + pad])
        ax.plot(
            span, span, color="black", linewidth=1.1, alpha=0.8, zorder=3, label="1:1"
        )
        if np.isfinite(era5_yhat) and np.isfinite(era5_obs):
            ax.scatter(
                era5_yhat,
                era5_obs,
                marker="*",
                s=220,
                color="black",
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
                label="ERA5",
            )
        ax.set_xlim(float(span[0]), float(span[1]))
        ax.set_ylim(float(span[0]), float(span[1]))
        ax.set_aspect("equal", adjustable="box")
        u = f" ({unit})" if unit else ""
        ax.set_xlabel(f"{n_pc}-PC prediction{u}", fontsize=AXIS_LABEL_SIZE_DENSE)
        ax.set_ylabel(f"Ensemble-member {ylabel}{u}", fontsize=AXIS_LABEL_SIZE_DENSE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE_DENSE)
        ax.grid(True, alpha=0.25)
        r2_in_txt = f"{r2_in:.2f}" if np.isfinite(r2_in) else "n/a"
        r2_cv_txt = f"{r2_cv:.2f}" if np.isfinite(r2_cv) else "n/a"
        if not np.isfinite(p_value):
            p_txt = "p = n/a"
        elif p_value < 1e-3:
            p_txt = "p < 0.001"
        else:
            p_txt = f"p = {p_value:.3f}"
        ax.text(
            0.97,
            0.03,
            f"N = {n}\n$R^2$ in = {r2_in_txt} ({p_txt})\n$R^2$ CV = {r2_cv_txt}",
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=TICK_LABEL_SIZE_DENSE,
            bbox=dict(
                facecolor="white", alpha=0.85, edgecolor="0.7", boxstyle="round,pad=0.3"
            ),
            zorder=6,
        )
    # ``add_subplot_labels(placement="inside")`` pins each label to its axes'
    # top-left corner, which is where a plain ``loc="upper left"`` legend also
    # lands -- so panel (a), the only panel carrying a legend, printed "a.)"
    # underneath the legend frame and lost it.  Nudge just this legend down by
    # one label's height; the corner stays empty of data in every case and
    # mode, so no scatter is covered.
    flat_axes[0].legend(
        fontsize=LEGEND_SIZE_DENSE,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.93),
        framealpha=0.9,
    )
    add_subplot_labels(flat_axes, placement="inside", fontsize=SUBPLOT_LABEL_SIZE_DENSE)
    fig.suptitle(
        f"{case_name}\n"
        f"{_mode_label(mode).capitalize()} · {_free_end_role(mode)} free-end "
        f"track vs {n_pc}-PC regression (analyzable; {n_dropped} excluded)",
        fontsize=SUPTITLE_SIZE_DENSE,
        color="black",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"[aggregate_synoptic_pca] wrote {out_path}  "
        f"({n_pc} PCs; {', '.join(summary)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--n-eof-figures",
        type=int,
        default=None,
        help="Number of leading EOFs to render a combined figure for, per "
        "mode (default: every EOF baked into the npz, i.e. n_eof_show). "
        "An explicit value is capped at the per-mode n_eof_show.",
    )
    parser.add_argument(
        "--pair-eofs",
        default=None,
        help="Comma-separated 1-based EOF numbers (e.g. '1,2') to render as "
        "a minimal loading + track-sorting figure, one row per EOF "
        "(synoptic_pca_eof_tracks_<mode>_eof<a>-<b>.png), one file per "
        "mode. A fast path: renders only these figures and exits, skipping "
        "the per-EOF figures, videos, and supplements.",
    )
    parser.add_argument(
        "--track-regression-only",
        action="store_true",
        help="Render only the predicted-vs-ensemble-member track-regression figure "
        "(synoptic_pca_track_regression_<mode>.png) per mode and exit -- a "
        "fast path for refreshing that panel after a track-QC change "
        "(plot_track_regression re-resolves the storm of interest from the "
        "parquet, so no npz rebuild is needed).",
    )
    args = parser.parse_args()
    if args.n_eof_figures is not None and args.n_eof_figures < 1:
        raise SystemExit(
            f"ERROR: --n-eof-figures must be >= 1 (got {args.n_eof_figures})"
        )
    pair_eofs: List[int] | None = None
    if args.pair_eofs is not None:
        try:
            pair_eofs = [
                int(tok) - 1 for tok in args.pair_eofs.split(",") if tok.strip()
            ]
        except ValueError:
            raise SystemExit(
                f"ERROR: --pair-eofs must be comma-separated 1-based EOF "
                f"numbers (got {args.pair_eofs!r})"
            )
        if not pair_eofs or any(ei < 0 for ei in pair_eofs):
            raise SystemExit(
                "ERROR: --pair-eofs needs 1-based EOF numbers (e.g. '1,2')"
            )

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"ERROR: --output-dir does not exist: {output_dir}")

    per_mode: Dict[str, dict] = {}
    for mode in MODE_ORDER:
        d = _load_mode(output_dir, mode)
        if d is not None:
            per_mode[mode] = d
    if not per_mode:
        raise SystemExit(f"ERROR: no per-mode npz files found in {output_dir}")
    print(f"[aggregate_synoptic_pca] modes: {list(per_mode)}")

    any_diag = per_mode[next(iter(per_mode))]
    variables: List[str] = [str(v) for v in np.asarray(any_diag["variables"])]
    print(f"[aggregate_synoptic_pca] variables: {variables}")

    # The combined figure renders one precursor variable -- z500 if present
    # (the headline steering/height field), else the first PCA variable.
    precursor_var_idx = variables.index("z500") if "z500" in variables else 0
    precursor_var = variables[precursor_var_idx]

    if pair_eofs is not None or args.track_regression_only:
        if pair_eofs is not None:
            tag = "-".join(str(ei + 1) for ei in pair_eofs)
            for mode, diag in per_mode.items():
                plot_eof_track_pair(
                    diag,
                    precursor_var_idx,
                    precursor_var,
                    mode,
                    pair_eofs,
                    output_dir / f"synoptic_pca_eof_tracks_{mode}_eof{tag}.png",
                )
        if args.track_regression_only:
            for mode, diag in per_mode.items():
                plot_track_regression(
                    diag,
                    mode,
                    output_dir / f"synoptic_pca_track_regression_{mode}.png",
                )
        return

    for mode, diag in per_mode.items():
        n_eof_show = int(_scalar(diag["n_eof_show"]))
        # Default (None) renders one combined figure per EOF baked into the
        # npz (n_eof_show); an explicit --n-eof-figures caps that count.
        n_eof = (
            n_eof_show
            if args.n_eof_figures is None
            else min(args.n_eof_figures, n_eof_show)
        )
        for ei in range(n_eof):
            plot_combined(
                diag,
                precursor_var_idx,
                precursor_var,
                mode,
                ei,
                output_dir / f"synoptic_pca_combined_{mode}_eof{ei + 1}.png",
            )
            render_member_videos(
                diag,
                mode,
                ei,
                output_dir / f"synoptic_pca_video_{mode}_eof{ei + 1}",
            )
        plot_pc_scatter(diag, mode, output_dir / f"synoptic_pca_pc_scatter_{mode}.png")
        plot_precursor_impact(
            diag,
            precursor_var_idx,
            precursor_var,
            mode,
            output_dir / f"synoptic_pca_precursor_impact_{mode}.png",
        )
        plot_domain_ladder(
            diag,
            precursor_var_idx,
            precursor_var,
            mode,
            output_dir / f"synoptic_pca_domain_ladder_{mode}.png",
        )
        plot_track_regression(
            diag,
            mode,
            output_dir / f"synoptic_pca_track_regression_{mode}.png",
        )

    for vi, vname in enumerate(variables):
        plot_eofs(per_mode, vi, vname, output_dir / f"synoptic_pca_eofs_{vname}.png")

    plot_diversity_metrics(per_mode, output_dir / "synoptic_pca_diversity.png")


if __name__ == "__main__":
    main()
