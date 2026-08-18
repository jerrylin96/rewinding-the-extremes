"""End-conditioned landfall-split TC diagnostic (Hurricane Ian).

A single-case, single-mode companion to ``aggregate_tc_tracks.py`` that
exists to show the one thing Ian's ``tc_tracks.png`` cannot say without
just repeating the Sandy figure: under ``end``-only conditioning the
ensemble is anchored to a *weak* landfall (Ian's Cat-1 South Carolina
landfall, frame 11) while the initial condition is left free, so the
backward trajectories reveal how many members would have driven a far
stronger impact through Florida on the way there.

Reads the ``end``-mode ``tc_tracks_end.parquet`` written upstream by
``plot_ensemble_tc_tracks.py`` and renders a 2-row figure:

``track_qc`` resolves each member's storm of interest anchor-first (the
tracker path reaching the ERA5 SC-landfall fix at ``--landfall-frame``)
and marks a member analyzable when that path has >=3 valid fixes in the
track detection box.  The split is over the analyzable population, on the
resolved path:

    a.) members that make landfall in Florida (a resolved-path fix in the
        Florida box), and
    b.) members that do not (bypass).  Because analyzability already
        requires reaching the end-conditioned SC-landfall anchor, every
        bypass member is anchored toward South Carolina.

The excluded members (no_track / unanchored / insufficient_fixes) form
the inspect set -- the ones outside the clean FL/SC story, logged so they
can be eyeballed with animate_ensemble.py.

Every track is colored by its central MSL at the *initial condition* —
the free (unconstrained) end under end-conditioning — on a colorbar
shared across both panels, so the contrast is literally "how strong was
the storm the model freely chose to start from, given a weak SC
landfall."  The free-end fix falls back to the nearest valid frame when
the resolved path has no IC-frame fix (flagged).  The Florida box (which
defines the split) and the South Carolina box (illustrating the
end-conditioning landfall region) are drawn on both panels in the two
group colors, and ERA5 (which did make Florida landfall) is overlaid on
the Florida panel.

Row 2 — the central-MSL-vs-local-time trajectory (the same data
``aggregate_tc_tracks.py`` plots) for the same two groups, but with one
color per group plus a per-group mean line and the ERA5 reference, so the
deep Florida-landfall intensities at the intermediate frames read against
the shallow bypass trajectories that converge to the same weak SC
landfall at the free-fixed end.

Outputs (in --output-dir):
    tc_landfall_split.png

Usage:
    python tc_landfall_split.py \\
        --output-dir /path/to/diagnostics/tc_tracks/<tracker> \\
        --lon-min -90 --lon-max -70 --lat-min 20 --lat-max 36 \\
        --fl-lon-min -83.5 --fl-lon-max -80.0 --fl-lat-min 24.5 --fl-lat-max 30.0 \\
        --sc-lon-min -81.5 --sc-lon-max -78.5 --sc-lat-min 32.0 --sc-lat-max 34.5 \\
        --fl-label Florida --sc-label "South Carolina" \\
        --landfall-frame 11 \\
        --title "Hurricane Ian — End-Conditioned Landfall Split" \\
        --start-time 2022-09-28T00:00:00Z \\
        --timezone America/New_York
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "_shared"))

from _dispatch_lib import wrap_lon_0_360  # noqa: E402
from local_time_axis import format_local_time_axis  # noqa: E402
from plot_style import (  # noqa: E402
    AXIS_LABEL_SIZE,
    COLORBAR_LABEL_SIZE,
    COLORBAR_TICK_SIZE,
    LEGEND_SIZE,
    SUPTITLE_SIZE,
    TICK_LABEL_SIZE,
    TITLE_SIZE,
    add_subplot_labels,
    draw_domain_box,
    format_bbox,
)
from tc_tracks_io import (  # noqa: E402
    load_tracks_parquet,
    tracks_in_domain,
)
from track_qc import (  # noqa: E402
    ANCHOR_RADIUS_KM,
    TrackQC,
    compute_track_qc_arrays,
)

# The resolved storm-of-interest path is placed at index 0 by
# ``TrackQC.storm_of_interest_array`` (see track_qc.py); a track needs at least
# this many valid fixes to be drawn (analyzable members already clear it).
PRIMARY_PATH_ID = 0
MIN_PATH_LENGTH = 3

# Categorical group colors.  Florida = warm (greater impact), bypass-to-SC
# = cool.  Chosen distinct from the sequential intensity colormap below so
# the row-2 group lines never read as an intensity value, and reused as the
# row-1 classification-box outline colors so "red box / red group" and
# "blue box / blue group" tie the two rows together.
GROUP_FL_COLOR = "#C44E52"
GROUP_BYPASS_COLOR = "#4C72B0"

GROUP_LABELS_FALLBACK = ("Florida", "South Carolina")

# Sequential colormap for the per-track IC (free-end) central MSL.  viridis
# maps deep pressure (strong) to dark purple — high contrast on the pale
# land/ocean basemap so the strongest free-end storms stand out — and weak
# (high MSL) to yellow, which appropriately recedes.
INTENSITY_CMAP = plt.cm.viridis

# Map-panel member-track style.
MAP_TRACK_LW = 0.9
MAP_TRACK_ALPHA = 0.5
FREE_END_MS = 4.0

# Row-2 spaghetti style.  Alpha is set per group at draw time (see
# _spaghetti_alpha) so the smaller bypass group is not buried under the
# much larger Florida group; only the linewidth is fixed here.
SPAGHETTI_LW = 0.5
GROUP_MEAN_LW = 2.6

# Default cBottle output cadence; Ian uses the standard 6 h grid.
DEFAULT_TIMESTEP_HOURS = 6.0


def _first_valid_fix(lats: np.ndarray, lons: np.ndarray) -> int | None:
    """Index of the earliest step with a finite (lat, lon); ``None`` if none."""
    valid = np.flatnonzero(np.isfinite(lats) & np.isfinite(lons))
    return int(valid[0]) if valid.size else None


def _basemap(ax, extent: tuple[float, float, float, float]) -> None:
    """Draw the shared land/ocean/coastline basemap + gridlines on ``ax``.

    Lifted verbatim from aggregate_tc_tracks.py so the landfall-split map
    panels are visually identical to the per-case tc_tracks.png panels.
    """
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="#555555")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.4, color="#888888")
    ax.add_feature(cfeature.STATES, linewidth=0.2, alpha=0.3, color="#aaaaaa")
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.4,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 11}
    gl.ylabel_style = {"size": 11}
    ax.set_extent(list(extent), crs=ccrs.PlateCarree())


def _draw_classification_boxes(
    ax,
    fl_extent: tuple[float, float, float, float],
    sc_extent: tuple[float, float, float, float],
) -> None:
    """Outline the FL / SC landfall boxes in the two group colors with labels.

    Box longitudes are wrapped to 0-360 to match the map's set_extent
    convention (mirrors the domain-box drawing in aggregate_tc_tracks.py);
    classification upstream uses the raw boxes, which is wrap-safe.
    """
    for extent, color, tag in (
        (fl_extent, GROUP_FL_COLOR, "FL"),
        (sc_extent, GROUP_BYPASS_COLOR, "SC"),
    ):
        lon_min = wrap_lon_0_360(extent[0])
        lon_max = wrap_lon_0_360(extent[1])
        lat_min, lat_max = extent[2], extent[3]
        draw_domain_box(
            ax,
            [lon_min, lon_max, lat_min, lat_max],
            edgecolor=color,
            linewidth=1.6,
            zorder=6,
        )
        ax.text(
            0.5 * (lon_min + lon_max),
            lat_max,
            tag,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="bottom",
            fontsize=max(9, LEGEND_SIZE - 1),
            fontweight="bold",
            color=color,
            zorder=7,
            clip_on=False,
        )


def _draw_era5(ax, era5_tracks: np.ndarray) -> None:
    """Overlay the ERA5 reference track(s) as a dashed black line on a map."""
    for path_id in range(era5_tracks.shape[1]):
        era5_lats = era5_tracks[0, path_id, :, 0]
        era5_lons = era5_tracks[0, path_id, :, 1]
        valid = ~np.isnan(era5_lats) & ~np.isnan(era5_lons)
        if valid.sum() < MIN_PATH_LENGTH:
            continue
        label = "ERA5" if path_id == 0 else None
        ax.plot(
            era5_lons[valid],
            era5_lats[valid],
            color="black",
            linewidth=2.0,
            linestyle="--",
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            zorder=5,
            label=label,
        )
        ax.plot(
            era5_lons[valid],
            era5_lats[valid],
            "D",
            color="black",
            markersize=4,
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )


def _draw_landfall_map_panel(
    ax,
    ens: np.ndarray,
    member_indices: np.ndarray,
    ic_msl_hpa: np.ndarray,
    *,
    norm: Normalize,
    extent: tuple[float, float, float, float],
    fl_extent: tuple[float, float, float, float],
    sc_extent: tuple[float, float, float, float],
    era5_tracks: np.ndarray | None,
    title: str,
) -> int:
    """Render one landfall-group map panel; return the number of tracks drawn.

    Each member's primary track is drawn in a single color taken from
    ``INTENSITY_CMAP(norm(ic_msl_hpa))`` — its free-end (IC) central MSL —
    with a dot marking that free end.  Members whose IC intensity is NaN
    (no fix anywhere) are drawn in neutral gray.
    """
    _basemap(ax, extent)
    _draw_classification_boxes(ax, fl_extent, sc_extent)

    n_plotted = 0
    for j, m in enumerate(member_indices):
        tc_lats = ens[m, PRIMARY_PATH_ID, :, 0]
        tc_lons = ens[m, PRIMARY_PATH_ID, :, 1]
        valid = ~np.isnan(tc_lats) & ~np.isnan(tc_lons)
        if valid.sum() < MIN_PATH_LENGTH:
            continue
        msl = ic_msl_hpa[j]
        color = INTENSITY_CMAP(norm(msl)) if np.isfinite(msl) else "0.6"
        ax.plot(
            tc_lons[valid],
            tc_lats[valid],
            color=color,
            linewidth=MAP_TRACK_LW,
            alpha=MAP_TRACK_ALPHA,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        fe = _first_valid_fix(tc_lats, tc_lons)
        if fe is not None:
            ax.plot(
                tc_lons[fe],
                tc_lats[fe],
                "o",
                color=color,
                markersize=FREE_END_MS,
                alpha=0.8,
                markeredgecolor="black",
                markeredgewidth=0.3,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
        n_plotted += 1

    if era5_tracks is not None:
        _draw_era5(ax, era5_tracks)

    if n_plotted == 0:
        ax.text(
            0.5,
            0.5,
            "no members",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="#888888",
        )

    # cartopy's gridliner + constrained_layout silently drops ax.set_title on
    # a GeoAxes (the title artist exists but never renders), so place the
    # panel title as axes-fraction text, which always shows.
    ax.text(
        0.5,
        1.04,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    if era5_tracks is not None:
        ax.legend(loc="lower left", fontsize=LEGEND_SIZE, framealpha=0.8)
    return n_plotted


def _spaghetti_alpha(n: int) -> float:
    """Per-group spaghetti alpha that scales like ``1/sqrt(n)``, clamped.

    A fixed alpha buries a small group under a much larger one (Ian: ~110
    bypass members vs ~880 Florida).  Scaling by ``1/sqrt(n)`` gives the two
    groups comparable total ink so the bypass spaghetti stays legible.
    """
    if n <= 0:
        return 0.0
    return float(np.clip(2.2 / np.sqrt(n), 0.03, 0.30))


def _draw_group_trajectory(
    ax,
    ens: np.ndarray,
    fl_idx: np.ndarray,
    bypass_idx: np.ndarray,
    era5_tracks: np.ndarray | None,
    *,
    lead_hours: np.ndarray,
    start_time_iso: str,
    timezone_name: str,
    fl_label: str,
    sc_label: str,
    fl_pct: float,
    bypass_pct: float,
) -> None:
    """Draw the two-group central-MSL trajectory row.

    Per-group spaghetti in the group color (with a ``1/sqrt(n)`` alpha and
    the smaller group drawn on top so it is not buried), an opaque per-group
    mean line on top (nan-aware across that group's members at each step),
    and the ERA5 reference as a dashed black line.  Y-axis is
    low-to-high MSL (not inverted), matching aggregate_tc_tracks.py.  The
    leftmost step is the free (IC) end the row-1 colors key off; the
    rightmost is the anchored end (≈ the SC landfall the ensemble is
    conditioned on).  Legend labels carry each group's share of the
    ensemble.
    """
    msl_hpa = ens[:, PRIMARY_PATH_ID, :, 2] / 100.0

    # Spaghetti: draw the larger group first (lower) and the smaller group on
    # top, each with its own 1/sqrt(n) alpha, so the bypass group is legible.
    spaghetti_groups = sorted(
        ((fl_idx, GROUP_FL_COLOR), (bypass_idx, GROUP_BYPASS_COLOR)),
        key=lambda g: -g[0].size,
    )
    for z, (idx, color) in enumerate(spaghetti_groups, start=1):
        alpha = _spaghetti_alpha(idx.size)
        for m in idx:
            finite = np.isfinite(msl_hpa[m])
            if finite.sum() < 2:
                continue
            ax.plot(
                lead_hours,
                msl_hpa[m],
                color=color,
                alpha=alpha,
                linewidth=SPAGHETTI_LW,
                zorder=z,
            )

    # Per-group mean (nanmean so members missing a fix at a step don't pull
    # the mean to NaN).  Drawn opaque and on top (high zorder) of the faded
    # spaghetti, so it stays legible without an outline.
    for idx, color, z in (
        (fl_idx, GROUP_FL_COLOR, 5.0),
        (bypass_idx, GROUP_BYPASS_COLOR, 6.0),
    ):
        if idx.size == 0:
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            grp_mean = np.nanmean(msl_hpa[idx], axis=0)
        finite_mean = np.isfinite(grp_mean)
        if finite_mean.any():
            ax.plot(
                lead_hours[finite_mean],
                grp_mean[finite_mean],
                color=color,
                linewidth=GROUP_MEAN_LW,
                zorder=z,
            )

    if era5_tracks is not None and PRIMARY_PATH_ID < era5_tracks.shape[1]:
        era5_msl_hpa = era5_tracks[0, PRIMARY_PATH_ID, :, 2] / 100.0
        finite_era5 = np.isfinite(era5_msl_hpa)
        if finite_era5.sum() >= 2:
            ax.plot(
                lead_hours[finite_era5],
                era5_msl_hpa[finite_era5],
                color="black",
                linewidth=2.4,
                linestyle="--",
                zorder=7,
            )

    format_local_time_axis(ax, lead_hours, start_time_iso, timezone_name)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel("Storm central MSLP (hPa)", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(float(lead_hours[0]), float(lead_hours[-1]))
    ax.grid(True, alpha=0.3)
    ax.set_title(
        "End Conditioning — Storm central MSLP trajectory",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )

    handles = [
        Line2D([0], [0], color=GROUP_FL_COLOR, linewidth=GROUP_MEAN_LW),
        Line2D([0], [0], color=GROUP_BYPASS_COLOR, linewidth=GROUP_MEAN_LW),
        Line2D([0], [0], color="black", linewidth=2.4, linestyle="--"),
    ]
    labels = [
        f"{fl_label} landfall ({fl_pct:.1f}%)",
        f"Bypass {fl_label} → {sc_label} ({bypass_pct:.1f}%)",
        "ERA5",
    ]
    # Lower-right is the emptiest quadrant: members converge to the weak,
    # anchored SC-landfall MSL at the final (right) lead, so nothing sits at
    # low MSL / late time.
    ax.legend(handles, labels, loc="lower right", fontsize=LEGEND_SIZE, framealpha=0.9)


def _intensity_norm(
    ens: np.ndarray,
    fl_idx: np.ndarray,
    bypass_idx: np.ndarray,
    fl_ic: np.ndarray,
    bypass_ic: np.ndarray,
    era5_tracks: np.ndarray | None,
) -> Normalize:
    """Shared IC-MSL color normalization across both panels (+ ERA5)."""
    samples = [fl_ic[np.isfinite(fl_ic)], bypass_ic[np.isfinite(bypass_ic)]]
    if era5_tracks is not None and PRIMARY_PATH_ID < era5_tracks.shape[1]:
        era5_ic = era5_tracks[0, PRIMARY_PATH_ID, :, 2] / 100.0
        valid = era5_ic[np.isfinite(era5_ic)]
        if valid.size:
            samples.append(valid[:1])
    pooled = (
        np.concatenate([s for s in samples if s.size])
        if any(s.size for s in samples)
        else np.array([])
    )
    if pooled.size == 0:
        return Normalize(vmin=950.0, vmax=1010.0)
    vmin = float(np.nanmin(pooled))
    vmax = float(np.nanmax(pooled))
    if vmin == vmax:
        vmin, vmax = vmin - 5.0, vmax + 5.0
    return Normalize(vmin=vmin, vmax=vmax)


def make_figure(
    ens: np.ndarray,
    era5_tracks: np.ndarray | None,
    qc: TrackQC,
    *,
    extent: tuple[float, float, float, float],
    fl_extent: tuple[float, float, float, float],
    sc_extent: tuple[float, float, float, float],
    fl_label: str,
    sc_label: str,
    out_path: Path,
    title: str,
    start_time_iso: str,
    timezone_name: str,
    timestep_hours: float,
) -> None:
    """Render the 2-row end-conditioned landfall-split figure.

    ``ens`` is the anchor-resolved ``[n_members, 1, n_steps, 4]`` array (index 0
    is each member's storm of interest, from ``track_qc``); ``qc`` is the QC
    table that resolved it.  ``extent`` / ``fl_extent`` / ``sc_extent`` are raw
    lon/lat boxes (lon may be negative); the panels wrap longitude to 0-360 for
    cartopy internally.
    """
    # cartopy's set_extent wants 0-360 for a W-hemisphere view; keep the raw
    # extent for the (negative-lon) animate_ensemble.py hint below.
    map_extent = (
        wrap_lon_0_360(extent[0]),
        wrap_lon_0_360(extent[1]),
        extent[2],
        extent[3],
    )
    # Split the analyzable population on the resolved storm-of-interest path:
    # Florida = a fix in fl_box, bypass = analyzable but no fl_box fix.  Because
    # analyzable already requires reaching the end-conditioned SC-landfall
    # anchor, every bypass member is anchored toward SC.  The inspect set is the
    # excluded members (no_track / unanchored / insufficient_fixes) -- the ones
    # outside the clean FL/SC story, worth eyeballing with animate_ensemble.py.
    analyzable = qc.analyzable
    hit_fl = tracks_in_domain(ens, *fl_extent)[:, PRIMARY_PATH_ID]
    fl_idx = np.flatnonzero(analyzable & hit_fl)
    bypass_idx = np.flatnonzero(analyzable & ~hit_fl)
    inspect_idx = np.flatnonzero(~analyzable)

    fe_msl = qc.table["fe_msl_hpa"].to_numpy(dtype=float)
    fe_fallback = qc.fe_fallback
    fl_ic, bypass_ic = fe_msl[fl_idx], fe_msl[bypass_idx]
    fl_fb = int(fe_fallback[fl_idx].sum())
    bypass_fb = int(fe_fallback[bypass_idx].sum())

    qc_counts = qc.counts()
    total = qc_counts["total"]
    n_fl, n_bypass, n_excl = fl_idx.size, bypass_idx.size, inspect_idx.size

    def _pct(n: int) -> float:
        return (100.0 * n / total) if total else 0.0

    # Florida / not-Florida split over the analyzable population (fl + bypass +
    # excluded == total).
    print(
        f"[landfall_split] of {total} end-conditioned members: "
        f"{fl_label}={n_fl} ({_pct(n_fl):.1f}%), "
        f"bypass (not {fl_label})={n_bypass} ({_pct(n_bypass):.1f}%), "
        f"excluded={n_excl} ({_pct(n_excl):.1f}%)"
    )
    print(
        f"  excluded taxonomy: no_track={qc_counts['no_track']} / "
        f"unanchored={qc_counts['unanchored']} / "
        f"insufficient_fixes={qc_counts['insufficient_fixes']}"
    )
    print(
        f"  free-end fix fallbacks (nearest valid frame): "
        f"{fl_label}={fl_fb}, bypass={bypass_fb}"
    )
    # Inspect set = the excluded members.  Surface the indices + a ready-to-run
    # animate_ensemble.py command so the MSL field evolution can be eyeballed
    # (several have little or no usable track).
    print(f"  inspect set (excluded members): {n_excl} members")
    if inspect_idx.size:
        members_arg = " ".join(str(m) for m in inspect_idx.tolist())
        print(
            "    inspect with: python animate_ensemble.py --ensemble-zarr "
            "<end-mode ensemble zarr> --variable msl --members "
            f"{members_arg} --extent {extent[0]} {extent[1]} {extent[2]} "
            f"{extent[3]} --output-dir <dir>"
        )

    norm = _intensity_norm(ens, fl_idx, bypass_idx, fl_ic, bypass_ic, era5_tracks)

    fig = plt.figure(figsize=(14.0, 10.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[6.0, 3.2])

    proj = ccrs.PlateCarree()
    ax_fl = fig.add_subplot(gs[0, 0], projection=proj)
    ax_by = fig.add_subplot(gs[0, 1], projection=proj)

    n_fl_drawn = _draw_landfall_map_panel(
        ax_fl,
        ens,
        fl_idx,
        fl_ic,
        norm=norm,
        extent=map_extent,
        fl_extent=fl_extent,
        sc_extent=sc_extent,
        era5_tracks=era5_tracks,
        title=f"{fl_label} landfall",
    )
    n_by_drawn = _draw_landfall_map_panel(
        ax_by,
        ens,
        bypass_idx,
        bypass_ic,
        norm=norm,
        extent=map_extent,
        fl_extent=fl_extent,
        sc_extent=sc_extent,
        era5_tracks=None,  # ERA5 made FL landfall — it is the FL-panel reference.
        title=f"Bypass {fl_label} → {sc_label} landfall",
    )
    print(f"  [map] drew {n_fl_drawn} {fl_label} + {n_by_drawn} bypass member tracks")

    # Shared IC-intensity colorbar to the right of the two map panels.
    sm = ScalarMappable(norm=norm, cmap=INTENSITY_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_fl, ax_by], location="right", shrink=0.85, pad=0.02)
    cbar.set_label(
        "Storm central MSLP at free end / IC (hPa)", fontsize=COLORBAR_LABEL_SIZE
    )
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    n_steps = ens.shape[2]
    lead_hours = np.arange(n_steps, dtype=float) * timestep_hours
    ax_traj = fig.add_subplot(gs[1, :])
    _draw_group_trajectory(
        ax_traj,
        ens,
        fl_idx,
        bypass_idx,
        era5_tracks,
        lead_hours=lead_hours,
        start_time_iso=start_time_iso,
        timezone_name=timezone_name,
        fl_label=fl_label,
        sc_label=sc_label,
        fl_pct=_pct(n_fl),
        bypass_pct=_pct(n_bypass),
    )

    add_subplot_labels([ax_fl, ax_by, ax_traj], placement="inside")
    fig.suptitle(title, fontsize=SUPTITLE_SIZE, fontweight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Tracker dir holding tc_tracks_end.parquet; the figure is "
        "written here as tc_landfall_split.png.",
    )
    # Track detection box (map extent + path-0 domain restriction).
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    # Florida landfall classification box.
    parser.add_argument("--fl-lon-min", type=float, required=True)
    parser.add_argument("--fl-lon-max", type=float, required=True)
    parser.add_argument("--fl-lat-min", type=float, required=True)
    parser.add_argument("--fl-lat-max", type=float, required=True)
    # South Carolina landfall classification box.
    parser.add_argument("--sc-lon-min", type=float, required=True)
    parser.add_argument("--sc-lon-max", type=float, required=True)
    parser.add_argument("--sc-lat-min", type=float, required=True)
    parser.add_argument("--sc-lat-max", type=float, required=True)
    parser.add_argument("--fl-label", default=GROUP_LABELS_FALLBACK[0])
    parser.add_argument("--sc-label", default=GROUP_LABELS_FALLBACK[1])
    parser.add_argument("--title", default="End-Conditioned Landfall Split")
    parser.add_argument(
        "--start-time",
        type=str,
        default="1970-01-01T00:00:00Z",
        help="Ensemble IC as ISO 8601 UTC (drives the local wall-clock x-axis).",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="IANA timezone for the lead-axis labels (e.g. America/New_York).",
    )
    parser.add_argument(
        "--timestep-hours",
        type=float,
        default=DEFAULT_TIMESTEP_HOURS,
        help=f"Lead-time step in hours (default: {DEFAULT_TIMESTEP_HOURS}).",
    )
    parser.add_argument(
        "--landfall-frame",
        type=int,
        required=True,
        help="Pinned-end anchor frame for the end-conditioned split (the case "
        "YAML's tc_tracks.landfall_frame; the SC landfall frame for Ian).  A "
        "member is analyzable when its anchor-resolved storm path has >=3 "
        "in-domain fixes.",
    )
    parser.add_argument(
        "--anchor-radius-km",
        type=float,
        default=ANCHOR_RADIUS_KM,
        help="Great-circle radius (km) within which a member's path must "
        f"approach the ERA5 landfall fix to reach the anchor (default "
        f"{ANCHOR_RADIUS_KM:g}).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"ERROR: --output-dir does not exist: {output_dir}")

    parquet = output_dir / "tc_tracks_end.parquet"
    if not parquet.exists():
        raise SystemExit(
            f"ERROR: {parquet} not found.  Run dispatch_tc_tracks.py "
            "(end mode) for this case first."
        )

    extent = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    fl_extent = (args.fl_lon_min, args.fl_lon_max, args.fl_lat_min, args.fl_lat_max)
    sc_extent = (args.sc_lon_min, args.sc_lon_max, args.sc_lat_min, args.sc_lat_max)

    tracks = load_tracks_parquet(str(parquet))
    ens = tracks.get("ensemble")
    if ens is None:
        raise SystemExit(f"ERROR: no ensemble rows in {parquet}")
    era5 = tracks.get("era5")

    # Unified QC: resolve each member's storm of interest anchor-first (end
    # mode -> the landfall frame is the anchor, frame 0 the free end), then
    # reduce the raw all-paths array to that one path for drawing.
    qc = compute_track_qc_arrays(
        ens,
        era5,
        track_box=extent,
        anchor_frame=args.landfall_frame,
        free_end_frame=0,
        anchor_radius_km=args.anchor_radius_km,
    )
    counts = qc.counts()
    print(
        f"[landfall_split] track domain {format_bbox(list(extent))}, anchor "
        f"frame {args.landfall_frame}: {counts['analyzable']}/{counts['total']} "
        f"members analyzable ({counts['excluded']} excluded)"
    )
    ens = qc.storm_of_interest_array(ens)
    # ERA5's storm of interest is its path 0 (the fix that defines the anchor).
    era5 = era5[:, 0:1] if era5 is not None else None

    make_figure(
        ens,
        era5,
        qc,
        extent=extent,
        fl_extent=fl_extent,
        sc_extent=sc_extent,
        fl_label=args.fl_label,
        sc_label=args.sc_label,
        out_path=output_dir / "tc_landfall_split.png",
        title=args.title,
        start_time_iso=args.start_time,
        timezone_name=args.timezone,
        timestep_hours=args.timestep_hours,
    )


if __name__ == "__main__":
    main()
