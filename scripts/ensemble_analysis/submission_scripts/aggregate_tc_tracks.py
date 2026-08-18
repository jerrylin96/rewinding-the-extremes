"""Aggregate ensemble TC tracks across conditioning modes.

Reads the three per-mode ``tc_tracks_<mode>.parquet`` files (written by
``plot_ensemble_tc_tracks.py``) and renders one multi-row figure per case.
``track_qc`` resolves each member's storm of interest anchor-first (the
tracker path reaching the ERA5 fix at the pinned end -- the landfall frame
under end-conditioning, frame 0 otherwise) and reduces the raw all-paths
array to that one path; the maps and trajectories draw the resolved storm.

Row 1 is the 3-panel side-by-side track map: one panel per available
conditioning mode (end / start / both), each member's resolved storm
track in the mode-specific color, the per-step ensemble-mean track in
crimson, and the ERA5 reference overlaid.

Rows 2, 3 and 4 plot the resolved storm's ``tc_msl`` trajectory — the
same data the intensity-CI analysis bootstraps — under end-mode (row 2),
start-mode (row 3) and both-end-mode (row 4) conditioning.  Each is a
full-width panel showing every member as a gray spaghetti, the ensemble
mean as a solid black line, ERA5 as a dashed black line, and the member
whose trajectory has the lowest MSE against ERA5 (over all 12 lead
steps, restricted to members with a fix at every step) highlighted in
gold.  A shaded 5–95% band shows the member envelope at each step, and
on the end- and start-conditioned rows a whisker + star marks where ERA5
falls within that envelope at the *free* (unconditioned) end — lead 0
under end-conditioning, the final lead under start-conditioning — labelled
with track_qc's exact-frame ``ERA5 pN`` percentile.  The x-axis is the
local wall-clock keyed to the case's IANA timezone.

Outputs (in --output-dir):
    tc_tracks.png   - multi-row case figure (track maps + tc_msl trajectories)

Usage:
    python aggregate_tc_tracks.py \\
        --output-dir /path/to/diagnostics/tc_tracks/<tracker> \\
        --lon-min -105 --lon-max -60 --lat-min 10 --lat-max 40 \\
        --title "Hurricane Sandy Storm Tracks" \\
        --start-time 2012-10-27T06:00:00Z \\
        --timezone America/New_York
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "_shared"))

from _dispatch_lib import wrap_lon_0_360  # noqa: E402
from local_time_axis import format_local_time_axis  # noqa: E402
from plot_style import (  # noqa: E402
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    SUPTITLE_SIZE,
    TICK_LABEL_SIZE,
    TITLE_SIZE,
    add_subplot_labels,
    draw_domain_box,
)
from tc_tracks_io import (  # noqa: E402
    ensemble_mean_track,
    load_tracks_parquet,
)
from track_qc import ANCHOR_RADIUS_KM, compute_track_qc_arrays  # noqa: E402

MODE_LABELS = {
    "start": "Start Conditioning",
    "end": "End Conditioning",
    "both": "Both-End Conditioning",
}

MEMBER_TRACK_CMAP = plt.cm.jet

MIN_PATH_LENGTH = 3

# Index of the resolved storm-of-interest path after track_qc reduces each
# member to one path (TrackQC.storm_of_interest_array puts it at index 0).
PRIMARY_PATH_ID = 0

# Free-end frame per conditioning mode (the unconditioned end): the IC (frame
# 0) under end-conditioning, the last lead under start-conditioning.  Both-end
# conditioning pins both ends -- resolve against the IC anchor, no free end.
FREE_END_FRAME = {"end": 0, "start": 11}

# Spaghetti style mirrors aggregate_free_end_states.py so the two
# figures read as the same family of plot.
SPAGHETTI_COLOR = "0.55"
SPAGHETTI_ALPHA = 0.08
SPAGHETTI_LW = 0.4

# Crimson ensemble-mean track on the map row: thicker than the per-
# member spaghetti and a clearly non-rainbow hue so it reads as a
# distinct overlay rather than another member, but thinner than the
# ERA5 reference so ERA5 stays visually dominant.
ENSEMBLE_MEAN_TRACK_COLOR = "crimson"
ENSEMBLE_MEAN_TRACK_LW = 2.2

# Gold highlight for the ensemble member whose central-MSL trajectory is
# closest (lowest MSE) to ERA5 — meaningfully thicker than the spaghetti
# but still thinner than the ensemble-mean / ERA5 reference lines so the
# eye reads it as "one of the members."
GOLD_COLOR = "gold"
GOLD_LW = 1.6
GOLD_ALPHA = 1.0

# 5–95% member envelope on the tc_msl trajectory rows.  Soft blue so it
# reads as a backdrop under the gray spaghetti; the darker edge color is
# reused for the free-end whisker that calls out where ERA5 sits inside
# the band.
ENVELOPE_COLOR = "#4C72B0"
ENVELOPE_ALPHA = 0.18
ENVELOPE_EDGE = "#2F4B7C"

# Free-end percentile band: central 90% (5th–95th) of the surviving
# members at each lead step.
ENVELOPE_LO_PCT = 5.0
ENVELOPE_HI_PCT = 95.0

# Default cBottle output timestep; sandy + ian both use the standard
# 6 h cadence.  Override via --timestep-hours if a future case needs it.
DEFAULT_TIMESTEP_HOURS = 6.0


def _load_mode_tracks(
    output_dir: Path, mode: str
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return ``(ensemble_tracks, era5_tracks)`` for one mode, or ``(None, None)``."""
    parquet = output_dir / f"tc_tracks_{mode}.parquet"
    if not parquet.exists():
        print(f"WARNING: missing {parquet}, skipping mode '{mode}'")
        return None, None
    tracks = load_tracks_parquet(str(parquet))
    ens = tracks.get("ensemble")
    era5 = tracks.get("era5")
    return ens, era5


def _draw_track_map_panel(
    ax,
    tc_tracks_np: np.ndarray,
    era5_tracks: Optional[np.ndarray],
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    mode: str,
) -> None:
    """Render one mode's track map panel.  Extracted from the prior layout
    so the cross-mode row sits inside the multi-row gridspec without
    duplicating the cartopy boilerplate."""
    n_members = tc_tracks_np.shape[0]
    _lon_min = wrap_lon_0_360(lon_min)
    _lon_max = wrap_lon_0_360(lon_max)

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
    ax.set_extent([_lon_min, _lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    # Outline the tracking domain: tracks are restricted to this box, so the
    # dashed frame marks exactly the region the in-domain counts cover.
    draw_domain_box(ax, [_lon_min, _lon_max, lat_min, lat_max])

    # The "free end" — the end of each track that wasn't anchored to ERA5
    # by the conditioning — is the most informative endpoint to highlight:
    # under end-conditioning, the free end is the *start* (members
    # diverge backwards from a common landfall fix); under start-
    # conditioning, the free end is the *last* fix; under both-end
    # conditioning, neither end is free, so no marker is drawn.
    free_end_idx_lookup = {"end": "first", "start": "last"}
    free_end_pick = free_end_idx_lookup.get(mode)

    n_plotted = 0
    n_free_end_dots = 0
    for path_id in range(tc_tracks_np.shape[1]):
        for member_idx in range(n_members):
            tc_lats = tc_tracks_np[member_idx, path_id, :, 0]
            tc_lons = tc_tracks_np[member_idx, path_id, :, 1]
            valid = ~np.isnan(tc_lats) & ~np.isnan(tc_lons)
            if valid.sum() < MIN_PATH_LENGTH:
                continue
            member_color = MEMBER_TRACK_CMAP(member_idx / max(1, n_members - 1))
            ax.plot(
                tc_lons[valid],
                tc_lats[valid],
                color=member_color,
                linewidth=0.8,
                alpha=0.3,
                transform=ccrs.PlateCarree(),
                zorder=2,
            )
            if free_end_pick is not None:
                end_i = 0 if free_end_pick == "first" else -1
                ax.plot(
                    tc_lons[valid][end_i],
                    tc_lats[valid][end_i],
                    "o",
                    color=member_color,
                    markersize=4.5,
                    alpha=0.75,
                    markeredgecolor="black",
                    markeredgewidth=0.3,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                )
                n_free_end_dots += 1
            n_plotted += 1

    # Ensemble-mean track at path 0 (circular lon mean handles dateline).
    mean_lats, mean_lons = ensemble_mean_track(tc_tracks_np, path_id=0)
    mean_valid_n = 0
    if mean_lats.size:
        valid_mean = ~np.isnan(mean_lats) & ~np.isnan(mean_lons)
        if valid_mean.sum() >= MIN_PATH_LENGTH:
            ax.plot(
                mean_lons[valid_mean],
                mean_lats[valid_mean],
                color=ENSEMBLE_MEAN_TRACK_COLOR,
                linewidth=ENSEMBLE_MEAN_TRACK_LW,
                alpha=0.95,
                transform=ccrs.PlateCarree(),
                zorder=4,
                label="Ensemble mean",
            )
            ax.plot(
                mean_lons[valid_mean],
                mean_lats[valid_mean],
                "o",
                color=ENSEMBLE_MEAN_TRACK_COLOR,
                markersize=3.5,
                markeredgecolor="white",
                markeredgewidth=0.4,
                alpha=0.95,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
            mean_valid_n = int(valid_mean.sum())

    if era5_tracks is not None:
        for path_id in range(era5_tracks.shape[1]):
            era5_lats = era5_tracks[0, path_id, :, 0]
            era5_lons = era5_tracks[0, path_id, :, 1]
            valid_era5 = ~np.isnan(era5_lats) & ~np.isnan(era5_lons)
            if valid_era5.sum() < MIN_PATH_LENGTH:
                continue
            label = "ERA5" if path_id == 0 else None
            ax.plot(
                era5_lons[valid_era5],
                era5_lats[valid_era5],
                color="black",
                linewidth=2.0,
                linestyle="--",
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                zorder=5,
                label=label,
            )
            ax.plot(
                era5_lons[valid_era5],
                era5_lats[valid_era5],
                "D",
                color="black",
                markersize=4,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )

    ax.set_title(
        MODE_LABELS.get(mode, mode.title()), fontsize=TITLE_SIZE, fontweight="bold"
    )
    ax.legend(loc="lower left", fontsize=LEGEND_SIZE, framealpha=0.8)
    free_end_msg = (
        f" + {n_free_end_dots} free-end dots ({free_end_pick})"
        if free_end_pick is not None
        else ""
    )
    print(
        f"  [{mode}] plotted {n_plotted} member tracks"
        + (f" + ensemble mean ({mean_valid_n} valid steps)" if mean_valid_n else "")
        + free_end_msg
    )


def _draw_tc_msl_trajectory(
    ax,
    ens_arr: np.ndarray,
    era5_arr: Optional[np.ndarray],
    *,
    lead_hours: np.ndarray,
    start_time_iso: str,
    timezone_name: str,
    mode: str,
    era5_pct: Optional[float] = None,
    era5_pct_n: int = 0,
) -> bool:
    """Draw the per-mode tc_msl trajectory on ``ax``.

    Pulls central MSL at ``PRIMARY_PATH_ID`` (the index every downstream
    intensity diagnostic uses), converts Pa → hPa, and renders the
    member spaghetti + ensemble mean + ERA5.  Also highlights in gold
    the eligible member whose MSL trajectory has the lowest MSE against
    ERA5; eligibility requires a hurricane fix at every lead step (no
    NaN in the member's path-0 row).  A shaded 5–95% percentile band
    (nan-aware across the surviving members at each step) shows the
    ensemble envelope, and for end-/start-conditioning a whisker + star
    annotates where ERA5 sits within that envelope at the free
    (unconditioned) end — lead 0 for end-conditioning, the final lead for
    start-conditioning — labelled with ERA5's percentile rank among the
    surviving members (the value, band bounds and member count go to the
    run log).  Returns ``True`` if any member or
    ERA5 trajectory was plotted; ``False`` if the parquet had no usable
    tc_msl at this path id (e.g. tracker missed it entirely).

    Y-axis runs low-to-high MSL (i.e. NOT inverted); deeper troughs sit
    at the bottom, which inverts the usual meteorological convention but
    matches how raw pressure values would intuitively be read.
    """
    n_members, n_paths, n_steps, _ = ens_arr.shape
    if PRIMARY_PATH_ID >= n_paths:
        print(
            f"  [{mode}] tc_msl: parquet has no path_id={PRIMARY_PATH_ID} "
            f"(only {n_paths} paths); skipping trajectory row"
        )
        return False

    # tc_tracks_io trailing axis: [tc_lat, tc_lon, tc_msl, tc_w10m].
    # Drop members with no fix at path 0 anywhere along the lead window —
    # an all-NaN row would otherwise contribute a flat gap in the
    # spaghetti.
    msl_pa = ens_arr[:, PRIMARY_PATH_ID, :, 2]
    msl_hpa = msl_pa / 100.0

    # Pre-compute the gold-highlight member: the eligible member (one with
    # a fix at every lead step) whose MSL trajectory minimises MSE vs
    # ERA5 over all n_steps frames.  Requires ERA5 to also be fully
    # finite at path 0; otherwise we skip the highlight and just render
    # the spaghetti normally.
    era5_msl_hpa_full: Optional[np.ndarray] = None
    if era5_arr is not None and PRIMARY_PATH_ID < era5_arr.shape[1]:
        candidate = era5_arr[0, PRIMARY_PATH_ID, :, 2] / 100.0
        if np.isfinite(candidate).all():
            era5_msl_hpa_full = candidate

    gold_idx: Optional[int] = None
    if era5_msl_hpa_full is not None:
        eligible_mask = np.isfinite(msl_hpa).all(axis=1)
        if eligible_mask.any():
            mse = np.full(n_members, np.inf)
            diff = msl_hpa[eligible_mask] - era5_msl_hpa_full
            mse[eligible_mask] = np.mean(diff * diff, axis=1)
            gold_idx = int(np.argmin(mse))

    # 5–95% ensemble envelope across surviving members at each lead step.
    # nan-aware: members without a fix at a step are dropped, so the band
    # narrows where the storm weakens below the tracker threshold and the
    # surviving sample shrinks (the surviving count is annotated at the
    # free end below).  All-NaN steps yield a NaN bound and are masked out
    # of the fill.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        env_lo = np.nanpercentile(msl_hpa, ENVELOPE_LO_PCT, axis=0)
        env_hi = np.nanpercentile(msl_hpa, ENVELOPE_HI_PCT, axis=0)
    env_valid = np.isfinite(env_lo) & np.isfinite(env_hi)
    if env_valid.any():
        ax.fill_between(
            lead_hours,
            env_lo,
            env_hi,
            where=env_valid,
            color=ENVELOPE_COLOR,
            alpha=ENVELOPE_ALPHA,
            linewidth=0,
            zorder=0.5,
        )

    n_plotted_members = 0
    for i in range(n_members):
        finite = np.isfinite(msl_hpa[i])
        if finite.sum() < 2:
            continue
        if i == gold_idx:
            # Defer the gold member to a separate draw so it sits on top
            # of the spaghetti with its own color / linewidth.
            n_plotted_members += 1
            continue
        ax.plot(
            lead_hours,
            msl_hpa[i],
            color=SPAGHETTI_COLOR,
            alpha=SPAGHETTI_ALPHA,
            linewidth=SPAGHETTI_LW,
            zorder=1,
        )
        n_plotted_members += 1

    if gold_idx is not None:
        ax.plot(
            lead_hours,
            msl_hpa[gold_idx],
            color=GOLD_COLOR,
            alpha=GOLD_ALPHA,
            linewidth=GOLD_LW,
            zorder=4,
        )

    # Ensemble mean across members at each step; nanmean so that members
    # missing a fix at one step don't pull the mean to NaN.  Suppress
    # the "Mean of empty slice" RuntimeWarning that fires when every
    # member is NaN at a step (degenerate parquet, or the tracker has
    # not yet locked onto the storm at the first frame).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        ens_mean_hpa = np.nanmean(msl_hpa, axis=0)
    finite_mean = np.isfinite(ens_mean_hpa)
    if finite_mean.any():
        ax.plot(
            lead_hours[finite_mean],
            ens_mean_hpa[finite_mean],
            color="black",
            linewidth=2.4,
            zorder=5,
        )

    era5_plotted = False
    if era5_arr is not None and PRIMARY_PATH_ID < era5_arr.shape[1]:
        era5_msl_pa = era5_arr[0, PRIMARY_PATH_ID, :, 2]
        era5_msl_hpa = era5_msl_pa / 100.0
        finite_era5 = np.isfinite(era5_msl_hpa)
        if finite_era5.sum() >= 2:
            ax.plot(
                lead_hours[finite_era5],
                era5_msl_hpa[finite_era5],
                color="black",
                linewidth=2.4,
                linestyle="--",
                zorder=6,
            )
            era5_plotted = True

    if n_plotted_members == 0 and not era5_plotted:
        print(f"  [{mode}] tc_msl: no plottable members or ERA5 at path 0")
        return False

    # --- Free-end envelope call-out -------------------------------------
    # The "free end" is the trajectory endpoint not pinned by the
    # conditioning (mirrors the map panels' free_end_idx_lookup): under
    # end-conditioning members fan out backward from the anchored landfall,
    # so the free end is lead 0 (first step); under start-conditioning they
    # fan out forward from the anchored IC, so it is the final lead step.
    # Both-end conditioning pins both ends — no free end — so we skip it.
    free_end_pick = {"end": "first", "start": "last"}.get(mode)
    if free_end_pick is not None:
        free_idx = 0 if free_end_pick == "first" else n_steps - 1
        free_lead = float(lead_hours[free_idx])
        lo_fe = env_lo[free_idx]
        hi_fe = env_hi[free_idx]
        col = msl_hpa[:, free_idx]
        members_fe = col[np.isfinite(col)]
        n_surv_fe = int(members_fe.size)

        era5_fe = np.nan
        if era5_arr is not None and PRIMARY_PATH_ID < era5_arr.shape[1]:
            era5_fe = float(era5_arr[0, PRIMARY_PATH_ID, free_idx, 2]) / 100.0

        if np.isfinite(lo_fe) and np.isfinite(hi_fe) and n_surv_fe > 0:
            # Whisker spanning the 5–95% band at the free-end lead, with
            # short end caps so the band bounds read as discrete values.
            ax.vlines(
                free_lead, lo_fe, hi_fe, color=ENVELOPE_EDGE, linewidth=1.6, zorder=7
            )
            cap = 0.012 * (float(lead_hours[-1]) - float(lead_hours[0]))
            for yy in (lo_fe, hi_fe):
                ax.plot(
                    [free_lead - cap, free_lead + cap],
                    [yy, yy],
                    color=ENVELOPE_EDGE,
                    linewidth=1.6,
                    zorder=7,
                )

            if np.isfinite(era5_fe):
                inside = lo_fe <= era5_fe <= hi_fe
                # Percentile rank of ERA5 among the members plotted at the free
                # end (fraction with MSL at or below ERA5).  Lower MSL = deeper
                # = stronger, so a small rank means ERA5 is stronger than most.
                rank = 100.0 * float(np.mean(members_fe <= era5_fe))
                # Prefer the unified-QC percentile (track_qc's exact-frame
                # analyzable subset -- the single source of truth the track-
                # stats report quotes); fall back to the plotted-member rank
                # when it is unavailable.
                have_qc = era5_pct is not None and np.isfinite(era5_pct)
                label_pct = float(era5_pct) if have_qc else rank
                ax.plot(
                    free_lead,
                    era5_fe,
                    marker="*",
                    markersize=13,
                    color="black",
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    zorder=8,
                )
                # Anchor the call-out toward the panel interior so it does
                # not run off the free-end edge (lead 0 -> text to the
                # right; final lead -> text to the left).
                on_left = free_idx == 0
                dx = 45 if on_left else -45
                ha = "left" if on_left else "right"
                ax.annotate(
                    f"ERA5 p{label_pct:.0f}",
                    xy=(free_lead, era5_fe),
                    xytext=(dx, 30),
                    textcoords="offset points",
                    ha=ha,
                    va="bottom",
                    fontsize=max(8, LEGEND_SIZE - 2),
                    zorder=9,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor=ENVELOPE_EDGE,
                        alpha=0.9,
                        linewidth=0.8,
                    ),
                    arrowprops=dict(arrowstyle="-", color=ENVELOPE_EDGE, linewidth=0.8),
                )
                verdict = (
                    "inside" if inside else ("below" if era5_fe < lo_fe else "above")
                )
                print(
                    f"  [{mode}] tc_msl free end (lead {free_lead:.0f}h, "
                    f"idx {free_idx}): ERA5={era5_fe:.2f} hPa, "
                    f"5-95%=[{lo_fe:.2f}, {hi_fe:.2f}] hPa, {verdict} band, "
                    f"p{label_pct:.0f} "
                    f"({'QC exact-frame n=' + str(era5_pct_n) if have_qc else 'plotted-member'}"
                    f"; plotted-member rank p{rank:.0f}), n_surv={n_surv_fe}"
                )
            else:
                print(
                    f"  [{mode}] tc_msl free end (lead {free_lead:.0f}h): "
                    f"5-95%=[{lo_fe:.2f}, {hi_fe:.2f}] hPa, n_surv={n_surv_fe}, "
                    "ERA5 missing at free end"
                )

    format_local_time_axis(ax, lead_hours, start_time_iso, timezone_name)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel("Storm central MSLP (hPa)", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(float(lead_hours[0]), float(lead_hours[-1]))
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"{MODE_LABELS.get(mode, mode.title())} — Storm central MSLP trajectory",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    print(
        f"  [{mode}] tc_msl: plotted {n_plotted_members} members"
        + (" + ERA5" if era5_plotted else "")
        + (f" + gold member #{gold_idx}" if gold_idx is not None else "")
    )
    return True


def _build_trajectory_legend_handles() -> tuple[list, list]:
    """Static legend proxies for the tc_msl trajectory rows."""
    handles = [
        Line2D([0], [0], color="black", linewidth=2.4),
        Line2D([0], [0], color="black", linewidth=2.4, linestyle="--"),
        Line2D(
            [0],
            [0],
            color=GOLD_COLOR,
            alpha=GOLD_ALPHA,
            linewidth=GOLD_LW,
        ),
        Line2D(
            [0],
            [0],
            color=SPAGHETTI_COLOR,
            # Bump alpha so the swatch is legible at legend scale.
            alpha=min(1.0, SPAGHETTI_ALPHA * 6),
            linewidth=SPAGHETTI_LW * 2,
        ),
        Patch(facecolor=ENVELOPE_COLOR, alpha=ENVELOPE_ALPHA, edgecolor="none"),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            color="black",
            markeredgecolor="white",
            markersize=12,
        ),
    ]
    labels = [
        "Ensemble mean",
        "ERA5",
        "Closest to ERA5 trajectory (MSE)",
        "Members",
        "5–95% envelope",
        "ERA5 at free end",
    ]
    return handles, labels


def make_figure(
    mode_tracks: dict[str, np.ndarray],
    mode_era5: dict[str, Optional[np.ndarray]],
    *,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    out_path: Path,
    title: str,
    start_time_iso: str,
    timezone_name: str,
    timestep_hours: float,
    era5_pct_by_mode: Optional[dict[str, tuple[float, int]]] = None,
) -> None:
    """Render the multi-row case figure: track-map row + tc_msl trajectory rows.

    The track-map row always renders for whatever modes are present.
    Rows 2 / 3 / 4 render the end-, start-, and both-end-mode tc_msl
    trajectories respectively; any row is skipped silently if the
    corresponding parquet was missing or had no fix at
    ``PRIMARY_PATH_ID``.
    """
    map_modes = [m for m in ("end", "start", "both") if m in mode_tracks]
    if not map_modes:
        raise SystemExit("ERROR: no modes with track data to plot.")
    n_cols = len(map_modes)

    # Determine which trajectory rows will be drawn (need both ensemble
    # parquet present *and* a path-0 fix).  We pre-decide by checking the
    # ensemble shape so we can size the gridspec correctly; the actual
    # draw call returns the same boolean.
    traj_modes = [
        m
        for m in ("end", "start", "both")
        if m in mode_tracks and PRIMARY_PATH_ID < mode_tracks[m].shape[1]
    ]
    n_traj = len(traj_modes)
    n_rows = 1 + n_traj

    # Row heights tuned so each cartopy panel reads at its natural aspect
    # ratio (~6" tall for the bbox sizes sandy / ian use) and each
    # trajectory row gets ~3" — wide-and-shallow so the 12-step lead
    # window doesn't compress into a column.  constrained_layout drives
    # the actual spacing; the height ratios just set the relative sizes.
    map_row_h = 6.0
    traj_row_h = 3.0

    fig_w = 7.0 * n_cols
    # 1.4" of vertical slack covers the suptitle (top) and figure-level
    # legend (bottom); constrained_layout absorbs anything left over.
    fig_h = map_row_h + traj_row_h * n_traj + 1.4
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)

    height_ratios = [map_row_h] + [traj_row_h] * n_traj
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        height_ratios=height_ratios,
    )

    # Row 1: per-mode track maps.
    proj = ccrs.PlateCarree()
    map_axes = [fig.add_subplot(gs[0, c], projection=proj) for c in range(n_cols)]
    for ax, mode in zip(map_axes, map_modes):
        _draw_track_map_panel(
            ax,
            mode_tracks[mode],
            mode_era5.get(mode),
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            mode=mode,
        )

    # Rows 2..: tc_msl trajectory per mode, full width.
    traj_axes: list = []
    drew_any_traj = False
    for i, mode in enumerate(traj_modes):
        ens = mode_tracks[mode]
        n_steps = ens.shape[2]
        lead_hours = np.arange(n_steps, dtype=float) * timestep_hours
        ax = fig.add_subplot(gs[1 + i, :])
        pct, pct_n = (era5_pct_by_mode or {}).get(mode, (None, 0))
        ok = _draw_tc_msl_trajectory(
            ax,
            ens,
            mode_era5.get(mode),
            lead_hours=lead_hours,
            start_time_iso=start_time_iso,
            timezone_name=timezone_name,
            mode=mode,
            era5_pct=pct,
            era5_pct_n=pct_n,
        )
        if ok:
            traj_axes.append(ax)
            drew_any_traj = True
        else:
            # Nothing to show; remove the axes so the figure doesn't have
            # an empty white strip.
            ax.remove()

    all_axes = list(map_axes) + traj_axes
    add_subplot_labels(all_axes, placement="inside")

    fig.suptitle(title, fontsize=SUPTITLE_SIZE, fontweight="bold")

    if drew_any_traj:
        handles, labels = _build_trajectory_legend_handles()
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=len(labels),
            fontsize=LEGEND_SIZE,
            framealpha=0.9,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _mode_anchor_frame(mode: str, landfall_frame: int, n_leads: int) -> int:
    """Pinned-end anchor frame for one mode (matches track_qc consumers).

    End-conditioning pins the landfall frame (last lead if the case sets
    none); start- and both-conditioning pin frame 0.
    """
    if mode == "start" or mode == "both":
        return 0
    return landfall_frame if 0 <= landfall_frame < n_leads else n_leads - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--title", default="Ensemble Storm Tracks")
    parser.add_argument(
        "--start-time",
        type=str,
        default="1970-01-01T00:00:00Z",
        help="Ensemble IC as ISO 8601 UTC (drives the local wall-clock x-axis "
        "on the tc_msl trajectory rows).  Falls back to epoch when omitted.",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="IANA timezone for the lead-axis labels (e.g. "
        "'America/New_York').  Falls back to UTC when omitted.",
    )
    parser.add_argument(
        "--timestep-hours",
        type=float,
        default=DEFAULT_TIMESTEP_HOURS,
        help=f"Lead-time step in hours (default: {DEFAULT_TIMESTEP_HOURS}, "
        "matches cBottle's 6 h cadence).",
    )
    parser.add_argument(
        "--landfall-frame",
        type=int,
        default=-1,
        help="Pinned-end anchor frame for end-conditioning (the case YAML's "
        "tc_tracks.landfall_frame; last lead when unset).  Drives track_qc's "
        "storm-of-interest resolution and the 'ERA5 pN' annotation.",
    )
    parser.add_argument(
        "--anchor-radius-km",
        type=float,
        default=ANCHOR_RADIUS_KM,
        help="Great-circle radius (km) within which a member's path must "
        f"approach the ERA5 pinned-end fix to reach the anchor (default "
        f"{ANCHOR_RADIUS_KM:g}).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"ERROR: --output-dir does not exist: {output_dir}")

    track_box = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    mode_tracks: dict[str, np.ndarray] = {}
    mode_era5: dict[str, Optional[np.ndarray]] = {}
    era5_pct_by_mode: dict[str, tuple[float, int]] = {}
    for mode in ("end", "start", "both"):
        ens, era5 = _load_mode_tracks(output_dir, mode)
        if ens is None:
            continue
        # Unified QC: resolve each member's storm of interest anchor-first,
        # then reduce to that one path for drawing.  The MSL-percentile
        # annotation uses the same QC's exact-frame analyzable subset -- one
        # definition shared with the track-stats report (no baked-npz read).
        n_total = ens.shape[0]
        n_leads = ens.shape[2]
        anchor_frame = _mode_anchor_frame(mode, args.landfall_frame, n_leads)
        try:
            qc = compute_track_qc_arrays(
                ens,
                era5,
                track_box=track_box,
                anchor_frame=anchor_frame,
                free_end_frame=FREE_END_FRAME.get(mode, 0),
                anchor_radius_km=args.anchor_radius_km,
            )
        except ValueError as exc:
            print(f"  [{mode}] track_qc skipped ({exc}); dropping this mode.")
            continue
        counts = qc.counts()
        print(
            f"  [{mode}] unified QC (anchor frame {anchor_frame}): "
            f"{counts['analyzable']}/{n_total} members analyzable "
            f"({counts['excluded']} excluded)"
        )
        mode_tracks[mode] = qc.storm_of_interest_array(ens)
        # ERA5's storm of interest is its path 0 (the fix that defines the anchor).
        mode_era5[mode] = era5[:, 0:1] if era5 is not None else None
        if mode in FREE_END_FRAME:
            era5_pct_by_mode[mode] = qc.era5_free_end_msl_pct()

    if not mode_tracks:
        raise SystemExit(f"ERROR: no per-mode parquet files found in {output_dir}")

    print(f"[aggregate_tc_tracks] modes: {list(mode_tracks)}")
    out_path = output_dir / "tc_tracks.png"
    make_figure(
        mode_tracks,
        mode_era5,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        out_path=out_path,
        title=args.title,
        start_time_iso=args.start_time,
        timezone_name=args.timezone,
        timestep_hours=args.timestep_hours,
        era5_pct_by_mode=era5_pct_by_mode,
    )


if __name__ == "__main__":
    main()
