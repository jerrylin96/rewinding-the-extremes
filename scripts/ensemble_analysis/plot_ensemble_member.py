#!/usr/bin/env python3
"""Animate ensemble members side-by-side with the ERA5 reference.

Each output is an MP4 that steps through all lead-time frames, with the
ERA5 reference on the left and the chosen ensemble member on the right.

Supports two modes:

  1. Event mode  — batch-render every (mode, variable, member) combination
     configured under a case YAML in ``cases/<event>.yaml``.  Variable
     style defaults come from ``cases/_palettes.yaml`` and can be
     overridden inline in the case YAML.  Outputs land under
     ``output/<event>_plots/<mode>/<variable>_member<N>.mp4``.

  2. CLI mode    — specify a single animation via command-line arguments.

Event examples:
    python plot_ensemble_member.py --event katrina
    python plot_ensemble_member.py --event katrina --mode end
    python plot_ensemble_member.py --event katrina --variable msl
    python plot_ensemble_member.py --event katrina --members 0-99
    python plot_ensemble_member.py --event katrina --members 5,6,7
    python plot_ensemble_member.py --event teleport_gulf --layout grid \
        --variable msl --members 0-3 --wind-overlay wind500

CLI examples:
    python plot_ensemble_member.py \
        --ensemble-zarr output/ensemble_f0_11_n64.zarr \
        --era5-zarr output/era5_reference.zarr \
        --variable t2m --member 0

    python plot_ensemble_member.py \
        --ensemble-zarr output/ensemble_f0_11_n64.zarr \
        --era5-zarr output/era5_reference.zarr \
        --variable msl --member 5 \
        --cmap viridis --vmin 96000 --vmax 104000 \
        --extent -100 -50 10 45
"""

import argparse
import os
import sys

# scripts/_shared/ holds _dispatch_lib.py and a custom_utils symlink.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "_shared"))
sys.path.insert(0, SHARED_DIR)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import cmasher  # noqa: F401, E402 — registers "cmr.*" colormap names with matplotlib
from matplotlib.animation import FuncAnimation  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402

from _dispatch_lib import (  # noqa: E402
    ensemble_zarr_path,
    era5_zarr_path,
    iter_modes,
    list_cases,
    load_case,
    load_palettes,
    parse_member_spec,
    resolve_view,
)
from custom_utils.ensemble_loading import variable_index  # noqa: E402
from custom_utils.interpolation_utils import CBottleUtils  # noqa: E402
from derived_fields import relative_vorticity  # noqa: E402
from plot_style import (  # noqa: E402
    COLORBAR_LABEL_SIZE,
    COLORBAR_LABEL_SIZE_DENSE,
    COLORBAR_TICK_SIZE,
    COLORBAR_TICK_SIZE_DENSE,
    QUIVER_REF_LEN_FRAC,
    SUPTITLE_SIZE_DENSE,
    TICK_LABEL_SIZE_DENSE,
    TITLE_SIZE,
    TITLE_SIZE_DENSE,
    add_subplot_labels,
    draw_graticule,
    draw_quiver,
    draw_quiver_key,
    nice_speed,
    quiver_step,
)

import torch  # noqa: E402

# ── Derived variables ──────────────────────────────────────────────────────
# Each entry maps a derived name to (list of source vars, combining function,
# display label).  Source vars must exist in the zarr store.
#
# A spec may set ``needs_coords: True``, in which case ``func`` is called as
# ``func(*components, lat, lon)`` — required by any derivation that takes a
# spatial gradient rather than combining components pointwise.

DERIVED_VARIABLES = {
    "wspd10m": {
        "components": ["u10m", "v10m"],
        "func": lambda u, v: np.sqrt(u**2 + v**2),
        "label": "10-m wind speed (m/s)",
    },
    "wspd850": {
        "components": ["u850", "v850"],
        "func": lambda u, v: np.sqrt(u**2 + v**2),
        "label": "850 hPa wind speed (m/s)",
    },
    "vort850": {
        "components": ["u850", "v850"],
        "func": relative_vorticity,
        "needs_coords": True,
        "label": "850 hPa relative vorticity (s$^{-1}$)",
    },
}


# ── Wind overlays ──────────────────────────────────────────────────────────
#
# Optional flow arrows over the member panels, drawn with the shared quiver
# convention from plot_style — the same helpers, constants and reference-key
# placement the synoptic-PCA precursor figure uses.
#
# One definition only: raw u500/v500, straight off the store with no
# combination step of any kind.  That is *exactly* the quantity the
# synoptic-PCA figure draws (`DEFAULT_WIND_VARIABLES = ("u500", "v500")` in
# dispatch_synoptic_pca.py, unoverridden by any case YAML), so the arrows here
# need no defence the paper's existing figure does not already make.
#
# A multi-level deep-layer mean was built first and REMOVED at the author's
# direction: its weighting was not traceable to a citable source, and nothing
# undefendable in review belongs in the figure path.  Do not reintroduce a
# combined wind here without one — see
# paper/proposals/stress-test-member-grids.md (commit 04a04fe has the code).
#
# Both components are in custom_utils.interpolation_utils.STORED_VARIABLES, so
# any ensemble store written by this pipeline carries them.
#
# Deliberately NOT a tracker: a quiver needs no detection step, and the parked
# figure design bans one ("a tracker stub near each storm is indistinguishable
# from the tracker losing the storm").

WIND_OVERLAYS = {
    "wind500": {"u": "u500", "v": "v500", "label": "500 hPa wind"},
}
WIND_OVERLAY_CHOICES = ("none",) + tuple(WIND_OVERLAYS)

# The overlay grid is cropped to the panels' view box before the stride and the
# reference speed are derived from it — see _crop_to_extent.
WIND_OVERLAY_PAD_FRAC = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate ensemble members side-by-side with the ERA5 "
        "reference.  Use --event for batch rendering of a case "
        "from cases/<name>.yaml, or pass arguments directly for "
        "a single animation."
    )

    # Event (batch) mode
    parser.add_argument(
        "--event",
        type=str,
        default=None,
        help="Case name (e.g. 'katrina').  Reads "
        "cases/<event>.yaml and renders every "
        "(mode, variable, member) combination.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available cases and exit."
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        choices=["start", "end", "both"],
        help="Event mode only: restrict to one or more "
        "conditioning modes (repeatable).",
    )
    parser.add_argument(
        "--members",
        type=str,
        default=None,
        help="Event mode only: override plots.members.  Accepts 'all', a "
        "range ('0-99'), a list ('5,6,7'), or a mix ('0-3,10,20-25').  "
        "Shard a large render across jobs by giving each a disjoint range.",
    )

    # Single-animation CLI arguments (ignored when --event is used)
    parser.add_argument(
        "--ensemble-zarr",
        type=str,
        default=None,
        help="Path to ensemble zarr store (CLI mode)",
    )
    parser.add_argument(
        "--era5-zarr",
        type=str,
        default=None,
        help="Path to ERA5 reference zarr store (CLI mode)",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default=None,
        help="Variable to animate (e.g. t2m, z500, msl). "
        "Derived variables are also supported: "
        + ", ".join(DERIVED_VARIABLES.keys())
        + ". In event mode, restricts to one variable.",
    )
    parser.add_argument(
        "--member",
        type=int,
        default=0,
        help="Ensemble member index (default: 0, CLI mode)",
    )
    parser.add_argument(
        "--cmap", type=str, default="viridis", help="Colormap name (default: viridis)"
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Colorbar minimum (default: auto from data)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Colorbar maximum (default: auto from data)",
    )
    parser.add_argument(
        "--coastline-color",
        type=str,
        default="black",
        help="Coastline color (default: black)",
    )
    parser.add_argument(
        "--gridline-color",
        type=str,
        default="gray",
        help="Gridline color (default: gray)",
    )
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Geographic extent in degrees "
        "(lon_min lon_max lat_min lat_max). "
        "Longitudes in [-180, 180] convention.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="video",
        choices=["video", "grid"],
        help="'video' (default): MP4 stepping through frames, ERA5 beside the "
        "member.  'grid': a single PNG with all 12 frames as panels, pinned "
        "frames marked — the member alone, since the pinned panels already "
        "state what they were pinned to.",
    )
    parser.add_argument(
        "--grid-title",
        type=str,
        default=None,
        help="Suptitle for --layout grid (default: the case display_name).",
    )
    parser.add_argument(
        "--wind-overlay",
        type=str,
        default="none",
        choices=list(WIND_OVERLAY_CHOICES),
        help="--layout grid only.  Overlay flow arrows on every panel: 'none' "
        "(default) leaves the field alone; 'wind500' draws raw u500/v500 — the "
        "same quantity the synoptic-PCA precursor figure draws, with no "
        "averaging or other combination.  One quiver scale and one reference "
        "key are shared across all 12 panels, so arrows are comparable frame "
        "to frame (but not between two members' figures, which each derive "
        "their own scale).  Written to a separate filename, so the plain grid "
        "is not overwritten.",
    )
    parser.add_argument(
        "--fps", type=int, default=2, help="Frames per second (default: 2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: auto-generated)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="Figure DPI (default: 150)"
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=None,
        help="Event mode only: override ensemble.size when "
        "resolving upstream zarr paths.",
    )

    args = parser.parse_args()

    if args.list:
        return args

    # The overlay is a grid-layout feature: the video's arrows would need a
    # scale shared across frames *and* across the ERA5 panel, and the videos
    # are a QC tool, not a figure.  Fail loudly rather than silently dropping
    # the flag.
    if args.wind_overlay != "none" and args.layout != "grid":
        parser.error(
            f"--wind-overlay {args.wind_overlay} requires --layout grid "
            f"(got --layout {args.layout})"
        )

    if args.event is None:
        if (
            args.ensemble_zarr is None
            or args.era5_zarr is None
            or args.variable is None
        ):
            parser.error(
                "--ensemble-zarr, --era5-zarr, and --variable are required "
                "when not using --event"
            )

    return args


def get_valid_time_labels(root, n_frames, is_era5=False):
    """Return a list of human-readable valid-time strings for all frames."""
    times = np.array(root["time"][:])
    lead_times = np.array(root["lead_time"][:])
    labels = []
    for f in range(n_frames):
        if is_era5:
            valid = times[f]
        else:
            valid = times[0] + lead_times[f]
        labels.append(str(np.datetime_as_string(valid, unit="h")))
    return labels


def _hpx_level_from_root(root):
    level = root.attrs.get("hpx_level")
    if level is not None:
        return int(level)
    return int(np.log2(root["hpx"].shape[0] // 12) / 2)


def _round_trip_era5_field(arr, root, hpx_level):
    """Project an ERA5 ``[n_frames, n_lat, n_lon]`` array onto the
    HPX-resolvable subspace and back to lat/lon.

    Done so the ERA5 panel doesn't look visibly sharper than the cBottle
    panel — both fields end up filtered through the same HPX kernel.
    Returns ``(rt_arr, lat, lon)``.
    """
    t = torch.from_numpy(arr).float()
    coords = {
        "time": np.array(root["time"][: t.shape[0]]),
        "lead_time": np.array([np.timedelta64(0, "h")]),
        "variable": np.array(["_"]),
        "lat": np.array(root["lat"][:]),
        "lon": np.array(root["lon"][:]),
    }
    # Insert lead_time + variable dims to match the regridder contract:
    # [n_frames, 1, 1, n_lat, n_lon].
    t4 = t.unsqueeze(1).unsqueeze(2)
    hpx_data, hpx_coords = CBottleUtils.regrid_latlon_to_healpix_nest(
        t4,
        coords,
        hpx_level=hpx_level,
    )
    rt_data, rt_coords = CBottleUtils.regrid_healpix_nest_to_latlon(
        hpx_data,
        hpx_coords,
        hpx_level=hpx_level,
    )
    return rt_data[:, 0, 0, :, :].cpu().numpy(), rt_coords["lat"], rt_coords["lon"]


# The ERA5 round-trip is member-independent but sits inside the per-member
# load path, so a 100-member render would repeat the same global regrid 100
# times per variable.  Event mode iterates members innermost, so a handful of
# entries (one per variable in flight, including the components of a derived
# one) is enough to make every repeat a hit.  Values are treated as read-only
# by every caller — each consumer re-sorts or recombines into a fresh array.
_ERA5_RT_CACHE = {}
_ERA5_RT_CACHE_MAX = 4


def _round_trip_era5_cached(root, var_idx, n_frames, hpx_level):
    """Memoized :func:`_round_trip_era5_field` for one ERA5 variable."""
    key = (str(root.store), var_idx, n_frames, hpx_level)
    hit = _ERA5_RT_CACHE.get(key)
    if hit is not None:
        return hit

    raw = np.array(root["data"][:n_frames, 0, var_idx, :, :])
    result = _round_trip_era5_field(raw, root, hpx_level)

    if len(_ERA5_RT_CACHE) >= _ERA5_RT_CACHE_MAX:
        _ERA5_RT_CACHE.pop(next(iter(_ERA5_RT_CACHE)))
    _ERA5_RT_CACHE[key] = result
    return result


def _regrid_member_field(arr_hpx, root, hpx_level):
    """Regrid an ensemble member's field ``[n_frames, n_pix]`` to
    ``[n_frames, n_lat, n_lon]``.

    Returns ``(arr_latlon, lat, lon)``.
    """
    t = torch.from_numpy(arr_hpx).float()
    coords = {
        "time": np.array(root["time"][: t.shape[0]]),
        "lead_time": np.array([np.timedelta64(0, "h")]),
        "variable": np.array(["_"]),
        "hpx": np.array(root["hpx"][:]),
    }
    t4 = t.unsqueeze(1).unsqueeze(2)  # [n_frames, 1, 1, n_pix]
    latlon_data, latlon_coords = CBottleUtils.regrid_healpix_nest_to_latlon(
        t4,
        coords,
        hpx_level=hpx_level,
    )
    return (
        latlon_data[:, 0, 0, :, :].cpu().numpy(),
        latlon_coords["lat"],
        latlon_coords["lon"],
    )


def load_field(root, var, member, n_frames, is_era5, hpx_level):
    """Load a raw or derived variable from a zarr store, on lat/lon.

    Returns ``(field, lat, lon)`` where ``field`` has shape
    ``[n_frames, n_lat, n_lon]``.  ERA5 is round-tripped through HEALPix
    (at the ensemble's ``hpx_level``) for visual fairness with the
    ensemble; ensemble members are stored on HEALPix and regridded to
    lat/lon at compute time.
    """
    if var in DERIVED_VARIABLES:
        spec = DERIVED_VARIABLES[var]
        components = []
        comp_lat = comp_lon = None
        for comp in spec["components"]:
            arr, comp_lat, comp_lon = load_field(
                root,
                comp,
                member,
                n_frames,
                is_era5,
                hpx_level,
            )
            components.append(arr)
        if spec.get("needs_coords"):
            return (
                spec["func"](*components, comp_lat, comp_lon),
                comp_lat,
                comp_lon,
            )
        return spec["func"](*components), comp_lat, comp_lon

    var_idx = variable_index(root, var)
    if is_era5:
        # ERA5 shape: [time, lead_time=1, variable, lat, lon]
        return _round_trip_era5_cached(root, var_idx, n_frames, hpx_level)

    # Ensemble — HPX (5D: [ensemble, time=1, lead_time, variable, n_pix]).
    if "hpx" not in root.array_keys():
        raise ValueError(
            f"plot_ensemble_member.py requires HEALPix ensemble zarrs; "
            f"got arrays {sorted(root.array_keys())}"
        )
    arr_hpx = np.array(root["data"][member, 0, :n_frames, var_idx, :])
    return _regrid_member_field(arr_hpx, root, hpx_level)


def get_display_label(var):
    """Return a human-readable label for a variable."""
    if var in DERIVED_VARIABLES:
        return DERIVED_VARIABLES[var]["label"]
    return var


def animate_member(ens_root, era5_root, var, member, args):
    """Create an MP4 animation stepping through all lead-time frames."""
    if "hpx" not in ens_root.array_keys():
        raise SystemExit(
            "plot_ensemble_member.py requires HEALPix ensemble zarrs; "
            f"got arrays {sorted(ens_root.array_keys())}"
        )
    hpx_level = _hpx_level_from_root(ens_root)

    n_members = ens_root["data"].shape[0]
    n_lead = ens_root["data"].shape[2]
    n_era5_frames = era5_root["data"].shape[0]
    n_frames = min(n_lead, n_era5_frames)

    if member >= n_members:
        raise ValueError(
            f"Member {member} out of range (ensemble has {n_members} members)"
        )

    display_label = get_display_label(var)

    # Load all frames at once — both end up on the same lat/lon grid: ERA5
    # round-tripped through HPX, ensemble regridded HPX -> lat/lon.
    ens_all, ens_lat, ens_lon = load_field(
        ens_root,
        var,
        member,
        n_frames,
        is_era5=False,
        hpx_level=hpx_level,
    )
    era5_all, era5_lat, era5_lon = load_field(
        era5_root,
        var,
        member,
        n_frames,
        is_era5=True,
        hpx_level=hpx_level,
    )

    # Convert lon from [0, 360) to [-180, 180) and sort
    ens_plot_lon = np.where(ens_lon >= 180, ens_lon - 360, ens_lon)
    era5_plot_lon = np.where(era5_lon >= 180, era5_lon - 360, era5_lon)
    ens_sort = np.argsort(ens_plot_lon)
    era5_sort = np.argsort(era5_plot_lon)
    ens_plot_lon = ens_plot_lon[ens_sort]
    era5_plot_lon = era5_plot_lon[era5_sort]
    ens_all = ens_all[:, :, ens_sort]
    era5_all = era5_all[:, :, era5_sort]

    # Color range
    vmin = args.vmin
    vmax = args.vmax
    if vmin is None:
        vmin = min(float(np.nanmin(era5_all)), float(np.nanmin(ens_all)))
    if vmax is None:
        vmax = max(float(np.nanmax(era5_all)), float(np.nanmax(ens_all)))

    ens_time_labels = get_valid_time_labels(ens_root, n_frames, is_era5=False)
    era5_time_labels = get_valid_time_labels(era5_root, n_frames, is_era5=True)

    # Build figure and initial frame
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    era5_lon_grid, era5_lat_grid = np.meshgrid(era5_plot_lon, era5_lat)
    ens_lon_grid, ens_lat_grid = np.meshgrid(ens_plot_lon, ens_lat)

    panels = []
    for ax, lon_grid, lat_grid, field in [
        (axes[0], era5_lon_grid, era5_lat_grid, era5_all[0]),
        (axes[1], ens_lon_grid, ens_lat_grid, ens_all[0]),
    ]:
        im = ax.pcolormesh(
            lon_grid,
            lat_grid,
            field,
            transform=ccrs.PlateCarree(),
            cmap=args.cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.add_feature(
            cfeature.COASTLINE, linewidth=0.7, edgecolor=args.coastline_color
        )
        ax.add_feature(
            cfeature.BORDERS,
            linestyle=":",
            linewidth=0.5,
            edgecolor=args.coastline_color,
        )
        if args.extent is not None:
            ax.set_extent(args.extent, crs=ccrs.PlateCarree())
            ax.add_feature(
                cfeature.STATES, linewidth=0.4, edgecolor=args.coastline_color
            )
        ax.gridlines(
            draw_labels=True,
            linewidth=0.4,
            alpha=0.5,
            linestyle="--",
            color=args.gridline_color,
        )
        panels.append(im)

    # y=1.0 for the same reason as the grid layout: these axes carry gridline
    # labels, and cartopy's Gridliner drops an auto-placed title on them.  See
    # draw_graticule's caller obligation in plot_style.
    axes[0].set_title(
        f"ERA5 (HPX round-trip)\n{display_label} — {era5_time_labels[0]}",
        fontsize=TITLE_SIZE,
        y=1.0,
    )
    axes[1].set_title(
        f"Ensemble Member {member}\n{display_label} — {ens_time_labels[0]}",
        fontsize=TITLE_SIZE,
        y=1.0,
    )
    # Inside placement keeps the labels visible over the maps without
    # overlapping the multi-line per-panel titles.
    add_subplot_labels(axes, placement="inside")

    cbar = fig.colorbar(
        panels[0],
        ax=axes,
        orientation="horizontal",
        shrink=0.6,
        pad=0.04,
        label=display_label,
    )
    cbar.ax.xaxis.label.set_fontsize(COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    def update(frame_idx):
        panels[0].set_array(era5_all[frame_idx].ravel())
        panels[1].set_array(ens_all[frame_idx].ravel())
        axes[0].set_title(
            f"ERA5 (HPX round-trip)\n{display_label} — {era5_time_labels[frame_idx]}",
            fontsize=TITLE_SIZE,
            y=1.0,
        )
        axes[1].set_title(
            f"Ensemble Member {member}\n{display_label} — {ens_time_labels[frame_idx]}",
            fontsize=TITLE_SIZE,
            y=1.0,
        )
        return panels

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

    out_name = args.output or f"{var}_member{member}.mp4"
    out_path = os.path.join(args.output_dir, out_name)
    anim.save(out_path, writer="ffmpeg", fps=args.fps, dpi=args.dpi)
    plt.close(fig)
    return out_path


# Stores already warned about, so a 100-member render prints the banner once
# rather than a hundred times.
_INCOMPLETE_WARNED = set()


def _warn_if_incomplete(ens_root, path):
    """Warn once per store when the writing job never marked it complete.

    ensemble_interpolation.py sets ``attrs["complete"]`` only after its final
    micro-batch is written, so a job killed at the wall clock leaves the flag
    off and the unwritten tail sitting at zarr's fill value.  Those members
    render as a flat panel at the bottom of the colour scale, which is easy to
    mistake for a model failure.  merge_ensemble_zarrs.py already refuses to
    merge a chunk store without the flag; an unpartitioned run (any --quick
    smoke test) never goes through the merge, so this is the only check it
    gets.
    """
    if path in _INCOMPLETE_WARNED or ens_root.attrs.get("complete", False):
        return
    _INCOMPLETE_WARNED.add(path)
    print(
        f"WARNING: {path} is not marked complete — the writing job did not "
        f"finish.\n"
        f"         Trailing members are unwritten and will render as a flat "
        f"panel at the colour-scale floor.\n"
        f"         Run member_field_qc.py on this store before trusting the "
        f"videos."
    )


# ── Frame-grid layout ──────────────────────────────────────────────────────
#
# The 12-panel figure from the parked D3 design: one panel per frame, in the
# model's own output unit, with the pinned frames marked.  Given-vs-invented is
# the whole point of the figure — a reader who cannot see which panels were
# constraints cannot read it — so the pinned panels get a coloured border and
# name the ERA5 field they were pinned to.

GRID_ROWS, GRID_COLS = 3, 4
PINNED_EDGE_COLOR = "#c1272d"
PINNED_EDGE_WIDTH = 3.0
FREE_EDGE_COLOR = "#999999"
FREE_EDGE_WIDTH = 0.8


def _frame_provenance(ens_root, n_frames):
    """Per-frame ``(is_pinned, source_label)`` from the store's own attrs.

    ``conditioning_frames`` names which frames were pinned; ``custom_frames``
    maps a frame to the ERA5 stamp its field was borrowed from.  Reading both
    from the zarr rather than the case YAML keeps the figure honest about what
    was actually run.

    The borrowed stamp matters: a custom frame's field comes from another date
    entirely, while its lead time still says ``start + 6h*frame``.  Labelling
    frame 11 of the Atlantic teleport with its natural 2010-09-05 valid time
    would name a state that appears nowhere in the panel.
    """
    conditioning = ens_root.attrs.get("conditioning_frames") or []
    custom = ens_root.attrs.get("custom_frames") or {}
    start_date = ens_root.attrs.get("start_date")
    pinned = {int(f) for f in conditioning}

    def _stamp(iso):
        # "2013-09-12T18:00" -> "2013-09-12 18Z".  Slicing, not rstrip: the
        # obvious rstrip(":00") also eats the hour of any frame pinned at 00Z,
        # which is exactly frame 0 of the Atlantic teleport.
        text = str(iso)
        date, _, clock = text.partition("T")
        return f"{date} {clock[:2]}Z" if clock else date

    labels = []
    for f in range(n_frames):
        if f not in pinned:
            labels.append((False, None))
            continue
        # A custom frame borrows its field from another date; an ordinary
        # pinned frame is the run's own start_date.
        borrowed = custom.get(str(f), custom.get(f))
        if borrowed is not None:
            labels.append((True, f"ERA5 {_stamp(borrowed)}"))
        elif f == 0 and start_date:
            labels.append((True, f"ERA5 {_stamp(start_date)}"))
        else:
            labels.append((True, "ERA5"))
    return labels


def _sort_by_plot_lon(fields, lon):
    """Convert ``lon`` to the [-180, 180) convention and reorder ``fields``.

    Each field is indexed on its trailing axis, so this handles
    ``[n_frames, n_lat, n_lon]`` and ``[n_lat, n_lon]`` alike.  Returns
    ``(sorted_fields, plot_lon)``.
    """
    plot_lon = np.where(lon >= 180, lon - 360, lon)
    order = np.argsort(plot_lon)
    return [np.asarray(f)[..., order] for f in fields], plot_lon[order]


def _load_wind_overlay(ens_root, overlay, member, n_frames, hpx_level):
    """Load a wind overlay's ``(u, v, lat, plot_lon)`` for one member.

    ``u``/``v`` come back as ``[n_frames, n_lat, n_lon]`` on the same lat/lon
    grid and the same longitude convention as the panels' field — the stored
    components themselves, with no averaging, scaling or other combination
    step.  There is deliberately no arithmetic here to defend: what the arrows
    show is the variable named in :data:`WIND_OVERLAYS`, which is also what the
    suptitle says they are.
    """
    spec = WIND_OVERLAYS[overlay]
    try:
        u, lat, lon = load_field(
            ens_root, spec["u"], member, n_frames, is_era5=False, hpx_level=hpx_level
        )
        v, _, _ = load_field(
            ens_root, spec["v"], member, n_frames, is_era5=False, hpx_level=hpx_level
        )
    except ValueError as exc:
        # Both are in STORED_VARIABLES, so this means an older or hand-built
        # store — name the overlay that wanted them.
        raise SystemExit(
            f"--wind-overlay {overlay} needs {spec['u']}/{spec['v']}: {exc}"
        ) from exc

    (u, v), plot_lon = _sort_by_plot_lon([u, v], lon)
    return u, v, lat, plot_lon


def _monotonic_slice(coord, lo, hi):
    """Contiguous slice of a monotonic coordinate covering ``[lo, hi]``.

    A mask on a monotonic array is contiguous, so its first and last true
    index bound the slice — which keeps the result a strided view, as
    ``draw_quiver``'s meshgrid subsampling needs, whether the coordinate
    ascends or descends.  Falls back to the full range when nothing is inside,
    so a view box off the grid degrades to "all the arrows" rather than none.
    """
    inside = np.flatnonzero((coord >= lo) & (coord <= hi))
    if inside.size == 0:
        return slice(None)
    return slice(int(inside[0]), int(inside[-1]) + 1)


def _crop_to_extent(u, v, lat, lon, extent):
    """Restrict an overlay grid to the panels' view box, plus a small margin.

    A member field arrives global, while the panels show one box, and both the
    arrow stride and the shared reference speed are derived from whatever grid
    they are handed:

    * ``quiver_step`` targets ~22 arrows across the grid's longer dimension.
      On a global grid that is ~22 arrows around the planet — one or two inside
      the view.
    * the reference speed sets the arrow length for every panel.  Taken
      globally it would be set by the midlatitude jet, rendering the in-box
      arrows as stubs.

    Reading a global field against a range chosen for the view box is exactly
    the trap the member QC screen fell into, so crop first, then derive.
    """
    if extent is None:
        return u, v, lat, lon
    lon_min, lon_max, lat_min, lat_max = (float(x) for x in extent)
    lon_pad = WIND_OVERLAY_PAD_FRAC * (lon_max - lon_min)
    lat_pad = WIND_OVERLAY_PAD_FRAC * (lat_max - lat_min)
    lat_sl = _monotonic_slice(lat, lat_min - lat_pad, lat_max + lat_pad)
    lon_sl = _monotonic_slice(lon, lon_min - lon_pad, lon_max + lon_pad)
    return (
        u[:, lat_sl, lon_sl],
        v[:, lat_sl, lon_sl],
        lat[lat_sl],
        lon[lon_sl],
    )


def _prepare_wind_overlay(ens_root, overlay, member, n_frames, hpx_level, extent):
    """Everything the panel loop needs to draw one comparable quiver per frame.

    Returns ``None`` when no overlay was requested, else a dict carrying the
    cropped ``u``/``v``/``lat``/``lon``, the ``scale`` and ``stride`` shared by
    every panel, the reference-key magnitude, and the label naming what the
    arrows are.
    """
    if overlay == "none":
        return None

    u, v, lat, lon = _load_wind_overlay(ens_root, overlay, member, n_frames, hpx_level)
    u, v, lat, lon = _crop_to_extent(u, v, lat, lon, extent)

    # One scale for all 12 panels, from the 90th-percentile in-box speed over
    # every frame — the same reference the synoptic-PCA figure derives, so one
    # arrow length means one speed throughout the figure.  Frame-to-frame
    # comparability is the whole point: a per-panel scale would hide exactly
    # the change in the flow the figure exists to show.
    ref_speed = float(np.nanpercentile(np.hypot(u, v), 90))
    if not np.isfinite(ref_speed) or ref_speed <= 0:
        # A dead member (an unwritten tail renders as exact zeros) would give a
        # zero scale and a division by zero inside matplotlib.  _warn_if_incomplete
        # has already said so; keep the panels renderable.
        ref_speed = 1.0
    return {
        "u": u,
        "v": v,
        "lat": lat,
        "lon": lon,
        "scale": ref_speed / QUIVER_REF_LEN_FRAC,
        "stride": quiver_step(lat, lon),
        "key_mag": nice_speed(ref_speed),
        "label": WIND_OVERLAYS[overlay]["label"],
    }


def plot_member_grid(ens_root, era5_root, var, member, args):
    """Render all 12 frames of one member as a labelled panel grid."""
    if "hpx" not in ens_root.array_keys():
        raise SystemExit(
            "plot_ensemble_member.py requires HEALPix ensemble zarrs; "
            f"got arrays {sorted(ens_root.array_keys())}"
        )
    hpx_level = _hpx_level_from_root(ens_root)

    n_members = ens_root["data"].shape[0]
    if member >= n_members:
        raise ValueError(
            f"Member {member} out of range (ensemble has {n_members} members)"
        )
    n_frames = ens_root["data"].shape[2]

    display_label = get_display_label(var)
    ens_all, lat, lon = load_field(
        ens_root, var, member, n_frames, is_era5=False, hpx_level=hpx_level
    )

    (ens_all,), plot_lon = _sort_by_plot_lon([ens_all], lon)

    vmin = args.vmin if args.vmin is not None else float(np.nanmin(ens_all))
    vmax = args.vmax if args.vmax is not None else float(np.nanmax(ens_all))

    wind = _prepare_wind_overlay(
        ens_root,
        getattr(args, "wind_overlay", "none") or "none",
        member,
        n_frames,
        hpx_level,
        args.extent,
    )
    wind_key_done = False

    time_labels = get_valid_time_labels(ens_root, n_frames, is_era5=False)
    provenance = _frame_provenance(ens_root, n_frames)

    fig, axes = plt.subplots(
        GRID_ROWS,
        GRID_COLS,
        figsize=(4.2 * GRID_COLS, 3.4 * GRID_ROWS),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    lon_grid, lat_grid = np.meshgrid(plot_lon, lat)

    im = None
    for f in range(n_frames):
        ax = flat_axes[f]
        im = ax.pcolormesh(
            lon_grid,
            lat_grid,
            ens_all[f],
            transform=ccrs.PlateCarree(),
            cmap=args.cmap,
            vmin=vmin,
            vmax=vmax,
        )
        # Geographic reference furniture, matching the video layout and every
        # other map figure in the pipeline: coastlines, national borders, and
        # (only when zoomed to a region, where they are signal rather than
        # noise) state/province lines.
        ax.add_feature(
            cfeature.COASTLINE, linewidth=0.6, edgecolor=args.coastline_color
        )
        ax.add_feature(
            cfeature.BORDERS,
            linestyle=":",
            linewidth=0.5,
            edgecolor=args.coastline_color,
        )
        if args.extent is not None:
            ax.set_extent(args.extent, crs=ccrs.PlateCarree())
            ax.add_feature(
                cfeature.STATES, linewidth=0.3, edgecolor=args.coastline_color
            )

        # Labels on the outer edges only — twelve fully labelled panels would
        # cost more legibility than they buy, and interior panels share their
        # neighbours' axes.
        # "Nothing drawn below me" rather than "I am in the last row", so a
        # truncated store (fewer than GRID_ROWS*GRID_COLS frames) still labels
        # its bottom-most visible panel in every column.
        draw_graticule(
            ax,
            show_xlabels=f + GRID_COLS >= n_frames,
            show_ylabels=f % GRID_COLS == 0,
            color=args.gridline_color,
            max_ticks=4,  # a twelfth of a page fits about four per axis
        )

        # Flow arrows, if asked for: one shared scale across all panels and a
        # single reference key on the first panel that drew any, so the arrows
        # are comparable frame to frame.  No detection step anywhere — a quiver
        # needs none, and the figure design bans a tracker.
        if wind is not None:
            handle = draw_quiver(
                ax,
                wind["u"][f],
                wind["v"][f],
                wind["lat"],
                wind["lon"],
                scale=wind["scale"],
                step=wind["stride"],
            )
            if handle is not None and not wind_key_done and wind["key_mag"] > 0:
                draw_quiver_key(
                    ax, handle, wind["key_mag"], fontsize=TICK_LABEL_SIZE_DENSE
                )
                wind_key_done = True

        is_pinned, source = provenance[f]
        # Panel titles carry the frame index and valid time; a pinned panel
        # replaces its valid time with the stamp of the field it was pinned
        # to, which is the only date visible in it.
        #
        # y=1.0 is required, not cosmetic: see draw_graticule's caller
        # obligation — cartopy's Gridliner resolves an auto-placed title to
        # `inf` on a labelled axes, which drops the title and breaks the tight
        # bbox.  It reproduces the automatic placement when nothing sits above.
        if is_pinned:
            ax.set_title(
                f"F{f} — PINNED\n{source}",
                fontsize=TITLE_SIZE_DENSE,
                color=PINNED_EDGE_COLOR,
                fontweight="bold",
                y=1.0,
            )
        else:
            ax.set_title(
                f"F{f} — free\n{time_labels[f]}", fontsize=TITLE_SIZE_DENSE, y=1.0
            )

        # cartopy exposes the map frame as the "geo" spine.
        spine = ax.spines["geo"]
        spine.set_edgecolor(PINNED_EDGE_COLOR if is_pinned else FREE_EDGE_COLOR)
        spine.set_linewidth(PINNED_EDGE_WIDTH if is_pinned else FREE_EDGE_WIDTH)

    for ax in flat_axes[n_frames:]:
        ax.set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="horizontal",
        shrink=0.5,
        pad=0.02,
        aspect=40,
        label=display_label,
    )
    cbar.ax.xaxis.label.set_fontsize(COLORBAR_LABEL_SIZE_DENSE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE_DENSE)

    title = getattr(args, "grid_title", None) or display_label
    # Which wind the arrows are belongs in the figure, not only in the
    # filename: a reader looking at the panel alone must be able to say what
    # the arrows show, and a reviewer must be able to check it.
    suptitle = f"{title}  —  member {member}"
    if wind is not None:
        suptitle += f"\narrows: {wind['label']}"
    fig.suptitle(suptitle, fontsize=SUPTITLE_SIZE_DENSE)

    out_name = args.output or f"{var}_member{member}_grid.png"
    out_path = os.path.join(args.output_dir, out_name)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_single(args):
    """Generate an animation for a single set of arguments."""
    os.makedirs(args.output_dir, exist_ok=True)

    ens_root = zarr.open_group(args.ensemble_zarr, mode="r")
    era5_root = zarr.open_group(args.era5_zarr, mode="r")
    _warn_if_incomplete(ens_root, args.ensemble_zarr)

    renderer = plot_member_grid if args.layout == "grid" else animate_member
    out_path = renderer(ens_root, era5_root, args.variable, args.member, args)
    print(f"Saved: {out_path}")


# ── Event mode ─────────────────────────────────────────────────────────────


def _spec_to_namespace(spec):
    """Turn a fully-merged plot spec dict into an argparse-style Namespace."""
    ns = argparse.Namespace()
    ns.ensemble_zarr = spec["ensemble_zarr"]
    ns.era5_zarr = spec["era5_zarr"]
    ns.variable = spec["variable"]
    ns.member = spec["member"]
    ns.cmap = spec.get("cmap", "viridis")
    ns.vmin = spec.get("vmin")
    ns.vmax = spec.get("vmax")
    ns.coastline_color = spec.get("coastline_color", "black")
    ns.gridline_color = spec.get("gridline_color", "gray")
    ns.extent = spec.get("extent")
    ns.fps = int(spec.get("fps", 2))
    ns.output = spec.get("output")
    ns.output_dir = spec["output_dir"]
    ns.dpi = int(spec.get("dpi", 150))
    ns.layout = spec.get("layout", "video")
    ns.grid_title = spec.get("grid_title")
    ns.wind_overlay = spec.get("wind_overlay", "none")
    return ns


def _expand_event_plots(
    case,
    palettes,
    *,
    only_modes=None,
    only_variable=None,
    ensemble_size_override=None,
    members_override=None,
    layout="video",
    extent_override=None,
    vmin_override=None,
    vmax_override=None,
    wind_overlay="none",
):
    """Expand a case YAML into a flat list of per-animation spec dicts."""
    var_defaults = palettes.get("variable_defaults", {})
    plots_cfg = case.get("plots") or {}

    members = plots_cfg.get("members", [0, 1, 2, 3])
    # Member animations are presentational: their viewport defaults to
    # regions.impact (overridable via plots.view -- a region name or a custom
    # [lon_min, lon_max, lat_min, lat_max]).  Per-variable plots.variables[].extent
    # still overrides this below.
    extent = resolve_view(case)
    fps = plots_cfg.get("fps", 2)
    dpi = plots_cfg.get("dpi", 150)
    variables = plots_cfg.get("variables")
    if not variables:
        variables = [{"name": v} for v in palettes["variable_sets"]["standard"]]
    if only_variable:
        variables = [v for v in variables if v.get("name") == only_variable]
        if not variables:
            raise ValueError(
                f"case '{case['name']}': no variable '{only_variable}' "
                f"configured in plots.variables"
            )

    size = (
        ensemble_size_override
        if ensemble_size_override is not None
        else int(case["ensemble"]["size"])
    )

    # Bounded by the configured ensemble size so a typo'd range fails before
    # any rendering; animate_member re-checks against what is actually on disk.
    if members_override:
        members = parse_member_spec(members_override, size)

    specs = []
    for mode, _frames in iter_modes(case):
        if only_modes and mode not in only_modes:
            continue
        ens_zarr = ensemble_zarr_path(case["base"], mode, size)
        era5_zarr = era5_zarr_path(case["base"], mode)
        output_dir = f"output/{case['name']}_plots/{mode}"

        for var_entry in variables:
            name = var_entry["name"]
            base_style = dict(var_defaults.get(name, {}))
            # Per-case overrides beat palette defaults.
            merged = {**base_style, **var_entry}
            merged.pop("name", None)

            for member in members:
                # A CLI --extent beats everything, including a per-variable
                # entry, so a domain can be tried without editing the YAML.
                if extent_override is not None:
                    panel_extent = extent_override
                else:
                    panel_extent = merged.get("extent", extent)
                # The overlay goes in the filename so an overlaid render sits
                # beside the plain grid instead of overwriting it — which
                # version reaches the SI is the author's call to make by
                # looking at both.
                if layout != "grid":
                    suffix = ".mp4"
                elif wind_overlay == "none":
                    suffix = "_grid.png"
                else:
                    suffix = f"_grid_{wind_overlay}.png"
                # Colour-scale overrides land after **merged for the same
                # reason as the extent: iterating on the scale is the fastest
                # way to make small panels readable, and it should not require
                # editing the shared palette.
                scale_override = {}
                if vmin_override is not None:
                    scale_override["vmin"] = vmin_override
                if vmax_override is not None:
                    scale_override["vmax"] = vmax_override
                specs.append(
                    {
                        "variable": name,
                        "member": int(member),
                        "ensemble_zarr": ens_zarr,
                        "era5_zarr": era5_zarr,
                        "output_dir": output_dir,
                        "output": f"{name}_member{member}{suffix}",
                        "fps": merged.get("fps", fps),
                        "dpi": merged.get("dpi", dpi),
                        "layout": layout,
                        "wind_overlay": wind_overlay,
                        "grid_title": case["display_name"],
                        **merged,
                        "extent": panel_extent,
                        **scale_override,
                    }
                )
    return specs


def plot_event(
    event_name,
    *,
    only_modes=None,
    only_variable=None,
    ensemble_size_override=None,
    members_override=None,
    layout="video",
    extent_override=None,
    vmin_override=None,
    vmax_override=None,
    wind_overlay="none",
):
    """Render every configured (mode, variable, member) plot for a case."""
    case = load_case(event_name)
    palettes = load_palettes()

    specs = _expand_event_plots(
        case,
        palettes,
        only_modes=only_modes,
        only_variable=only_variable,
        ensemble_size_override=ensemble_size_override,
        members_override=members_override,
        layout=layout,
        extent_override=extent_override,
        vmin_override=vmin_override,
        vmax_override=vmax_override,
        wind_overlay=wind_overlay,
    )
    if not specs:
        print(
            f"No plots to render for '{event_name}' "
            f"(check plots.variables and plots.members in the case YAML)."
        )
        return

    print(
        f"Rendering {len(specs)} animation(s) for case '{event_name}' "
        f"({case['display_name']})"
    )
    for i, spec in enumerate(specs):
        print(f"\n[{i + 1}/{len(specs)}] {spec['output_dir']}/{spec['output']}")
        run_single(_spec_to_namespace(spec))
    print(f"\nDone — {len(specs)} animation(s) generated.")


def main():
    args = parse_args()

    if args.list:
        print("Available cases:")
        for n in list_cases():
            print(f"  {n}")
        return

    if args.event:
        plot_event(
            args.event,
            only_modes=args.mode,
            only_variable=args.variable,
            ensemble_size_override=args.ensemble_size,
            members_override=args.members,
            layout=args.layout,
            extent_override=args.extent,
            vmin_override=args.vmin,
            vmax_override=args.vmax,
            wind_overlay=args.wind_overlay,
        )
    else:
        run_single(args)


if __name__ == "__main__":
    main()
