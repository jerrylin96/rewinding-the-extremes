#!/usr/bin/env python3
"""Explain the pinned-end anchor classification of conditioned TC tracks.

The synoptic-PCA track-stats report classifies every conditioned ensemble
member by whether its storm reaches the ERA5 fix at the *pinned* end of the
window (see ``tc_track_targets._anchor_check``): ``anchored`` (the tracker's
nominal primary path -- raw ``path_id=0`` -- reaches the anchor), ``alt_path``
(the storm is found within the anchor radius but on a secondary path), and
``lost`` (no stitched path reaches the anchor at all).  For end-conditioned
Sandy that split is 812 / 144 / 44 -- numbers the paper quotes but that, until
now, could only be inspected as counts.

This script makes the three populations inspectable.  It re-derives the
classification from the same raw all-paths ``tc_tracks_<mode>.parquet`` with
the same shared helpers (one source of truth -- the counts here match the
track-stats report exactly), then writes, into per-category subfolders of the
tracker's ``tc_tracks`` directory:

1. ``member_anchor_categories.csv`` -- one row per member: category, number
   of stitched paths, each path's closest approach to the anchor, the raw
   primary path's fix count / first / last step, and the best
   (closest-approach) path's identity and stats.
2. ``anchor_summary.txt`` -- counts, per-category distance/length
   distributions, and a guide to the artifacts.
3. ``<category>/overview.png`` -- all members of the category on one map
   (nominal primary paths + the secondary path that actually holds the
   storm), so the failure mode reads at a glance.
4. ``<category>/track_maps_pNN.png`` -- per-member panels (auto-zoomed, all
   stitched paths, ERA5 track, anchor radius ring) for every member of the
   category.
5. ``<category>/msl_member_XXXX.mp4`` -- for a representative subset (spread
   across the category's closest-approach range), an animation of that
   member's own MSL field with its stitched paths, the ERA5 track, the
   anchor ring, the per-frame minimum-MSL location inside the track box,
   and -- when the TempestExtremes binaries are available -- the raw
   DetectNodes candidates at each frame, with and without the warm-core
   thickness criterion.  The candidate overlay separates "DetectNodes
   rejected the low (warm-core failure during extratropical transition)"
   from "StitchNodes could not stitch a long-enough path".

Requires only the parquet for (1)-(4); (5) additionally needs
``--ensemble-zarr`` (and ``--era5-zarr`` to reproduce the pinned-frame ERA5
replacement the tracker saw).

Usage (normally submitted via dispatch_tc_track_anchor_diagnostics.py):
    python tc_track_anchor_diagnostics.py \
        --tracks-parquet .../tc_tracks/tempest/tc_tracks_end.parquet \
        --output-dir .../tc_tracks/tempest/anchor_diagnostics_end \
        --mode end --anchor-frame 11 \
        --lon-min -85 --lon-max -60 --lat-min 25 --lat-max 45 \
        --ensemble-zarr .../end/ensemble_f11_n1000.zarr \
        --era5-zarr .../end/era5_reference.zarr --warm-core \
        --title "Hurricane Sandy"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
SHARED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "_shared"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, SHARED_DIR)
sys.path.insert(0, PARENT_DIR)

from tc_track_targets import (  # noqa: E402
    ANCHOR_FRAME_WINDOW,
    ANCHOR_RADIUS_KM,
    EARTH_RADIUS_M,
    _fix_at_frame,
    _norm_lon,
    anchor_min_distances,
)
from tc_tracks_io import load_tracks_parquet  # noqa: E402

CATEGORIES: Tuple[str, ...] = ("anchored", "alt_path", "lost")
CATEGORY_LABELS: Dict[str, str] = {
    "anchored": "path 0 reaches the anchor (normal tracking)",
    "alt_path": "storm within the anchor radius, but on a secondary path",
    "lost": "no stitched path reaches the anchor",
}

# Consistent colors across overview maps, per-member panels, and animations.
PATH0_COLOR = "tab:blue"
BEST_PATH_COLOR = "tab:orange"
OTHER_PATH_COLOR = "0.6"
ANCHOR_COLOR = "crimson"

# Degrees of padding around the auto-zoomed per-member extent.
EXTENT_PAD_DEG = 3.0

MSL_VMIN_HPA = 920.0
MSL_VMAX_HPA = 1030.0


# ---------------------------------------------------------------------------
# Classification (parquet only -- no matplotlib / zarr imports)
# ---------------------------------------------------------------------------


def load_all_paths(parquet_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Raw all-paths tracks from the parquet, normalized for analysis.

    Returns ``(ensemble [N, P, T, 4], era5 [T, 4])`` with longitudes on
    [-180, 180) and MSL in hPa -- the same normalization
    ``tc_track_targets.load_track_targets`` applies, so distances computed
    here match the track-stats report.
    """
    tracks = load_tracks_parquet(parquet_path)
    ens = tracks.get("ensemble")
    era5 = tracks.get("era5")
    if ens is None or era5 is None:
        raise ValueError(f"{parquet_path}: missing ensemble or era5 tracks")
    ens = ens.astype(np.float64)
    ens[..., 1] = _norm_lon(ens[..., 1])
    ens[..., 2] /= 100.0
    era5_track = era5[0, 0].astype(np.float64)
    era5_track[:, 1] = _norm_lon(era5_track[:, 1])
    era5_track[:, 2] /= 100.0
    return ens, era5_track


def classify_members(
    ens: np.ndarray,
    era5_track: np.ndarray,
    anchor_frame: int,
    radius_km: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Anchor-classify every member, keeping the per-path distances.

    Thin wrapper over the shared :func:`anchor_min_distances` (the same
    helper ``_anchor_check`` thresholds), so the categories here reproduce
    the track-stats counts exactly.  Returns ``(category [N] str,
    dmin [N, P] km, anchor_fix [3], anchor_fell_back)`` where ``dmin`` is
    each path's closest approach to the anchor inside the
    ``+-ANCHOR_FRAME_WINDOW`` frame window (``np.inf`` = no fix there).
    """
    anchor_fix, fell_back = _fix_at_frame(era5_track[:, :3], anchor_frame)
    if not np.isfinite(anchor_fix[:2]).all():
        raise ValueError(
            f"ERA5 has no usable anchor fix at frame {anchor_frame} "
            f"(nor any valid fallback fix)"
        )
    dmin = anchor_min_distances(
        ens, float(anchor_fix[0]), float(anchor_fix[1]), anchor_frame
    )
    if dmin is None:
        raise ValueError(
            f"anchor window around frame {anchor_frame} lies wholly beyond "
            f"the parquet's {ens.shape[2]}-step lead axis"
        )
    path0_ok = dmin[:, 0] <= radius_km
    any_ok = (dmin <= radius_km).any(axis=1)
    category = np.where(path0_ok, "anchored", np.where(any_ok, "alt_path", "lost"))
    return category.astype(object), dmin, anchor_fix, fell_back


def _path_step_stats(valid: np.ndarray) -> Tuple[int, int, int]:
    """``(n_fixes, first_step, last_step)`` for one path's validity mask."""
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return 0, -1, -1
    return int(idx.size), int(idx[0]), int(idx[-1])


def member_table(
    ens: np.ndarray, category: np.ndarray, dmin: np.ndarray
) -> pd.DataFrame:
    """Per-member classification table (one row per member).

    ``best_path_id`` is the path with the closest approach to the anchor
    (-1 when no path has a fix inside the anchor window at all);
    infinite distances are written as NaN so the CSV stays readable.
    """
    valid = np.isfinite(ens[..., 0]) & np.isfinite(ens[..., 1])  # [N, P, T]
    n_paths = (valid.any(axis=2)).sum(axis=1)  # [N]
    best_path = np.argmin(dmin, axis=1)  # [N]

    rows: List[Dict[str, object]] = []
    for m in range(ens.shape[0]):
        b = int(best_path[m])
        has_best = bool(np.isfinite(dmin[m, b]))
        n0, f0, l0 = _path_step_stats(valid[m, 0])
        nb, fb, lb = _path_step_stats(valid[m, b]) if has_best else (0, -1, -1)
        rows.append(
            {
                "member": m,
                "category": str(category[m]),
                "n_paths": int(n_paths[m]),
                "path0_min_anchor_km": (
                    float(dmin[m, 0]) if np.isfinite(dmin[m, 0]) else float("nan")
                ),
                "path0_n_fixes": n0,
                "path0_first_step": f0,
                "path0_last_step": l0,
                "best_path_id": b if has_best else -1,
                "best_path_min_anchor_km": (
                    float(dmin[m, b]) if has_best else float("nan")
                ),
                "best_path_n_fixes": nb,
                "best_path_first_step": fb,
                "best_path_last_step": lb,
            }
        )
    return pd.DataFrame(rows)


def pick_spread(members: np.ndarray, key: np.ndarray, n: int) -> List[int]:
    """Deterministic subset of ``n`` members spread across sorted ``key``.

    Sorts ``members`` by ``key`` (stable; ``inf`` -- e.g. "no fix in the
    anchor window at all" -- sorts last, so the most extreme failures are
    always represented) and takes evenly spaced entries including both
    ends.  Returns everything when ``n >= len(members)``.
    """
    if n <= 0 or members.size == 0:
        return []
    order = np.argsort(key, kind="stable")
    ordered = np.asarray(members)[order]
    if n >= ordered.size:
        return [int(m) for m in ordered]
    idx = np.unique(np.round(np.linspace(0, ordered.size - 1, n)).astype(int))
    return [int(m) for m in ordered[idx]]


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def _dist_quantiles(vals: np.ndarray) -> str:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return "all inf (no fix in the anchor window)"
    q = np.percentile(finite, [5, 50, 95])
    extra = f" ({vals.size - finite.size} inf)" if finite.size < vals.size else ""
    return f"p5={q[0]:.0f} p50={q[1]:.0f} p95={q[2]:.0f} km{extra}"


def write_summary(
    path: Path,
    *,
    title: str,
    mode: str,
    parquet_path: str,
    df: pd.DataFrame,
    dmin: np.ndarray,
    anchor_fix: np.ndarray,
    anchor_frame: int,
    anchor_fell_back: bool,
    radius_km: float,
    animated: Dict[str, List[int]],
    candidates_note: str,
) -> None:
    """Human-readable companion to ``member_anchor_categories.csv``."""
    n = len(df)
    lines = [
        f"TC track pinned-end anchor diagnostics -- {title}",
        f"mode={mode}, parquet={parquet_path}",
        f"n_members={n}, n_paths={dmin.shape[1]} (tracker path slots)",
        f"Anchor: ERA5 fix {anchor_fix[0]:.2f}N {anchor_fix[1]:.2f}E at frame "
        f"{anchor_frame} +-{ANCHOR_FRAME_WINDOW}, radius {radius_km:g} km"
        + (
            " [ERA5 anchor fix fell back to nearest valid frame]"
            if anchor_fell_back
            else ""
        ),
        "",
        "Classification (same definition as tc_track_targets._anchor_check,",
        "so the counts match the synoptic-PCA track-stats report):",
    ]
    for cat in CATEGORIES:
        k = int((df["category"] == cat).sum())
        lines.append(
            f"  {cat:<9}: {k:4d} ({100.0 * k / max(n, 1):.1f}%)"
            f"  -- {CATEGORY_LABELS[cat]}"
        )

    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        if len(sub) == 0:
            continue
        lines += [
            "",
            f"[{cat}] n={len(sub)}",
            "  path 0 closest approach to anchor : "
            + _dist_quantiles(sub["path0_min_anchor_km"].to_numpy()),
            "  best path closest approach        : "
            + _dist_quantiles(sub["best_path_min_anchor_km"].to_numpy()),
            f"  path 0 length (fixes): median "
            f"{sub['path0_n_fixes'].median():.0f}, "
            f"last step: median {sub['path0_last_step'].median():.0f}",
            f"  members with no stitched path at all: "
            f"{int((sub['n_paths'] == 0).sum())}",
        ]
        if animated.get(cat):
            lines.append(
                "  animated members: " + ", ".join(str(m) for m in animated[cat])
            )

    lines += [
        "",
        "Artifacts:",
        "  member_anchor_categories.csv  per-member classification, per-path",
        "                                closest approach, path lengths",
        "  <category>/overview.png       all members of the category on one",
        "                                map (fixed track-box view)",
        "  <category>/track_maps_pNN.png per-member panels, auto-zoomed so a",
        "                                nominal primary path that wandered",
        "                                to a different system stays visible",
        "                                (anchored capped by --map-anchored,",
        "                                spread across its anchor distances)",
        "  <category>/msl_member_*.mp4   per-member MSL animation for a",
        "                                subset spread across the category's",
        "                                closest-approach range",
        f"  candidate overlay: {candidates_note}",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"[anchor-diag] wrote {path}")


# ---------------------------------------------------------------------------
# Maps (matplotlib / cartopy imported lazily)
# ---------------------------------------------------------------------------


def _geodesic_circle(
    lat0: float, lon0: float, radius_km: float, n: int = 181
) -> Tuple[np.ndarray, np.ndarray]:
    """Points (lats, lons) of a great-circle ring around ``(lat0, lon0)``."""
    ang = radius_km * 1e3 / EARTH_RADIUS_M
    bearings = np.linspace(0.0, 2.0 * np.pi, n)
    p0 = np.radians(lat0)
    lats = np.arcsin(
        np.sin(p0) * np.cos(ang) + np.cos(p0) * np.sin(ang) * np.cos(bearings)
    )
    lons = np.radians(lon0) + np.arctan2(
        np.sin(bearings) * np.sin(ang) * np.cos(p0),
        np.cos(ang) - np.sin(p0) * np.sin(lats),
    )
    return np.degrees(lats), _norm_lon(np.degrees(lons))


def _member_extent(
    member_paths: np.ndarray,
    era5_track: np.ndarray,
    box: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Auto-zoomed extent: track box grown to include every member fix.

    A misassociated primary path can sit thousands of km from the track
    box (the paper quotes a 15,400 km 95th-percentile free-end distance);
    growing the extent keeps that path on the map, which *is* the
    diagnosis.  Assumes fixes on [-180, 180) that do not straddle the
    antimeridian (true for the Atlantic cases this pipeline covers).
    """
    lon_min, lon_max, lat_min, lat_max = box
    lats = np.concatenate([member_paths[..., 0].ravel(), era5_track[:, 0]])
    lons = np.concatenate([member_paths[..., 1].ravel(), era5_track[:, 1]])
    lats, lons = lats[np.isfinite(lats)], lons[np.isfinite(lons)]
    if lats.size:
        lon_min = min(lon_min, float(lons.min()))
        lon_max = max(lon_max, float(lons.max()))
        lat_min = min(lat_min, float(lats.min()))
        lat_max = max(lat_max, float(lats.max()))
    return (
        max(lon_min - EXTENT_PAD_DEG, -180.0),
        min(lon_max + EXTENT_PAD_DEG, 180.0),
        max(lat_min - EXTENT_PAD_DEG, -90.0),
        min(lat_max + EXTENT_PAD_DEG, 90.0),
    )


def _draw_track_context(
    ax,
    era5_track: np.ndarray,
    anchor_fix: np.ndarray,
    radius_km: float,
    box: Tuple[float, float, float, float],
) -> None:
    """Shared per-axes context: coastline, track box, ERA5 track, anchor."""
    import cartopy.crs as ccrs
    from plot_style import draw_domain_box

    pc = ccrs.PlateCarree()
    ax.coastlines(linewidth=0.6, color="0.3")
    draw_domain_box(ax, [box[0], box[1], box[2], box[3]], linewidth=0.9)
    ev = np.isfinite(era5_track[:, 0]) & np.isfinite(era5_track[:, 1])
    if ev.sum() >= 2:
        ax.plot(
            era5_track[ev, 1],
            era5_track[ev, 0],
            color="black",
            linestyle="--",
            linewidth=1.4,
            marker="D",
            markersize=2.5,
            transform=pc,
            zorder=8,
        )
    ring_lat, ring_lon = _geodesic_circle(
        float(anchor_fix[0]), float(anchor_fix[1]), radius_km
    )
    ax.plot(
        ring_lon,
        ring_lat,
        color=ANCHOR_COLOR,
        linestyle=":",
        linewidth=1.2,
        transform=pc,
        zorder=8,
    )
    ax.plot(
        float(anchor_fix[1]),
        float(anchor_fix[0]),
        "*",
        color=ANCHOR_COLOR,
        markersize=11,
        markeredgecolor="black",
        markeredgewidth=0.5,
        transform=pc,
        zorder=9,
    )


def _path_color_lw(p: int, best: int, has_best: bool) -> Tuple[str, float, float]:
    """(color, linewidth, zorder) for path ``p`` of one member."""
    if p == 0:
        return PATH0_COLOR, 1.8, 6.0
    if has_best and p == best:
        return BEST_PATH_COLOR, 1.8, 5.0
    return OTHER_PATH_COLOR, 0.9, 4.0


def plot_member_grid(
    ens: np.ndarray,
    era5_track: np.ndarray,
    members: Sequence[int],
    df: pd.DataFrame,
    anchor_fix: np.ndarray,
    radius_km: float,
    box: Tuple[float, float, float, float],
    out_dir: Path,
    page_title: str,
    members_per_page: int,
) -> None:
    """Per-member track panels, paged into ``track_maps_pNN.png`` files."""
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    pc = ccrs.PlateCarree()
    n_cols = 4
    members = list(members)
    n_pages = (len(members) + members_per_page - 1) // members_per_page
    for page in range(n_pages):
        chunk = members[page * members_per_page : (page + 1) * members_per_page]
        n_rows = (len(chunk) + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(4.4 * n_cols, 3.4 * n_rows))
        for i, m in enumerate(chunk):
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection=pc)
            row = df.loc[df["member"] == m].iloc[0]
            best = int(row["best_path_id"])
            has_best = best >= 0
            ax.set_extent(_member_extent(ens[m], era5_track, box), crs=pc)
            _draw_track_context(ax, era5_track, anchor_fix, radius_km, box)
            for p in range(ens.shape[1]):
                pl = ens[m, p]
                pv = np.isfinite(pl[:, 0]) & np.isfinite(pl[:, 1])
                if not pv.any():
                    continue
                color, lw, z = _path_color_lw(p, best, has_best)
                ax.plot(
                    pl[pv, 1],
                    pl[pv, 0],
                    color=color,
                    linewidth=lw,
                    marker="o",
                    markersize=2.5,
                    transform=pc,
                    zorder=z,
                )
            d0 = row["path0_min_anchor_km"]
            db = row["best_path_min_anchor_km"]
            d0_s = f"{d0:.0f} km" if np.isfinite(d0) else "--"
            db_s = f"best p{best}={db:.0f} km" if has_best else "no path in window"
            ax.set_title(f"m{m:04d}  d0={d0_s}  {db_s}", fontsize=8)
        fig.suptitle(f"{page_title} (page {page + 1}/{n_pages})", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_png = out_dir / f"track_maps_p{page + 1:02d}.png"
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[anchor-diag] wrote {out_png}")


def plot_category_overview(
    ens: np.ndarray,
    era5_track: np.ndarray,
    members: Sequence[int],
    df: pd.DataFrame,
    anchor_fix: np.ndarray,
    radius_km: float,
    box: Tuple[float, float, float, float],
    out_png: Path,
    title: str,
) -> None:
    """One fixed-view map with every member of a category overlaid.

    Nominal primary paths draw in blue, the closest-approach secondary
    path (when it differs from path 0) in orange; paths that leave the
    fixed view simply exit the frame -- the per-member pages carry the
    auto-zoomed view.
    """
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection=pc)
    lon_min, lon_max, lat_min, lat_max = box
    ax.set_extent(
        [
            lon_min - 2 * EXTENT_PAD_DEG,
            lon_max + 2 * EXTENT_PAD_DEG,
            lat_min - 2 * EXTENT_PAD_DEG,
            lat_max + 2 * EXTENT_PAD_DEG,
        ],
        crs=pc,
    )
    _draw_track_context(ax, era5_track, anchor_fix, radius_km, box)
    for m in members:
        row = df.loc[df["member"] == m].iloc[0]
        best = int(row["best_path_id"])
        p0 = ens[m, 0]
        pv = np.isfinite(p0[:, 0]) & np.isfinite(p0[:, 1])
        if pv.any():
            ax.plot(
                p0[pv, 1],
                p0[pv, 0],
                color=PATH0_COLOR,
                linewidth=0.8,
                alpha=0.35,
                transform=pc,
                zorder=5,
            )
        if best > 0:
            pb = ens[m, best]
            bv = np.isfinite(pb[:, 0]) & np.isfinite(pb[:, 1])
            if bv.any():
                ax.plot(
                    pb[bv, 1],
                    pb[bv, 0],
                    color=BEST_PATH_COLOR,
                    linewidth=0.8,
                    alpha=0.5,
                    transform=pc,
                    zorder=6,
                )
    handles = [
        mlines.Line2D([], [], color=PATH0_COLOR, label="nominal primary (path 0)"),
        mlines.Line2D(
            [],
            [],
            color=BEST_PATH_COLOR,
            label="closest-approach secondary path",
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            linestyle="--",
            marker="D",
            markersize=3,
            label="ERA5 track",
        ),
        mlines.Line2D(
            [],
            [],
            color=ANCHOR_COLOR,
            linestyle=":",
            label=f"anchor radius ({radius_km:g} km)",
        ),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85)
    ax.set_title(title, fontsize=11)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[anchor-diag] wrote {out_png}")


# ---------------------------------------------------------------------------
# Per-member MSL animations (zarr / torch / TempestExtremes imported lazily)
# ---------------------------------------------------------------------------


def _detect_candidates_for_member(
    member_data: np.ndarray,
    variables: Sequence[str],
    lats: np.ndarray,
    lons: np.ndarray,
    timestep_hours: float,
    warm_core: bool,
    work_dir: Path,
    member_idx: int,
) -> Optional[Dict[str, List[np.ndarray]]]:
    """Raw DetectNodes candidates per step, with/without the warm-core test.

    Stages the member NetCDF once and runs DetectNodes per variant:
    ``msl-only`` (plain Z&U closed-contour criterion -- "where is a
    detectable low") and, when ``warm_core``, ``warm-core`` (the criterion
    the production tracking ran with -- "what survived the thickness
    test").  A candidate present in ``msl-only`` but absent in
    ``warm-core`` marks a warm-core rejection at that frame.  Returns None
    when the TempestExtremes binaries are not on PATH.
    """
    from tempest_extremes_runner import (
        DEFAULT_DETECT_FLAGS,
        DETECT_NODES_BIN,
        WARM_CORE_CONTOUR,
        _run_detect_nodes,
        _stage_member_netcdf,
        parse_candidates_file,
    )

    if shutil.which(DETECT_NODES_BIN) is None:
        return None

    variants = {"msl-only": dict(DEFAULT_DETECT_FLAGS)}
    if warm_core:
        flags = dict(DEFAULT_DETECT_FLAGS)
        flags["closedcontourcmd"] += ";" + WARM_CORE_CONTOUR
        variants["warm-core"] = flags

    n_steps = member_data.shape[0]
    nc_path = work_dir / f"anchor_diag_member_{member_idx:04d}.nc"
    out: Dict[str, List[np.ndarray]] = {}
    try:
        _stage_member_netcdf(
            member_data, variables, lats, lons, timestep_hours, nc_path
        )
        for name, flags in variants.items():
            cand_path = work_dir / (
                f"anchor_diag_member_{member_idx:04d}_{name}.candidates.txt"
            )
            try:
                _run_detect_nodes(nc_path, cand_path, flags)
                out[name] = parse_candidates_file(cand_path, n_steps, timestep_hours)
            finally:
                cand_path.unlink(missing_ok=True)
    finally:
        nc_path.unlink(missing_ok=True)
    return out


def animate_members(
    ens: np.ndarray,
    era5_track: np.ndarray,
    to_animate: Dict[str, List[int]],
    df: pd.DataFrame,
    anchor_fix: np.ndarray,
    radius_km: float,
    box: Tuple[float, float, float, float],
    out_root: Path,
    *,
    ensemble_zarr: str,
    era5_zarr: Optional[str],
    mode: str,
    warm_core: bool,
    with_candidates: bool,
    title: str,
    start_time_iso: Optional[str],
    tz_name: Optional[str],
) -> str:
    """Per-member MSL animations for the selected members of each category.

    Reads each member's own MSL field from the ensemble zarr (with the
    pinned frames replaced by ERA5 when ``era5_zarr`` is given, matching
    what the tracker saw), overlays the member's stitched paths, the ERA5
    track, the anchor ring, the per-frame minimum-MSL position inside the
    track box, and (optionally) the raw DetectNodes candidates.  Returns a
    one-line note describing whether the candidate overlay ran, for the
    summary report.
    """
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.animation as animation
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    import torch
    from _dispatch_lib import lead_hours, mode_frames, zarr_variable_index
    from local_time_axis import format_local_time_title
    from plot_ensemble_tc_tracks import (
        MSL_CMAP,
        _regrid_hpx_batch_to_latlon,
        _tempest_tracker_variables,
        load_era5_replacement,
        load_zarr_data,
    )
    from tempest_extremes_runner import infer_timestep_hours

    root, coords, time_steps = load_zarr_data(ensemble_zarr)
    hpx_level = coords.get("hpx_level")
    lats, lons = np.asarray(coords["lat"]), np.asarray(coords["lon"])
    lead_h = lead_hours(time_steps)

    # Monotonic [-180, 180) longitude axis for pcolormesh: the stores (and
    # the regrid target grid) use [0, 360), which normalizes to a wrapped,
    # non-monotonic axis.  Column-reorder once here; the raw grid is kept
    # for the DetectNodes staging, which must see the production layout.
    lon_norm = _norm_lon(lons)
    lon_order = np.argsort(lon_norm)
    lon_grid = lon_norm[lon_order]

    # Candidate detection needs the full tracker variable set; MSL-only
    # animations just need msl.
    if with_candidates:
        variables = _tempest_tracker_variables(warm_core=warm_core)
        try:
            var_indices = [zarr_variable_index(root, v) for v in variables]
        except Exception as exc:  # missing z300/z500 in archived stores
            print(
                f"[anchor-diag] candidate overlay disabled: {exc}; "
                f"falling back to msl-only animations."
            )
            with_candidates = False
    if not with_candidates:
        variables = ["msl"]
        var_indices = [zarr_variable_index(root, "msl")]
    msl_pos = variables.index("msl")

    era5_replacements = None
    if era5_zarr:
        era5_replacements = load_era5_replacement(
            era5_zarr, coords, variables, mode_frames(mode), hpx_level
        )

    timestep_hours = infer_timestep_hours(time_steps)

    # Track-box mask (on the reordered longitude axis) for the per-frame
    # minimum-MSL marker.
    lon_min, lon_max, lat_min, lat_max = box
    in_box_lat = (lats >= lat_min) & (lats <= lat_max)
    in_box_lon = (lon_grid >= lon_min) & (lon_grid <= lon_max)

    candidates_ran = False
    work_dir = Path(tempfile.mkdtemp(prefix="anchor_diag_"))
    pc = ccrs.PlateCarree()
    try:
        for cat, members in to_animate.items():
            for m in members:
                raw = root["data"]
                if raw.shape[1] == 1:
                    slab = np.asarray(raw[m, 0])  # [T, V(, ...spatial)]
                else:
                    slab = np.asarray(raw[m, :, 0])
                slab = slab[:, var_indices].astype(np.float32)
                if hpx_level is not None:
                    slab = (
                        _regrid_hpx_batch_to_latlon(
                            torch.from_numpy(slab).float(), hpx_level
                        )
                        .cpu()
                        .numpy()
                    )
                if era5_replacements:
                    for t, frame in era5_replacements.items():
                        slab[t] = frame.cpu().numpy()

                cands = None
                if with_candidates:
                    cands = _detect_candidates_for_member(
                        slab,
                        variables,
                        lats,
                        lons,
                        timestep_hours,
                        warm_core,
                        work_dir,
                        m,
                    )
                    if cands is None:
                        print(
                            "[anchor-diag] DetectNodes not on PATH; "
                            "animations continue without candidates."
                        )
                        with_candidates = False
                    else:
                        candidates_ran = True

                out_mp4 = out_root / cat / f"msl_member_{m:04d}.mp4"
                _animate_one_member(
                    m,
                    cat,
                    slab[:, msl_pos][..., lon_order],
                    cands,
                    ens[m],
                    era5_track,
                    df,
                    anchor_fix,
                    radius_km,
                    box,
                    lats,
                    lon_grid,
                    in_box_lat,
                    in_box_lon,
                    out_mp4,
                    title,
                    lead_h,
                    start_time_iso,
                    tz_name,
                    plt=plt,
                    animation=animation,
                    mlines=mlines,
                    pc=pc,
                    msl_cmap=MSL_CMAP,
                    format_local_time_title=format_local_time_title,
                )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if candidates_ran:
        variants = "msl-only vs warm-core variants" if warm_core else "msl-only"
        return f"DetectNodes candidates drawn per frame ({variants})"
    return "not run (disabled, binaries missing, or variables unavailable)"


def _animate_one_member(
    m: int,
    cat: str,
    msl: np.ndarray,
    cands: Optional[Dict[str, List[np.ndarray]]],
    member_paths: np.ndarray,
    era5_track: np.ndarray,
    df: pd.DataFrame,
    anchor_fix: np.ndarray,
    radius_km: float,
    box: Tuple[float, float, float, float],
    lats: np.ndarray,
    lon_grid: np.ndarray,
    in_box_lat: np.ndarray,
    in_box_lon: np.ndarray,
    out_mp4: Path,
    title: str,
    lead_h: np.ndarray,
    start_time_iso: Optional[str],
    tz_name: Optional[str],
    *,
    plt,
    animation,
    mlines,
    pc,
    msl_cmap,
    format_local_time_title,
) -> None:
    """Render one member's MSL + tracks + candidates animation."""
    row = df.loc[df["member"] == m].iloc[0]
    best = int(row["best_path_id"])
    has_best = best >= 0
    extent = _member_extent(member_paths, era5_track, box)
    n_steps = msl.shape[0]

    fig = plt.figure(figsize=(11, 7.5))
    writer = animation.FFMpegWriter(
        fps=2, metadata=dict(artist="earth2studio"), bitrate=1800
    )
    with writer.saving(fig, str(out_mp4), dpi=120):
        for t in range(n_steps):
            fig.clf()
            ax = fig.add_subplot(1, 1, 1, projection=pc)
            ax.set_extent(extent, crs=pc)
            im = ax.pcolormesh(
                lon_grid,
                lats,
                msl[t] / 100.0,
                cmap=msl_cmap,
                vmin=MSL_VMIN_HPA,
                vmax=MSL_VMAX_HPA,
                transform=pc,
                alpha=0.65,
            )
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
            cb.set_label("Member mean sea level pressure (hPa)")
            _draw_track_context(ax, era5_track, anchor_fix, radius_km, box)

            handles = []
            for p in range(member_paths.shape[0]):
                pl = member_paths[p]
                pv = np.isfinite(pl[:, 0]) & np.isfinite(pl[:, 1])
                if not pv.any():
                    continue
                color, lw, z = _path_color_lw(p, best, has_best)
                ax.plot(
                    pl[pv, 1],
                    pl[pv, 0],
                    color=color,
                    linewidth=lw,
                    transform=pc,
                    zorder=z,
                    alpha=0.9,
                )
                # t indexes the zarr's lead axis, which can outrun a
                # truncated tracker parquet -- guard the fix lookup.
                if t < pv.size and pv[t]:
                    ax.plot(
                        pl[t, 1],
                        pl[t, 0],
                        "o",
                        color=color,
                        markersize=8,
                        markeredgecolor="black",
                        markeredgewidth=0.6,
                        transform=pc,
                        zorder=z + 3,
                    )
            handles.append(mlines.Line2D([], [], color=PATH0_COLOR, label="path 0"))
            if has_best and best > 0:
                handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=BEST_PATH_COLOR,
                        label=f"closest path (p{best})",
                    )
                )

            # ERA5 fix at the current step, on top of the dashed full track.
            if t < era5_track.shape[0] and np.isfinite(era5_track[t, :2]).all():
                ax.plot(
                    era5_track[t, 1],
                    era5_track[t, 0],
                    "D",
                    color="black",
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    transform=pc,
                    zorder=10,
                )
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="black",
                    linestyle="--",
                    marker="D",
                    markersize=4,
                    label="ERA5 track",
                )
            )

            # Minimum member MSL inside the track box: where the deepest low
            # actually is, independent of any tracker decision.
            sub = msl[t][np.ix_(in_box_lat, in_box_lon)]
            if sub.size and np.isfinite(sub).any():
                ij = np.unravel_index(np.nanargmin(sub), sub.shape)
                ax.plot(
                    lon_grid[in_box_lon][ij[1]],
                    lats[in_box_lat][ij[0]],
                    "x",
                    color="cyan",
                    markersize=10,
                    markeredgewidth=2.5,
                    transform=pc,
                    zorder=11,
                )
                handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color="cyan",
                        linestyle="none",
                        marker="x",
                        markersize=7,
                        label="min MSL in track box",
                    )
                )

            if cands:
                cand_styles = {
                    "msl-only": dict(
                        marker="o",
                        color="lime",
                        facecolors="none",
                        label="DetectNodes candidate (msl-only)",
                    ),
                    "warm-core": dict(
                        marker="^",
                        color="magenta",
                        label="DetectNodes candidate (warm-core)",
                    ),
                }
                for name, per_step in cands.items():
                    pts = per_step[t]
                    style = cand_styles[name]
                    if pts.shape[0]:
                        ax.scatter(
                            _norm_lon(pts[:, 1]),
                            pts[:, 0],
                            s=55,
                            marker=style["marker"],
                            facecolors=style.get("facecolors", style["color"]),
                            edgecolors=style["color"],
                            linewidths=1.6,
                            transform=pc,
                            zorder=12,
                        )
                    handles.append(
                        mlines.Line2D(
                            [],
                            [],
                            color=style["color"],
                            linestyle="none",
                            marker=style["marker"],
                            markerfacecolor=style.get("facecolors", style["color"]),
                            markersize=7,
                            label=style["label"],
                        )
                    )

            ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.85)
            if start_time_iso:
                time_label = format_local_time_title(
                    start_time_iso, float(lead_h[t]), tz_name
                )
            else:
                time_label = f"step {t}"
            ax.set_title(
                f"{title} -- member {m:04d} [{cat}]\n"
                f"step {t + 1}/{n_steps} ({time_label})",
                fontsize=11,
            )
            writer.grab_frame()
    plt.close(fig)
    print(f"[anchor-diag] wrote {out_mp4}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-member pinned-end anchor diagnostics for conditioned "
        "TC tracks (anchored / alt_path / lost subfolders).",
    )
    parser.add_argument(
        "--tracks-parquet",
        required=True,
        help="Raw all-paths tc_tracks_<mode>.parquet from "
        "plot_ensemble_tc_tracks.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory (per-category subfolders are created inside).",
    )
    parser.add_argument("--mode", required=True, choices=["start", "end"])
    parser.add_argument(
        "--anchor-frame",
        type=int,
        required=True,
        help="Pinned-end frame index (landfall frame under end-conditioning, "
        "0 under start-conditioning).",
    )
    parser.add_argument(
        "--anchor-radius-km",
        type=float,
        default=ANCHOR_RADIUS_KM,
        help="Anchor radius (km); default matches the track-stats report.",
    )
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--title", default="TC track anchor diagnostics")
    parser.add_argument(
        "--members-per-page",
        type=int,
        default=16,
        help="Per-member map panels per track_maps page.",
    )
    parser.add_argument(
        "--map-anchored",
        type=int,
        default=64,
        help="Cap on per-member map panels for the (large, healthy) anchored "
        "category, spread across its path-0 anchor distances; 0 = map every "
        "member.  alt_path and lost members are always all mapped.",
    )
    parser.add_argument(
        "--ensemble-zarr",
        default=None,
        help="Ensemble zarr store; enables the per-member MSL animations.",
    )
    parser.add_argument(
        "--era5-zarr",
        default=None,
        help="ERA5 reference zarr; when given, the pinned frames in the "
        "animations use ERA5 data, matching what the tracker saw.",
    )
    parser.add_argument(
        "--warm-core",
        action="store_true",
        help="Run the candidate overlay with the Z&U 2017 warm-core variant "
        "alongside the msl-only variant (requires z300/z500 in the zarr).",
    )
    parser.add_argument(
        "--no-candidates",
        action="store_true",
        help="Skip the DetectNodes candidate overlay in the animations.",
    )
    parser.add_argument(
        "--animate-anchored",
        type=int,
        default=2,
        help="Members of the anchored category to animate (0 disables).",
    )
    parser.add_argument(
        "--animate-alt",
        type=int,
        default=4,
        help="Members of the alt_path category to animate.",
    )
    parser.add_argument(
        "--animate-lost",
        type=int,
        default=8,
        help="Members of the lost category to animate.",
    )
    parser.add_argument(
        "--animate-members",
        type=int,
        nargs="*",
        default=None,
        help="Explicit member indices to animate (overrides the per-category "
        "counts).",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="ISO UTC window start; enables local wall-clock frame titles.",
    )
    parser.add_argument(
        "--timezone",
        default=None,
        help="IANA timezone for the frame titles (with --start-time).",
    )
    args = parser.parse_args()

    box = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    ens, era5_track = load_all_paths(args.tracks_parquet)
    category, dmin, anchor_fix, anchor_fell_back = classify_members(
        ens, era5_track, args.anchor_frame, args.anchor_radius_km
    )
    df = member_table(ens, category, dmin)

    csv_path = out_root / "member_anchor_categories.csv"
    df.to_csv(csv_path, index=False)
    print(f"[anchor-diag] wrote {csv_path}")
    counts = {cat: int((df["category"] == cat).sum()) for cat in CATEGORIES}
    print(f"[anchor-diag] classification: {counts}")

    # Representative members per category, spread across the category's
    # closest-approach range so marginal and extreme failures both appear.
    to_animate: Dict[str, List[int]] = {}
    spread_key = np.where(np.isfinite(dmin.min(axis=1)), dmin.min(axis=1), np.inf)
    per_cat_n = {
        "anchored": args.animate_anchored,
        "alt_path": args.animate_alt,
        "lost": args.animate_lost,
    }
    explicit = set(args.animate_members or [])
    for cat in CATEGORIES:
        cat_members = df.loc[df["category"] == cat, "member"].to_numpy()
        if args.animate_members is not None:
            picked = [int(m) for m in cat_members if int(m) in explicit]
        else:
            key = (
                dmin[cat_members, 0] if cat == "anchored" else (spread_key[cat_members])
            )
            picked = pick_spread(cat_members, key, per_cat_n[cat])
        to_animate[cat] = picked

    for cat in CATEGORIES:
        cat_dir = out_root / cat
        cat_dir.mkdir(exist_ok=True)
        cat_members = df.loc[df["category"] == cat, "member"].to_numpy()
        if cat_members.size == 0:
            continue
        plot_category_overview(
            ens,
            era5_track,
            cat_members,
            df,
            anchor_fix,
            args.anchor_radius_km,
            box,
            cat_dir / "overview.png",
            f"{args.title} ({args.mode}) -- {cat}: "
            f"{CATEGORY_LABELS[cat]} (n={cat_members.size})",
        )
        map_members: Sequence[int] = cat_members.tolist()
        if cat == "anchored" and 0 < args.map_anchored < cat_members.size:
            map_members = pick_spread(
                cat_members, dmin[cat_members, 0], args.map_anchored
            )
        plot_member_grid(
            ens,
            era5_track,
            map_members,
            df,
            anchor_fix,
            args.anchor_radius_km,
            box,
            cat_dir,
            f"{args.title} ({args.mode}) -- {cat}",
            args.members_per_page,
        )

    candidates_note = "not run (no --ensemble-zarr)"
    if args.ensemble_zarr and any(to_animate.values()):
        # Animations are the secondary artifact -- a failure there (a bad
        # zarr chunk, an ffmpeg hiccup) must not take down the CSV / maps /
        # summary that constitute the diagnosis.
        try:
            candidates_note = animate_members(
                ens,
                era5_track,
                to_animate,
                df,
                anchor_fix,
                args.anchor_radius_km,
                box,
                out_root,
                ensemble_zarr=args.ensemble_zarr,
                era5_zarr=args.era5_zarr,
                mode=args.mode,
                warm_core=args.warm_core,
                with_candidates=not args.no_candidates,
                title=args.title,
                start_time_iso=args.start_time,
                tz_name=args.timezone,
            )
        except Exception:
            import traceback

            traceback.print_exc()
            candidates_note = "animations failed (see job log traceback)"
            print("[anchor-diag] WARNING: animations failed; summary/maps kept.")

    write_summary(
        out_root / "anchor_summary.txt",
        title=args.title,
        mode=args.mode,
        parquet_path=args.tracks_parquet,
        df=df,
        dmin=dmin,
        anchor_fix=anchor_fix,
        anchor_frame=args.anchor_frame,
        anchor_fell_back=anchor_fell_back,
        radius_km=args.anchor_radius_km,
        animated=to_animate,
        candidates_note=candidates_note,
    )
    print("[anchor-diag] done.")


if __name__ == "__main__":
    main()
