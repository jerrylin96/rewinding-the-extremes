"""Shared TC free-end target extraction + pinned-end anchoring.

Single definition of the per-member free-end track/intensity targets (storm
latitude, longitude, great-circle distance from the ERA5 free-end fix, and
central MSL) and the pinned-end anchoring check, read from the raw all-paths
TempestExtremes parquet (``path_id=0`` is each member's tracker-primary path;
anchoring flags the members whose primary path never reaches the ERA5 fix at
the pinned end -- wrong-system or lost-storm contamination).

Imported by both ``compute_synoptic_pca.py`` -- which bakes the anchored
targets + the ERA5 free-end MSL percentile into ``synoptic_pca_<mode>.npz`` so
the aggregators read one source of truth -- and ``synoptic_pca_track_stats.py``
-- the thin QC CLI that renders the human-readable report. Keeping the
extraction here (rather than in the npz's domain-restricted ``path_id=0``,
which cannot anchor) is what lets the figure and the report quote identical
numbers.

Kept dependency-light (NumPy + the parquet reader) so a stats-only login-node
run does not pull cartopy/matplotlib.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from tc_tracks_io import load_tracks_parquet

EARTH_RADIUS_M = 6.371e6

# The pinned-end anchor tolerates the tracker losing the storm this many
# frames short of the anchor frame (e.g. warm-core rejection during
# extratropical transition near landfall truncating the primary path).
ANCHOR_FRAME_WINDOW = 2


def _great_circle_km(
    lat_a: np.ndarray, lon_a: np.ndarray, lat_b: float, lon_b: float
) -> np.ndarray:
    """Great-circle distance (km) from points ``(lat_a, lon_a)`` to one fix."""
    pa, pb = np.radians(lat_a), np.radians(lat_b)
    dphi = pb - pa
    dlam = np.radians(lon_b) - np.radians(lon_a)
    h = np.sin(dphi / 2.0) ** 2 + np.cos(pa) * np.cos(pb) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M / 1e3 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def _norm_lon(lon):
    """Normalize longitudes to [-180, 180).

    TempestExtremes fixes come on the tracker grid's [0, 360) convention while
    the npz precursor grid uses [-180, 180]; everything downstream (storm-box
    selection, reporting) works on the normalized convention.  Mirrors
    ``compute_synoptic_pca._hpx_bbox_mask``.
    """
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def _fix_at_frame(track: np.ndarray, frame: int) -> Tuple[np.ndarray, bool]:
    """One member's ``[lat, lon, msl]`` at ``frame``, nearest valid fallback.

    ``track`` is ``[n_steps, 3]`` NaN-padded.  Returns the fix and whether a
    fallback (nearest valid step) was needed; all-NaN tracks return NaN.  A
    ``frame`` beyond the array (a truncated tracker run) also falls back to
    the nearest valid step.
    """
    if 0 <= frame < track.shape[0] and np.isfinite(track[frame]).all():
        return track[frame], False
    valid = np.flatnonzero(np.isfinite(track).all(axis=1))
    if valid.size == 0:
        return np.full(3, np.nan, dtype=np.float64), False
    return track[valid[np.argmin(np.abs(valid - frame))]], True


def anchor_min_distances(
    ens_all_paths: np.ndarray,
    anchor_lat: float,
    anchor_lon: float,
    anchor_frame: int,
) -> np.ndarray | None:
    """Per-path closest approach (km) to the anchor fix inside the window.

    ``ens_all_paths`` is ``[n_members, n_paths, n_steps, >=2]`` with
    normalized longitudes.  Returns ``[n_members, n_paths]`` -- the minimum
    great-circle distance from any of a path's fixes within
    ``+-ANCHOR_FRAME_WINDOW`` frames of ``anchor_frame`` to the anchor;
    ``np.inf`` where a path has no finite fix in the window.  Returns None
    when the window lies wholly beyond the lead axis (see
    :func:`_anchor_check` for why that is *undefined*, not failed).

    Shared by :func:`_anchor_check` (which thresholds it at the anchor
    radius) and the anchor-diagnostics script (which reports the raw
    distances per member), so both quote the same numbers.
    """
    n_members, n_paths, n_steps, _ = ens_all_paths.shape
    lo = max(0, anchor_frame - ANCHOR_FRAME_WINDOW)
    hi = min(n_steps, anchor_frame + ANCHOR_FRAME_WINDOW + 1)
    if lo >= hi:
        return None
    window = ens_all_paths[:, :, lo:hi, :2]  # [N, P, W, (lat, lon)]
    dist = _great_circle_km(
        window[..., 0].ravel(), window[..., 1].ravel(), anchor_lat, anchor_lon
    ).reshape(n_members, n_paths, hi - lo)
    return np.min(np.where(np.isfinite(dist), dist, np.inf), axis=2)  # [N, P]


def _anchor_check(
    ens_all_paths: np.ndarray,
    anchor_lat: float,
    anchor_lon: float,
    anchor_frame: int,
    radius_km: float,
) -> Tuple[np.ndarray, Dict[str, int]] | None:
    """Classify members by whether their storm reaches the pinned-end anchor.

    ``ens_all_paths`` is ``[n_members, n_paths, n_steps, >=2]`` with
    normalized longitudes.  A path "reaches the anchor" when any of its fixes
    within ``+-ANCHOR_FRAME_WINDOW`` frames of ``anchor_frame`` lies within
    ``radius_km`` of the ERA5 anchor fix (the window tolerates the tracker
    dropping the storm a step or two short of the anchor, e.g. warm-core
    rejection during extratropical transition).

    Returns ``(path0_anchored [n_members] bool, counts)`` where counts splits
    the failures into ``alt_path`` (another path reaches the anchor: the
    tracker's primary-path choice, not the ensemble, is suspect) and ``lost``
    (no path does).  Returns None when the anchor window lies wholly beyond
    the parquet's lead axis (a truncated tracker run): the check is then
    *undefined*, not failed -- classifying every member as un-anchored would
    silently empty the analysis under ``--drop-unanchored``.
    """
    dmin = anchor_min_distances(ens_all_paths, anchor_lat, anchor_lon, anchor_frame)
    if dmin is None:
        return None
    path_reaches = dmin <= radius_km  # [N, P]
    path0_anchored = path_reaches[:, 0]
    any_anchored = path_reaches.any(axis=1)
    counts = {
        "anchored": int(path0_anchored.sum()),
        "alt_path": int((~path0_anchored & any_anchored).sum()),
        "lost": int((~any_anchored).sum()),
    }
    return path0_anchored, counts


def load_track_targets(
    parquet_path: str,
    free_end_frame: int,
    n_members: int,
    *,
    anchor_frame: int,
    anchor_radius_km: float,
    drop_unanchored: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, object]]:
    """Per-member free-end track/intensity targets from the parquet.

    Returns ``(targets, era5_ref, stats)``:

    * ``targets`` maps target name -> ``[n_members]`` (NaN where the member
      has no usable fix, or is un-anchored under ``drop_unanchored``);
    * ``era5_ref`` holds the ERA5 free-end fix;
    * ``stats`` carries the report bookkeeping: fallback count, the
      pinned-end anchor classification, the number of members dropped, and
      the ERA5 free-end MSL percentile (exact-frame definition, matching
      ``aggregate_tc_tracks``'s ``ERA5 p<N>`` annotation).
    """
    tracks = load_tracks_parquet(parquet_path)
    ens = tracks.get("ensemble")
    era5 = tracks.get("era5")
    if ens is None or era5 is None:
        raise ValueError(f"{parquet_path}: missing ensemble or era5 tracks")

    # All paths, [member, path, step, (lat, lon, msl)]; msl Pa -> hPa,
    # tracker-grid [0, 360) longitudes -> [-180, 180).
    ens_paths = ens[:, :, :, :3].astype(np.float64)
    ens_paths[:, :, :, 1] = _norm_lon(ens_paths[:, :, :, 1])
    ens_paths[:, :, :, 2] /= 100.0
    ens_track = ens_paths[:, 0]  # primary path
    era5_track = era5[0, 0, :, :3].astype(np.float64)
    era5_track[:, 1] = _norm_lon(era5_track[:, 1])
    era5_track[:, 2] /= 100.0

    era5_fix, _ = _fix_at_frame(era5_track, free_end_frame)
    if not np.isfinite(era5_fix).all():
        raise ValueError(f"{parquet_path}: ERA5 has no valid free-end fix")

    stats: Dict[str, object] = {}

    # Pinned-end anchoring: ERA5's fix at the anchor frame is the point every
    # conditioned member's storm must approach.  The check is skipped (never
    # failed wholesale) when ERA5 has no usable anchor fix or the anchor
    # window lies beyond the parquet's lead axis.
    anchor_result = None
    anchor_fix, anchor_fell_back = _fix_at_frame(era5_track, anchor_frame)
    if np.isfinite(anchor_fix[:2]).all():
        anchor_result = _anchor_check(
            ens_paths,
            float(anchor_fix[0]),
            float(anchor_fix[1]),
            anchor_frame,
            anchor_radius_km,
        )
    if anchor_result is not None:
        path0_anchored, counts = anchor_result
        stats["anchor"] = {
            "frame": anchor_frame,
            "lat": float(anchor_fix[0]),
            "lon": float(anchor_fix[1]),
            "radius_km": anchor_radius_km,
            "era5_fallback": anchor_fell_back,
            **counts,
        }
    else:
        path0_anchored = np.ones(ens_paths.shape[0], dtype=bool)
        stats["anchor"] = None
        print(
            f"[tc-targets] anchor: no usable ERA5 fix / track window at "
            f"frame {anchor_frame}; skipping the pinned-end anchoring check."
        )

    m = min(n_members, ens_track.shape[0])
    fe = np.full((n_members, 3), np.nan, dtype=np.float64)
    n_fallback = 0
    n_dropped = 0
    for i in range(m):
        if drop_unanchored and not path0_anchored[i]:
            n_dropped += 1
            continue
        fix, fell_back = _fix_at_frame(ens_track[i], free_end_frame)
        fe[i] = fix
        n_fallback += int(fell_back and np.isfinite(fix).all())
    stats["n_fallback"] = n_fallback
    stats["n_dropped"] = n_dropped
    # Per-member anchoring mask (True = path 0 reaches the pinned-end anchor);
    # consumed by ``free_end_target_arrays`` to bake the mask into the npz. Not
    # read by the track-stats report, so exposing it changes nothing there.
    stats["path0_anchored"] = np.asarray(path0_anchored, dtype=bool)

    # ERA5 free-end MSL percentile, exact-frame definition (fraction of
    # members with a finite fix at the free-end frame whose central MSL is at
    # or below ERA5's) -- matches aggregate_tc_tracks's "ERA5 p<N>" label.
    # Under --drop-unanchored the dropped members are excluded here too.
    stats["era5_fe_msl_pct"] = float("nan")
    stats["era5_fe_msl_n"] = 0
    if 0 <= free_end_frame < ens_track.shape[1] and np.isfinite(
        era5_track[free_end_frame, 2]
    ):
        col = ens_track[:m, free_end_frame, 2].copy()
        if drop_unanchored:
            col[~path0_anchored[:m]] = np.nan
        col = col[np.isfinite(col)]
        if col.size:
            stats["era5_fe_msl_pct"] = float(
                100.0 * (col <= era5_track[free_end_frame, 2]).mean()
            )
            stats["era5_fe_msl_n"] = int(col.size)

    targets = {
        "fe_lat_deg": fe[:, 0],
        "fe_lon_deg": fe[:, 1],
        "fe_dist_era5_km": _great_circle_km(
            fe[:, 0], fe[:, 1], float(era5_fix[0]), float(era5_fix[1])
        ),
        "fe_msl_hpa": fe[:, 2],
    }
    era5_ref = {
        "lat": float(era5_fix[0]),
        "lon": float(era5_fix[1]),
        "msl_hpa": float(era5_fix[2]),
    }
    return targets, era5_ref, stats


# Mean km per degree of great-circle arc (2*pi*R / 360).
KM_PER_DEG = 2.0 * np.pi * EARTH_RADIUS_M / 1e3 / 360.0


def rotated_track_targets(
    parquet_path: str,
    targets: Dict[str, np.ndarray],
    free_end_frame: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Coordinate-free free-end track targets: track-relative + fix-cloud axes.

    The baked targets (``fe_lat_deg``/``fe_lon_deg``) measure displacement
    along fixed geographic axes, so a PC that shifts storms along a rotated
    axis has its signal split (and diluted) between the two coordinates --
    e.g. a cross-track mode is invisible in longitude when each coordinate's
    variance is dominated by along-track spread.  This helper re-expresses
    each member's free-end fix in two rotated frames:

    * **ERA5-track frame** -- ``fe_along_km`` / ``fe_cross_km``: signed
      displacement of the member fix from the ERA5 free-end fix, resolved
      along and across ERA5's direction of motion at the free-end frame
      (``cross > 0`` = left of ERA5's motion).
    * **Fix-cloud frame** -- ``fe_cloud_major_km`` / ``fe_cloud_minor_km``:
      the fix projected onto the principal axes of the member fix cloud
      itself (centered on the cloud mean), i.e. the data-driven along- and
      cross-spread coordinates.  The major axis is signed to point with
      ERA5's motion (northward when that is undefined).

    Positions use a local equirectangular projection about the ERA5 free-end
    fix (exact enough at the few-hundred-km scale of the fix cloud).  Members
    whose baked targets are NaN (dropped / no fix) stay NaN.  Returns
    ``(rotated_targets, info)`` where ``info`` carries the frame metadata for
    the report: ERA5 motion bearing, cloud major-axis bearing and variance
    share, the lat--lon correlation of the fix cloud (the dilution culprit),
    and ERA5's own coordinates in the cloud frame.  Both are empty when no
    ERA5 motion vector or too few fixes exist.
    """
    fe_lat = np.asarray(targets["fe_lat_deg"], dtype=np.float64)
    fe_lon = np.asarray(targets["fe_lon_deg"], dtype=np.float64)

    tracks = load_tracks_parquet(parquet_path)
    era5 = tracks.get("era5")
    if era5 is None:
        return {}, {}
    era5_track = era5[0, 0, :, :3].astype(np.float64)
    era5_track[:, 1] = _norm_lon(era5_track[:, 1])

    era5_fix, _ = _fix_at_frame(era5_track, free_end_frame)
    if not np.isfinite(era5_fix[:2]).all():
        return {}, {}
    lat0, lon0 = float(era5_fix[0]), float(era5_fix[1])

    def _to_km(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_east = _norm_lon(lon - lon0) * KM_PER_DEG * np.cos(np.radians(lat0))
        y_north = (lat - lat0) * KM_PER_DEG
        return x_east, y_north

    # ERA5 motion tangent at the free end: earlier -> later fix, preferring
    # the frame pair leaving the free end, else the pair arriving at it.
    tangent = None
    n_steps = era5_track.shape[0]
    for fa, fb in ((free_end_frame, free_end_frame + 1), (free_end_frame - 1, free_end_frame)):
        if 0 <= fa < n_steps and 0 <= fb < n_steps:
            a, b = era5_track[fa], era5_track[fb]
            if np.isfinite(a[:2]).all() and np.isfinite(b[:2]).all():
                dx, dy = _to_km(np.array([b[0]]), np.array([b[1]]))
                ax, ay = _to_km(np.array([a[0]]), np.array([a[1]]))
                vx, vy = float(dx[0] - ax[0]), float(dy[0] - ay[0])
                norm = float(np.hypot(vx, vy))
                if norm > 1e-6:
                    tangent = (vx / norm, vy / norm)
                    break
    if tangent is None:
        return {}, {}
    tx, ty = tangent
    nx, ny = -ty, tx  # left of motion

    x, y = _to_km(fe_lat, fe_lon)
    rotated: Dict[str, np.ndarray] = {
        "fe_along_km": x * tx + y * ty,
        "fe_cross_km": x * nx + y * ny,
    }
    info: Dict[str, float] = {
        # Compass bearing (deg clockwise from north) of ERA5's motion.
        "era5_motion_bearing_deg": float(np.degrees(np.arctan2(tx, ty)) % 360.0),
    }

    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) >= 3:
        pts = np.column_stack([x[ok], y[ok]])
        center = pts.mean(axis=0)
        evals, evecs = np.linalg.eigh(np.cov((pts - center).T))
        major = evecs[:, int(np.argmax(evals))]
        ref = np.array([tx, ty])
        if float(major @ ref) < 0.0 or (
            float(major @ ref) == 0.0 and float(major[1]) < 0.0
        ):
            major = -major
        minor = np.array([-major[1], major[0]])  # left of major
        xc, yc = x - center[0], y - center[1]
        rotated["fe_cloud_major_km"] = xc * major[0] + yc * major[1]
        rotated["fe_cloud_minor_km"] = xc * minor[0] + yc * minor[1]
        info.update(
            {
                "cloud_major_bearing_deg": float(
                    np.degrees(np.arctan2(major[0], major[1])) % 360.0
                ),
                "cloud_major_var_pct": float(
                    100.0 * evals.max() / max(evals.sum(), 1e-12)
                ),
                "cloud_latlon_corr": float(np.corrcoef(pts[:, 0], pts[:, 1])[0, 1]),
                # ERA5 fix (projection origin) expressed in the cloud frame.
                "era5_cloud_major_km": float(-center @ major),
                "era5_cloud_minor_km": float(-center @ minor),
            }
        )
    return rotated, info


def _mean_storm_position(
    targets: Dict[str, np.ndarray],
) -> Tuple[float, float] | None:
    """Ensemble-mean free-end storm fix, or None when no member has one.

    Longitudes are averaged on the unit circle (sin/cos components, the same
    convention as ``tc_tracks_io.ensemble_mean_track``) so a storm cluster
    straddling the antimeridian does not collapse to a spurious
    mid-longitude mean.
    """
    fe_lat = targets["fe_lat_deg"]
    fe_lon = targets["fe_lon_deg"]
    ok = np.isfinite(fe_lat) & np.isfinite(fe_lon)
    if not ok.any():
        return None
    lon_rad = np.radians(fe_lon[ok])
    lon0 = np.degrees(np.arctan2(np.sin(lon_rad).mean(), np.cos(lon_rad).mean()))
    return float(fe_lat[ok].mean()), float(_norm_lon(lon0))


# Great-circle radius (km) within which a member's path must approach the ERA5
# fix at the pinned-end frame to count as anchored.  Matches the track-stats
# CLI default (--anchor-radius-km) so the baked targets and the report agree.
ANCHOR_RADIUS_KM = 500.0

# Column order of the baked per-member free-end target array.
TARGET_COLUMNS: Tuple[str, ...] = (
    "fe_lat_deg",
    "fe_lon_deg",
    "fe_dist_era5_km",
    "fe_msl_hpa",
)


def free_end_target_arrays(
    parquet_path: str,
    free_end_frame: int,
    n_members: int,
    *,
    anchor_frame: int,
    anchor_radius_km: float = ANCHOR_RADIUS_KM,
) -> Dict[str, np.ndarray]:
    """npz-ready free-end target arrays for ``compute_synoptic_pca`` to bake.

    Runs the shared :func:`load_track_targets` twice on the raw all-paths
    parquet -- unanchored for the per-member targets, the anchoring mask and
    the unanchored ERA5 MSL percentile, and ``drop_unanchored=True`` for the
    anchored percentile (the same exact-frame definition the track-stats report
    and the ``aggregate_tc_tracks`` ``ERA5 p<N>`` annotation use).  Baking the
    all-member targets plus the mask lets a reader reconstruct either variant
    (anchored = targets with ``~mask`` set to NaN) from one source of truth.

    Returns fixed-shape arrays keyed exactly as the npz stores them:

    * ``track_target_fe``            ``[n_members, 4]`` -- columns
      ``TARGET_COLUMNS`` (lat, lon, great-circle dist from ERA5, central MSL);
    * ``track_path0_anchored``       ``[n_members]`` bool;
    * ``track_era5_fe``              ``[3]`` -- ERA5 free-end (lat, lon, MSL);
    * ``track_anchor_frame``         scalar int;
    * ``track_anchor_counts``        ``[3]`` int (anchored, alt_path, lost);
    * ``track_era5_fe_msl_pct``      scalar (unanchored percentile);
    * ``track_era5_fe_msl_pct_anchored`` scalar (anchored percentile);
    * ``track_n_fallback``           scalar int (unanchored fallback count).
    """
    targets, era5_ref, stats = load_track_targets(
        parquet_path,
        free_end_frame,
        n_members,
        anchor_frame=anchor_frame,
        anchor_radius_km=anchor_radius_km,
        drop_unanchored=False,
    )
    _, _, stats_anchored = load_track_targets(
        parquet_path,
        free_end_frame,
        n_members,
        anchor_frame=anchor_frame,
        anchor_radius_km=anchor_radius_km,
        drop_unanchored=True,
    )

    fe = np.column_stack([targets[c] for c in TARGET_COLUMNS]).astype(np.float32)

    # Align the mask (length = parquet member count) to n_members: members
    # beyond the parquet have no track and are not anchored.
    mask = np.asarray(stats["path0_anchored"], dtype=bool)
    aligned = np.zeros(n_members, dtype=bool)
    k = min(mask.shape[0], n_members)
    aligned[:k] = mask[:k]

    anchor = stats["anchor"]
    counts = (
        np.array(
            [anchor["anchored"], anchor["alt_path"], anchor["lost"]], dtype=np.int32
        )
        if anchor is not None
        else np.array([int(aligned.sum()), 0, 0], dtype=np.int32)
    )

    return {
        "track_target_fe": fe,
        "track_path0_anchored": aligned,
        "track_era5_fe": np.array(
            [era5_ref["lat"], era5_ref["lon"], era5_ref["msl_hpa"]], dtype=np.float32
        ),
        "track_anchor_frame": np.int32(anchor_frame),
        "track_anchor_counts": counts,
        "track_era5_fe_msl_pct": np.float32(stats["era5_fe_msl_pct"]),
        "track_era5_fe_msl_pct_anchored": np.float32(stats_anchored["era5_fe_msl_pct"]),
        "track_n_fallback": np.int32(stats["n_fallback"]),
    }
