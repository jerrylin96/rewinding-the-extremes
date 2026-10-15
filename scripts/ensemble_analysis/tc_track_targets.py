"""Shared TC track geometry + the path-0 pinned-end anchoring diagnostic.

Great-circle / longitude / nearest-fix helpers shared by ``track_qc`` and the
track analyses, the rotated free-end target frames used by
``synoptic_pca_track_stats.py``, and the path-0 anchoring check
(:func:`_anchor_check`) that the standalone anchor diagnostic reports: whether
each member's tracker-primary path (``path_id=0``) reaches the ERA5 fix at the
pinned end.

Free-end targets are NOT extracted here.  The storm of interest is resolved by
the round-5 track QC (``track_qc.py``), because tracking is global and
TempestExtremes numbers concurrent systems arbitrarily: path 0 can be another
storm (19 of 1000 Hurricane Ian members).  The path-0 extraction that used to
live here (``load_track_targets`` / ``free_end_target_arrays``) was removed
once ``compute_synoptic_pca.py`` stopped baking its output.

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
    misreport a truncated run as a wholesale tracking failure.
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
# CLI default (--anchor-radius-km); track_qc uses the same value.
ANCHOR_RADIUS_KM = 500.0
