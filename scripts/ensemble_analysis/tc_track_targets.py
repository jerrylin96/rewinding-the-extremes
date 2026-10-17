"""Shared TC track geometry for ``track_qc`` and the track analyses.

Great-circle / longitude helpers, the anchor constants ``track_qc`` applies,
and the rotated free-end target frames used by ``synoptic_pca_track_stats.py``.
Storm-of-interest resolution and every free-end fix live in ``track_qc.py``.

Kept dependency-light (NumPy only) so a stats-only login-node run does not pull
cartopy/matplotlib.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

EARTH_RADIUS_M = 6.371e6

# The pinned-end anchor tolerates the tracker losing the storm this many
# frames short of the anchor frame (e.g. warm-core rejection during
# extratropical transition near landfall truncating the path).
ANCHOR_FRAME_WINDOW = 2

# Great-circle radius (km) within which a member's path must approach the ERA5
# fix at the anchor frame to reach the anchor.
ANCHOR_RADIUS_KM = 500.0


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


# Mean km per degree of great-circle arc (2*pi*R / 360).
KM_PER_DEG = 2.0 * np.pi * EARTH_RADIUS_M / 1e3 / 360.0


def rotated_track_targets(
    era5_all_paths: np.ndarray,
    targets: Dict[str, np.ndarray],
    era5_free_end: Dict[str, float],
    free_end_frame: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Coordinate-free free-end track targets: track-relative + fix-cloud axes.

    The targets ``fe_lat_deg``/``fe_lon_deg`` measure displacement along fixed
    geographic axes, so a PC that shifts storms along a rotated axis has its
    signal split (and diluted) between the two coordinates -- e.g. a
    cross-track mode is invisible in longitude when each coordinate's variance
    is dominated by along-track spread.  This helper re-expresses each member's
    free-end fix in two rotated frames:

    * **ERA5-track frame** -- ``fe_along_km`` / ``fe_cross_km``: signed
      displacement of the member fix from the ERA5 free-end fix, resolved
      along and across ERA5's direction of motion at the free-end frame
      (``cross > 0`` = left of ERA5's motion).
    * **Fix-cloud frame** -- ``fe_cloud_major_km`` / ``fe_cloud_minor_km``:
      the fix projected onto the principal axes of the member fix cloud
      itself (centered on the cloud mean), i.e. the data-driven along- and
      cross-spread coordinates.  The major axis is signed to point with
      ERA5's motion (northward if exactly perpendicular to it).

    ``era5_all_paths`` is the raw ERA5 tracker array and ``era5_free_end`` the
    reference fix ``track_qc`` resolved from its path 0.  ERA5's motion is the
    step between the free end and its neighbour inside the window.  Positions
    use a local equirectangular projection about the ERA5 free-end fix (exact
    enough at the few-hundred-km scale of the fix cloud).  Excluded members
    (NaN targets) stay NaN.  Returns ``(rotated_targets, info)`` where ``info``
    carries the frame metadata for the report: ERA5 motion bearing, cloud
    major-axis bearing and variance share, the lat--lon correlation of the fix
    cloud (the dilution culprit), and ERA5's own coordinates in the cloud
    frame.  Raises ``ValueError`` when ERA5 has no fix at the neighbouring
    frame or fewer than three members have a free-end fix.
    """
    fe_lat = np.asarray(targets["fe_lat_deg"], dtype=np.float64)
    fe_lon = np.asarray(targets["fe_lon_deg"], dtype=np.float64)
    lat0, lon0 = float(era5_free_end["lat"]), float(era5_free_end["lon"])

    def _to_km(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_east = _norm_lon(lon - lon0) * KM_PER_DEG * np.cos(np.radians(lat0))
        y_north = (lat - lat0) * KM_PER_DEG
        return x_east, y_north

    era5_ll = era5_all_paths[0, 0, :, :2].astype(np.float64)
    era5_ll[:, 1] = _norm_lon(era5_ll[:, 1])
    # Earlier -> later ERA5 fix across the free end's step into the window.
    fa, fb = (
        (free_end_frame, free_end_frame + 1)
        if free_end_frame == 0
        else (free_end_frame - 1, free_end_frame)
    )
    if not (fb < era5_ll.shape[0] and np.isfinite(era5_ll[[fa, fb]]).all()):
        raise ValueError(
            f"rotated_track_targets: ERA5 path 0 has no fixes at frames {fa}, {fb}"
        )
    (ax,), (ay,) = _to_km(era5_ll[fa : fa + 1, 0], era5_ll[fa : fa + 1, 1])
    (bx,), (by,) = _to_km(era5_ll[fb : fb + 1, 0], era5_ll[fb : fb + 1, 1])
    norm = float(np.hypot(bx - ax, by - ay))
    if norm <= 1e-6:
        raise ValueError("rotated_track_targets: ERA5 does not move across the free end")
    tangent = ((bx - ax) / norm, (by - ay) / norm)
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
    if int(ok.sum()) < 3:
        raise ValueError(
            f"rotated_track_targets: {int(ok.sum())} free-end fixes; need >= 3"
        )
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
