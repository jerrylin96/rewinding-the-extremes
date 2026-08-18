"""Parquet I/O for TC tracker output.

Schema (long format)::

    source   str    "ensemble" | "era5"
    member   int    member index (0 for era5)
    path_id  int    tracker path index
    step     int    timestep index along the lead-time axis
    tc_lat   float  latitude (degN; NaN where the tracker has no fix)
    tc_lon   float  longitude (deg; same)
    tc_msl   float  central mean sea level pressure (Pa)
    tc_w10m  float  10 m wind speed (m/s)

The ``earth2studio.models.dx`` trackers natively output a 4-D array of
shape ``[member, path_id, step, 4]`` with the trailing axis ordered
``[tc_lat, tc_lon, tc_msl, tc_w10m]``.  This module is the single
canonical place for converting that array to/from parquet so writers
(``plot_ensemble_tc_tracks.py``) and readers (``aggregate_tc_tracks.py``,
analysis notebooks) stay in sync.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

COLUMNS = [
    "source",
    "member",
    "path_id",
    "step",
    "tc_lat",
    "tc_lon",
    "tc_msl",
    "tc_w10m",
]

# Order of the trailing axis in the tracker's native array output.
# Matches earth2studio.models.dx.tc_tracking.OUT_VARIABLES.
_VAR_AXIS = ("tc_lat", "tc_lon", "tc_msl", "tc_w10m")


def tracks_array_to_dataframe(arr: np.ndarray, source: str) -> pd.DataFrame:
    """Flatten a tracker output array to long-format DataFrame.

    Parameters
    ----------
    arr : np.ndarray
        Shape ``[n_members, n_paths, n_steps, 4]`` (the trailing axis
        is ``[tc_lat, tc_lon, tc_msl, tc_w10m]``).  NaNs mark
        unresolved fixes — they are preserved in the output.
    source : str
        ``"ensemble"`` or ``"era5"``.
    """
    if arr.ndim != 4 or arr.shape[-1] != 4:
        raise ValueError(f"tracks array must have shape [N, P, T, 4]; got {arr.shape}")
    n_members, n_paths, n_steps, _ = arr.shape

    members = np.repeat(np.arange(n_members), n_paths * n_steps)
    paths = np.tile(np.repeat(np.arange(n_paths), n_steps), n_members)
    steps = np.tile(np.arange(n_steps), n_members * n_paths)
    flat = arr.reshape(-1, 4)

    return pd.DataFrame(
        {
            "source": np.full(flat.shape[0], source, dtype=object),
            "member": members.astype(np.int32),
            "path_id": paths.astype(np.int32),
            "step": steps.astype(np.int32),
            "tc_lat": flat[:, _VAR_AXIS.index("tc_lat")].astype(np.float32),
            "tc_lon": flat[:, _VAR_AXIS.index("tc_lon")].astype(np.float32),
            "tc_msl": flat[:, _VAR_AXIS.index("tc_msl")].astype(np.float32),
            "tc_w10m": flat[:, _VAR_AXIS.index("tc_w10m")].astype(np.float32),
        }
    )


def dataframe_to_tracks_array(df: pd.DataFrame, source: str) -> Optional[np.ndarray]:
    """Rebuild ``[n_members, n_paths, n_steps, 4]`` from the long DataFrame.

    Returns ``None`` if no rows match ``source``.  Missing
    ``(member, path_id, step)`` triples are filled with NaN.
    """
    sub = df[df["source"] == source]
    if len(sub) == 0:
        return None

    n_members = int(sub["member"].max()) + 1
    n_paths = int(sub["path_id"].max()) + 1
    n_steps = int(sub["step"].max()) + 1

    arr = np.full((n_members, n_paths, n_steps, 4), np.nan, dtype=np.float32)
    arr[sub["member"], sub["path_id"], sub["step"], 0] = sub["tc_lat"].to_numpy()
    arr[sub["member"], sub["path_id"], sub["step"], 1] = sub["tc_lon"].to_numpy()
    arr[sub["member"], sub["path_id"], sub["step"], 2] = sub["tc_msl"].to_numpy()
    arr[sub["member"], sub["path_id"], sub["step"], 3] = sub["tc_w10m"].to_numpy()
    return arr


def save_tracks_parquet(
    path: str,
    ensemble_arr: Optional[np.ndarray] = None,
    era5_arr: Optional[np.ndarray] = None,
) -> str:
    """Write ensemble and/or ERA5 tracks to a single parquet file.

    At least one of ``ensemble_arr`` / ``era5_arr`` must be supplied.
    """
    parts = []
    if ensemble_arr is not None:
        parts.append(tracks_array_to_dataframe(ensemble_arr, "ensemble"))
    if era5_arr is not None:
        parts.append(tracks_array_to_dataframe(era5_arr, "era5"))
    if not parts:
        raise ValueError("save_tracks_parquet: both ensemble_arr and era5_arr are None")

    df = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _fixes_in_bbox(
    arr: np.ndarray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> np.ndarray:
    """Per-fix ``[n_members, n_paths, n_steps]`` mask of valid in-bbox fixes.

    Longitude wrapping is handled by reducing the test to
    ``(lon - lon_min) % 360 <= (lon_max - lon_min) % 360`` so it stays
    correct when the bbox spans the antimeridian or when track / bbox
    longitudes use different conventions ([-180, 180] vs [0, 360]).  A
    degenerate ``lon_min == lon_max`` is treated as a full longitude band
    rather than a thin sliver so we don't silently filter every track.
    """
    lats = arr[..., 0]
    lons = arr[..., 1]
    lon_width = (lon_max - lon_min) % 360
    if lon_width == 0:
        lon_width = 360.0
    dlon = (lons - lon_min) % 360
    in_lat = (lats >= lat_min) & (lats <= lat_max)
    in_lon = dlon <= lon_width
    valid = ~(np.isnan(lats) | np.isnan(lons))
    return valid & in_lat & in_lon


def tracks_in_domain(
    arr: np.ndarray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> np.ndarray:
    """Boolean ``[n_members, n_paths]`` mask of tracks touching the bbox.

    A ``(member, path_id)`` is True when at least one valid fix lies
    inside ``[lon_min, lon_max] x [lat_min, lat_max]``.  See
    ``_fixes_in_bbox`` for the longitude-wrapping convention.
    """
    if arr.ndim != 4 or arr.shape[-1] != 4:
        raise ValueError(f"tracks array must have shape [N, P, T, 4]; got {arr.shape}")
    return _fixes_in_bbox(arr, lon_min, lon_max, lat_min, lat_max).any(axis=-1)


def restrict_to_domain(
    arr: np.ndarray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> np.ndarray:
    """Restrict to the bbox and order each member's paths primary-first.

    Returns a copy of ``arr`` with the same ``[n_members, n_paths,
    n_steps, 4]`` shape.  For every member:

    * paths with no valid fix in ``[lon_min, lon_max] x [lat_min,
      lat_max]`` have their whole ``[n_steps, 4]`` slab replaced with NaN
      (as before), and
    * the surviving in-domain paths are moved to the front of the
      ``path`` axis, ordered by how many fixes each places inside the
      bbox (most first; ties keep the lowest original ``path_id``).

    The reordering is what makes ``path_id=0`` mean "this member's storm
    of interest" downstream.  TempestExtremes numbers concurrent systems
    arbitrarily, so without it a member whose storm landed on
    ``path_id>=1`` -- while an unrelated low claimed ``path_id=0`` -- is
    silently dropped by every reader that takes path 0 (the overlay, the
    ensemble-mean track, the tc_msl rows): ~1 in 7 members for Hurricane
    Sandy.  Out-of-domain paths sort to the back, where they stay NaN, so
    NaN-aware plotting / scoring code is unaffected.
    """
    if arr.ndim != 4 or arr.shape[-1] != 4:
        raise ValueError(f"tracks array must have shape [N, P, T, 4]; got {arr.shape}")
    counts = _fixes_in_bbox(arr, lon_min, lon_max, lat_min, lat_max).sum(axis=-1)
    order = np.argsort(-counts, axis=1, kind="stable")  # most in-bbox fixes first
    member_idx = np.arange(arr.shape[0])[:, None]
    out = arr[member_idx, order].copy()
    out[counts[member_idx, order] == 0] = np.nan  # paths with no in-bbox fix
    return out


def ensemble_mean_track(
    arr: np.ndarray, path_id: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean ``(lats, lons)`` over the member axis at one path.

    Longitudes are averaged on the unit circle (sin/cos components) so
    members that straddle the dateline don't collapse to spurious mid-
    longitude values; the returned longitudes are in ``[-180, 180]``.
    Steps where every member is NaN at this path come back as NaN.
    Returns empty arrays when ``path_id`` is out of range.
    """
    if arr.ndim != 4 or arr.shape[-1] != 4:
        raise ValueError(f"tracks array must have shape [N, P, T, 4]; got {arr.shape}")
    if path_id >= arr.shape[1]:
        return np.array([]), np.array([])
    lats = arr[:, path_id, :, 0]
    lons = arr[:, path_id, :, 1]
    lon_rad = np.deg2rad(lons)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mean_lats = np.nanmean(lats, axis=0)
        mean_sin = np.nanmean(np.sin(lon_rad), axis=0)
        mean_cos = np.nanmean(np.cos(lon_rad), axis=0)
    mean_lons = np.rad2deg(np.arctan2(mean_sin, mean_cos))
    return mean_lats, mean_lons


def load_tracks_parquet(path: str) -> Dict[str, np.ndarray]:
    """Read a tracks parquet and split it back into ``{source: array}``.

    Sources without rows in the file are omitted from the result.
    """
    df = pd.read_parquet(path)
    out: Dict[str, np.ndarray] = {}
    for src in df["source"].unique():
        arr = dataframe_to_tracks_array(df, str(src))
        if arr is not None:
            out[str(src)] = arr
    return out
