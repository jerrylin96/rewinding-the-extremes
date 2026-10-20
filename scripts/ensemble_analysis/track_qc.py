"""Unified anchor-first TC track QC.

The single quality-control authority for the conditioned-ensemble TC track
analyses: which tracker path is each member's storm, and which members every
track metric and figure is computed over.  One definition, stated once and
applied everywhere; nothing in it substitutes one frame, path or population for
another.

Definition (do not weaken or extend it):

  Frames come from the conditioning mode
  (``_dispatch_lib.anchor_free_end_frames``).  The anchor is the pinned frame
  where every member coincides with ERA5 (end: the last frame; start and both:
  frame 0).  The free end is the other end of the window (end: frame 0; start:
  the last frame); both-conditioning pins both ends and has none.

  The ERA5 reference is ERA5's path 0.  Its fixes at the anchor and free-end
  frames must exist and lie inside the case's track box, else ``ValueError``.

  A member's storm of interest is the tracker path that reaches the anchor: at
  least one fix within ``anchor_radius_km`` of the ERA5 anchor fix, at the
  anchor frame +-``ANCHOR_FRAME_WINDOW`` frames.  When several paths reach it,
  the one with the most valid fixes in the track box wins (ties: lowest original
  ``path_id``).  Path numbering in the parquet is a TempestExtremes
  implementation detail; this module owns resolution.

  A member is ANALYZABLE iff its storm of interest has at least
  ``min_domain_fixes`` (3) valid fixes in the track box and, when the mode has a
  free end, a valid fix at exactly the free-end frame.  Exclusion reasons,
  checked in this order (deterministic, mutually exclusive):
    1. ``no_track``           -- no path has a valid fix in the track box.
    2. ``unanchored``         -- no path reaches the anchor.
    3. ``insufficient_fixes`` -- the storm of interest has fewer than
                                 ``min_domain_fixes`` valid in-box fixes.
    4. ``no_free_end_fix``    -- the storm of interest has no fix at the
                                 free-end frame (it formed after, or decayed
                                 before, the free end).

Every TC track analysis (``synoptic_pca_track_stats``, ``tc_landfall_split``,
``aggregate_tc_tracks``, ``combine_tc_tracks``, ``plot_ensemble_tc_tracks``,
``aggregate_synoptic_pca`` and the backward-emulator comparison) reads this
table's ``analyzable`` mask and resolved path and reports its own ``n``; no
other track filter exists in the analysis layer, and no track target or track
is stored for later reuse.

Kept dependency-light (NumPy + pandas + the parquet reader) so a stats-only
login-node run does not pull cartopy / matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from _dispatch_lib import anchor_free_end_frames
from tc_track_targets import (
    ANCHOR_FRAME_WINDOW,
    ANCHOR_RADIUS_KM,
    _great_circle_km,
    _norm_lon,
)
from tc_tracks_io import _fixes_in_bbox, load_tracks_parquet

# A resolved storm-of-interest path needs at least this many valid fixes inside
# the track domain to be analyzable; fewer is tracker noise, not a trajectory.
MIN_DOMAIN_FIXES = 3

# Exclusion-reason labels (empty string == analyzable).  Checked in this order.
EXCL_NONE = ""
EXCL_NO_TRACK = "no_track"
EXCL_UNANCHORED = "unanchored"
EXCL_INSUFFICIENT = "insufficient_fixes"
EXCL_NO_FREE_END = "no_free_end_fix"

EXCLUSION_ORDER: Tuple[str, ...] = (
    EXCL_NO_TRACK,
    EXCL_UNANCHORED,
    EXCL_INSUFFICIENT,
    EXCL_NO_FREE_END,
)

# Free-end target columns, in the order the regressions use them.
TARGET_COLUMNS: Tuple[str, ...] = (
    "fe_lat_deg",
    "fe_lon_deg",
    "fe_dist_era5_km",
    "fe_msl_hpa",
)

# Per-member QC table columns, in report order.
QC_COLUMNS: Tuple[str, ...] = (
    "member",
    "resolved_path_id",
    "n_domain_fixes",
    "anchored",
    "analyzable",
    "exclusion",
    "fe_lat_deg",
    "fe_lon_deg",
    "fe_msl_hpa",
)


def _fix_at(track: np.ndarray, frame: int) -> np.ndarray:
    """``[lat, lon, msl]`` of ``track [n_steps, 3]`` at ``frame``; NaN if none."""
    if 0 <= frame < track.shape[0] and np.isfinite(track[frame]).all():
        return track[frame]
    return np.full(3, np.nan)


def _in_box(lat: float, lon: float, box: Tuple[float, float, float, float]) -> bool:
    point = np.array([[[[lat, lon]]]], dtype=np.float64)
    return bool(_fixes_in_bbox(point, *box)[0, 0, 0])


def _anchor_min_distances(
    ens_ll: np.ndarray,
    anchor_lat: float,
    anchor_lon: float,
    anchor_frame: int,
) -> np.ndarray:
    """Per-path closest approach (km) to the anchor fix inside the frame window.

    ``ens_ll`` is ``[n_members, n_paths, n_steps, >=2]`` with normalized
    longitudes.  Returns ``[n_members, n_paths]`` -- the minimum great-circle
    distance from any of a path's fixes within ``+-ANCHOR_FRAME_WINDOW`` frames
    of ``anchor_frame`` to the anchor fix; ``np.inf`` where a path has no finite
    fix in the window (including a window past the end of the array).
    """
    n_members, n_paths, n_steps, _ = ens_ll.shape
    lo = max(0, anchor_frame - ANCHOR_FRAME_WINDOW)
    hi = min(n_steps, anchor_frame + ANCHOR_FRAME_WINDOW + 1)
    if lo >= hi:
        return np.full((n_members, n_paths), np.inf)
    window = ens_ll[:, :, lo:hi, :2]  # [N, P, W, (lat, lon)]
    dist = _great_circle_km(
        window[..., 0].ravel(), window[..., 1].ravel(), anchor_lat, anchor_lon
    ).reshape(n_members, n_paths, hi - lo)
    return np.min(np.where(np.isfinite(dist), dist, np.inf), axis=2)  # [N, P]


@dataclass(frozen=True)
class TrackQC:
    """Per-member QC table + the metadata the consumers and report quote.

    ``table`` has one row per member with the :data:`QC_COLUMNS` columns.
    ``era5_anchor`` / ``era5_free_end`` carry the reference fixes
    (``era5_free_end`` is ``None`` for both-conditioning, which has no free end).
    ``config`` records the resolution parameters for provenance.
    """

    table: pd.DataFrame
    era5_anchor: Dict[str, float]
    era5_free_end: Optional[Dict[str, float]]
    config: Dict[str, object]

    # -- masks --------------------------------------------------------------
    @property
    def n_members(self) -> int:
        return int(len(self.table))

    @property
    def analyzable(self) -> np.ndarray:
        """Boolean ``[n_members]`` mask of analyzable members."""
        return self.table["analyzable"].to_numpy(dtype=bool)

    @property
    def anchored(self) -> np.ndarray:
        """Boolean ``[n_members]`` mask of members with an anchor-reaching path."""
        return self.table["anchored"].to_numpy(dtype=bool)

    @property
    def resolved_path_id(self) -> np.ndarray:
        """Resolved original ``path_id`` per member (``-1`` where none)."""
        return self.table["resolved_path_id"].to_numpy(dtype=int)

    # -- drawing ------------------------------------------------------------
    def storm_of_interest_array(self, ens_all_paths: np.ndarray) -> np.ndarray:
        """Reduce a raw tracker array to the analyzable members' storms.

        Returns a ``[n_members, 1, n_steps, 4]`` copy of ``ens_all_paths`` whose
        single path is each analyzable member's storm of interest; every
        excluded member is all-NaN, so a figure draws exactly the population the
        statistics use.
        """
        n_members, _, n_steps, _ = ens_all_paths.shape
        out = np.full((n_members, 1, n_steps, 4), np.nan, dtype=ens_all_paths.dtype)
        resolved = self.resolved_path_id
        keep = self.analyzable
        for m in range(min(n_members, resolved.shape[0])):
            if keep[m]:
                out[m, 0] = ens_all_paths[m, resolved[m]]
        return out

    # -- summaries ----------------------------------------------------------
    def counts(self) -> Dict[str, int]:
        """Taxonomy counts: total, analyzable, excluded, and per-reason."""
        excl = self.table["exclusion"].to_numpy()
        out = {
            "total": self.n_members,
            "analyzable": int(self.analyzable.sum()),
            "excluded": int((~self.analyzable).sum()),
        }
        for reason in EXCLUSION_ORDER:
            out[reason] = int((excl == reason).sum())
        return out

    def summary(self) -> str:
        """One-line population summary for logs."""
        c = self.counts()
        reasons = " ".join(f"{r}={c[r]}" for r in EXCLUSION_ORDER)
        return f"{c['analyzable']}/{c['total']} analyzable (excluded: {reasons})"

    def _require_free_end(self) -> Dict[str, float]:
        if self.era5_free_end is None:
            raise ValueError(f"track_qc: mode '{self.config['mode']}' has no free end")
        return self.era5_free_end

    def free_end_targets(self) -> Dict[str, np.ndarray]:
        """Free-end track/intensity targets aligned to the member axis.

        Returns ``fe_lat_deg`` / ``fe_lon_deg`` / ``fe_dist_era5_km`` (great-
        circle distance from the ERA5 free-end fix) / ``fe_msl_hpa``; every
        excluded member is NaN.
        """
        ref = self._require_free_end()
        keep = self.analyzable
        cols = {}
        for c in ("fe_lat_deg", "fe_lon_deg", "fe_msl_hpa"):
            v = self.table[c].to_numpy(dtype=float).copy()
            v[~keep] = np.nan
            cols[c] = v
        dist = _great_circle_km(
            cols["fe_lat_deg"], cols["fe_lon_deg"], ref["lat"], ref["lon"]
        )
        return {
            "fe_lat_deg": cols["fe_lat_deg"],
            "fe_lon_deg": cols["fe_lon_deg"],
            "fe_dist_era5_km": dist,
            "fe_msl_hpa": cols["fe_msl_hpa"],
        }

    def era5_free_end_msl_pct(self) -> Tuple[float, int]:
        """ERA5 free-end central-MSL percentile over the analyzable members.

        Fraction (in percent) of analyzable members whose free-end central MSL
        is at or below ERA5's -- the ``ERA5 p<N>`` the track figures and the
        paper quote.  Returns ``(nan, 0)`` when no member is analyzable.
        """
        era5_msl = self._require_free_end()["msl_hpa"]
        col = self.free_end_targets()["fe_msl_hpa"]
        col = col[np.isfinite(col)]
        if col.size == 0:
            return float("nan"), 0
        return float(100.0 * (col <= era5_msl).mean()), int(col.size)


def compute_track_qc_arrays(
    ens_all_paths: np.ndarray,
    era5_all_paths: Optional[np.ndarray],
    *,
    track_box: Tuple[float, float, float, float],
    mode: str,
    n_members: Optional[int] = None,
    anchor_radius_km: float = ANCHOR_RADIUS_KM,
    min_domain_fixes: int = MIN_DOMAIN_FIXES,
) -> TrackQC:
    """Anchor-first QC on in-memory tracker arrays.

    ``ens_all_paths`` is the raw ``[n_members, n_paths, n_steps, 4]`` tracker
    array (trailing axis ``[tc_lat, tc_lon, tc_msl(Pa), tc_w10m]``, tracker-grid
    longitudes); ``era5_all_paths`` is the ``[1, n_paths, n_steps, 4]`` ERA5
    reference.  ``track_box`` is ``(lon_min, lon_max, lat_min, lat_max)``.
    ``n_members`` pads the member axis: the parquet has no rows for a member
    whose tracker found nothing, so trailing members are ``no_track``.  See the
    module docstring for the rule.
    """
    if ens_all_paths.ndim != 4 or ens_all_paths.shape[-1] != 4:
        raise ValueError(
            f"ens_all_paths must have shape [N, P, T, 4]; got {ens_all_paths.shape}"
        )
    anchor_frame, free_end_frame = anchor_free_end_frames(mode)
    n_ens = int(ens_all_paths.shape[0])
    n_out = int(n_members) if n_members is not None else n_ens
    if n_out < n_ens:
        raise ValueError(
            f"track_qc: n_members={n_out} but the tracks hold {n_ens} members"
        )

    # [member, path, step, (lat, lon, msl_hPa)] with normalized longitudes.
    ens = ens_all_paths[:, :, :, :3].astype(np.float64)
    ens[:, :, :, 1] = _norm_lon(ens[:, :, :, 1])
    ens[:, :, :, 2] /= 100.0

    if era5_all_paths is None or era5_all_paths.shape[0] == 0:
        raise ValueError("track_qc: the ERA5 reference track is required")
    era5_track = era5_all_paths[0, 0, :, :3].astype(np.float64)
    era5_track[:, 1] = _norm_lon(era5_track[:, 1])
    era5_track[:, 2] /= 100.0

    def _era5_fix(frame: int, role: str) -> Dict[str, float]:
        fix = _fix_at(era5_track, frame)
        if not np.isfinite(fix).all():
            raise ValueError(
                f"track_qc: ERA5 path 0 has no fix at the {role} frame {frame}"
            )
        if not _in_box(fix[0], fix[1], track_box):
            raise ValueError(
                f"track_qc: ERA5 path 0's {role} fix ({fix[0]:.2f}, {fix[1]:.2f}) "
                f"lies outside the track box {track_box}; path 0 is not the storm"
            )
        return {"lat": float(fix[0]), "lon": float(fix[1]), "msl_hpa": float(fix[2])}

    era5_anchor = _era5_fix(anchor_frame, "anchor")
    era5_free_end = (
        _era5_fix(free_end_frame, "free-end") if free_end_frame is not None else None
    )

    # In-domain valid-fix count per path, and which paths reach the anchor.
    n_dom = _fixes_in_bbox(ens[:, :, :, :2], *track_box).sum(axis=-1)  # [N, P]
    reaches = (
        _anchor_min_distances(ens, era5_anchor["lat"], era5_anchor["lon"], anchor_frame)
        <= anchor_radius_km
    )

    nan3 = np.full(3, np.nan)
    rows = []
    for m in range(n_out):
        if m >= n_ens or not (n_dom[m] > 0).any():
            rows.append(_row(m, -1, 0, False, EXCL_NO_TRACK, nan3))
            continue
        cand = np.flatnonzero(reaches[m])
        if cand.size == 0:
            rows.append(_row(m, -1, 0, False, EXCL_UNANCHORED, nan3))
            continue
        # Most in-domain fixes among anchor-reaching paths; ties keep the
        # lowest original path id (cand is ascending, argmax returns the first).
        resolved = int(cand[int(np.argmax(n_dom[m, cand]))])
        ndf = int(n_dom[m, resolved])
        fix = (
            nan3
            if free_end_frame is None
            else _fix_at(ens[m, resolved], free_end_frame)
        )
        if ndf < min_domain_fixes:
            reason = EXCL_INSUFFICIENT
        elif free_end_frame is not None and not np.isfinite(fix).all():
            reason = EXCL_NO_FREE_END
        else:
            reason = EXCL_NONE
        rows.append(_row(m, resolved, ndf, True, reason, fix))

    table = pd.DataFrame(rows, columns=QC_COLUMNS)
    config = {
        "mode": mode,
        "anchor_frame": int(anchor_frame),
        "free_end_frame": free_end_frame,
        "anchor_radius_km": float(anchor_radius_km),
        "min_domain_fixes": int(min_domain_fixes),
        "track_box": tuple(float(v) for v in track_box),
        "n_members": n_out,
    }
    return TrackQC(table, era5_anchor, era5_free_end, config)


def _row(
    member: int,
    resolved_path_id: int,
    n_domain_fixes: int,
    anchored: bool,
    exclusion: str,
    fix: np.ndarray,
) -> Dict[str, object]:
    """One QC-table row (free-end fix split into lat/lon/msl)."""
    return {
        "member": int(member),
        "resolved_path_id": int(resolved_path_id),
        "n_domain_fixes": int(n_domain_fixes),
        "anchored": bool(anchored),
        "analyzable": exclusion == EXCL_NONE,
        "exclusion": exclusion,
        "fe_lat_deg": float(fix[0]),
        "fe_lon_deg": float(fix[1]),
        "fe_msl_hpa": float(fix[2]),
    }


def compute_track_qc(
    parquet_path: str,
    *,
    track_box: Tuple[float, float, float, float],
    mode: str,
    n_members: Optional[int] = None,
    anchor_radius_km: float = ANCHOR_RADIUS_KM,
    min_domain_fixes: int = MIN_DOMAIN_FIXES,
) -> Tuple[np.ndarray, np.ndarray, TrackQC]:
    """``(ens, era5, qc)`` for a mode's raw all-paths ``tc_tracks_<mode>.parquet``.

    Returns the raw tracker arrays with the QC so a figure can draw
    ``qc.storm_of_interest_array(ens)`` without reading the parquet twice.
    """
    tracks = load_tracks_parquet(parquet_path)
    ens = tracks.get("ensemble")
    era5 = tracks.get("era5")
    if ens is None or era5 is None:
        raise ValueError(f"{parquet_path}: missing ensemble or era5 tracks")
    qc = compute_track_qc_arrays(
        ens,
        era5,
        track_box=track_box,
        mode=mode,
        n_members=n_members,
        anchor_radius_km=anchor_radius_km,
        min_domain_fixes=min_domain_fixes,
    )
    return ens, era5, qc
