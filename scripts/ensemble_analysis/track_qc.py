"""Unified anchor-first TC track QC (round 5).

The single quality-control authority for the conditioned-ensemble TC track
analyses.  Round 5 replaced a layer of per-analysis QC filters that had drifted
out of agreement (a landfall split that required >=3 in-domain fixes on the
tracker's primary path but ignored the pinned anchor; a regression QC that
required the primary path to reach the anchor but had no minimum length) with
one definition, stated once and applied everywhere.

Definition (do not weaken or extend it):

  Storm-of-interest resolution is anchor-first.  A member's storm of interest
  is the tracker path that reaches the pinned anchor state -- at least one fix
  within ``anchor_radius_km`` of the ERA5 fix at the anchor frame
  ``+-ANCHOR_FRAME_WINDOW`` frames (end mode: the case's landfall frame; start
  mode: frame 0).  When several paths reach the anchor, the resolved path is
  the one with the most valid fixes in the case's track domain (ties: lowest
  original ``path_id``).  Path numbering in the parquet is a TempestExtremes
  implementation detail; this module owns resolution.

  A member is ANALYZABLE iff its resolved storm-of-interest path has at least
  ``min_domain_fixes`` (3) valid fixes in the track domain.

  Exclusion taxonomy, checked in this order (deterministic, mutually
  exclusive):
    1. ``no_track``           -- no path has any valid fix in the track domain.
    2. ``unanchored``         -- has an in-domain path, but no path reaches the
                                 pinned anchor.
    3. ``insufficient_fixes`` -- the resolved anchor-reaching path has fewer
                                 than ``min_domain_fixes`` valid in-domain fixes.

  A member's free-end fix is read on its resolved path at ``free_end_frame``,
  falling back to the nearest valid frame with the fallback flagged.  Members
  with an exact-frame free-end fix form the exact-frame subset (the ERA5
  intensity-percentile population).

Every parquet-consuming TC analysis (``synoptic_pca_track_stats``,
``tc_landfall_split``, ``aggregate_tc_tracks``, ``plot_ensemble_tc_tracks``)
reads this table's ``analyzable`` mask and resolved path and reports its own
``n``; no other track filter exists in the analysis layer.

Scope note.  ``tc_track_targets.py`` retains the *old* path-0 anchoring only to
serve the upstream npz-baking in ``compute_synoptic_pca.py`` (a frozen input
this post-processing refactor does not regenerate) and the standalone anchor
diagnostic; that path is not part of the round-5 analysis QC and must not be
called by the four consumers above.  Only the pure-geometry helpers are shared.

Kept dependency-light (NumPy + pandas + the parquet reader) so a stats-only
login-node run does not pull cartopy / matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from tc_track_targets import (
    ANCHOR_FRAME_WINDOW,
    ANCHOR_RADIUS_KM,
    _fix_at_frame,
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

EXCLUSION_ORDER: Tuple[str, ...] = (
    EXCL_NO_TRACK,
    EXCL_UNANCHORED,
    EXCL_INSUFFICIENT,
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
    "fe_fallback",
)


def _anchor_min_distances(
    ens_ll: np.ndarray,
    anchor_lat: float,
    anchor_lon: float,
    anchor_frame: int,
) -> Optional[np.ndarray]:
    """Per-path closest approach (km) to the anchor fix inside the frame window.

    ``ens_ll`` is ``[n_members, n_paths, n_steps, >=2]`` with normalized
    longitudes.  Returns ``[n_members, n_paths]`` -- the minimum great-circle
    distance from any of a path's fixes within ``+-ANCHOR_FRAME_WINDOW`` frames
    of ``anchor_frame`` to the anchor fix; ``np.inf`` where a path has no finite
    fix in the window.  Returns ``None`` when the window lies wholly beyond the
    lead axis (a truncated tracker run), so resolution can degrade to a
    domain-only rule rather than declaring every member un-anchored.
    """
    n_members, n_paths, n_steps, _ = ens_ll.shape
    lo = max(0, anchor_frame - ANCHOR_FRAME_WINDOW)
    hi = min(n_steps, anchor_frame + ANCHOR_FRAME_WINDOW + 1)
    if lo >= hi:
        return None
    window = ens_ll[:, :, lo:hi, :2]  # [N, P, W, (lat, lon)]
    dist = _great_circle_km(
        window[..., 0].ravel(), window[..., 1].ravel(), anchor_lat, anchor_lon
    ).reshape(n_members, n_paths, hi - lo)
    return np.min(np.where(np.isfinite(dist), dist, np.inf), axis=2)  # [N, P]


@dataclass(frozen=True)
class TrackQC:
    """Per-member QC table + the metadata the consumers and report quote.

    ``table`` has one row per member with the :data:`QC_COLUMNS` columns.
    ``era5_free_end`` / ``era5_anchor`` carry the reference fixes;
    ``era5_anchor`` is ``None`` when the anchor window was undefined (a
    truncated tracker run) and resolution fell back to a domain-only rule.
    ``config`` records the resolution parameters for provenance.
    """

    table: pd.DataFrame
    era5_free_end: Dict[str, float]
    era5_anchor: Optional[Dict[str, float]]
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
    def fe_fallback(self) -> np.ndarray:
        return self.table["fe_fallback"].to_numpy(dtype=bool)

    @property
    def exact_frame(self) -> np.ndarray:
        """Analyzable members whose free-end fix is at the exact free-end frame."""
        return self.analyzable & ~self.fe_fallback

    @property
    def resolved_path_id(self) -> np.ndarray:
        """Resolved original ``path_id`` per member (``-1`` where none)."""
        return self.table["resolved_path_id"].to_numpy(dtype=int)

    # -- drawing ------------------------------------------------------------
    def storm_of_interest_array(self, ens_all_paths: np.ndarray) -> np.ndarray:
        """Reduce a raw tracker array to one resolved storm path per member.

        Returns a ``[n_members, 1, n_steps, 4]`` copy of ``ens_all_paths`` whose
        single path is each member's anchor-resolved storm of interest.
        Members with no resolved anchor-reaching path (``no_track`` /
        ``unanchored``) come back all-NaN, so NaN-aware drawing / mean code
        skips them; ``insufficient_fixes`` members keep their short resolved
        path (the figure's own minimum-length check drops it).  This is the
        anchor-first replacement for ``tc_tracks_io.restrict_to_domain`` in the
        figure scripts: ``path_id=0`` now means "this member's storm of
        interest" under the round-5 rule.
        """
        n_members, _, n_steps, _ = ens_all_paths.shape
        out = np.full((n_members, 1, n_steps, 4), np.nan, dtype=ens_all_paths.dtype)
        resolved = self.resolved_path_id
        for m in range(min(n_members, resolved.shape[0])):
            if resolved[m] >= 0:
                out[m, 0] = ens_all_paths[m, resolved[m]]
        return out

    # -- summaries ----------------------------------------------------------
    def counts(self) -> Dict[str, int]:
        """Taxonomy counts: total, analyzable, excluded, and per-reason.

        The prose quotes ``analyzable`` and ``excluded`` (one population, one
        total exclusion count); the per-reason keys populate the QC/SI table.
        """
        excl = self.table["exclusion"].to_numpy()
        out = {
            "total": self.n_members,
            "analyzable": int(self.analyzable.sum()),
            "excluded": int((~self.analyzable).sum()),
            "exact_frame": int(self.exact_frame.sum()),
            "fe_fallback": int((self.analyzable & self.fe_fallback).sum()),
        }
        for reason in EXCLUSION_ORDER:
            out[reason] = int((excl == reason).sum())
        return out

    def free_end_targets(self, analyzable_only: bool = True) -> Dict[str, np.ndarray]:
        """Free-end track/intensity targets aligned to the member axis.

        Returns ``fe_lat_deg`` / ``fe_lon_deg`` / ``fe_dist_era5_km`` (great-
        circle distance from the ERA5 free-end fix) / ``fe_msl_hpa``.  With
        ``analyzable_only`` (default) every excluded member is set to NaN so a
        consumer that forgets the mask cannot contaminate a statistic with an
        unresolved storm.
        """
        fe_lat = self.table["fe_lat_deg"].to_numpy(dtype=float).copy()
        fe_lon = self.table["fe_lon_deg"].to_numpy(dtype=float).copy()
        fe_msl = self.table["fe_msl_hpa"].to_numpy(dtype=float).copy()
        if analyzable_only:
            drop = ~self.analyzable
            fe_lat[drop] = np.nan
            fe_lon[drop] = np.nan
            fe_msl[drop] = np.nan
        dist = _great_circle_km(
            fe_lat, fe_lon, self.era5_free_end["lat"], self.era5_free_end["lon"]
        )
        return {
            "fe_lat_deg": fe_lat,
            "fe_lon_deg": fe_lon,
            "fe_dist_era5_km": dist,
            "fe_msl_hpa": fe_msl,
        }

    def era5_free_end_msl_pct(self) -> Tuple[float, int]:
        """ERA5 free-end central-MSL percentile over the exact-frame subset.

        Fraction (in percent) of exact-frame analyzable members whose free-end
        central MSL is at or below ERA5's -- the ``ERA5 p<N>`` definition the
        track figures and the paper quote.  Returns ``(nan, 0)`` when no
        exact-frame member or no finite ERA5 free-end MSL exists.
        """
        era5_msl = self.era5_free_end["msl_hpa"]
        if not np.isfinite(era5_msl):
            return float("nan"), 0
        col = self.table["fe_msl_hpa"].to_numpy(dtype=float)[self.exact_frame]
        col = col[np.isfinite(col)]
        if col.size == 0:
            return float("nan"), 0
        return float(100.0 * (col <= era5_msl).mean()), int(col.size)


def compute_track_qc_arrays(
    ens_all_paths: np.ndarray,
    era5_all_paths: Optional[np.ndarray],
    *,
    track_box: Tuple[float, float, float, float],
    anchor_frame: int,
    free_end_frame: int,
    n_members: Optional[int] = None,
    anchor_radius_km: float = ANCHOR_RADIUS_KM,
    min_domain_fixes: int = MIN_DOMAIN_FIXES,
) -> TrackQC:
    """Anchor-first QC on in-memory tracker arrays.

    ``ens_all_paths`` is the raw ``[n_members, n_paths, n_steps, 4]`` tracker
    array (trailing axis ``[tc_lat, tc_lon, tc_msl(Pa), tc_w10m]``, tracker-grid
    longitudes); ``era5_all_paths`` is the ``[1, n_paths, n_steps, 4]`` ERA5
    reference (its path 0 supplies the anchor and free-end reference fixes).
    ``track_box`` is ``(lon_min, lon_max, lat_min, lat_max)``.  See the module
    docstring for the resolution rule and taxonomy.
    """
    if ens_all_paths.ndim != 4 or ens_all_paths.shape[-1] != 4:
        raise ValueError(
            f"ens_all_paths must have shape [N, P, T, 4]; got {ens_all_paths.shape}"
        )
    n_ens = int(ens_all_paths.shape[0])
    n_out = int(n_members) if n_members is not None else n_ens

    # [member, path, step, (lat, lon, msl_hPa)] with normalized longitudes.
    ens = ens_all_paths[:, :, :, :3].astype(np.float64)
    ens[:, :, :, 1] = _norm_lon(ens[:, :, :, 1])
    ens[:, :, :, 2] /= 100.0

    if era5_all_paths is None or era5_all_paths.shape[0] == 0:
        raise ValueError("compute_track_qc_arrays: ERA5 reference track is required")
    era5_track = era5_all_paths[0, 0, :, :3].astype(np.float64)
    era5_track[:, 1] = _norm_lon(era5_track[:, 1])
    era5_track[:, 2] /= 100.0

    era5_fe, _ = _fix_at_frame(era5_track, free_end_frame)
    if not np.isfinite(era5_fe).all():
        raise ValueError("compute_track_qc_arrays: ERA5 has no valid free-end fix")
    era5_free_end = {
        "lat": float(era5_fe[0]),
        "lon": float(era5_fe[1]),
        "msl_hpa": float(era5_fe[2]),
    }

    # In-domain valid-fix count per path (convention-agnostic bbox test).
    n_dom = _fixes_in_bbox(ens[:, :, :, :2], *track_box).sum(axis=-1)  # [N, P]
    member_has_domain = (n_dom > 0).any(axis=1)  # [N]

    # Anchor: ERA5's fix at the pinned frame is the point every conditioned
    # member's storm must approach.  Undefined (no ERA5 anchor fix, or the
    # window lies beyond the lead axis) -> degrade to a domain-only rule.
    era5_anchor: Optional[Dict[str, float]] = None
    reaches = np.zeros(n_dom.shape, dtype=bool)
    anchor_fix, anchor_fell_back = _fix_at_frame(era5_track, anchor_frame)
    anchor_defined = False
    if np.isfinite(anchor_fix[:2]).all():
        dmin = _anchor_min_distances(
            ens, float(anchor_fix[0]), float(anchor_fix[1]), anchor_frame
        )
        if dmin is not None:
            reaches = dmin <= anchor_radius_km
            anchor_defined = True
            era5_anchor = {
                "lat": float(anchor_fix[0]),
                "lon": float(anchor_fix[1]),
                "frame": int(anchor_frame),
                "radius_km": float(anchor_radius_km),
                "era5_fallback": bool(anchor_fell_back),
            }
    if not anchor_defined:
        # No usable anchor: treat every in-domain path as reaching so members
        # still resolve by fix count (no member is spuriously "unanchored").
        reaches = n_dom > 0

    nan3 = np.full(3, np.nan)
    rows = []
    for m in range(n_out):
        # Taxonomy is checked in the spec's deterministic order:
        # (1) no_track, (2) unanchored, (3) insufficient_fixes; only then
        # analyzable.  no_track has priority, so a path reaching the anchor
        # solely via an out-of-domain window fix is still no_track when the
        # member has no in-domain fix anywhere.
        if m >= n_ens or not member_has_domain[m]:
            rows.append(_row(m, -1, 0, False, False, EXCL_NO_TRACK, nan3, False))
            continue
        reaches_m = reaches[m]
        if not reaches_m.any():
            rows.append(_row(m, -1, 0, False, False, EXCL_UNANCHORED, nan3, False))
            continue
        n_dom_m = n_dom[m]
        cand = np.flatnonzero(reaches_m)
        # Most in-domain fixes among anchor-reaching paths; ties keep the
        # lowest original path id (cand is ascending, argmax returns the
        # first maximum).
        resolved = int(cand[int(np.argmax(n_dom_m[cand]))])
        ndf = int(n_dom_m[resolved])
        fix, fell_back = _fix_at_frame(ens[m, resolved], free_end_frame)
        has_fe = bool(np.isfinite(fix).all())
        analyzable = ndf >= min_domain_fixes
        reason = EXCL_NONE if analyzable else EXCL_INSUFFICIENT
        rows.append(
            _row(m, resolved, ndf, True, analyzable, reason, fix, fell_back and has_fe)
        )

    table = pd.DataFrame(rows, columns=QC_COLUMNS)
    config = {
        "mode_anchor_frame": int(anchor_frame),
        "free_end_frame": int(free_end_frame),
        "anchor_radius_km": float(anchor_radius_km),
        "min_domain_fixes": int(min_domain_fixes),
        "track_box": tuple(float(v) for v in track_box),
        "n_members": n_out,
        "anchor_defined": anchor_defined,
    }
    return TrackQC(table, era5_free_end, era5_anchor, config)


def _row(
    member: int,
    resolved_path_id: int,
    n_domain_fixes: int,
    anchored: bool,
    analyzable: bool,
    exclusion: str,
    fix: np.ndarray,
    fe_fallback: bool,
) -> Dict[str, object]:
    """One QC-table row (free-end fix split into lat/lon/msl)."""
    return {
        "member": int(member),
        "resolved_path_id": int(resolved_path_id),
        "n_domain_fixes": int(n_domain_fixes),
        "anchored": bool(anchored),
        "analyzable": bool(analyzable),
        "exclusion": exclusion,
        "fe_lat_deg": float(fix[0]),
        "fe_lon_deg": float(fix[1]),
        "fe_msl_hpa": float(fix[2]),
        "fe_fallback": bool(fe_fallback),
    }


def compute_track_qc(
    parquet_path: str,
    *,
    track_box: Tuple[float, float, float, float],
    anchor_frame: int,
    free_end_frame: int,
    n_members: Optional[int] = None,
    anchor_radius_km: float = ANCHOR_RADIUS_KM,
    min_domain_fixes: int = MIN_DOMAIN_FIXES,
) -> TrackQC:
    """Anchor-first QC read straight from a mode's ``tc_tracks_<mode>.parquet``.

    Loads the raw all-paths parquet once and delegates to
    :func:`compute_track_qc_arrays`.  This is the entry point every consumer
    calls: one parquet read, one QC table, one definition.
    """
    tracks = load_tracks_parquet(parquet_path)
    ens = tracks.get("ensemble")
    era5 = tracks.get("era5")
    if ens is None or era5 is None:
        raise ValueError(f"{parquet_path}: missing ensemble or era5 tracks")
    return compute_track_qc_arrays(
        ens,
        era5,
        track_box=track_box,
        anchor_frame=anchor_frame,
        free_end_frame=free_end_frame,
        n_members=n_members,
        anchor_radius_km=anchor_radius_km,
        min_domain_fixes=min_domain_fixes,
    )


def resolve_storm_of_interest(
    ens_all_paths: np.ndarray,
    era5_all_paths: Optional[np.ndarray],
    *,
    track_box: Tuple[float, float, float, float],
    anchor_frame: int,
    free_end_frame: int,
    anchor_radius_km: float = ANCHOR_RADIUS_KM,
    min_domain_fixes: int = MIN_DOMAIN_FIXES,
) -> Tuple[np.ndarray, TrackQC]:
    """QC an in-memory tracker array and return its storm-of-interest reduction.

    Convenience wrapper for the figure scripts: runs
    :func:`compute_track_qc_arrays` once and returns
    ``(storm_of_interest_array, qc)`` -- the ``[n_members, 1, n_steps, 4]``
    anchor-resolved array (see :meth:`TrackQC.storm_of_interest_array`) plus the
    full QC table, so a consumer gets both the drawing array and the analyzable
    mask / counts from a single resolution pass.
    """
    qc = compute_track_qc_arrays(
        ens_all_paths,
        era5_all_paths,
        track_box=track_box,
        anchor_frame=anchor_frame,
        free_end_frame=free_end_frame,
        anchor_radius_km=anchor_radius_km,
        min_domain_fixes=min_domain_fixes,
    )
    return qc.storm_of_interest_array(ens_all_paths), qc
