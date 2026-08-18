"""Quantify how synoptic-PCA modes sort TC track / intensity diversity.

A TC-case companion to ``aggregate_synoptic_pca.py`` that turns the
qualitative "tracks coloured by PC percentile" inset of the combined
figure into numbers a paper can quote.  The scalar-impact (heatwave)
cases already get this from the precursor -> impact regression figure;
TC cases have had no quantitative PC <-> track link until now.

Reads, per conditioning mode, the ``synoptic_pca_<mode>.npz`` written by
``compute_synoptic_pca.py`` (PC scores, EVR, EOF loadings, ERA5 PC
percentiles, track domain) and the raw all-paths
``tc_tracks_<mode>.parquet`` written by ``plot_ensemble_tc_tracks.py``.
``track_qc`` resolves each member's storm-of-interest path (anchor-first)
and marks it analyzable; every statistic below runs over that one
population.  Writes a single self-contained report per mode:

1.  **Target distribution summary** -- per-target min/max and
    percentiles across members, so heavy tails (e.g. free-end fixes far
    outside the track region) are visible in the report itself rather
    than needing a reader to notice a suspicious mean, and the full
    member range (e.g. deepest/weakest central MSL) is quotable
    directly.

2.  **Unified track QC (round 5)** -- ``track_qc.compute_track_qc``
    resolves each member's storm of interest anchor-first (the tracker
    path reaching the ERA5 fix at the pinned end -- landfall frame under
    end-conditioning, frame 0 under start-conditioning) and marks a
    member analyzable when that resolved path has at least three valid
    fixes in the track domain.  Every statistic below runs over the
    analyzable population; the exclusion taxonomy (no_track / unanchored
    / insufficient_fixes) is reported once.  No per-script anchoring or
    coherence filter remains here -- ``track_qc`` is the sole authority.

3.  **Per-PC sorting table** -- for each leading PC, the Spearman rank
    correlation between the PC score and each track/intensity target at
    the mode's free end (lat, lon, great-circle distance from the ERA5
    free-end fix, central MSL), plus the top-minus-bottom PC-decile gap
    in each target with a Welch t-test p-value.  Because fixed lat/lon
    axes dilute a mode that shifts storms along a rotated axis (the
    signal splits between coordinates whose variance is dominated by
    along-track spread), the targets also include coordinate-free
    variants: displacement from the ERA5 fix along/across ERA5's motion
    (``fe_along_km`` / ``fe_cross_km``; cross > 0 = left of motion) and
    the fix projected onto the member fix cloud's own principal axes
    (``fe_cloud_major_km`` / ``fe_cloud_minor_km``).

4.  **Multi-PC regression** -- each target regressed on the leading
    ``--n-pc`` PC scores: in-sample R^2, deterministic 10-fold
    cross-validated R^2, and the F-test p-value.  Directly comparable
    with the scalar-impact regression the heatwave figure reports.

5.  **ERA5 free-end MSL percentile** -- fraction of members with a
    finite exact-frame fix whose central MSL is at or below ERA5's,
    matching the definition behind the ``ERA5 p<N>`` annotation in
    ``aggregate_tc_tracks.py``.  Plus the fractions of finite free-end
    fixes east and north of the ERA5 free-end fix, so directional
    displacement claims ("X% of members end offshore to the east") can
    be quoted from the report.

6.  **EOF-implied geostrophic steering** -- for each EOF of a
    geopotential precursor, the geostrophic wind anomaly implied by the
    EOF loading gradient, averaged over a box centred on the ensemble's
    mean free-end storm position, scaled per +1 sigma of PC score and by
    the top-minus-bottom decile-mean score gap.  Computed at every box
    half-width in the sweep (default 2.0/2.5/3.5/5.0 deg) so the
    scale-robustness check is part of the standard output.

7.  **Loading scale check** -- per EOF, the loading's amplitude inside a
    +-5 deg storm-centred window vs. its domain-wide extrema.  A compact
    storm-scale feature dominating the window (position leakage into the
    EOF) disqualifies the steering reading; a weak, smooth tail of a
    domain-scale pattern supports it.

Track targets use each member's resolved storm-of-interest path.  The
free-end fix is the fix at the free-end frame on that path, falling back
to the nearest valid fix (fallbacks are flagged; the ERA5 MSL percentile
uses the exact-frame subset).  Non-analyzable members are NaN in the
targets, so every statistic runs over the analyzable population and
reports its n.

Outputs (in --output-dir):
    synoptic_pca_track_stats_<mode>.csv   per-PC sorting table
    synoptic_pca_track_stats_<mode>.txt   full human-readable report

Usage:
    python synoptic_pca_track_stats.py sandy
    python synoptic_pca_track_stats.py sandy --mode end --n-pc 8 \\
        --steering-box-half-deg 2.5
    python synoptic_pca_track_stats.py --list
    # --output-dir stays as an explicit override for a non-standard layout:
    python synoptic_pca_track_stats.py \\
        --output-dir /path/to/diagnostics/synoptic_pca
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ENSEMBLE_ANALYSIS_DIR = SCRIPT_DIR.parent
SHARED_DIR = SCRIPT_DIR.parent.parent / "_shared"
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(ENSEMBLE_ANALYSIS_DIR))

from _dispatch_lib import (  # noqa: E402
    list_cases,
    load_case,
    tc_tracks_parquet_path,
)
from pca_stats import (  # noqa: E402
    _kfold_r2,
    _ols_fit,
    _r2_score,
    _regression_f_pvalue,
    _spearman,
)
from tc_track_targets import (  # noqa: E402
    EARTH_RADIUS_M,
    _mean_storm_position,
    _norm_lon,
    rotated_track_targets,
)
from track_qc import ANCHOR_FRAME_WINDOW, TrackQC, compute_track_qc  # noqa: E402
from var_metadata import is_geopotential  # noqa: E402

OMEGA = 7.2921e-5  # Earth's rotation rate (rad/s)
STANDARD_GRAVITY = 9.80665  # geopotential (m^2/s^2) -> height (m)

# Modes whose free end is uniquely defined (mirrors dispatch_synoptic_pca).
STAT_MODES: Tuple[str, ...] = ("end", "start")

DECILE_FRAC = 0.1

# Default steering-box half-widths (deg): the scale-robustness sweep is part
# of the standard output, not a manual re-run loop.
DEFAULT_BOX_SWEEP: Tuple[float, ...] = (2.0, 2.5, 3.5, 5.0)

# Half-width (deg) of the storm-centred window for the loading scale check.
LOADING_BOX_HALF_DEG = 5.0

# Percentiles for the target distribution summary.
SUMMARY_PCTS: Tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _welch_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Welch t-test p-value (scipy, a declared dependency)."""
    from scipy import stats

    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def _decile_gap(scores: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
    """Top-minus-bottom PC-decile contrast in ``target``.

    Deciles are taken over the members with a finite target.  Returns
    ``(gap, p_value, delta_score)`` where ``delta_score`` is the
    top-minus-bottom decile-mean PC score (feeds the steering scaling).
    """
    ok = np.isfinite(target) & np.isfinite(scores)
    n = int(ok.sum())
    n_dec = max(int(round(n * DECILE_FRAC)), 1)
    if n < 2 * n_dec or n_dec < 2:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(scores[ok])
    t_ok = target[ok]
    bottom, top = t_ok[order[:n_dec]], t_ok[order[-n_dec:]]
    s_ok = scores[ok]
    delta_score = float(s_ok[order[-n_dec:]].mean() - s_ok[order[:n_dec]].mean())
    return float(top.mean() - bottom.mean()), _welch_p(bottom, top), delta_score


def target_summary(targets: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-target distribution summary (n, mean, min/max, SUMMARY_PCTS pcts).

    Makes heavy tails self-evident in the report: a mean far from the median
    (or an extreme p5/p95) flags free-end fixes far from the storm cluster
    without a reader having to notice it.  min/max bracket the percentile
    columns so the full member range (e.g. the deepest and weakest central
    MSL in the ensemble) is quotable directly.
    """
    rows: List[Dict[str, object]] = []
    for name, tgt in targets.items():
        ok = np.isfinite(tgt)
        row: Dict[str, object] = {"target": name, "n": int(ok.sum())}
        if ok.any():
            vals = tgt[ok]
            row["mean"] = float(vals.mean())
            row["min"] = float(vals.min())
            for p in SUMMARY_PCTS:
                row[f"p{p:g}"] = float(np.percentile(vals, p))
            row["max"] = float(vals.max())
        rows.append(row)
    return pd.DataFrame(rows)


def era5_relative_fractions(
    targets: Dict[str, np.ndarray], era5_ref: Dict[str, float]
) -> Dict[str, int]:
    """Counts of finite free-end fixes east / north of the ERA5 free-end fix.

    Longitude differences are wrapped onto [-180, 180) so the east/west split
    is meaningful on either longitude convention.  Returns an empty dict when
    no member has a finite fix.
    """
    lat = targets.get("fe_lat_deg")
    lon = targets.get("fe_lon_deg")
    if lat is None or lon is None:
        return {}
    ok = np.isfinite(lat) & np.isfinite(lon)
    n = int(ok.sum())
    if n == 0:
        return {}
    dlon = _norm_lon(np.asarray(lon)[ok] - era5_ref["lon"])
    return {
        "n": n,
        "east": int((dlon > 0).sum()),
        "north": int((np.asarray(lat)[ok] > era5_ref["lat"]).sum()),
    }


def per_pc_table(
    pc_scores: np.ndarray,
    evr: np.ndarray,
    era5_pc_pct: np.ndarray,
    targets: Dict[str, np.ndarray],
    mode: str,
) -> pd.DataFrame:
    """Per-PC Spearman + decile-gap sorting table across all targets."""
    n_pc = pc_scores.shape[1]
    rows: List[Dict[str, object]] = []
    for j in range(n_pc):
        pcj = pc_scores[:, j].astype(np.float64)
        row: Dict[str, object] = {
            "mode": mode,
            "eof": j + 1,
            "evr_pct": 100.0 * float(evr[j]),
            "era5_pc_pct": (
                float(era5_pc_pct[j]) if j < era5_pc_pct.size else float("nan")
            ),
        }
        for name, tgt in targets.items():
            ok = np.isfinite(tgt) & np.isfinite(pcj)
            row[f"{name}_n"] = int(ok.sum())
            row[f"{name}_spearman"] = _spearman(pcj, tgt)
            gap, p, _ = _decile_gap(pcj, tgt)
            row[f"{name}_decile_gap"] = gap
            row[f"{name}_decile_p"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def multi_pc_regressions(
    pc_scores: np.ndarray, targets: Dict[str, np.ndarray], n_pc: int
) -> pd.DataFrame:
    """Leading-``n_pc`` OLS of each target on the PC scores (+ 10-fold CV)."""
    k = min(n_pc, pc_scores.shape[1])
    rows: List[Dict[str, object]] = []
    for name, tgt in targets.items():
        ok = np.isfinite(tgt) & np.isfinite(pc_scores[:, :k]).all(axis=1)
        n = int(ok.sum())
        if n <= k + 1:
            rows.append({"target": name, "n": n, "n_pc": k, "r2_in": float("nan")})
            continue
        x, y = pc_scores[ok, :k].astype(np.float64), tgt[ok].astype(np.float64)
        beta, intercept = _ols_fit(x, y)
        r2_in = _r2_score(y, intercept + x @ beta)
        rows.append(
            {
                "target": name,
                "n": n,
                "n_pc": k,
                "r2_in": r2_in,
                "r2_cv10": _kfold_r2(x, y),
                "f_pvalue": _regression_f_pvalue(r2_in, n, k),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EOF-implied geostrophic steering at the storm position
# ---------------------------------------------------------------------------


def steering_table(
    diag: dict,
    pc_scores: np.ndarray,
    targets: Dict[str, np.ndarray],
    box_half_degs: Sequence[float],
    storm_pos: Tuple[float, float],
) -> pd.DataFrame | None:
    """Geostrophic wind anomaly implied by each EOF at the mean storm fix.

    The loading (``pc_components_latlon``, physical storage units per unit
    PC score) is differenced on the lat/lon grid; ``u_g = -(1/f) dPhi/dy``
    and ``v_g = (1/f) dPhi/dx`` (geopotential Phi in m^2 s^-2), averaged over
    a storm-centred box at every half-width in ``box_half_degs`` -- the
    scale-robustness sweep is part of the table (a domain-scale gradient is
    stable across box sizes; a compact storm-scale feature decays).  Only
    defined for geopotential precursors; returns None otherwise.
    """
    var0 = str(np.asarray(diag["variables"]).reshape(-1)[0])
    if not is_geopotential(var0):
        print(
            f"[track-stats] steering: precursor '{var0}' is not geopotential; "
            f"skipping the geostrophic-steering table."
        )
        return None

    lat = np.asarray(diag["domain_lat"], dtype=np.float64)
    lon = np.asarray(diag["domain_lon"], dtype=np.float64)
    loadings = np.asarray(diag["pc_components_latlon"], dtype=np.float64)[:, 0]

    lat0, lon0 = storm_pos
    f_cor = 2.0 * OMEGA * np.sin(np.radians(lat0))
    if abs(f_cor) < 1e-6:
        return None

    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    rows: List[Dict[str, object]] = []
    for j in range(loadings.shape[0]):
        phi = loadings[j]
        dphi_dy = np.gradient(phi, lat_rad, axis=0) / EARTH_RADIUS_M
        dphi_dx = np.gradient(phi, lon_rad, axis=1) / (
            EARTH_RADIUS_M * np.cos(lat_rad)[:, None]
        )
        sigma = float(np.std(pc_scores[:, j]))
        _, _, delta_score = _decile_gap(
            pc_scores[:, j].astype(np.float64), targets["fe_lat_deg"]
        )
        for half in box_half_degs:
            in_lat = np.abs(lat - lat0) <= half
            # Wrapped difference so a precursor grid on either longitude
            # convention ([-180, 180] or [0, 360)) matches the storm position.
            in_lon = np.abs(_norm_lon(lon - lon0)) <= half
            if in_lat.sum() < 2 or in_lon.sum() < 2:
                print(
                    f"[track-stats] steering: box +-{half} deg at "
                    f"({lat0:.1f}N, {lon0:.1f}E) has too few grid points; "
                    f"skipping this width."
                )
                continue
            u_g = float(np.mean(-dphi_dy[np.ix_(in_lat, in_lon)]) / f_cor)
            v_g = float(np.mean(dphi_dx[np.ix_(in_lat, in_lon)]) / f_cor)
            rows.append(
                {
                    "eof": j + 1,
                    "box_half_deg": half,
                    "u_g_per_sigma_ms": u_g * sigma,
                    "v_g_per_sigma_ms": v_g * sigma,
                    "speed_per_sigma_ms": float(np.hypot(u_g, v_g)) * sigma,
                    "u_g_decile_gap_ms": u_g * delta_score,
                    "v_g_decile_gap_ms": v_g * delta_score,
                    "speed_decile_gap_ms": float(np.hypot(u_g, v_g)) * abs(delta_score),
                }
            )
    if not rows:
        return None
    return pd.DataFrame(rows)


def loading_scale_table(
    diag: dict, storm_pos: Tuple[float, float]
) -> pd.DataFrame | None:
    """Per-EOF loading amplitude near the storm vs. domain-wide extrema.

    The position-leak check, mechanized: for each EOF, the loading's min /
    max / std inside a ``+-LOADING_BOX_HALF_DEG`` storm-centred window
    against its domain-wide min / max.  A window that contains amplitudes
    comparable to the domain extrema (or a sign flip across a compact
    feature) indicates the storm's own signature leaked into the EOF and the
    steering reading is circular; a weak, smooth tail supports it.
    Geopotential loadings are reported as height (m); other variables stay in
    storage units.
    """
    lat = np.asarray(diag["domain_lat"], dtype=np.float64)
    lon = np.asarray(diag["domain_lon"], dtype=np.float64)
    loadings = np.asarray(diag["pc_components_latlon"], dtype=np.float64)[:, 0]
    var0 = str(np.asarray(diag["variables"]).reshape(-1)[0])
    scale = STANDARD_GRAVITY if is_geopotential(var0) else 1.0

    lat0, lon0 = storm_pos
    in_lat = np.abs(lat - lat0) <= LOADING_BOX_HALF_DEG
    in_lon = np.abs(_norm_lon(lon - lon0)) <= LOADING_BOX_HALF_DEG
    if in_lat.sum() < 2 or in_lon.sum() < 2:
        return None

    rows: List[Dict[str, object]] = []
    for j in range(loadings.shape[0]):
        field = loadings[j] / scale
        box = field[np.ix_(in_lat, in_lon)]
        rows.append(
            {
                "eof": j + 1,
                "domain_min": float(field.min()),
                "domain_max": float(field.max()),
                "box_min": float(box.min()),
                "box_max": float(box.max()),
                "box_std": float(box.std()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_df(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.3g}")


def write_report(
    path: Path,
    mode: str,
    case_name: str,
    n_members: int,
    free_end_frame: int,
    era5_ref: Dict[str, float],
    qc: TrackQC,
    summary: pd.DataFrame,
    table: pd.DataFrame,
    regressions: pd.DataFrame,
    steering: pd.DataFrame | None,
    loading_scale: pd.DataFrame | None,
    storm_pos: Tuple[float, float] | None,
    rotated_info: Dict[str, float] | None = None,
    era5_rel: Dict[str, int] | None = None,
) -> None:
    """Write the human-readable per-mode report (analyzable population)."""
    counts = qc.counts()
    msl_pct, msl_n = qc.era5_free_end_msl_pct()
    lines = [
        f"Synoptic-PCA track/intensity sorting report -- {case_name}",
        f"mode={mode} (free end = frame {free_end_frame}), n_members={n_members}",
        f"analyzable={counts['analyzable']}, excluded={counts['excluded']} "
        f"(no_track {counts['no_track']} / unanchored {counts['unanchored']} / "
        f"insufficient_fixes {counts['insufficient_fixes']}); "
        f"exact-frame free-end fixes={counts['exact_frame']}, "
        f"nearest-frame fallbacks={counts['fe_fallback']}",
        f"ERA5 free-end fix: {era5_ref['lat']:.2f}N {era5_ref['lon']:.2f}E, "
        f"{era5_ref['msl_hpa']:.1f} hPa",
    ]

    anchor = qc.era5_anchor
    if anchor is not None:
        lines += [
            "",
            "Unified track QC -- anchor-first storm-of-interest resolution "
            f"(ERA5 anchor fix {anchor['lat']:.2f}N {anchor['lon']:.2f}E at "
            f"frame {anchor['frame']} +-{ANCHOR_FRAME_WINDOW}, radius "
            f"{anchor['radius_km']:g} km):",
            f"  analyzable (resolved path, >=3 in-domain fixes): "
            f"{counts['analyzable']}",
            f"  excluded -- no_track (no in-domain fix)         : "
            f"{counts['no_track']}",
            f"  excluded -- unanchored (no path reaches anchor) : "
            f"{counts['unanchored']}",
            f"  excluded -- insufficient_fixes (<3 on resolved) : "
            f"{counts['insufficient_fixes']}",
        ]
    else:
        lines += [
            "",
            "Unified track QC: anchor window undefined for this mode; "
            "resolution fell back to a domain-only rule (most in-domain fixes).",
            f"  analyzable: {counts['analyzable']}, excluded: "
            f"{counts['excluded']} (no_track {counts['no_track']}, "
            f"insufficient_fixes {counts['insufficient_fixes']})",
        ]

    lines += [
        "",
        "Targets (per member, at the mode's free end, resolved storm path):",
        "  fe_lat_deg / fe_lon_deg  storm-centre fix",
        "  fe_dist_era5_km          great-circle distance from the ERA5 fix",
        "  fe_msl_hpa               central mean sea level pressure",
        "  fe_along_km / fe_cross_km          displacement from the ERA5 fix",
        "                           along/across ERA5's motion (cross > 0 =",
        "                           left of motion) -- coordinate-free",
        "  fe_cloud_major_km / fe_cloud_minor_km  fix projected onto the fix",
        "                           cloud's own principal axes (cloud-mean",
        "                           centered) -- coordinate-free",
    ]
    if rotated_info:
        lines.append(
            "Rotated frames: ERA5 motion bearing "
            f"{rotated_info['era5_motion_bearing_deg']:.0f} deg"
        )
        if "cloud_major_bearing_deg" in rotated_info:
            lines += [
                f"  fix-cloud major axis bearing "
                f"{rotated_info['cloud_major_bearing_deg']:.0f} deg "
                f"({rotated_info['cloud_major_var_pct']:.0f}% of cloud "
                f"variance); cloud lat-lon corr "
                f"{rotated_info['cloud_latlon_corr']:.2f}",
                f"  ERA5 fix in the cloud frame: major "
                f"{rotated_info['era5_cloud_major_km']:.0f} km, minor "
                f"{rotated_info['era5_cloud_minor_km']:.0f} km",
            ]
    lines += [
        "",
        "Target distribution across members (tail check -- a mean far from",
        "the median flags free-end fixes far outside the storm cluster):",
        _fmt_df(summary),
        "",
        f"ERA5 free-end central MSL percentile (exact-frame analyzable "
        f"members, n={msl_n}; matches aggregate_tc_tracks's 'ERA5 p<N>' "
        f"label): {msl_pct:.1f}",
    ]
    if era5_rel:
        n_rel = era5_rel["n"]
        lines += [
            "",
            "Free-end fixes relative to the ERA5 free-end fix "
            f"(finite-fix members, n={n_rel}):",
            f"  east of the ERA5 fix : {era5_rel['east']}/{n_rel} "
            f"({100.0 * era5_rel['east'] / n_rel:.1f}%)",
            f"  north of the ERA5 fix: {era5_rel['north']}/{n_rel} "
            f"({100.0 * era5_rel['north'] / n_rel:.1f}%)",
        ]
    lines += [
        "",
        "Per-PC sorting table (Spearman rho; top-minus-bottom decile gap):",
        _fmt_df(table.drop(columns=["mode"])),
        "",
        f"Multi-PC OLS (leading {int(regressions['n_pc'].iloc[0])} PCs; "
        f"r2_cv10 = deterministic 10-fold out-of-sample R^2):",
        _fmt_df(regressions),
    ]
    if storm_pos is not None and steering is not None:
        lines += [
            "",
            "EOF-implied geostrophic steering at the ensemble-mean free-end "
            f"storm position ({storm_pos[0]:.1f}N, {storm_pos[1]:.1f}E), "
            "per box half-width (deg):",
            "  per_sigma   = steering anomaly per +1 std-dev of the PC score",
            "  decile_gap  = steering anomaly across the top-minus-bottom "
            "PC decile gap",
            "  Stable across box sizes -> domain-scale gradient; decaying "
            "with box size -> compact (storm-scale) feature.",
            _fmt_df(steering),
        ]
    if storm_pos is not None and loading_scale is not None:
        lines += [
            "",
            f"Loading amplitude near the storm (+-{LOADING_BOX_HALF_DEG:g} deg "
            "box) vs. domain extrema (geopotential shown as height, m):",
            "  Box amplitudes comparable to the domain extrema (or a sign "
            "flip across a compact feature) = position leakage;",
            "  a weak smooth tail of the domain pattern = environmental.",
            _fmt_df(loading_scale),
        ]
    path.write_text("\n".join(lines) + "\n")
    print(f"[track-stats] wrote {path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_mode(
    output_dir: Path,
    mode: str,
    *,
    tracks_root: Path,
    tracker_override: str | None,
    n_pc: int,
    box_half_degs: Sequence[float],
    anchor_radius_km: float,
) -> None:
    """Compute and write the sorting report for one conditioning mode."""
    npz_path = output_dir / f"synoptic_pca_{mode}.npz"
    if not npz_path.exists():
        print(f"[track-stats] {npz_path} missing; skipping mode '{mode}'.")
        return
    with np.load(npz_path, allow_pickle=False) as data:
        diag = {k: data[k] for k in data.files}

    if str(np.asarray(diag["impact_kind"])) != "track":
        print(
            f"[track-stats] mode '{mode}' has impact_kind="
            f"{np.asarray(diag['impact_kind'])!s} (not 'track'); this "
            f"diagnostic is TC-only. Skipping."
        )
        return

    tracker = tracker_override or str(np.asarray(diag["tc_tracker"]))
    parquet = tc_tracks_parquet_path(str(tracks_root), mode, tracker)
    if not Path(parquet).exists():
        print(f"[track-stats] {parquet} missing; skipping mode '{mode}'.")
        return

    track_box_arr = np.asarray(diag.get("tc_domain", np.zeros(0)), dtype=float).reshape(-1)
    if track_box_arr.size != 4:
        print(
            f"[track-stats] mode '{mode}': npz has no track domain (tc_domain); "
            f"cannot resolve the storm of interest. Skipping."
        )
        return
    track_box = (
        float(track_box_arr[0]),
        float(track_box_arr[1]),
        float(track_box_arr[2]),
        float(track_box_arr[3]),
    )

    n_members = int(diag["n_members"])
    free_end_frame = int(diag["free_end_frame"])
    n_leads = int(np.asarray(diag["lead_hours"]).size)

    # The anchor is the *pinned* end -- the frame where every conditioned
    # member must coincide with ERA5: the landfall frame (last frame if the
    # case sets none) under end-conditioning, frame 0 under start-conditioning.
    landfall_frame = int(diag.get("landfall_frame", np.int32(-1)))
    if mode == "start":
        anchor_frame = 0
    else:
        anchor_frame = landfall_frame if 0 <= landfall_frame < n_leads else n_leads - 1

    # Single QC authority: anchor-first storm-of-interest resolution.  Targets
    # are NaN for non-analyzable members, so every statistic below runs over
    # the one analyzable population.
    qc = compute_track_qc(
        parquet,
        track_box=track_box,
        anchor_frame=anchor_frame,
        free_end_frame=free_end_frame,
        n_members=n_members,
        anchor_radius_km=anchor_radius_km,
    )
    era5_ref = qc.era5_free_end
    targets = qc.free_end_targets()
    rotated, rotated_info = rotated_track_targets(parquet, targets, free_end_frame)
    targets.update(rotated)
    pc_scores = np.asarray(diag["pc_scores"], dtype=np.float64)
    evr = np.asarray(diag["explained_variance_ratio"], dtype=np.float64)
    era5_pc_pct = np.asarray(diag["era5_pc_percentile"], dtype=np.float64)

    summary = target_summary(targets)
    era5_rel = era5_relative_fractions(targets, era5_ref)
    table = per_pc_table(pc_scores, evr, era5_pc_pct, targets, mode)
    stem = f"synoptic_pca_track_stats_{mode}"
    csv_path = output_dir / f"{stem}.csv"
    table.to_csv(csv_path, index=False)
    print(f"[track-stats] wrote {csv_path}")

    regressions = multi_pc_regressions(pc_scores, targets, n_pc)
    storm_pos = _mean_storm_position(targets)
    steering = None
    loading_scale = None
    if storm_pos is None:
        print("[track-stats] steering: no valid storm positions; skipping.")
    else:
        steering = steering_table(diag, pc_scores, targets, box_half_degs, storm_pos)
        loading_scale = loading_scale_table(diag, storm_pos)

    write_report(
        output_dir / f"{stem}.txt",
        mode,
        str(np.asarray(diag["case_name"])),
        n_members,
        free_end_frame,
        era5_ref,
        qc,
        summary,
        table,
        regressions,
        steering,
        loading_scale,
        storm_pos,
        rotated_info,
        era5_rel,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify PC <-> TC-track/intensity sorting for the "
        "synoptic-PCA diagnostic (TC cases only)."
    )
    parser.add_argument(
        "case",
        nargs="?",
        default=None,
        help="Case name (e.g. sandy, ian). Resolves the diagnostics paths from "
        "the case YAML, exactly like dispatch_synoptic_pca.py -- so "
        "`synoptic_pca_track_stats.py sandy` just works. Use --list to see "
        "available cases. Overridden by --output-dir when both are given.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available cases and exit."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory holding synoptic_pca_<mode>.npz (outputs land here). "
        "Optional escape hatch: overrides the path resolved from the case "
        "name for a non-standard layout.",
    )
    parser.add_argument(
        "--tc-tracks-root",
        default=None,
        help="Diagnostics root containing tc_tracks/<tracker>/ (default: the "
        "parent of the output dir).",
    )
    parser.add_argument(
        "--tracker",
        default=None,
        help="Tracker whose parquet to read (default: the npz's tc_tracker).",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        choices=list(STAT_MODES),
        help=f"Restrict to one or more modes (default: all of {list(STAT_MODES)}).",
    )
    parser.add_argument(
        "--n-pc",
        type=int,
        default=8,
        help="Leading PCs in the multi-PC regression (default 8, matching "
        "the scalar-impact precursor regression).",
    )
    parser.add_argument(
        "--steering-box-half-deg",
        type=float,
        action="append",
        default=None,
        dest="box_half_degs",
        help="Half-width (deg) of the storm-centred steering box (repeatable; "
        f"default sweep: {list(DEFAULT_BOX_SWEEP)}).",
    )
    parser.add_argument(
        "--anchor-radius-km",
        type=float,
        default=500.0,
        help="Great-circle radius (km) within which a member's path must "
        "approach the ERA5 fix at the pinned-end frame to reach the anchor "
        "(default 500).",
    )
    args = parser.parse_args()

    if args.list:
        print("Available cases:")
        for name in list_cases():
            print(f"  {name}")
        return

    # Resolve the diagnostics dir the same way dispatch_synoptic_pca.py does
    # (``<case base>/diagnostics/synoptic_pca``), so a bare case name is all
    # that is normally needed; --output-dir stays as an explicit override.
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.case is not None:
        case = load_case(args.case)
        output_dir = Path(case["base"]) / "diagnostics" / "synoptic_pca"
    else:
        parser.error("give a case name (e.g. 'sandy') or --output-dir; --list to list")

    tracks_root = (
        Path(args.tc_tracks_root) if args.tc_tracks_root else output_dir.parent
    )
    for mode in args.mode or list(STAT_MODES):
        run_mode(
            output_dir,
            mode,
            tracks_root=tracks_root,
            tracker_override=args.tracker,
            n_pc=args.n_pc,
            box_half_degs=args.box_half_degs or list(DEFAULT_BOX_SWEEP),
            anchor_radius_km=args.anchor_radius_km,
        )
    print("[track-stats] done.")


if __name__ == "__main__":
    main()
