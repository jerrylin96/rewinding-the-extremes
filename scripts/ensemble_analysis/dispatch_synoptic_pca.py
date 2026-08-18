"""Dispatch synoptic-PCA jobs for a named case study.

Reads ``cases/<event>.yaml`` and submits one qsub job per conditioning
mode (start, end) plus a single dependent aggregator job that renders the
combined precursor -> impact figures, the EOF-pattern supplements, and the
PC-scatter supplements.

The ``both`` conditioning mode is intentionally skipped: with both ends
pinned the choice of "free-end" frame is arbitrary.

The diagnostic does a PCA on a synoptic *precursor* field (z500 by
default; msl can be added via --variable) at the free-end frame, picks the
ensemble members closest to fixed PC percentiles of each leading EOF, and
shows how that precursor diversity manifests in the event *impact*:

  * cases with a ``tc_tracks`` block  -> TC track impact (the cached
    TempestExtremes parquet is reused; run the tc_tracks diagnostic first).
  * other cases                        -> scalar impact (bbox-mean impact
    variable, e.g. t2m land-masked, from ``regions.impact`` +
    ``diagnostics.free_end_states.mask``).

Output layout (modes aggregated)::

    <output_root>/synoptic_pca/
        synoptic_pca_start.npz
        synoptic_pca_end.npz
        synoptic_pca_combined_<mode>_eof<i>.png   (precursor maps + impact;
                                                   TC: steering arrows + track)
        synoptic_pca_video_<mode>_eof<i>.mp4      (TC: p5/p50/p95 MSL animation)
        synoptic_pca_eofs_<var>.png               (EOF loading patterns)
        synoptic_pca_pc_scatter_<mode>.png        (PC1-PC2 scatter)
        synoptic_pca_diversity.png                (spread + dimensionality)

Usage:
    python dispatch_synoptic_pca.py sandy
    python dispatch_synoptic_pca.py pnw_heatwave --dry-run
    python dispatch_synoptic_pca.py sandy --mode end
    python dispatch_synoptic_pca.py sandy --n-eof-figures 1
    python dispatch_synoptic_pca.py sandy --variable z500 --variable msl
    python dispatch_synoptic_pca.py sandy --lon-min -100 --lon-max -30 \\
        --lat-min 25 --lat-max 75
    python dispatch_synoptic_pca.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "_shared"))

from _dispatch_lib import (  # noqa: E402
    build_qsub_cmd,
    case_ensemble_size,
    ensemble_zarr_path,
    era5_zarr_path,
    iter_modes,
    list_cases,
    load_case,
    mode_frames,
    region_box,
    require_region,
    resolve_landfall_frame,
    resolve_timezone,
    start_time_iso_from_case,
    submit_qsub,
    tc_tracks_parquet_path,
    trackers_from_case,
    validate_zarrs_exist,
)

PCA_SCRIPT = str(SCRIPT_DIR / "submission_scripts" / "submit_synoptic_pca.sh")
AGG_SCRIPT = str(SCRIPT_DIR / "submission_scripts" / "submit_aggregate_synoptic_pca.sh")
PCA_LOG_DIR = str(SCRIPT_DIR / "submission_scripts" / "synoptic_pca_logs")

DEFAULT_H_RT = "2:00:00"
AGG_H_RT = "0:30:00"

# Generous synoptic box default — captures the upstream wave train /
# blocking patterns; the per-case YAML or CLI flags override these.
DEFAULT_LON_MIN = -130.0
DEFAULT_LON_MAX = -20.0
DEFAULT_LAT_MIN = 20.0
DEFAULT_LAT_MAX = 70.0

# Single-field precursor PCA on the synoptic height field.  z500 is the
# canonical steering/blocking precursor and carries the dynamical diversity
# for all cases; msl is largely its balanced twin and (for TCs) leaks the
# storm's own low into the precursor.  msl stays available ad hoc via
# --variable / the case YAML, but is not the default.
DEFAULT_VARIABLES: Tuple[str, ...] = ("z500",)

# Raw (u, v) wind components whose flow the aggregator draws as steering arrows
# on the TC precursor panels.  500 hPa matches the headline z500 precursor (the
# arrows are the flow around those height contours).  Only passed for TC cases;
# overridable via --wind-variable or the case YAML.
DEFAULT_WIND_VARIABLES: Tuple[str, ...] = ("u500", "v500")

# Variable for the per-EOF percentile-member animation; MSL shows the hurricane
# low most clearly.  Only passed for TC cases; overridable.
DEFAULT_VIDEO_VARIABLE = "msl"

DEFAULT_MAX_D = 10
DEFAULT_N_EOF = 3

# Modes whose free end is uniquely defined.
PCA_MODES: Tuple[str, ...] = ("start", "end")


def pca_block(case: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the optional ``synoptic_pca`` block from a case YAML."""
    return case.get("synoptic_pca") or {}


def resolve_domain(
    case: Dict[str, Any],
    *,
    lon_min: Optional[float],
    lon_max: Optional[float],
    lat_min: Optional[float],
    lat_max: Optional[float],
) -> Tuple[float, float, float, float]:
    """CLI > regions.synoptic > dispatcher default precedence for the lat/lon box."""
    yaml_box = region_box(case, "synoptic")
    y_lon_min, y_lon_max, y_lat_min, y_lat_max = (
        yaml_box if yaml_box is not None else (None, None, None, None)
    )

    return (
        (
            lon_min
            if lon_min is not None
            else (y_lon_min if y_lon_min is not None else DEFAULT_LON_MIN)
        ),
        (
            lon_max
            if lon_max is not None
            else (y_lon_max if y_lon_max is not None else DEFAULT_LON_MAX)
        ),
        (
            lat_min
            if lat_min is not None
            else (y_lat_min if y_lat_min is not None else DEFAULT_LAT_MIN)
        ),
        (
            lat_max
            if lat_max is not None
            else (y_lat_max if y_lat_max is not None else DEFAULT_LAT_MAX)
        ),
    )


def resolve_variables(
    case: Dict[str, Any], cli_variables: Optional[List[str]]
) -> List[str]:
    """CLI > case YAML > dispatcher default precedence for the PCA variables."""
    if cli_variables:
        return list(cli_variables)
    yaml_vars = pca_block(case).get("variables")
    if yaml_vars:
        return list(yaml_vars)
    return list(DEFAULT_VARIABLES)


def resolve_wind_variables(
    case: Dict[str, Any], cli_wind_variables: Optional[List[str]]
) -> List[str]:
    """CLI > case YAML > dispatcher default for the raw-wind overlay variables.

    These (u, v) components are drawn as steering arrows on the TC precursor
    panels.  Consumed only for TC cases (see ``submit_pca_for_mode``).
    """
    if cli_wind_variables:
        return list(cli_wind_variables)
    yaml_winds = pca_block(case).get("wind_variables")
    if yaml_winds:
        return list(yaml_winds)
    return list(DEFAULT_WIND_VARIABLES)


def resolve_video_variable(
    case: Dict[str, Any], cli_video_variable: Optional[str]
) -> str:
    """CLI > case YAML > dispatcher default for the animation variable.

    Drives the aggregator's per-EOF percentile-member animation; consumed only
    for TC cases.
    """
    if cli_video_variable:
        return cli_video_variable
    yaml_video = pca_block(case).get("video_variable")
    return str(yaml_video) if yaml_video else DEFAULT_VIDEO_VARIABLE


def resolve_n_eof_figures(
    case: Dict[str, Any], cli_value: Optional[int]
) -> Optional[int]:
    """CLI > case YAML > None precedence for the combined-figure EOF count.

    ``None`` lets the aggregator fall back to its own default of rendering
    one combined precursor -> impact figure per EOF baked into the npz
    (``n_eof_show`` = ``min(n_eof, max_d)``).
    """
    if cli_value is not None:
        return cli_value
    yaml_val = pca_block(case).get("n_eof_figures")
    return int(yaml_val) if yaml_val is not None else None


def resolve_n_eof(case: Dict[str, Any], cli_value: Optional[int]) -> int:
    """CLI > case YAML > dispatcher default precedence for the analyzed-EOF count.

    ``n_eof`` is the number of leading EOFs the compute job bakes
    percentile-member fields for (precursor maps, impact panels, EOF
    patterns).  Mirrors :func:`resolve_n_eof_figures`'s YAML hook so a case
    whose precursor diversity is high-dimensional (many modes clearing the
    North 1982 separation test) can analyze more than the conservative
    default without a standing CLI flag.  Falls back to
    :data:`DEFAULT_N_EOF` when neither CLI nor YAML sets it.
    """
    if cli_value is not None:
        return cli_value
    yaml_val = pca_block(case).get("n_eof")
    return int(yaml_val) if yaml_val is not None else DEFAULT_N_EOF


def _resolve_impact_variable(case: Dict[str, Any]) -> str:
    """Impact variable for scalar (non-TC) cases.

    Priority: ``synoptic_pca.impact_variable`` > the variable that
    ``diagnostics.free_end_states.mask`` masks (the case's intensity
    variable, e.g. t2m for the heatwave) > ``t2m``.
    """
    blk = pca_block(case)
    if blk.get("impact_variable"):
        return str(blk["impact_variable"])
    mask = ((case.get("diagnostics", {}) or {}).get("free_end_states", {}) or {}).get(
        "mask", {}
    ) or {}
    if mask:
        return str(next(iter(mask.keys())))
    return "t2m"


def resolve_impact_args(
    case: Dict[str, Any],
    mode: str,
    output_root: str,
    *,
    impact_variable_override: Optional[str],
    tracker_override: Optional[str],
) -> List[str]:
    """Build the impact-related compute args for one mode.

    Cases with a ``tc_tracks`` block get the TC-track impact (reusing the
    cached per-mode parquet); all others get the scalar impact.
    """
    tc_block = case.get("tc_tracks")
    if tc_block:
        trackers = trackers_from_case(
            case, override=[tracker_override] if tracker_override else None
        )
        tracker = trackers[0]
        lon_min, lon_max, lat_min, lat_max = require_region(case, "track")
        parquet = tc_tracks_parquet_path(output_root, mode, tracker)
        return [
            "--impact-kind",
            "track",
            "--tc-parquet",
            parquet,
            "--tc-tracker",
            tracker,
            "--tc-domain",
            str(lon_min),
            str(lon_max),
            str(lat_min),
            str(lat_max),
            "--landfall-frame",
            str(resolve_landfall_frame(case)),
        ]

    # Scalar impact: the single effect region (regions.impact), shared
    # pixel-for-pixel with the free_end_states bbox-mean so there is exactly
    # one impact averaging.  The mask comes from the same
    # diagnostics.free_end_states.mask the free_end_states diagnostic applies.
    impact_var = impact_variable_override or _resolve_impact_variable(case)
    box = region_box(case, "impact")
    if box is None:
        print(
            f"[{case['name']}] WARNING: no regions.impact for scalar impact; "
            f"the combined figure will omit the impact panel."
        )
        return ["--impact-kind", "none"]
    lon_min, lon_max, lat_min, lat_max = box
    mask = (
        ((case.get("diagnostics", {}) or {}).get("free_end_states", {}) or {}).get(
            "mask", {}
        )
        or {}
    ).get(impact_var, "none")
    return [
        "--impact-kind",
        "scalar",
        "--impact-variable",
        impact_var,
        "--impact-bbox",
        str(lon_min),
        str(lon_max),
        str(lat_min),
        str(lat_max),
        "--impact-mask",
        str(mask),
    ]


def submit_pca_for_mode(
    case: Dict[str, Any],
    mode: str,
    *,
    ensemble_size: int,
    diag_output_dir: str,
    output_root: str,
    variables: List[str],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    max_d: int,
    n_eof: int,
    impact_variable_override: Optional[str],
    tracker_override: Optional[str],
    wind_variables: List[str],
    video_variable: str,
    h_rt: str,
    dry_run: bool,
) -> Optional[str]:
    """Submit one qsub job for a single conditioning mode."""
    base = case["base"]
    ens_zarr = ensemble_zarr_path(base, mode, ensemble_size)
    era5_zarr = era5_zarr_path(base, mode)

    script_args = [
        "--ensemble-zarr",
        ens_zarr,
        "--era5-zarr",
        era5_zarr,
        "--output-dir",
        diag_output_dir,
        "--mode",
        mode,
        "--case-name",
        case["display_name"],
        "--start-time",
        start_time_iso_from_case(case),
        "--timezone",
        resolve_timezone(case),
        "--lon-min",
        str(lon_min),
        "--lon-max",
        str(lon_max),
        "--lat-min",
        str(lat_min),
        "--lat-max",
        str(lat_max),
        "--max-d",
        str(max_d),
        "--n-eof",
        str(n_eof),
    ]
    for v in variables:
        script_args += ["--variable", v]
    for frame in mode_frames(mode):
        script_args += ["--conditioning-frame", str(frame)]
    script_args += resolve_impact_args(
        case,
        mode,
        output_root,
        impact_variable_override=impact_variable_override,
        tracker_override=tracker_override,
    )
    # Raw-wind steering arrows and the MSL animation are a TC-track story (why
    # the storm goes one way vs another), so only TC cases pass these; others
    # omit them and the aggregator simply draws no arrows / renders no video.
    if case.get("tc_tracks"):
        for v in wind_variables:
            script_args += ["--wind-variable", v]
        if video_variable:
            script_args += ["--video-variable", video_variable]

    cmd = build_qsub_cmd(
        script=PCA_SCRIPT,
        job_name=f"SynPCA_{case['name']}_{mode}",
        h_rt=h_rt,
        script_args=script_args,
    )
    return submit_qsub(
        cmd,
        dry_run=dry_run,
        cwd=str(SCRIPT_DIR / "submission_scripts"),
        log_dir=PCA_LOG_DIR,
    )


def submit_aggregator(
    case: Dict[str, Any],
    *,
    diag_output_dir: str,
    hold_jids: List[str],
    n_eof_figures: Optional[int] = None,
    dry_run: bool,
) -> Optional[str]:
    """Submit the cross-mode aggregator job."""
    script_args = ["--output-dir", diag_output_dir]
    if n_eof_figures is not None:
        script_args += ["--n-eof-figures", str(n_eof_figures)]
    cmd = build_qsub_cmd(
        script=AGG_SCRIPT,
        job_name=f"AggSynPCA_{case['name']}",
        h_rt=AGG_H_RT,
        script_args=script_args,
        hold_jids=hold_jids,
    )
    return submit_qsub(
        cmd,
        dry_run=dry_run,
        cwd=str(SCRIPT_DIR / "submission_scripts"),
        log_dir=PCA_LOG_DIR,
    )


def dispatch(
    case_name: str,
    *,
    only_modes: Optional[List[str]] = None,
    ensemble_size_override: Optional[int] = None,
    output_root_override: Optional[str] = None,
    variables: Optional[List[str]] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    max_d: int = DEFAULT_MAX_D,
    n_eof: Optional[int] = None,
    n_eof_figures: Optional[int] = None,
    impact_variable: Optional[str] = None,
    tracker: Optional[str] = None,
    wind_variables: Optional[List[str]] = None,
    video_variable: Optional[str] = None,
    h_rt: str = DEFAULT_H_RT,
    dry_run: bool = False,
) -> Dict[str, Optional[str]]:
    """Submit synoptic-PCA jobs for a case."""
    case = load_case(case_name)
    ensemble_size = case_ensemble_size(case, override=ensemble_size_override)

    case_modes = [m for m, _ in iter_modes(case) if m in PCA_MODES]
    modes_to_run = case_modes
    if only_modes:
        modes_to_run = [m for m in case_modes if m in only_modes]

    if not modes_to_run:
        raise SystemExit(
            f"ERROR: no PCA-eligible modes to run for case '{case['name']}' "
            f"(eligible: {list(PCA_MODES)}, configured: {case_modes}, "
            f"--mode: {only_modes})."
        )

    domain = resolve_domain(
        case, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max
    )
    resolved_vars = resolve_variables(case, variables)
    resolved_wind_vars = resolve_wind_variables(case, wind_variables)
    resolved_video_var = resolve_video_variable(case, video_variable)
    resolved_n_eof_figures = resolve_n_eof_figures(case, n_eof_figures)
    resolved_n_eof = resolve_n_eof(case, n_eof)
    if resolved_n_eof > max_d:
        print(
            f"[{case['name']}] WARNING: n_eof={resolved_n_eof} exceeds "
            f"max_d={max_d}; the compute job will cap analyzed EOFs at {max_d}. "
            f"Raise --max-d to bake percentile-member fields for all "
            f"{resolved_n_eof} EOFs."
        )

    output_root = output_root_override or f"{case['base']}/diagnostics"
    diag_output_dir = f"{output_root}/synoptic_pca"

    is_tc = bool(case.get("tc_tracks"))
    print(
        f"[{case['name']}] submitting {len(modes_to_run)} synoptic-PCA "
        f"job(s): {modes_to_run}"
    )
    print(
        f"[{case['name']}] domain     : lon ∈ [{domain[0]:g}, {domain[1]:g}], "
        f"lat ∈ [{domain[2]:g}, {domain[3]:g}]"
    )
    print(
        f"[{case['name']}] PCA vars   : {resolved_vars}  "
        f"(max_d={max_d}, n_eof={resolved_n_eof})"
    )
    eof_fig_desc = (
        "all analyzed EOFs"
        if resolved_n_eof_figures is None
        else str(resolved_n_eof_figures)
    )
    print(f"[{case['name']}] EOF figures: {eof_fig_desc}")
    if is_tc:
        trackers = trackers_from_case(case, override=[tracker] if tracker else None)
        print(
            f"[{case['name']}] impact     : TC track (tracker={trackers[0]}); "
            f"reuses cached tc_tracks parquet -- run the tc_tracks diagnostic first"
        )
        print(
            f"[{case['name']}] winds      : raw {resolved_wind_vars} steering "
            f"arrows on the precursor panels"
        )
        print(
            f"[{case['name']}] video      : {resolved_video_var} percentile-member "
            f"animation (per EOF)"
        )
    else:
        print(
            f"[{case['name']}] impact     : scalar "
            f"({impact_variable or _resolve_impact_variable(case)} bbox-mean)"
        )

    validate_zarrs_exist(case, modes_to_run, ensemble_size)

    results: Dict[str, Optional[str]] = {}
    for mode in modes_to_run:
        jid = submit_pca_for_mode(
            case,
            mode,
            ensemble_size=ensemble_size,
            diag_output_dir=diag_output_dir,
            output_root=output_root,
            variables=resolved_vars,
            lon_min=domain[0],
            lon_max=domain[1],
            lat_min=domain[2],
            lat_max=domain[3],
            max_d=max_d,
            n_eof=resolved_n_eof,
            impact_variable_override=impact_variable,
            tracker_override=tracker,
            wind_variables=resolved_wind_vars,
            video_variable=resolved_video_var,
            h_rt=h_rt,
            dry_run=dry_run,
        )
        results[mode] = jid

    hold_jids = [j for j in results.values() if j]
    results["aggregate"] = submit_aggregator(
        case,
        diag_output_dir=diag_output_dir,
        hold_jids=hold_jids,
        n_eof_figures=resolved_n_eof_figures,
        dry_run=dry_run,
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch synoptic-PCA jobs for a named case.",
    )
    parser.add_argument("case", nargs="?", help="Case name (e.g. sandy, pnw_heatwave)")
    parser.add_argument(
        "--list", action="store_true", help="List available cases and exit."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        choices=list(PCA_MODES),
        help="Restrict to one or more conditioning modes (repeatable). "
        f"Valid: {list(PCA_MODES)}. 'both' is not supported.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=None,
        help="Override ensemble.size when resolving upstream zarr paths.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root output directory (default: <base>/diagnostics).",
    )
    parser.add_argument(
        "--variable",
        action="append",
        default=None,
        dest="variables",
        help="Precursor variable for the PCA (repeatable). "
        f"Default: {list(DEFAULT_VARIABLES)} (or case YAML synoptic_pca.variables).",
    )
    parser.add_argument(
        "--wind-variable",
        action="append",
        default=None,
        dest="wind_variables",
        help="Raw (u, v) wind component drawn as steering arrows on the "
        "TC precursor panels (repeatable). "
        f"Default: {list(DEFAULT_WIND_VARIABLES)} (or case YAML "
        "synoptic_pca.wind_variables).  TC cases only.",
    )
    parser.add_argument(
        "--video-variable",
        type=str,
        default=None,
        help="Variable for the per-EOF percentile-member animation. "
        f"Default: {DEFAULT_VIDEO_VARIABLE} (or case YAML "
        "synoptic_pca.video_variable).  TC cases only.",
    )
    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument(
        "--max-d",
        type=int,
        default=DEFAULT_MAX_D,
        help=f"Number of leading PCs to compute/store (default {DEFAULT_MAX_D}).",
    )
    parser.add_argument(
        "--n-eof",
        type=int,
        default=None,
        help=f"Number of leading EOFs to analyze (precursor maps + impact "
        f"panels are baked for this many). Overrides the case YAML "
        f"synoptic_pca.n_eof; falls back to {DEFAULT_N_EOF} when neither is set.",
    )
    parser.add_argument(
        "--n-eof-figures",
        type=int,
        default=None,
        help="Number of leading EOFs to render a combined precursor -> impact "
        "figure for (default: all analyzed EOFs, i.e. min(--n-eof, --max-d)). "
        "Overrides the case YAML synoptic_pca.n_eof_figures.",
    )
    parser.add_argument(
        "--impact-variable",
        type=str,
        default=None,
        help="Override the scalar impact variable (e.g. t2m). Ignored for TC "
        "cases, which use the track impact.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        help="Override which tc_tracks tracker's parquet to read for the "
        "track impact (default: the case's first configured tracker).",
    )
    parser.add_argument(
        "--h-rt",
        type=str,
        default=DEFAULT_H_RT,
        help=f"Per-mode job walltime (default {DEFAULT_H_RT}).",
    )
    args = parser.parse_args()

    if args.list:
        print("Available cases:")
        for n in list_cases():
            print(f"  {n}")
        return

    if not args.case:
        parser.error("case name required (or pass --list)")

    dispatch(
        args.case,
        only_modes=args.mode,
        ensemble_size_override=args.ensemble_size,
        output_root_override=args.output_root,
        variables=args.variables,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        max_d=args.max_d,
        n_eof=args.n_eof,
        n_eof_figures=args.n_eof_figures,
        impact_variable=args.impact_variable,
        tracker=args.tracker,
        wind_variables=args.wind_variables,
        video_variable=args.video_variable,
        h_rt=args.h_rt,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
