"""Dispatch ageostrophic-fraction jobs for a named case study.

Reads ``cases/<event>.yaml`` and submits one qsub job per conditioning
mode (start/end/both) plus a single dependent aggregator job that
renders all modes side-by-side as columns of a 4×N grid (one column per
mode; per-member spaghetti drawn in every cell).

Output layout (modes aggregated)::

    <output_root>/ageostrophic/
        ageostrophic_fraction_start.npz
        ageostrophic_fraction_end.npz
        ageostrophic_fraction_both.npz
        ageostrophic_fraction.png        (4-row × N-mode grid; spaghetti per cell)

The diagnostic is a domain-averaged ratio, so no storm-centric zoom is
produced: in a small box, members with the storm displaced outside the
box look very different from members with it inside, and the per-member
mean stops being comparable.  The midlatitude band sidesteps this
because the storm is always *somewhere* in the band.

Usage:
    python dispatch_ageostrophic.py sandy
    python dispatch_ageostrophic.py pnw_heatwave
    python dispatch_ageostrophic.py sandy --dry-run
    python dispatch_ageostrophic.py sandy --mode end
    python dispatch_ageostrophic.py sandy --lat-min 25 --lat-max 65
    python dispatch_ageostrophic.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    resolve_timezone,
    start_time_iso_from_case,
    submit_qsub,
    validate_zarrs_exist,
)

AGEO_SCRIPT = str(SCRIPT_DIR / "submission_scripts" / "submit_ageostrophic.sh")
AGG_SCRIPT = str(SCRIPT_DIR / "submission_scripts" / "submit_aggregate_ageostrophic.sh")
AGEO_LOG_DIR = str(SCRIPT_DIR / "submission_scripts" / "ageostrophic_logs")

DEFAULT_H_RT = "8:00:00"
AGG_H_RT = "0:30:00"
DEFAULT_LEVEL = 500
DEFAULT_LAT_MIN = 30.0
DEFAULT_LAT_MAX = 60.0


def submit_ageostrophic_for_mode(
    case: Dict[str, Any],
    mode: str,
    *,
    ensemble_size: int,
    diag_output_dir: str,
    level: int,
    lat_min: float,
    lat_max: float,
    lon_min: Optional[float],
    lon_max: Optional[float],
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
        "--level",
        str(level),
        "--lat-min",
        str(lat_min),
        "--lat-max",
        str(lat_max),
    ]
    if lon_min is not None:
        script_args += ["--lon-min", str(lon_min)]
    if lon_max is not None:
        script_args += ["--lon-max", str(lon_max)]
    for frame in mode_frames(mode):
        script_args += ["--conditioning-frame", str(frame)]

    cmd = build_qsub_cmd(
        script=AGEO_SCRIPT,
        job_name=f"Ageo_{case['name']}_{mode}",
        h_rt=h_rt,
        script_args=script_args,
    )
    return submit_qsub(
        cmd,
        dry_run=dry_run,
        cwd=str(SCRIPT_DIR / "submission_scripts"),
        log_dir=AGEO_LOG_DIR,
    )


def submit_aggregator(
    case: Dict[str, Any],
    *,
    diag_output_dir: str,
    hold_jids: List[str],
    dry_run: bool,
) -> Optional[str]:
    """Submit the cross-mode aggregator job."""
    script_args = ["--output-dir", diag_output_dir]
    cmd = build_qsub_cmd(
        script=AGG_SCRIPT,
        job_name=f"AggAgeo_{case['name']}",
        h_rt=AGG_H_RT,
        script_args=script_args,
        hold_jids=hold_jids,
    )
    return submit_qsub(
        cmd,
        dry_run=dry_run,
        cwd=str(SCRIPT_DIR / "submission_scripts"),
        log_dir=AGEO_LOG_DIR,
    )


def dispatch(
    case_name: str,
    *,
    only_modes: Optional[List[str]] = None,
    ensemble_size_override: Optional[int] = None,
    output_root_override: Optional[str] = None,
    level: int = DEFAULT_LEVEL,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    dry_run: bool = False,
) -> Dict[str, Optional[str]]:
    """Submit ageostrophic-fraction jobs for a case."""
    case = load_case(case_name)
    ensemble_size = case_ensemble_size(case, override=ensemble_size_override)

    modes_to_run = [m for m, _ in iter_modes(case)]
    if only_modes:
        modes_to_run = [m for m in modes_to_run if m in only_modes]

    output_root = output_root_override or f"{case['base']}/diagnostics"
    diag_output_dir = f"{output_root}/ageostrophic"

    print(
        f"[{case['name']}] submitting {len(modes_to_run)} ageostrophic job(s): "
        f"{modes_to_run}  (level={level} hPa, |lat| ∈ [{lat_min}, {lat_max}])"
    )

    validate_zarrs_exist(case, modes_to_run, ensemble_size)

    results: Dict[str, Optional[str]] = {}
    for mode in modes_to_run:
        jid = submit_ageostrophic_for_mode(
            case,
            mode,
            ensemble_size=ensemble_size,
            diag_output_dir=diag_output_dir,
            level=level,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            h_rt=DEFAULT_H_RT,
            dry_run=dry_run,
        )
        results[mode] = jid

    hold_jids = [j for j in results.values() if j]
    results["aggregate"] = submit_aggregator(
        case,
        diag_output_dir=diag_output_dir,
        hold_jids=hold_jids,
        dry_run=dry_run,
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch ageostrophic-fraction jobs for a named case.",
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
        choices=["start", "end", "both"],
        help="Restrict to one or more conditioning modes (repeatable).",
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
        "--level",
        type=int,
        default=DEFAULT_LEVEL,
        help=f"Pressure level in hPa (default {DEFAULT_LEVEL}).",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=DEFAULT_LAT_MIN,
        help=f"Lower edge of |lat| band (default {DEFAULT_LAT_MIN}°).",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=DEFAULT_LAT_MAX,
        help=f"Upper edge of |lat| band (default {DEFAULT_LAT_MAX}°).",
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=None,
        help="Optional western longitude bound (default: all longitudes).",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=None,
        help="Optional eastern longitude bound (default: all longitudes).",
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
        level=args.level,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
