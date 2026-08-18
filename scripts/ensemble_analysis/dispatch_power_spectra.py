"""Dispatch power-spectra jobs for a named case study.

Reads ``cases/<event>.yaml`` and submits one CPU qsub job per
conditioning mode (start/end/both) plus a single dependent aggregator
job that overlays the three modes on the same zonal-spectrum plot.

Output layout (per variable, modes aggregated)::

    <output_root>/power_spectra/<variable>/
        power_spectra_start.npz
        power_spectra_end.npz
        power_spectra_both.npz
        power_spectra_zonal.png    (4 lead-time panels, one ensemble curve per mode)

Usage:
    python dispatch_power_spectra.py sandy
    python dispatch_power_spectra.py sandy --dry-run
    python dispatch_power_spectra.py sandy --mode end
    python dispatch_power_spectra.py sandy --variables z500,t2m
    python dispatch_power_spectra.py --list
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
    resolve_timezone,
    resolve_variable_set,
    start_time_iso_from_case,
    submit_qsub,
    validate_zarrs_exist,
)

PSPEC_SCRIPT = str(SCRIPT_DIR / "submission_scripts" / "submit_power_spectra.sh")
AGG_SCRIPT = str(
    SCRIPT_DIR / "submission_scripts" / "submit_aggregate_power_spectra.sh"
)
PSPEC_LOG_DIR = str(SCRIPT_DIR / "submission_scripts" / "power_spectra_logs")

DEFAULT_H_RT = "8:00:00"
AGG_H_RT = "0:30:00"
DEFAULT_K_RANGE = "5,40"
DEFAULT_BATCH_SIZE = 32


def submit_for_mode(
    case: Dict[str, Any],
    mode: str,
    *,
    ensemble_size: int,
    variables: List[str],
    diag_output_root: str,
    k_range: str,
    batch_size: int,
    h_rt: str,
    dry_run: bool,
) -> Optional[str]:
    """Submit the per-mode power-spectra job."""
    base = case["base"]
    ens_zarr = ensemble_zarr_path(base, mode, ensemble_size)
    era5_zarr = era5_zarr_path(base, mode)

    script_args = [
        "--ensemble-zarr",
        ens_zarr,
        "--era5-zarr",
        era5_zarr,
        "--output-root",
        diag_output_root,
        "--variables",
        ",".join(variables),
        "--mode",
        mode,
        "--case-name",
        case["display_name"],
        "--start-time",
        start_time_iso_from_case(case),
        "--timezone",
        resolve_timezone(case),
        "--k-range",
        k_range,
        "--batch-size",
        str(batch_size),
    ]

    cmd = build_qsub_cmd(
        script=PSPEC_SCRIPT,
        job_name=f"PowerSpec_{case['name']}_{mode}",
        h_rt=h_rt,
        script_args=script_args,
    )
    return submit_qsub(
        cmd,
        dry_run=dry_run,
        cwd=str(SCRIPT_DIR / "submission_scripts"),
        log_dir=PSPEC_LOG_DIR,
    )


def submit_aggregator(
    case: Dict[str, Any],
    *,
    diag_output_root: str,
    variables: List[str],
    hold_jids: List[str],
    dry_run: bool,
) -> Optional[str]:
    """Submit the cross-mode aggregator job."""
    script_args = [
        "--output-root",
        diag_output_root,
        "--variables",
        ",".join(variables),
    ]
    cmd = build_qsub_cmd(
        script=AGG_SCRIPT,
        job_name=f"AggPowerSpec_{case['name']}",
        h_rt=AGG_H_RT,
        script_args=script_args,
        hold_jids=hold_jids,
    )
    return submit_qsub(
        cmd,
        dry_run=dry_run,
        cwd=str(SCRIPT_DIR / "submission_scripts"),
        log_dir=PSPEC_LOG_DIR,
    )


def dispatch(
    case_name: str,
    *,
    only_modes: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    ensemble_size_override: Optional[int] = None,
    output_root_override: Optional[str] = None,
    k_range: str = DEFAULT_K_RANGE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> Dict[str, Optional[str]]:
    """Submit power-spectra jobs for a case."""
    case = load_case(case_name)
    ensemble_size = case_ensemble_size(case, override=ensemble_size_override)

    variables = variables or resolve_variable_set(case)
    if not variables:
        raise ValueError(
            f"case '{case['name']}': no diagnostic variables "
            f"(set diagnostics.variables or palette 'standard')"
        )

    output_root = output_root_override or f"{case['base']}/diagnostics"
    diag_output_root = f"{output_root}/power_spectra"

    modes_to_run = [m for m, _ in iter_modes(case)]
    if only_modes:
        modes_to_run = [m for m in modes_to_run if m in only_modes]

    print(
        f"[{case['name']}] submitting {len(modes_to_run)} power-spectra "
        f"job(s): {modes_to_run}  ({len(variables)} variables)"
    )

    validate_zarrs_exist(case, modes_to_run, ensemble_size)

    results: Dict[str, Optional[str]] = {}
    for mode in modes_to_run:
        jid = submit_for_mode(
            case,
            mode,
            ensemble_size=ensemble_size,
            variables=variables,
            diag_output_root=diag_output_root,
            k_range=k_range,
            batch_size=batch_size,
            h_rt=DEFAULT_H_RT,
            dry_run=dry_run,
        )
        results[mode] = jid

    hold_jids = [j for j in results.values() if j]
    results["aggregate"] = submit_aggregator(
        case,
        diag_output_root=diag_output_root,
        variables=variables,
        hold_jids=hold_jids,
        dry_run=dry_run,
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch power-spectra jobs for a named case.",
    )
    parser.add_argument("case", nargs="?", help="Case name (e.g. sandy)")
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
        "--variables",
        type=str,
        default=None,
        help="Comma-separated variable list (default: resolved from the case YAML).",
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
        "--k-range",
        type=str,
        default=DEFAULT_K_RANGE,
        help=f"Wavenumber range for spectral slope fit (default: {DEFAULT_K_RANGE}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Members per batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    args = parser.parse_args()

    if args.list:
        print("Available cases:")
        for n in list_cases():
            print(f"  {n}")
        return

    if not args.case:
        parser.error("case name required (or pass --list)")

    variables = (
        [v.strip() for v in args.variables.split(",") if v.strip()]
        if args.variables
        else None
    )

    dispatch(
        args.case,
        only_modes=args.mode,
        variables=variables,
        ensemble_size_override=args.ensemble_size,
        output_root_override=args.output_root,
        k_range=args.k_range,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
