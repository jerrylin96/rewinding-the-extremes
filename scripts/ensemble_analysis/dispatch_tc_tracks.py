"""Dispatch TC / storm-track jobs for a named case study.

Submits one GPU/CPU qsub job per ``(conditioning mode, tracker)`` pair
plus a single dependent aggregator job per tracker that produces the
3-panel side-by-side static map across the three modes.

The case YAML's ``tc_tracks.trackers`` is a list — every named tracker
is dispatched per mode, with each writing into its own subdirectory.

Output layout (per tracker, modes aggregated)::

    <output_root>/tc_tracks/<tracker>/
        tc_tracks_start.parquet
        tc_tracks_end.parquet
        tc_tracks_both.parquet
        tc_tracks_<mode>.mp4   (if --animate, the default; one per mode)
        tc_tracks.png          (3-panel side-by-side static map)

Usage:
    python dispatch_tc_tracks.py katrina
    python dispatch_tc_tracks.py katrina --dry-run
    python dispatch_tc_tracks.py katrina --mode end
    python dispatch_tc_tracks.py katrina --tracker minmsl
    python dispatch_tc_tracks.py katrina --tracker tempest --warm-core
    python dispatch_tc_tracks.py sandy --no-warm-core   # override case default
    python dispatch_tc_tracks.py --list

The tempest warm-core thickness filter defaults to each case YAML's
``tc_tracks.warm_core`` (sandy / ian set it true); ``--warm-core`` /
``--no-warm-core`` override that default for a single run.
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
    require_region,
    resolve_landfall_frame,
    resolve_timezone,
    start_time_iso_from_case,
    submit_qsub,
    tc_tracks_dir,
    trackers_from_case,
    validate_zarrs_exist,
    warm_core_from_case,
)

TC_TRACKS_SCRIPT = str(SCRIPT_DIR / "submission_scripts" / "submit_tc_tracks.sh")
AGG_SCRIPT = str(SCRIPT_DIR / "submission_scripts" / "submit_aggregate_tc_tracks.sh")
TC_LOG_DIR = str(SCRIPT_DIR / "submission_scripts" / "tc_tracks_logs")

DEFAULT_H_RT = "5:00:00"
AGG_H_RT = "0:30:00"
DEFAULT_TRACKER = "minmsl"

# Per-tracker SGE wall-clock overrides.  tempest serializes through
# DetectNodes/StitchNodes binaries (one subprocess pair per member) and
# does an HPX -> lat/lon regrid per worker, so a 1000-member run on 4
# CPU cores takes hours even with single-threaded workers; minmsl /
# wuduan stream the whole ensemble through a single tensor op and
# finish inside the default budget.
TRACKER_H_RT: Dict[str, str] = {
    "tempest": "12:00:00",
}

# Per-tracker SGE resource overrides appended to the qsub command.
# tempest / minmsl are CPU-only; WuDuan uses cucim + cupy and needs a
# GPU. The GPU type list mirrors submit_ensemble.sh, which is known to
# schedule cleanly on the eb-general project's GPU queues.
TRACKER_EXTRA_RESOURCES: Dict[str, List[str]] = {
    "tempest": [],
    "minmsl": [],
    "wuduan": [
        "-l",
        "gpus=1",
        "-l",
        "gpu_type=A40|A6000|RTX6000|RTX6000ada|L40S|A100",
    ],
}


def submit_tc_tracks_for_mode(
    case: Dict[str, Any],
    mode: str,
    *,
    ensemble_size: int,
    output_root: str,
    tracker: str,
    box: Tuple[float, float, float, float],
    h_rt: str,
    warm_core: bool,
    dry_run: bool,
) -> Optional[str]:
    """Submit the per-mode TC tracking job."""
    base = case["base"]
    ens_zarr = ensemble_zarr_path(base, mode, ensemble_size)
    era5_zarr = era5_zarr_path(base, mode)
    output_dir = tc_tracks_dir(output_root, tracker)

    lon_min, lon_max, lat_min, lat_max = box

    title = (
        f"{case['display_name']} {mode.title()} Conditioning Storm Tracks ({tracker})"
    )

    script_args = [
        "--ensemble-zarr",
        ens_zarr,
        "--era5-zarr",
        era5_zarr,
        "--conditioning",
        mode,
        "--output-dir",
        output_dir,
        "--mode",
        mode,
        "--title",
        title,
        "--tracker",
        tracker,
        "--lon-min",
        str(lon_min),
        "--lon-max",
        str(lon_max),
        "--lat-min",
        str(lat_min),
        "--lat-max",
        str(lat_max),
        # Pinned-end anchor frame for the --animate track_qc resolution
        # (end-conditioning); -1 -> last lead.
        "--landfall-frame",
        str(resolve_landfall_frame(case)),
        "--animate",
    ]
    # The worker silently ignores --warm-core for non-tempest trackers
    # (and prints a NOTE).  Only forward it for tempest so the other
    # trackers' qsub args stay clean.
    if warm_core and tracker == "tempest":
        script_args.append("--warm-core")

    cmd = build_qsub_cmd(
        script=TC_TRACKS_SCRIPT,
        job_name=f"TC_{case['name']}_{mode}_{tracker}",
        h_rt=h_rt,
        script_args=script_args,
        extra_resource_flags=TRACKER_EXTRA_RESOURCES.get(tracker, []),
    )
    return submit_qsub(
        cmd,
        dry_run=dry_run,
        cwd=str(SCRIPT_DIR / "submission_scripts"),
        log_dir=TC_LOG_DIR,
    )


def submit_aggregator(
    case: Dict[str, Any],
    *,
    output_root: str,
    trackers: List[str],
    box: Tuple[float, float, float, float],
    hold_jids: List[str],
    dry_run: bool,
) -> Optional[str]:
    """Submit the cross-mode aggregator that builds the 3-panel static map."""
    diag_output_root = f"{output_root}/tc_tracks"
    lon_min, lon_max, lat_min, lat_max = box

    script_args = [
        "--output-root",
        diag_output_root,
        "--trackers",
        ",".join(trackers),
        "--lon-min",
        str(lon_min),
        "--lon-max",
        str(lon_max),
        "--lat-min",
        str(lat_min),
        "--lat-max",
        str(lat_max),
        # Pinned-end anchor frame for track_qc's storm-of-interest resolution
        # (end-conditioning) and the 'ERA5 pN' annotation; -1 -> last lead.
        "--landfall-frame",
        str(resolve_landfall_frame(case)),
        "--title",
        f"{case['display_name']} Storm Tracks",
        # start_time + timezone drive the local wall-clock x-axis on the
        # tc_msl trajectory rows of the multi-row case figure.
        "--start-time",
        start_time_iso_from_case(case),
        "--timezone",
        resolve_timezone(case),
    ]
    cmd = build_qsub_cmd(
        script=AGG_SCRIPT,
        job_name=f"AggTcTracks_{case['name']}",
        h_rt=AGG_H_RT,
        script_args=script_args,
        hold_jids=hold_jids,
    )
    return submit_qsub(
        cmd,
        dry_run=dry_run,
        cwd=str(SCRIPT_DIR / "submission_scripts"),
        log_dir=TC_LOG_DIR,
    )


def dispatch(
    case_name: str,
    *,
    only_modes: Optional[List[str]] = None,
    ensemble_size_override: Optional[int] = None,
    output_root_override: Optional[str] = None,
    tracker_override: Optional[List[str]] = None,
    warm_core: Optional[bool] = None,
    dry_run: bool = False,
) -> Dict[Any, Optional[str]]:
    """Submit TC tracking jobs for a case.

    ``warm_core`` tri-states the tempest warm-core filter: ``True`` / ``False``
    force it on / off (from an explicit ``--warm-core`` / ``--no-warm-core``),
    while ``None`` falls back to the case YAML's ``tc_tracks.warm_core``.
    """
    case = load_case(case_name)
    ensemble_size = case_ensemble_size(case, override=ensemble_size_override)

    # Explicit --warm-core/--no-warm-core wins; otherwise use the case default.
    if warm_core is None:
        warm_core = warm_core_from_case(case)

    box = require_region(case, "track")
    trackers = trackers_from_case(case, override=tracker_override)

    modes_to_run = [m for m, _ in iter_modes(case)]
    if only_modes:
        modes_to_run = [m for m in modes_to_run if m in only_modes]

    n_jobs = len(modes_to_run) * len(trackers)
    print(
        f"[{case['name']}] submitting {n_jobs} TC tracking job(s): "
        f"modes={modes_to_run}, trackers={trackers}, warm_core={warm_core}"
    )

    validate_zarrs_exist(case, modes_to_run, ensemble_size)

    results: Dict[Any, Optional[str]] = {}
    output_root = output_root_override or f"{case['base']}/diagnostics"
    for mode in modes_to_run:
        for tracker in trackers:
            jid = submit_tc_tracks_for_mode(
                case,
                mode,
                ensemble_size=ensemble_size,
                output_root=output_root,
                tracker=tracker,
                box=box,
                h_rt=TRACKER_H_RT.get(tracker, DEFAULT_H_RT),
                warm_core=warm_core,
                dry_run=dry_run,
            )
            results[(mode, tracker)] = jid

    hold_jids = [j for j in results.values() if j]
    results["aggregate"] = submit_aggregator(
        case,
        output_root=output_root,
        trackers=trackers,
        box=box,
        hold_jids=hold_jids,
        dry_run=dry_run,
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch TC / storm-track jobs for a named case.",
    )
    parser.add_argument("case", nargs="?", help="Case name (e.g. katrina)")
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
        "--tracker",
        action="append",
        default=None,
        choices=["wuduan", "minmsl", "tempest"],
        help="Restrict to one or more trackers (repeatable; default: all "
        "trackers configured in the case YAML's tc_tracks.trackers list).",
    )
    parser.add_argument(
        "--warm-core",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Tempest tracker only: add the Z&U 2017 _DIFF(z300,z500) "
        "thickness contour to DetectNodes' closedcontourcmd so cold-core "
        "lows are rejected.  Requires z300/z500 in the input zarr; ignored "
        "for the minmsl and wuduan trackers.  Defaults to the case YAML's "
        "tc_tracks.warm_core (false if unset); pass --warm-core / "
        "--no-warm-core to override it for a single run.",
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
        tracker_override=args.tracker,
        warm_core=args.warm_core,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
