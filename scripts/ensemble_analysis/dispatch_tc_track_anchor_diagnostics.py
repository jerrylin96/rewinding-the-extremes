"""Dispatch pinned-end anchor diagnostics for a named TC case study.

Submits one CPU qsub job per ``(conditioning mode, tracker)`` pair that
sorts the conditioned ensemble members into the three pinned-end anchor
categories the paper quotes (``anchored`` / ``alt_path`` / ``lost`` -- for
end-conditioned Sandy: 812 / 144 / 44) and renders per-category
diagnostics: a per-member classification CSV, category overview maps,
per-member track-map pages, and per-member MSL animations (with a
DetectNodes candidate overlay) for a representative subset of each
category.  See ``submission_scripts/tc_track_anchor_diagnostics.py`` for
the artifact details.

Requires the per-mode ``tc_tracks_<mode>.parquet`` files written by
``dispatch_tc_tracks.py``.

Output layout (per tracker and mode)::

    <output_root>/tc_tracks/<tracker>/anchor_diagnostics_<mode>/
        member_anchor_categories.csv
        anchor_summary.txt
        anchored/   overview.png, track_maps_pNN.png, msl_member_XXXX.mp4
        alt_path/   ...
        lost/       ...

Only the ``start`` and ``end`` conditioning modes have a uniquely defined
pinned-end anchor; ``both`` is skipped (mirrors synoptic_pca_track_stats).

Usage:
    python dispatch_tc_track_anchor_diagnostics.py sandy
    python dispatch_tc_track_anchor_diagnostics.py sandy --mode end
    python dispatch_tc_track_anchor_diagnostics.py sandy --animate-lost 12
    python dispatch_tc_track_anchor_diagnostics.py sandy --dry-run
    python dispatch_tc_track_anchor_diagnostics.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "_shared"))

from _dispatch_lib import (  # noqa: E402
    MODE_FRAMES,
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
    tc_tracks_parquet_path,
    trackers_from_case,
    warm_core_from_case,
)

DIAG_SCRIPT = str(
    SCRIPT_DIR / "submission_scripts" / "submit_tc_track_anchor_diagnostics.sh"
)
TC_LOG_DIR = str(SCRIPT_DIR / "submission_scripts" / "tc_tracks_logs")

# Per-member cartopy pages dominate; animations + DetectNodes reruns for
# the ~dozen selected members add minutes, not hours.
DEFAULT_H_RT = "4:00:00"

# Modes with a uniquely defined pinned end (mirrors synoptic_pca_track_stats).
ANCHOR_MODES: Tuple[str, ...] = ("start", "end")


def anchor_frame_for_mode(case: Dict[str, Any], mode: str) -> int:
    """Pinned-end frame: 0 under start-conditioning, else the landfall frame.

    Matches ``synoptic_pca_track_stats.run_mode``: under end-conditioning the
    anchor is the case's ``tc_tracks.landfall_frame`` when declared, falling
    back to the conditioned frame itself (``MODE_FRAMES['end'][0]``, the last
    frame of the standard 12-step window).
    """
    if mode == "start":
        return 0
    landfall = resolve_landfall_frame(case)
    return landfall if landfall >= 0 else MODE_FRAMES["end"][0]


def submit_anchor_diagnostics_for_mode(
    case: Dict[str, Any],
    mode: str,
    *,
    ensemble_size: int,
    output_root: str,
    tracker: str,
    box: Tuple[float, float, float, float],
    warm_core: bool,
    animate_counts: Tuple[int, int, int],
    no_candidates: bool,
    h_rt: str,
    dry_run: bool,
) -> Optional[str]:
    """Submit the per-(mode, tracker) anchor-diagnostics job."""
    base = case["base"]
    tracker_dir = tc_tracks_dir(output_root, tracker)
    parquet = tc_tracks_parquet_path(output_root, mode, tracker)
    lon_min, lon_max, lat_min, lat_max = box
    n_anchored, n_alt, n_lost = animate_counts

    script_args = [
        "--tracks-parquet",
        parquet,
        "--output-dir",
        f"{tracker_dir}/anchor_diagnostics_{mode}",
        "--mode",
        mode,
        "--anchor-frame",
        str(anchor_frame_for_mode(case, mode)),
        "--lon-min",
        str(lon_min),
        "--lon-max",
        str(lon_max),
        "--lat-min",
        str(lat_min),
        "--lat-max",
        str(lat_max),
        "--title",
        f"{case['display_name']} ({tracker})",
        "--ensemble-zarr",
        ensemble_zarr_path(base, mode, ensemble_size),
        "--era5-zarr",
        era5_zarr_path(base, mode),
        "--animate-anchored",
        str(n_anchored),
        "--animate-alt",
        str(n_alt),
        "--animate-lost",
        str(n_lost),
        "--start-time",
        start_time_iso_from_case(case),
        "--timezone",
        resolve_timezone(case),
    ]
    # The candidate overlay reruns DetectNodes with the same warm-core
    # setting the production tracking used, plus an msl-only variant, so
    # warm-core rejections are visible frame by frame.
    if warm_core and tracker == "tempest":
        script_args.append("--warm-core")
    if no_candidates or tracker != "tempest":
        script_args.append("--no-candidates")

    cmd = build_qsub_cmd(
        script=DIAG_SCRIPT,
        job_name=f"AnchorDiag_{case['name']}_{mode}_{tracker}",
        h_rt=h_rt,
        script_args=script_args,
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
    animate_counts: Tuple[int, int, int] = (2, 4, 8),
    no_candidates: bool = False,
    dry_run: bool = False,
) -> Dict[Any, Optional[str]]:
    """Submit anchor-diagnostics jobs for a case (start/end modes only)."""
    case = load_case(case_name)
    ensemble_size = case_ensemble_size(case, override=ensemble_size_override)
    if warm_core is None:
        warm_core = warm_core_from_case(case)
    box = require_region(case, "track")
    trackers = trackers_from_case(case, override=tracker_override)
    output_root = output_root_override or f"{case['base']}/diagnostics"

    modes_to_run = [m for m, _ in iter_modes(case) if m in ANCHOR_MODES]
    if only_modes:
        modes_to_run = [m for m in modes_to_run if m in only_modes]
    skipped = [m for m, _ in iter_modes(case) if m not in ANCHOR_MODES]
    if skipped:
        print(
            f"[{case['name']}] skipping mode(s) {skipped}: the pinned-end "
            f"anchor is only defined for {list(ANCHOR_MODES)}."
        )

    # The parquets are this diagnostic's real input; fail fast with a
    # pointer to the producing dispatcher instead of a mid-job crash.
    missing = [
        tc_tracks_parquet_path(output_root, mode, tracker)
        for mode in modes_to_run
        for tracker in trackers
        if not Path(tc_tracks_parquet_path(output_root, mode, tracker)).exists()
    ]
    if missing:
        print(
            "ERROR: missing tc_tracks parquet(s) " "(run dispatch_tc_tracks.py first):",
            file=sys.stderr,
        )
        print("\n".join(f"  {p}" for p in missing), file=sys.stderr)
        raise SystemExit(1)

    n_jobs = len(modes_to_run) * len(trackers)
    print(
        f"[{case['name']}] submitting {n_jobs} anchor-diagnostics job(s): "
        f"modes={modes_to_run}, trackers={trackers}, warm_core={warm_core}"
    )

    results: Dict[Any, Optional[str]] = {}
    for mode in modes_to_run:
        for tracker in trackers:
            results[(mode, tracker)] = submit_anchor_diagnostics_for_mode(
                case,
                mode,
                ensemble_size=ensemble_size,
                output_root=output_root,
                tracker=tracker,
                box=box,
                warm_core=warm_core,
                animate_counts=animate_counts,
                no_candidates=no_candidates,
                h_rt=DEFAULT_H_RT,
                dry_run=dry_run,
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch pinned-end anchor diagnostics for a named case.",
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
        choices=list(ANCHOR_MODES),
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
        help="Candidate overlay only: run the warm-core DetectNodes variant "
        "alongside msl-only.  Defaults to the case YAML's "
        "tc_tracks.warm_core so the overlay matches what the production "
        "tracking saw.",
    )
    parser.add_argument(
        "--animate-anchored",
        type=int,
        default=2,
        help="Anchored members to animate per job (default 2).",
    )
    parser.add_argument(
        "--animate-alt",
        type=int,
        default=4,
        help="alt_path members to animate per job (default 4).",
    )
    parser.add_argument(
        "--animate-lost",
        type=int,
        default=8,
        help="lost members to animate per job (default 8).",
    )
    parser.add_argument(
        "--no-candidates",
        action="store_true",
        help="Skip the DetectNodes candidate overlay in the animations.",
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
        animate_counts=(
            args.animate_anchored,
            args.animate_alt,
            args.animate_lost,
        ),
        no_candidates=args.no_candidates,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
