"""Dispatch the multi-case TC-tracks combiner.

This is the one diagnostic in scripts/ensemble_analysis that intentionally
spans multiple cases.  It reads each case's per-mode
``tc_tracks_<mode>.parquet`` (written upstream by ``dispatch_tc_tracks.py``)
and emits a single ``N x 3`` PNG where each row is a case and each column
is a conditioning mode — a publication-friendly substitute for stacking
the individual per-case ``tc_tracks.png`` files in the paper.

Pre-flight validates that every ``(case, mode)`` parquet exists before
submitting, so a missing upstream run fails loudly here rather than
inside the qsub job.

Output (default):
    <common_parent>/diagnostics_combined/tc_tracks/<tracker>/tc_tracks_<case1>_<case2>_...png

where ``<common_parent>`` is the deepest directory shared by every case's
``base`` path from the YAML.  Pass ``--output-path`` to override.

Usage:
    python dispatch_combined_tc_tracks.py                       # defaults: sandy + ian
    python dispatch_combined_tc_tracks.py --case sandy --case ian
    python dispatch_combined_tc_tracks.py --case sandy --case ian --dry-run
    python dispatch_combined_tc_tracks.py --case sandy --case ian \\
        --output-path /custom/path/combined.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "_shared"))

from _dispatch_lib import (  # noqa: E402
    build_qsub_cmd,
    list_cases,
    load_case,
    submit_qsub,
    tc_tracks_parquet_path,
    trackers_from_case,
)

COMBINE_SCRIPT = str(SCRIPT_DIR / "submission_scripts" / "submit_combined_tc_tracks.sh")
COMBINE_LOG_DIR = str(SCRIPT_DIR / "submission_scripts" / "tc_tracks_logs")

DEFAULT_CASES: List[str] = ["sandy", "ian"]
DEFAULT_TRACKER = "minmsl"
H_RT = "0:30:00"

# Mode list mirrors aggregate_tc_tracks.py / combine_tc_tracks.py.
MODES = ("end", "start", "both")


def _default_output_path(
    cases: List[dict],
    tracker: str,
) -> str:
    """Pick a sensible default output path under the cases' common parent.

    Falls back to the first case's ``base`` if the case bases don't share
    a common parent (e.g. lived on different filesystems).
    """
    bases = [c["base"] for c in cases]
    try:
        common = os.path.commonpath(bases)
        # ``commonpath`` may bottom out at "/" if the cases really diverge;
        # treat that as "no useful shared root" and fall back below.
        if common in ("", "/"):
            raise ValueError("trivial common path")
    except ValueError:
        common = cases[0]["base"]
    stem = "_".join(c["name"] for c in cases)
    return f"{common}/diagnostics_combined/tc_tracks/{tracker}/tc_tracks_{stem}.png"


def _validate_parquets_exist(cases: List[dict], tracker: str) -> None:
    """Friendly pre-flight: parquets for every (case, mode) must exist."""
    missing: List[str] = []
    for case in cases:
        diag_root = f"{case['base']}/diagnostics"
        for mode in MODES:
            parquet = Path(tc_tracks_parquet_path(diag_root, mode, tracker))
            if not parquet.exists():
                missing.append(f"  {case['name']}/{mode}: {parquet}")
    if missing:
        print(
            "ERROR: required tc_tracks parquet(s) not found.  Run "
            "dispatch_tc_tracks.py for each case first:",
            file=sys.stderr,
        )
        print("\n".join(missing), file=sys.stderr)
        raise SystemExit(1)


def _resolve_tracker(cases: List[dict], tracker_override: Optional[str]) -> str:
    """Pick a single tracker that every case configures."""
    common: Optional[set[str]] = None
    for case in cases:
        case_trackers = set(trackers_from_case(case))
        common = case_trackers if common is None else common & case_trackers

    if not common:
        raise SystemExit(
            "ERROR: requested cases do not share any tracker in their "
            "tc_tracks.trackers lists; cannot combine."
        )

    if tracker_override is not None:
        if tracker_override not in common:
            raise SystemExit(
                f"ERROR: tracker '{tracker_override}' is not configured on "
                f"every requested case.  Shared trackers: {sorted(common)}."
            )
        return tracker_override

    if DEFAULT_TRACKER in common:
        return DEFAULT_TRACKER
    return sorted(common)[0]


def dispatch(
    case_names: List[str],
    *,
    tracker_override: Optional[str] = None,
    output_path_override: Optional[str] = None,
    title: Optional[str] = None,
    dry_run: bool = False,
) -> Optional[str]:
    """Submit the combined TC-tracks plotting job.  Returns the SGE job id."""
    if len(case_names) < 2:
        raise SystemExit(
            f"ERROR: --case must be supplied at least twice; got {case_names!r}.  "
            "The single-case 3-panel plot is produced by dispatch_tc_tracks.py."
        )

    cases = [load_case(name) for name in case_names]
    tracker = _resolve_tracker(cases, tracker_override)

    _validate_parquets_exist(cases, tracker)

    output_path = output_path_override or _default_output_path(cases, tracker)

    script_args = []
    for name in case_names:
        script_args += ["--case", name]
    script_args += [
        "--tracker",
        tracker,
        "--output-path",
        output_path,
    ]
    if title:
        script_args += ["--title", title]

    print(
        f"[combined_tc_tracks] cases={case_names} tracker={tracker}\n"
        f"  -> {output_path}"
    )

    cmd = build_qsub_cmd(
        script=COMBINE_SCRIPT,
        job_name=f"CombinedTC_{'_'.join(case_names)}",
        h_rt=H_RT,
        script_args=script_args,
    )
    return submit_qsub(
        cmd,
        dry_run=dry_run,
        cwd=str(SCRIPT_DIR / "submission_scripts"),
        log_dir=COMBINE_LOG_DIR,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine per-case TC-track parquets into an N x 3 PNG.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help=f"Case name (repeatable; default: {DEFAULT_CASES}).",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available cases and exit."
    )
    parser.add_argument(
        "--tracker",
        default=None,
        choices=["wuduan", "minmsl", "tempest"],
        help="Tracker to combine across cases.  Must be configured on "
        f"every requested case.  Default: '{DEFAULT_TRACKER}' if shared, "
        "else the alphabetically first shared tracker.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Full path for the combined PNG (overrides the default under "
        "the cases' common parent).",
    )
    parser.add_argument(
        "--title",
        default="O(1000) Hurricane Tracks with Flexible Conditioning",
        help="Suptitle for the combined figure.  Pass an empty string "
        "(--title '') to suppress.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Available cases:")
        for n in list_cases():
            print(f"  {n}")
        return

    case_names = args.case or list(DEFAULT_CASES)

    dispatch(
        case_names,
        tracker_override=args.tracker,
        output_path_override=args.output_path,
        title=args.title,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
