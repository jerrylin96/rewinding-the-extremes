"""Dispatch ensemble interpolation jobs for a named case study.

Reads ``cases/<event>.yaml``, submits one qsub job per conditioning mode
defined in ``ensemble.modes``.  All task types (ERA5 infill, custom ERA5
frames, TC guidance) are configured through the case YAML.

Usage:
    python dispatch_ensemble.py katrina
    python dispatch_ensemble.py katrina --dry-run
    python dispatch_ensemble.py katrina --mode end
    python dispatch_ensemble.py katrina --quick
    python dispatch_ensemble.py --list

Defaults (overridable via CLI):
    ensemble.size       -> 1000               (from the case YAML)
    ensemble.batch_size -> 1
    members_per_job     -> 50                 (20 parallel chunk jobs)
    h_rt                -> 4:00:00
    quick ensemble_size -> 8
    quick h_rt          -> 0:15:00

Prints each submitted job's SGE job id to stdout on success.
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
    iter_modes,
    list_cases,
    load_case,
    submit_qsub,
)

BASH_SCRIPT = str(SCRIPT_DIR / "submit_ensemble.sh")
LOG_DIR = str(SCRIPT_DIR / "ensemble_logs")

DEFAULT_H_RT = "4:00:00"
DEFAULT_BATCH_SIZE = 1
DEFAULT_MEMBERS_PER_JOB = 50

QUICK_ENSEMBLE_SIZE = 8
QUICK_H_RT = "0:15:00"


def build_script_args(case: Dict[str, Any], mode: str, frames: List[int],
                      ensemble_size: int, batch_size: int) -> List[str]:
    """Build the CLI arg list forwarded to ``submit_ensemble.sh``."""
    ens = case["ensemble"]
    start = ens["start_time"]

    args: List[str] = [
        "--start-year", str(start["year"]),
        "--start-month", str(start["month"]),
        "--start-day", str(start["day"]),
        "--start-hour", str(start["hour"]),
        "--conditioning-frames", ",".join(str(f) for f in frames),
        "--ensemble-size", str(ensemble_size),
        "--batch-size", str(batch_size),
        "--output-dir", f"{case['base']}/{mode}",
    ]

    # Random seed (shared across conditioning modes of the same case so that
    # member k in `start`, `end`, and `both` receives correlated noise — a
    # prerequisite for cross-mode variance / pinch-point diagnostics).
    if ens.get("random_seed") is not None:
        args += ["--random-seed", str(ens["random_seed"])]

    # Custom ERA5 frames
    for cf in ens.get("custom_frames", []) or []:
        args += [
            "--custom-frame",
            str(cf["frame"]), str(cf["year"]), str(cf["month"]),
            str(cf["day"]), str(cf["hour"]),
        ]

    # TC guidance frames
    for tf in ens.get("tc_frames", []) or []:
        args += [
            "--tc-frame",
            str(tf["frame"]), tf["lats"], tf["lons"],
        ]

    return args


def submit_mode(
    case: Dict[str, Any],
    mode: str,
    frames: List[int],
    *,
    ensemble_size: int,
    batch_size: int,
    h_rt: str,
    dry_run: bool,
    members_per_job: Optional[int],
) -> Optional[str]:
    """Submit one qsub job for a single conditioning mode; return the job id."""
    if members_per_job is None or members_per_job >= ensemble_size:
        # Standard unpartitioned submission
        script_args = build_script_args(case, mode, frames, ensemble_size, batch_size)
        job_name = f"Ens_{case['name']}_{mode}_n{ensemble_size}"
        cmd = build_qsub_cmd(
            script=BASH_SCRIPT,
            job_name=job_name,
            h_rt=h_rt,
            script_args=script_args,
        )
        return submit_qsub(cmd, dry_run=dry_run, cwd=str(SCRIPT_DIR), log_dir=LOG_DIR)

    # Chunked submission.  Chunk offsets must align with the GPU micro-batch
    # size; otherwise adjacent jobs would reuse the same per-chunk seed and
    # produce correlated ensemble members (see ensemble_interpolation.py).
    if members_per_job % batch_size != 0:
        raise ValueError(
            f"--members-per-job ({members_per_job}) must be a multiple of "
            f"ensemble.batch_size ({batch_size}) to keep per-chunk seeds unique."
        )

    offsets = list(range(0, ensemble_size, members_per_job))
    chunk_jids = []
    for offset in offsets:
        count = min(members_per_job, ensemble_size - offset)
        script_args = build_script_args(case, mode, frames, ensemble_size, batch_size)
        script_args += [
            "--member-offset", str(offset),
            "--members-to-run", str(count),
        ]
        job_name = f"Ens_{case['name']}_{mode}_o{offset}"
        cmd = build_qsub_cmd(
            script=BASH_SCRIPT,
            job_name=job_name,
            h_rt=h_rt,
            script_args=script_args,
        )
        jid = submit_qsub(cmd, dry_run=dry_run, cwd=str(SCRIPT_DIR), log_dir=LOG_DIR)
        if jid:
            chunk_jids.append(jid)

    # Submit the merge job, held on every chunk job (intra-stage dependency).
    merge_script = str(SCRIPT_DIR / "submit_merge.sh")
    merge_name = f"EnsMrg_{case['name']}_{mode}"
    out_dir = f"{case['base']}/{mode}"
    merge_args = [
        out_dir,
        str(ensemble_size),
        ",".join(str(f) for f in frames),
        str(len(offsets)),
    ]
    merge_cmd = build_qsub_cmd(
        script=merge_script,
        job_name=merge_name,
        h_rt="00:30:00",
        script_args=merge_args,
        hold_jids=chunk_jids,
    )
    return submit_qsub(merge_cmd, dry_run=dry_run, cwd=str(SCRIPT_DIR), log_dir=LOG_DIR)


def dispatch(
    case_name: str,
    *,
    only_modes: Optional[List[str]] = None,
    ensemble_size_override: Optional[int] = None,
    quick: bool = False,
    dry_run: bool = False,
    members_per_job: Optional[int] = DEFAULT_MEMBERS_PER_JOB,
) -> Dict[str, Optional[str]]:
    """Submit all ensemble jobs for a case.  Returns ``{mode: job_id_or_None}``."""
    case = load_case(case_name)
    ens = case["ensemble"]

    if quick:
        ensemble_size = QUICK_ENSEMBLE_SIZE
        h_rt = QUICK_H_RT
    else:
        ensemble_size = case_ensemble_size(case, override=ensemble_size_override)
        h_rt = DEFAULT_H_RT

    batch_size = int(ens.get("batch_size", DEFAULT_BATCH_SIZE))

    modes_to_run = list(iter_modes(case))
    if only_modes:
        modes_to_run = [(m, f) for m, f in modes_to_run if m in only_modes]
        missing = set(only_modes) - {m for m, _ in modes_to_run}
        if missing:
            print(
                f"WARNING: case '{case['name']}' does not define modes: {sorted(missing)}",
                file=sys.stderr,
            )

    if quick:
        print(f"[QUICK MODE] ensemble_size={ensemble_size}, h_rt={h_rt}")
    print(
        f"[{case['name']}] submitting {len(modes_to_run)} job(s): "
        f"{[m for m, _ in modes_to_run]}  (n={ensemble_size}, h_rt={h_rt})"
    )

    results: Dict[str, Optional[str]] = {}
    for mode, frames in modes_to_run:
        results[mode] = submit_mode(
            case, mode, frames,
            ensemble_size=ensemble_size,
            batch_size=batch_size,
            h_rt=h_rt,
            dry_run=dry_run,
            members_per_job=members_per_job,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch ensemble interpolation jobs for a named case.",
    )
    parser.add_argument("case", nargs="?", help="Case name (e.g. katrina)")
    parser.add_argument("--list", action="store_true",
                        help="List available cases and exit.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mode", action="append", default=None,
        choices=["start", "end", "both"],
        help="Restrict to one or more conditioning modes (repeatable).",
    )
    parser.add_argument(
        "--ensemble-size", type=int, default=None,
        help="Override ensemble.size from the case YAML.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help=f"Quick test: ensemble_size={QUICK_ENSEMBLE_SIZE}, h_rt={QUICK_H_RT}.",
    )
    parser.add_argument(
        "--members-per-job", type=int, default=DEFAULT_MEMBERS_PER_JOB,
        help=f"Chunk large ensembles into multiple parallel jobs "
             f"(default: {DEFAULT_MEMBERS_PER_JOB}). Pass a value >= "
             f"ensemble size to disable chunking.",
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
        quick=args.quick,
        dry_run=args.dry_run,
        members_per_job=args.members_per_job,
    )


if __name__ == "__main__":
    main()
