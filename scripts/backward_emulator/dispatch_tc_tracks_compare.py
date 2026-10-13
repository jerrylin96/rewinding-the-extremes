"""Dispatch the makani TC-tracking job (TempestExtremes).

Only **Ian** has a tropical cyclone; PNW has none, so this submits one job for
Ian by default. The job writes a parquet in the paper's schema
(`tc_tracks_makani_ian_end.parquet`) that then feeds Ian's synoptic-PCA impact
target and the TC-track aggregator.

Pre-fetch the WB2 orography once on a login node first (see submit script /
`smoke_tc_tracks.py`).

Usage (from this directory):
    python dispatch_tc_tracks_compare.py --dry-run
    python dispatch_tc_tracks_compare.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

BW_DIR = Path(__file__).resolve().parent
SHARED = BW_DIR.parent / "_shared"
sys.path.insert(0, str(SHARED))

from _dispatch_lib import build_qsub_cmd, load_case, submit_qsub  # noqa: E402

SCRIPT = str(BW_DIR / "submit_tc_tracks_compare.sh")
LOG_DIR = str(BW_DIR / "compare_logs")
H_RT = "12:00:00"

CASE_DATA_SUBDIR = {"pnw_heatwave": "2021_pnw_heatwave", "ian": "ian"}
DEFAULT_CASES = ["ian"]  # only Ian has a TC
DEFAULT_DATA_ROOT = "/projectnb/eb-general/jlin404/scratch/backward_emulator_data"


def dispatch(case_name: str, *, data_root: str = DEFAULT_DATA_ROOT,
             out_root: str = "./out", dry_run: bool = False) -> Dict[str, Any]:
    load_case(case_name)
    subdir = CASE_DATA_SUBDIR.get(case_name, case_name)
    args = ["--makani-dir", f"{data_root}/{subdir}", "--case", case_name,
            "--out-dir", f"{out_root}/{case_name}/tc_tracks"]
    cmd = build_qsub_cmd(script=SCRIPT, job_name=f"TC_{case_name}",
                         h_rt=H_RT, script_args=args)
    jid = submit_qsub(cmd, dry_run=dry_run, cwd=str(BW_DIR), log_dir=LOG_DIR)
    print(f"[dispatch-tc] {case_name}: TempestExtremes tracking job")
    return {"case": case_name, "jid": jid, "cmd": cmd}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", action="append", default=None,
                   help="Case name (repeatable). Default: ian.")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--out-root", default="./out")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    recs: List[Dict[str, Any]] = []
    for case_name in (args.case or DEFAULT_CASES):
        recs.append(dispatch(case_name, data_root=args.data_root,
                             out_root=args.out_root, dry_run=args.dry_run))
    print(f"[dispatch-tc] {'previewed' if args.dry_run else 'submitted'} "
          f"{len(recs)} tracking job(s).")


if __name__ == "__main__":
    main()
