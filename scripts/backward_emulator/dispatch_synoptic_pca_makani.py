"""Dispatch the makani synoptic-PCA compute (Text S2), per case.

Resolves each case's precursor/impact from the case YAML and this project's
conventions:
* **PNW** -> scalar impact (land-mean t2m over regions.impact); needs the land mask
  from make_land_mask.py.
* **Ian** -> track impact (free-end storm properties); needs the makani TC parquet
  from the tc_tracks job -- pass `--tc-hold-jid <jobid>` so Ian's PCA waits for it.

Both load the shared `survival_<case>.npz` for the makani-specific free-end
divergence censoring (live in both cases: 778/1000 PNW and 867/1000 Ian members
survive the 66 h window).  The light compare step
(`synoptic_pca_compare.py`, makani vs the paper's cBottle npz) is run afterward,
interactively.

Usage (from this directory):
    python dispatch_synoptic_pca_makani.py --dry-run
    python dispatch_synoptic_pca_makani.py --case pnw_heatwave
    python dispatch_synoptic_pca_makani.py --case ian --tc-hold-jid 7323999
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BW_DIR = Path(__file__).resolve().parent
SHARED = BW_DIR.parent / "_shared"
sys.path.insert(0, str(SHARED))

from _dispatch_lib import build_qsub_cmd, load_case, submit_qsub  # noqa: E402

SCRIPT = str(BW_DIR / "submit_synoptic_pca_makani.sh")
LOG_DIR = str(BW_DIR / "compare_logs")
H_RT = "04:00:00"

CASE_DATA_SUBDIR = {"pnw_heatwave": "2021_pnw_heatwave", "ian": "ian"}
DEFAULT_CASES = ["pnw_heatwave", "ian"]
DEFAULT_DATA_ROOT = "/projectnb/eb-general/jlin404/scratch/backward_emulator_data"
DEFAULT_MAKANI_ANALYSIS = (
    "/projectnb/eb-general/jlin404/weather_interpolation/makani-aicr/analysis"
)


def dispatch(
    case_name: str, *, data_root: str = DEFAULT_DATA_ROOT, out_root: str = "./out",
    land_mask_dir: Optional[str] = None,
    makani_analysis_dir: str = DEFAULT_MAKANI_ANALYSIS,
    tc_hold_jid: Optional[str] = None, survival_hold_jid: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    case = load_case(case_name)
    subdir = CASE_DATA_SUBDIR.get(case_name, case_name)
    makani_dir = f"{data_root}/{subdir}"
    case_out = f"{out_root}/{case_name}"
    surv_npz = f"{case_out}/survival_{case_name}.npz"

    args = ["--makani-dir", makani_dir, "--case", case_name,
            "--out-dir", f"{case_out}/synoptic_pca",
            "--survival-npz", surv_npz,
            "--makani-analysis-dir", makani_analysis_dir]
    holds: List[str] = []
    if survival_hold_jid:  # shared survivor root (censoring cache)
        holds.append(str(survival_hold_jid))
    if case.get("tc_tracks"):  # track impact (Ian)
        parquet = f"{case_out}/tc_tracks/tc_tracks_makani_{case_name}_end.parquet"
        args += ["--impact-kind", "track", "--tc-parquet", parquet]
        if tc_hold_jid:
            holds.append(str(tc_hold_jid))
    else:  # scalar t2m impact (PNW)
        mask_dir = land_mask_dir or case_out
        args += ["--impact-kind", "scalar", "--impact-variable", "t2m",
                 "--land-mask", f"{mask_dir}/land_mask_{case_name}_721x1440.npy"]

    cmd = build_qsub_cmd(script=SCRIPT, job_name=f"SynPCA_{case_name}",
                         h_rt=H_RT, script_args=args, hold_jids=holds)
    jid = submit_qsub(cmd, dry_run=dry_run, cwd=str(BW_DIR), log_dir=LOG_DIR)
    hold_desc = f" (holds on {','.join(holds)})" if holds else ""
    print(f"[dispatch-pca] {case_name}: makani synoptic-PCA job{hold_desc}")
    return {"case": case_name, "jid": jid, "cmd": cmd}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", action="append", default=None,
                   help="Case name (repeatable). Default: pnw_heatwave and ian.")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--out-root", default="./out")
    p.add_argument("--land-mask-dir", default=None)
    p.add_argument("--makani-analysis-dir", default=DEFAULT_MAKANI_ANALYSIS)
    p.add_argument("--tc-hold-jid", default=None,
                   help="SGE job id of the Ian tc_tracks job; Ian's PCA holds on it.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    recs: List[Dict[str, Any]] = []
    for case_name in (args.case or DEFAULT_CASES):
        recs.append(dispatch(
            case_name, data_root=args.data_root, out_root=args.out_root,
            land_mask_dir=args.land_mask_dir,
            makani_analysis_dir=args.makani_analysis_dir,
            tc_hold_jid=args.tc_hold_jid, dry_run=args.dry_run))
    print(f"[dispatch-pca] {'previewed' if args.dry_run else 'submitted'} "
          f"{len(recs)} synoptic-PCA job(s).")


if __name__ == "__main__":
    main()
