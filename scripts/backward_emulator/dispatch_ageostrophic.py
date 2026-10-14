"""Dispatch the ageostrophic comparison job for one or more cases.

One job per case (the diagnostic is a single global-band scalar at 500 hPa, not
per-variable), running ``ageostrophic_compare.py`` (makani native vs cBottle
regridded).  Each job loads the shared survivor-mask cache ``survival_<case>.npz``
-- the same file the spread/RMSE dispatcher writes -- so makani's ensemble summary
is censored consistently; if that cache does not exist yet, the job computes it
once and writes it (and the spread jobs would then reuse it).  Ordering between the
two dispatchers therefore does not matter.

Usage (from this directory):
    python dispatch_ageostrophic.py --dry-run
    python dispatch_ageostrophic.py --case ian
    python dispatch_ageostrophic.py --no-cbottle          # makani-only
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

SCRIPT = str(BW_DIR / "submit_ageostrophic_compare.sh")
LOG_DIR = str(BW_DIR / "compare_logs")
H_RT = "08:00:00"

CASE_DATA_SUBDIR = {"pnw_heatwave": "2021_pnw_heatwave", "ian": "ian"}
DEFAULT_CASES = ["pnw_heatwave", "ian"]
DEFAULT_DATA_ROOT = "/projectnb/eb-general/jlin404/scratch/backward_emulator_data"
DEFAULT_MAKANI_ANALYSIS = (
    "/projectnb/eb-general/jlin404/weather_interpolation/makani-aicr/analysis"
)


def dispatch(
    case_name: str,
    *,
    data_root: str = DEFAULT_DATA_ROOT,
    out_root: str = "./out",
    makani_analysis_dir: str = DEFAULT_MAKANI_ANALYSIS,
    level: int = 500,
    cbottle: bool = True,
    survival_hold_jid: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Submit the ageostrophic comparison job for one case.

    ``survival_hold_jid``: hold on an externally-submitted survivor root (shared with
    spread/PCA by ``dispatch_all.py``) so the ~50 GB survivor scan runs once per case.
    """
    load_case(case_name)  # validate the case exists
    subdir = CASE_DATA_SUBDIR.get(case_name, case_name)
    makani_dir = f"{data_root}/{subdir}"
    case_out = f"{out_root}/{case_name}"
    surv_npz = f"{case_out}/survival_{case_name}.npz"

    args = ["--makani-dir", makani_dir, "--case", case_name,
            "--out-dir", f"{case_out}/ageostrophic", "--level", str(level),
            "--survival-npz", surv_npz,
            "--makani-analysis-dir", makani_analysis_dir]
    if not cbottle:
        args += ["--no-cbottle"]
    cmd = build_qsub_cmd(script=SCRIPT, job_name=f"Ageo_{case_name}",
                         h_rt=H_RT, script_args=args,
                         hold_jids=[survival_hold_jid] if survival_hold_jid else ())
    jid = submit_qsub(cmd, dry_run=dry_run, cwd=str(BW_DIR), log_dir=LOG_DIR)
    print(f"[dispatch-ageo] {case_name}: ageostrophic job "
          f"{'(makani-only)' if not cbottle else '(makani+cBottle)'}")
    return {"case": case_name, "jid": jid, "cmd": cmd}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", action="append", default=None,
                   help="Case name (repeatable). Default: pnw_heatwave and ian.")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--out-root", default="./out")
    p.add_argument("--makani-analysis-dir", default=DEFAULT_MAKANI_ANALYSIS)
    p.add_argument("--level", type=int, default=500)
    p.add_argument("--no-cbottle", dest="cbottle", action="store_false", default=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cases = args.case or DEFAULT_CASES
    recs: List[Dict[str, Any]] = []
    for case_name in cases:
        recs.append(dispatch(
            case_name, data_root=args.data_root, out_root=args.out_root,
            makani_analysis_dir=args.makani_analysis_dir, level=args.level,
            cbottle=args.cbottle, dry_run=args.dry_run))
    print(f"[dispatch-ageo] {'previewed' if args.dry_run else 'submitted'} "
          f"{len(recs)} ageostrophic job(s).")


if __name__ == "__main__":
    main()
