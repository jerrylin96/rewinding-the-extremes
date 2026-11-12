"""Dispatch the ageostrophic comparison job for one or more cases.

Per case: the survivor-mask job (``survival_<case>.npz``, the same file the
spread/RMSE jobs use) and one ``ageostrophic_compare.py`` job held on it (makani
native vs cBottle regridded; the diagnostic is one global-band scalar at 500 hPa,
not per-variable).  ``dispatch_all.py`` submits the survivor job itself and passes
it in, so the scan still runs once per case.  The figure is drawn afterwards, on
the login node, by ``plot_ageostrophic_compare.py``.

Usage (from this directory):
    python dispatch_ageostrophic.py --dry-run
    python dispatch_ageostrophic.py --case ian
    python dispatch_ageostrophic.py --no-cbottle          # makani-only
    python plot_ageostrophic_compare.py --case ian --case pnw_heatwave --case sandy
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
from dispatch_compare import submit_survival_root  # noqa: E402

SCRIPT = str(BW_DIR / "submit_ageostrophic_compare.sh")
LOG_DIR = str(BW_DIR / "compare_logs")
H_RT = "08:00:00"

CASE_DATA_SUBDIR = {"pnw_heatwave": "2021_pnw_heatwave", "ian": "ian"}
DEFAULT_CASES = ["pnw_heatwave", "ian", "sandy"]
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
    """Submit the survivor-mask job and the ageostrophic job held on it, for one case.

    ``survival_hold_jid``: hold on an externally-submitted survivor root instead
    (``dispatch_all.py`` shares one with spread/PCA) and submit none here.
    """
    load_case(case_name)  # validate the case exists
    subdir = CASE_DATA_SUBDIR.get(case_name, case_name)
    makani_dir = f"{data_root}/{subdir}"
    case_out = f"{out_root}/{case_name}"
    surv_npz = f"{case_out}/survival_{case_name}.npz"

    if survival_hold_jid is None:
        survival_hold_jid, _ = submit_survival_root(
            case_name, makani_dir, surv_npz, makani_analysis_dir, dry_run)
    args = ["--makani-dir", makani_dir, "--case", case_name,
            "--out-dir", f"{case_out}/ageostrophic", "--level", str(level),
            "--survival-npz", surv_npz,
            "--makani-analysis-dir", makani_analysis_dir]
    if not cbottle:
        args += ["--no-cbottle"]
    cmd = build_qsub_cmd(script=SCRIPT, job_name=f"Ageo_{case_name}",
                         h_rt=H_RT, script_args=args,
                         hold_jids=[survival_hold_jid])
    jid = submit_qsub(cmd, dry_run=dry_run, cwd=str(BW_DIR), log_dir=LOG_DIR)
    print(f"[dispatch-ageo] {case_name}: ageostrophic job "
          f"{'(makani-only)' if not cbottle else '(makani+cBottle)'}")
    return {"case": case_name, "jid": jid, "cmd": cmd,
            "survival_jid": survival_hold_jid}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", action="append", default=None,
                   help="Case name (repeatable). Default: pnw_heatwave, ian and sandy.")
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
