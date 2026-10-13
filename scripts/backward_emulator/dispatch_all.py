"""One-shot overnight submit for every backward-emulator comparison.

Submits the whole battery with correct dependency ordering, so the overnight run is
fire-and-forget.  Per case:

  1. ONE survivor-mask root job (the ~50 GB t2m scan) -> `survival_<case>.npz`;
  2. spread/RMSE/CRPS jobs (`dispatch_compare`), ageostrophic (`dispatch_ageostrophic`),
     and the synoptic-PCA job (`dispatch_synoptic_pca_makani`) ALL `-hold_jid` on that
     one root, so the scan is computed once and shared (not recomputed per diagnostic);
  3. Ian additionally gets the TC-tracks job, and Ian's PCA `-hold_jid`s on it too
     (its impact target is read from the tracks parquet).

Everything independent runs in parallel; only the two real dependencies
(survivor root -> {spread, ageo, PCA}; Ian TC -> Ian PCA) are serialized.

Run the login-node pre-steps first (land mask, glue gates, WB2 orography cache) --
see README.md.  `--dry-run` prints the full plan without submitting.

Usage (from this directory):
    python dispatch_all.py --dry-run
    python dispatch_all.py
    python dispatch_all.py --no-cbottle          # makani-only everywhere
    python dispatch_all.py --case ian
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BW_DIR = Path(__file__).resolve().parent
SHARED = BW_DIR.parent / "_shared"
for _p in (str(BW_DIR), str(SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _dispatch_lib import build_qsub_cmd, load_case, submit_qsub  # noqa: E402

import dispatch_ageostrophic as d_ageo  # noqa: E402
import dispatch_compare as d_cmp  # noqa: E402
import dispatch_synoptic_pca_makani as d_pca  # noqa: E402
import dispatch_tc_tracks_compare as d_tc  # noqa: E402

SURV_SCRIPT = str(BW_DIR / "submit_survival.sh")
LOG_DIR = str(BW_DIR / "compare_logs")
SURV_H_RT = "03:00:00"

CASE_DATA_SUBDIR = {"pnw_heatwave": "2021_pnw_heatwave", "ian": "ian"}
DEFAULT_CASES = ["pnw_heatwave", "ian"]
DEFAULT_DATA_ROOT = "/projectnb/eb-general/jlin404/scratch/backward_emulator_data"
DEFAULT_MAKANI_ANALYSIS = (
    "/projectnb/eb-general/jlin404/weather_interpolation/makani-aicr/analysis"
)


def _submit_survival_root(case_name, makani_dir, surv_npz, mk_dir, dry_run) -> str:
    cmd = build_qsub_cmd(
        script=SURV_SCRIPT, job_name=f"Surv_{case_name}", h_rt=SURV_H_RT,
        script_args=["--makani-dir", makani_dir, "--out", surv_npz,
                     "--makani-analysis-dir", mk_dir])
    jid = submit_qsub(cmd, dry_run=dry_run, cwd=str(BW_DIR), log_dir=LOG_DIR)
    return jid or f"<Surv_{case_name}>"  # placeholder keeps holds visible in dry-run


def run(
    cases: List[str], *, data_root: str = DEFAULT_DATA_ROOT, out_root: str = "./out",
    makani_analysis_dir: str = DEFAULT_MAKANI_ANALYSIS, cbottle: bool = True,
    dry_run: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Submit the full battery; return a per-case record of what went out."""
    # Pre-flight (real submits only): a case with a land-masked variable needs its
    # mask produced first (make_land_mask.py). Check ALL up front so we never submit
    # a survivor root and then abort mid-case, leaving an orphan job queued.
    if not dry_run:
        missing = []
        for case_name in cases:
            case = load_case(case_name)
            domains = (((case.get("diagnostics") or {}).get("spread_rmse_crps")
                        or {}).get("domains") or {})
            if any((e or {}).get("mask") == "land" for e in domains.values()):
                land = f"{out_root}/{case_name}/land_mask_{case_name}_721x1440.npy"
                if not Path(land).exists():
                    missing.append((case_name, land))
        if missing:
            lines = "\n".join(
                f"    python make_land_mask.py --case {c} --out {p}"
                for c, p in missing)
            raise SystemExit(
                "ERROR: land mask(s) missing before submit — produce them first "
                "(seconds, login node), then re-run:\n" + lines)

    summary: Dict[str, Dict[str, Any]] = {}
    for case_name in cases:
        case = load_case(case_name)
        subdir = CASE_DATA_SUBDIR.get(case_name, case_name)
        makani_dir = f"{data_root}/{subdir}"
        surv_npz = f"{out_root}/{case_name}/survival_{case_name}.npz"

        print(f"\n===== {case_name} =====")
        surv_jid = _submit_survival_root(case_name, makani_dir, surv_npz,
                                         makani_analysis_dir, dry_run)
        common = dict(data_root=data_root, out_root=out_root,
                      makani_analysis_dir=makani_analysis_dir, dry_run=dry_run)

        spread = d_cmp.dispatch(case_name, cbottle=cbottle,
                                survival_hold_jid=surv_jid, **common)
        ageo = d_ageo.dispatch(case_name, cbottle=cbottle,
                               survival_hold_jid=surv_jid, **common)

        tc_jid: Optional[str] = None
        tc = None
        if case.get("tc_tracks"):  # Ian: tracks first, PCA holds on it
            tc = d_tc.dispatch(case_name, data_root=data_root, out_root=out_root,
                               dry_run=dry_run)
            tc_jid = tc["jid"] or f"<TC_{case_name}>"

        pca = d_pca.dispatch(case_name, survival_hold_jid=surv_jid,
                             tc_hold_jid=tc_jid, **common)

        summary[case_name] = {"survival_jid": surv_jid, "spread": spread,
                              "ageo": ageo, "tc": tc, "pca": pca}
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", action="append", default=None,
                   help="Case name (repeatable). Default: pnw_heatwave and ian.")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--out-root", default="./out")
    p.add_argument("--makani-analysis-dir", default=DEFAULT_MAKANI_ANALYSIS)
    p.add_argument("--no-cbottle", dest="cbottle", action="store_false", default=True,
                   help="makani-only everywhere (not gated on the cBottle glue).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    summary = run(args.case or DEFAULT_CASES, data_root=args.data_root,
                  out_root=args.out_root, makani_analysis_dir=args.makani_analysis_dir,
                  cbottle=args.cbottle, dry_run=args.dry_run)
    n_cmp = sum(len([r for r in c["spread"] if r["role"] == "compare"])
                for c in summary.values())
    n_other = sum((1 + (1 if c["tc"] else 0) + 1) for c in summary.values())  # ageo+tc?+pca
    print(f"\n[dispatch-all] {'previewed' if args.dry_run else 'submitted'}: "
          f"{len(summary)} case(s), {len(summary)} survivor root(s), "
          f"{n_cmp} spread + {n_other} (ageo/tc/pca) job(s).")
    if args.dry_run:
        print("[dispatch-all] dry-run only; re-run without --dry-run to submit.")


if __name__ == "__main__":
    main()
