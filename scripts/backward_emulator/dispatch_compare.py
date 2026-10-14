"""Dispatch the backward-emulator comparison jobs for one or more cases.

For each case this submits:

  1. ONE survivor-mask root job (``submit_survival.sh``) that writes
     ``survival_<case>.npz`` -- the ~50 GB t2m scan, done once.
  2. One ``submit_compare_spread_rmse_crps.sh`` job per variable, each
     ``-hold_jid``'d on the root so it starts only after the mask exists and
     LOADS it via ``--survival-npz`` (no rescan).

Which ``(region, mask)`` each variable lands on is read from the SAME case-YAML
block the paper's own dispatcher uses --
``diagnostics.spread_rmse_crps.domains`` -- so the comparison reduces each
variable over exactly the box (and land mask) the submitted paper did:

    diagnostics:
      spread_rmse_crps:
        domains:
          z500: {region: synoptic}
          msl:  {region: synoptic}   # (impact for a TC)
          t2m:  {region: impact, mask: land}

A ``mask: land`` row is translated to ``--land-mask <dir>/land_mask_<case>_721x1440.npy``
(produced once by ``make_land_mask.py``); the dispatcher checks that file exists
up front so a missing mask fails in a second at the login node, not hours into a
queue slot.  Only ``end`` conditioning is dispatched (makani is backward-from-present).

Usage (from this directory):
    python dispatch_compare.py --dry-run                 # both cases, preview
    python dispatch_compare.py --case ian
    python dispatch_compare.py --case pnw_heatwave --variables z500,t2m
    python dispatch_compare.py --no-cbottle              # makani-only (not gated on cBottle)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BW_DIR = Path(__file__).resolve().parent
SHARED = BW_DIR.parent / "_shared"
sys.path.insert(0, str(SHARED))

from _dispatch_lib import (  # noqa: E402
    build_qsub_cmd,
    load_case,
    region_box,
    resolve_variable_set,
    submit_qsub,
    validate_zarrs_exist,
)

CMP_SCRIPT = str(BW_DIR / "submit_compare_spread_rmse_crps.sh")
SURV_SCRIPT = str(BW_DIR / "submit_survival.sh")
LOG_DIR = str(BW_DIR / "compare_logs")

CMP_H_RT = "06:00:00"
SURV_H_RT = "03:00:00"

# makani data lives under <data-root>/<subdir>; the case YAML name and the
# transfer subdir differ only for PNW (the dir carries the year prefix).
CASE_DATA_SUBDIR = {"pnw_heatwave": "2021_pnw_heatwave", "ian": "ian"}
DEFAULT_CASES = ["pnw_heatwave", "ian"]
DEFAULT_DATA_ROOT = "/projectnb/eb-general/jlin404/scratch/backward_emulator_data"
DEFAULT_MAKANI_ANALYSIS = (
    "/projectnb/eb-general/jlin404/weather_interpolation/makani-aicr/analysis"
)

# A resolved per-variable domain: (region name or None for global, mask kind).
Domain = Tuple[Optional[str], str]


def _resolve_domain(
    case: Dict[str, Any], var: str,
    region_override: Optional[str], mask_override: Optional[str],
) -> Domain:
    """(region_name|None, mask_kind) for a variable, from the case YAML domains.

    Priority (independently for region and mask): CLI override > YAML
    ``diagnostics.spread_rmse_crps.domains.<var>`` > global / unmasked.  Region
    names are validated against the case's ``regions`` block here so a typo fails
    at the login node, not four hours into a queue slot.
    """
    diag = (case.get("diagnostics") or {}).get("spread_rmse_crps") or {}
    entry = (diag.get("domains") or {}).get(var) or {}

    region_name = region_override or entry.get("region")
    if region_name in ("global", "none"):
        region_name = None
    elif region_name is not None:
        region_name = str(region_name)
        if region_box(case, region_name) is None:
            raise SystemExit(
                f"ERROR: case '{case['name']}': variable '{var}' asks for region "
                f"'{region_name}', not defined in the case YAML regions block.")

    mask_kind = str(mask_override or entry.get("mask") or "none")
    if mask_kind not in ("none", "land"):
        raise SystemExit(
            f"ERROR: case '{case['name']}' variable '{var}' mask={mask_kind!r}; "
            f"this comparison only produces a 'land' mask (make_land_mask.py). "
            f"'sea' is not supported.")
    return region_name, mask_kind


def _domain_label(domain: Domain) -> str:
    region_name, mask_kind = domain
    label = region_name or "global"
    return f"{label}-{mask_kind}" if mask_kind != "none" else label


def dispatch(
    case_name: str,
    *,
    data_root: str = DEFAULT_DATA_ROOT,
    out_root: str = "./out",
    land_mask_dir: Optional[str] = None,
    makani_analysis_dir: str = DEFAULT_MAKANI_ANALYSIS,
    variables: Optional[List[str]] = None,
    region_override: Optional[str] = None,
    mask_override: Optional[str] = None,
    cbottle: bool = True,
    survival_hold_jid: Optional[str] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Submit the survivor root + per-variable compare jobs for one case.

    Returns a list of ``{role, case, variable, region, mask, jid, cmd}`` records
    (also used by the tests to assert the matrix without touching qsub).

    ``survival_hold_jid``: when given (by ``dispatch_all.py``), the per-case survivor
    root is NOT submitted here -- the compare jobs hold on this externally-submitted
    root instead, so one survivor scan is shared across spread/ageo/PCA.
    """
    case = load_case(case_name)
    subdir = CASE_DATA_SUBDIR.get(case_name, case_name)
    makani_dir = f"{data_root}/{subdir}"
    case_out = f"{out_root}/{case_name}"
    surv_npz = f"{case_out}/survival_{case_name}.npz"
    mask_dir = land_mask_dir or case_out
    land_npy = f"{mask_dir}/land_mask_{case_name}_721x1440.npy"

    if variables:
        var_list = variables
    else:
        domains = (
            (case.get("diagnostics") or {}).get("spread_rmse_crps") or {}
        ).get("domains") or {}
        var_list = list(domains) or resolve_variable_set(case)

    domains_by_var = {
        v: _resolve_domain(case, v, region_override, mask_override) for v in var_list
    }

    # Pre-flight (skipped in dry-run so a preview works off the analysis HPC):
    # cBottle stores must exist, and any land row needs its mask produced first.
    if not dry_run:
        if cbottle:
            validate_zarrs_exist(case, ["end"], int(case["ensemble"]["size"]))
        if any(m == "land" for _, m in domains_by_var.values()) \
                and not Path(land_npy).exists():
            raise SystemExit(
                f"ERROR: {case_name} has land-masked variable(s) but the mask "
                f"{land_npy} does not exist. Produce it first (seconds, login node):\n"
                f"    python make_land_mask.py --case {case_name} --out {land_npy}")

    print(f"[dispatch] {case_name}: survivor root + {len(var_list)} compare job(s) "
          f"{'(makani-only)' if not cbottle else '(makani+cBottle)'}; "
          f"vars={ {v: _domain_label(d) for v, d in domains_by_var.items()} }")

    records: List[Dict[str, Any]] = []

    # 1. survivor-mask root job -- unless a shared root was submitted externally.
    if survival_hold_jid is not None:
        root_jid = str(survival_hold_jid)  # hold on the externally-submitted root
    else:
        surv_cmd = build_qsub_cmd(
            script=SURV_SCRIPT, job_name=f"Surv_{case_name}", h_rt=SURV_H_RT,
            script_args=["--makani-dir", makani_dir, "--case", case_name,
                         "--out", surv_npz, "--makani-analysis-dir", makani_analysis_dir])
        root_jid = submit_qsub(surv_cmd, dry_run=dry_run, cwd=str(BW_DIR),
                               log_dir=LOG_DIR)
        if dry_run:
            root_jid = f"<Surv_{case_name}>"  # placeholder so holds show in preview
        records.append({"role": "survival", "case": case_name, "variable": None,
                        "region": None, "mask": None, "jid": root_jid, "cmd": surv_cmd})

    # 2. one compare job per variable, held on the root.
    for var, domain in domains_by_var.items():
        region_name, mask_kind = domain
        args = ["--makani-dir", makani_dir, "--case", case_name,
                "--variable", var, "--region-name", region_name or "global",
                "--out-dir", f"{case_out}/{var}",
                "--makani-analysis-dir", makani_analysis_dir,
                "--survival-npz", surv_npz]
        if mask_kind == "land":
            args += ["--land-mask", land_npy]
        if not cbottle:
            args += ["--no-cbottle"]
        cmd = build_qsub_cmd(
            script=CMP_SCRIPT,
            job_name=f"Cmp_{case_name}_{var}_{_domain_label(domain)}",
            h_rt=CMP_H_RT, script_args=args,
            hold_jids=[root_jid] if root_jid else ())
        jid = submit_qsub(cmd, dry_run=dry_run, cwd=str(BW_DIR), log_dir=LOG_DIR)
        records.append({"role": "compare", "case": case_name, "variable": var,
                        "region": region_name, "mask": mask_kind,
                        "jid": jid, "cmd": cmd})
    return records


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", action="append", default=None,
                   help="Case name (repeatable). Default: pnw_heatwave and ian.")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                   help="Root of the makani transfer dirs (<root>/<case-subdir>).")
    p.add_argument("--out-root", default="./out")
    p.add_argument("--land-mask-dir", default=None,
                   help="Directory holding land_mask_<case>_721x1440.npy "
                   "(default: <out-root>/<case>).")
    p.add_argument("--makani-analysis-dir", default=DEFAULT_MAKANI_ANALYSIS)
    p.add_argument("--variables", default=None,
                   help="Comma-separated variable list (default: the case YAML's "
                   "spread_rmse_crps.domains keys).")
    p.add_argument("--region", default=None,
                   help="Override the region for EVERY variable (regions.<NAME>, "
                   "or 'global').")
    p.add_argument("--mask", default=None, choices=["none", "land"],
                   help="Override the land mask for EVERY variable.")
    p.add_argument("--no-cbottle", dest="cbottle", action="store_false", default=True,
                   help="makani-only jobs (not gated on the cBottle regrid path).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the qsub commands without submitting.")
    args = p.parse_args(argv)

    cases = args.case or DEFAULT_CASES
    variables = ([v.strip() for v in args.variables.split(",") if v.strip()]
                 if args.variables else None)

    total = 0
    for case_name in cases:
        recs = dispatch(
            case_name, data_root=args.data_root, out_root=args.out_root,
            land_mask_dir=args.land_mask_dir,
            makani_analysis_dir=args.makani_analysis_dir, variables=variables,
            region_override=args.region, mask_override=args.mask,
            cbottle=args.cbottle, dry_run=args.dry_run)
        total += sum(1 for r in recs if r["role"] == "compare")
    print(f"[dispatch] {'previewed' if args.dry_run else 'submitted'} "
          f"{total} compare job(s) across {len(cases)} case(s).")


if __name__ == "__main__":
    main()
