"""Dispatch spread / RMSE / mean error / CRPS jobs for a named case study.

Reads ``cases/<event>.yaml`` and submits one CPU qsub job per
conditioning mode (start/end/both) per *domain group* (see below), plus a
dependent aggregator job per domain group that overlays the three modes on
the same plots.

Output layout (per variable, modes aggregated)::

    <output_root>/spread_rmse_crps/<variable>/
        spread_rmse_crps_<mode>[_<region>[_<mask>]].npz
        spread_rmse_crps[_<region>[_<mask>]].png
                                     (3-row combined figure: spread/skill
                                      calibration on top, signed
                                      ensemble-mean error in the middle,
                                      3-panel spread / per-member
                                      RMSE / CRPS on the bottom)

Domains are per *variable*, not per run
---------------------------------------
A case declares purpose-named boxes (``regions.synoptic`` = cause,
``regions.impact`` = effect), and which one a variable belongs on depends on
the variable: a height field is a synoptic-scale precursor and belongs on
``synoptic``, while a variable carrying the event's own signature -- MSLP for
a TC, t2m for a heatwave -- belongs on ``impact``, because averaging its
ensemble spread over the wider box dilutes the event with background.  Some
variables additionally need a land/sea mask (``t2m`` over ocean is driven by
the prescribed member-identical SST, so unmasked ocean pixels carry full
weight in the pixel mean while contributing almost no spread).

Declare the mapping in the case YAML::

    diagnostics:
      spread_rmse_crps:
        domains:
          z500: {region: synoptic}
          msl:  {region: impact}
          t2m:  {region: impact, mask: land}

Variables absent from ``domains`` reduce globally with no mask, which is what
this dispatcher did before the block existed.  Because the bash submit script
applies a single ``--region`` / ``--mask`` to every variable it iterates over,
the dispatcher submits one qsub per distinct ``(region, mask)`` pair -- the
same grouping ``dispatch_free_end_states.py`` uses for its per-variable masks.

The CLI ``--region`` / ``--mask`` override the YAML for *all* variables.
``--region global`` and ``--mask none`` are the escape hatches back to an
unrestricted reduction, so a global-vs-regional comparison stays one flag
away and the pre-domains figures remain reproducible from the CLI.

Usage:
    python dispatch_spread_rmse_crps.py sandy
    python dispatch_spread_rmse_crps.py sandy --dry-run
    python dispatch_spread_rmse_crps.py sandy --mode end
    python dispatch_spread_rmse_crps.py sandy --variables z500,t2m
    python dispatch_spread_rmse_crps.py sandy --region synoptic   # all vars
    python dispatch_spread_rmse_crps.py --list
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
    region_box,
    resolve_landfall_frame,
    resolve_timezone,
    resolve_variable_set,
    start_time_iso_from_case,
    submit_qsub,
    validate_zarrs_exist,
)

SRC_SCRIPT = str(SCRIPT_DIR / "submission_scripts" / "submit_spread_rmse_crps.sh")
AGG_SCRIPT = str(
    SCRIPT_DIR / "submission_scripts" / "submit_aggregate_spread_rmse_crps.sh"
)
SRC_LOG_DIR = str(SCRIPT_DIR / "submission_scripts" / "spread_rmse_crps_logs")

DEFAULT_H_RT = "4:00:00"
AGG_H_RT = "0:30:00"

SUPPORTED_MASKS = ("none", "land", "sea")

# Reserved ``--region`` value meaning "no box at all".  Needed because a
# variable that declares a region in the YAML would otherwise have no way
# back to a global reduction, and the global-vs-regional comparison is
# exactly what the [R1] framing decision turns on.
GLOBAL_REGION = "global"

# A resolved domain: (region name or None for global, mask kind).
Domain = Tuple[Optional[str], str]


def _resolve_var_domains(
    case: Dict[str, Any],
    variables: List[str],
    region_override: Optional[str],
    mask_override: Optional[str],
) -> Dict[str, Domain]:
    """Resolve the ``{variable -> (region_name, mask_kind)}`` mapping.

    Priority, independently for the region and the mask: explicit CLI
    override (applies to every variable) > ``diagnostics.spread_rmse_crps.
    domains.<var>`` in the case YAML > global / unmasked.  The two are
    independent so ``--mask land`` can be layered onto YAML-declared regions
    without restating them.

    Region names are validated against the case's ``regions`` block here
    rather than in the qsub job, so a typo fails in a second at the login
    node instead of four hours into a queue slot.
    """
    if mask_override is not None and mask_override not in SUPPORTED_MASKS:
        raise SystemExit(
            f"ERROR: --mask must be one of {list(SUPPORTED_MASKS)}; "
            f"got {mask_override!r}"
        )

    if (case.get("regions") or {}).get(GLOBAL_REGION) is not None:
        raise SystemExit(
            f"ERROR: case '{case['name']}' defines a region named "
            f"'{GLOBAL_REGION}', which is reserved here as the sentinel for "
            f"'no box at all'. Rename it."
        )

    diag = case.get("diagnostics", {}) or {}
    src = diag.get("spread_rmse_crps", {}) or {}
    yaml_domains = src.get("domains")
    if yaml_domains is None:
        yaml_domains = {}
    elif not isinstance(yaml_domains, dict):
        raise SystemExit(
            f"ERROR: case '{case['name']}' diagnostics.spread_rmse_crps.domains "
            f"must be a mapping of variable -> {{region, mask}}; got "
            f"{type(yaml_domains).__name__}."
        )

    out: Dict[str, Domain] = {}
    for v in variables:
        entry = yaml_domains.get(v) or {}
        if not isinstance(entry, dict):
            raise SystemExit(
                f"ERROR: case '{case['name']}' diagnostics.spread_rmse_crps."
                f"domains.{v} must be a mapping with optional 'region' and "
                f"'mask' keys; got {entry!r}."
            )

        region_name = region_override or entry.get("region")
        if region_name == GLOBAL_REGION:
            # The sentinel clears the YAML region rather than naming a box.
            region_name = None
        elif region_name is not None:
            region_name = str(region_name)
            if region_box(case, region_name) is None:
                raise SystemExit(
                    f"ERROR: case '{case['name']}': variable '{v}' asks for "
                    f"region '{region_name}', which is not defined in the "
                    f"case YAML's regions block."
                )

        mask_kind = str(mask_override or entry.get("mask") or "none")
        if mask_kind not in SUPPORTED_MASKS:
            raise SystemExit(
                f"ERROR: case '{case['name']}' diagnostics.spread_rmse_crps."
                f"domains.{v}.mask = {mask_kind!r}; must be one of "
                f"{list(SUPPORTED_MASKS)}."
            )
        out[v] = (region_name, mask_kind)

    # Warn only for keys the CASE does not have at all -- i.e. actual typos.
    # Checking against ``variables`` instead would fire on every deliberate
    # ``--variables`` subset, which is the normal way this is run, and a
    # warning that cries wolf on the common path trains people to ignore it.
    case_vars = set(resolve_variable_set(case))
    unknown_keys = [k for k in yaml_domains if k not in case_vars]
    if unknown_keys:
        print(
            f"[{case['name']}] WARNING: diagnostics.spread_rmse_crps.domains "
            f"has entries for variables this case does not declare: "
            f"{unknown_keys}"
        )
    return out


def _group_variables_by_domain(
    var_domains: Dict[str, Domain],
) -> Dict[Domain, List[str]]:
    """Group variables by resolved domain, preserving input order.

    One qsub is submitted per group: the submit script applies a single
    ``--region`` / ``--mask`` to every variable it iterates over, so
    variables on different domains cannot share a job.
    """
    groups: Dict[Domain, List[str]] = {}
    for v, d in var_domains.items():
        groups.setdefault(d, []).append(v)
    return groups


def _domain_label(domain: Domain) -> str:
    """Short human-readable domain, for log lines and job names."""
    region_name, mask_kind = domain
    label = region_name or "global"
    if mask_kind != "none":
        label += f"-{mask_kind}"
    return label


def _domain_script_args(case: Dict[str, Any], domain: Domain) -> List[str]:
    """Translate a resolved domain into ``compute_spread_rmse_crps.py`` flags.

    A global unmasked domain contributes no flags at all, so the legacy
    filenames (and therefore the already-committed global figures) are
    reproduced byte-for-byte.
    """
    region_name, mask_kind = domain
    args: List[str] = []
    if region_name is not None:
        box = region_box(case, region_name)
        if box is None:  # pragma: no cover - validated in _resolve_var_domains
            raise SystemExit(
                f"case '{case['name']}': regions.{region_name} is not defined "
                f"in the case YAML."
            )
        args += ["--region", *[str(v) for v in box], "--region-name", region_name]
    if mask_kind != "none":
        args += ["--mask", mask_kind]
    return args


def submit_for_mode(
    case: Dict[str, Any],
    mode: str,
    *,
    ensemble_size: int,
    variables: List[str],
    diag_output_root: str,
    h_rt: str,
    dry_run: bool,
    domain: Domain = (None, "none"),
) -> Optional[str]:
    """Submit the per-mode, per-domain-group spread/RMSE/CRPS job."""
    base = case["base"]
    ens_zarr = ensemble_zarr_path(base, mode, ensemble_size)
    era5_zarr = era5_zarr_path(base, mode)
    landfall_frame = resolve_landfall_frame(case)

    script_args = [
        "--ensemble-zarr",
        ens_zarr,
        "--era5-zarr",
        era5_zarr,
        "--output-root",
        diag_output_root,
        "--variables",
        ",".join(variables),
        "--mode",
        mode,
        "--case-name",
        case["display_name"],
        "--start-time",
        start_time_iso_from_case(case),
        "--timezone",
        resolve_timezone(case),
        "--landfall-frame",
        str(landfall_frame),
    ]

    script_args += _domain_script_args(case, domain)

    cmd = build_qsub_cmd(
        script=SRC_SCRIPT,
        job_name=f"SpreadRMSE_{case['name']}_{mode}_{_domain_label(domain)}",
        h_rt=h_rt,
        script_args=script_args,
    )
    return submit_qsub(
        cmd,
        dry_run=dry_run,
        cwd=str(SCRIPT_DIR / "submission_scripts"),
        log_dir=SRC_LOG_DIR,
    )


def submit_aggregator(
    case: Dict[str, Any],
    *,
    diag_output_root: str,
    variables: List[str],
    hold_jids: List[str],
    dry_run: bool,
    domain: Domain = (None, "none"),
) -> Optional[str]:
    """Submit the cross-mode aggregator that overlays start/end/both.

    One aggregator per domain group: the npz filenames encode the domain,
    so an aggregator must be told which set to read.
    """
    region_name, mask_kind = domain
    script_args = [
        "--output-root",
        diag_output_root,
        "--variables",
        ",".join(variables),
    ]
    if region_name:
        script_args += ["--region-name", region_name]
    if mask_kind != "none":
        script_args += ["--mask", mask_kind]
    cmd = build_qsub_cmd(
        script=AGG_SCRIPT,
        job_name=f"AggSpreadRMSE_{case['name']}_{_domain_label(domain)}",
        h_rt=AGG_H_RT,
        script_args=script_args,
        hold_jids=hold_jids,
    )
    return submit_qsub(
        cmd,
        dry_run=dry_run,
        cwd=str(SCRIPT_DIR / "submission_scripts"),
        log_dir=SRC_LOG_DIR,
    )


def dispatch(
    case_name: str,
    *,
    only_modes: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    ensemble_size_override: Optional[int] = None,
    output_root_override: Optional[str] = None,
    dry_run: bool = False,
    region_name: Optional[str] = None,
    mask: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Submit spread/RMSE/CRPS jobs for a case.

    Submits ``len(modes) * len(domain_groups)`` compute jobs and one
    aggregator per domain group.  Returns ``{key: jid_or_None}``, keyed by
    ``mode`` (or ``mode/domain`` when more than one domain group is in play)
    plus ``aggregate`` / ``aggregate/domain``.
    """
    case = load_case(case_name)
    ensemble_size = case_ensemble_size(case, override=ensemble_size_override)

    variables = variables or resolve_variable_set(case)
    if not variables:
        raise ValueError(
            f"case '{case['name']}': no diagnostic variables "
            f"(set diagnostics.variables or palette 'standard')"
        )

    output_root = output_root_override or f"{case['base']}/diagnostics"
    diag_output_root = f"{output_root}/spread_rmse_crps"

    modes_to_run = [m for m, _ in iter_modes(case)]
    if only_modes:
        modes_to_run = [m for m in modes_to_run if m in only_modes]

    var_domains = _resolve_var_domains(case, variables, region_name, mask)
    domain_groups = _group_variables_by_domain(var_domains)

    group_desc = {_domain_label(d): v for d, v in domain_groups.items()}
    print(
        f"[{case['name']}] submitting "
        f"{len(modes_to_run) * len(domain_groups)} spread/RMSE/CRPS job(s): "
        f"modes={modes_to_run}, domain-groups={group_desc}"
    )

    validate_zarrs_exist(case, modes_to_run, ensemble_size)

    results: Dict[str, Optional[str]] = {}
    multi = len(domain_groups) > 1
    for domain, group_vars in domain_groups.items():
        label = _domain_label(domain)
        group_jids: List[str] = []
        for mode in modes_to_run:
            jid = submit_for_mode(
                case,
                mode,
                ensemble_size=ensemble_size,
                variables=group_vars,
                diag_output_root=diag_output_root,
                h_rt=DEFAULT_H_RT,
                dry_run=dry_run,
                domain=domain,
            )
            results[f"{mode}/{label}" if multi else mode] = jid
            if jid:
                group_jids.append(jid)

        # Each aggregator holds only on its own group's compute jobs: the
        # groups write disjoint npz filenames, so a group must not wait on
        # another group's queue time.
        agg_key = f"aggregate/{label}" if multi else "aggregate"
        results[agg_key] = submit_aggregator(
            case,
            diag_output_root=diag_output_root,
            variables=group_vars,
            hold_jids=group_jids,
            dry_run=dry_run,
            domain=domain,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch spread/RMSE/CRPS jobs for a named case.",
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
        choices=["start", "end", "both"],
        help="Restrict to one or more conditioning modes (repeatable).",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default=None,
        help="Comma-separated variable list (default: resolved from the case YAML).",
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
        "--region",
        type=str,
        default=None,
        metavar="NAME",
        help="Override the per-variable region for EVERY variable, using "
        "regions.<NAME> from the case YAML. Pass 'global' to clear the "
        "YAML-declared regions and reduce over the whole globe, which is how "
        "the pre-domains figures are reproduced. Default: the region each "
        "variable declares under diagnostics.spread_rmse_crps.domains, or "
        "global for variables that declare none.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        choices=list(SUPPORTED_MASKS),
        help="Override the per-variable land/sea mask for EVERY variable. "
        "Default: the mask each variable declares under "
        "diagnostics.spread_rmse_crps.domains, or none.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available cases:")
        for n in list_cases():
            print(f"  {n}")
        return

    if not args.case:
        parser.error("case name required (or pass --list)")

    variables = (
        [v.strip() for v in args.variables.split(",") if v.strip()]
        if args.variables
        else None
    )

    dispatch(
        args.case,
        only_modes=args.mode,
        variables=variables,
        ensemble_size_override=args.ensemble_size,
        output_root_override=args.output_root,
        dry_run=args.dry_run,
        region_name=args.region,
        mask=args.mask,
    )


if __name__ == "__main__":
    main()
