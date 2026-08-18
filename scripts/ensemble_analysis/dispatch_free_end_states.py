"""Dispatch free-end-state ranking jobs for a named case study.

Reads ``cases/<event>.yaml`` and submits one CPU qsub job per conditioning
mode (``start`` and ``end`` — ``both`` has no free end and is skipped)
plus a single dependent aggregator job that renders the per-variable
4-row figure.

For each mode, the job ranks every ensemble member at the *free-end*
frame (frame 11 for ``start`` conditioning, frame 0 for ``end``
conditioning) by its signed box-mean departure from ERA5 (member box-mean
minus ERA5 box-mean) inside the case's domain of interest
(``regions.impact`` from the YAML, overridable via ``--bbox``).  The top-K
coolest/lowest and top-K warmest/highest member fields are saved alongside
ERA5 so the aggregator can plot them without re-touching the multi-GB
ensemble zarr.

The saved fields are cropped to a wider plotting field of view
(``regions.synoptic`` from the YAML, else the impact box padded out,
overridable via ``--view-bbox``) so the aggregator can show synoptic
context around the dashed impact box — matching the synoptic-PCA precursor
panels.  The bias ranking and the bbox-mean timeseries still reduce over
``regions.impact``.

Output layout (per variable, modes aggregated)::

    <output_root>/free_end_states/<variable>/
        free_end_states_start.npz
        free_end_states_end.npz
        free_end_states.png           (4-row figure; end on top, start below)

Default variables: ``z500`` and ``msl``.  Override on the command line or
via ``diagnostics.variables`` in the case YAML.

``--mask`` restricts the RMSE + bbox-mean reductions to ``land`` or ``sea``
pixels (via the ERA5-derived HPX land mask in ``compute_free_end_states``);
without it the metric uses every HPX pixel inside the bbox.  Cases can also
declare per-variable masks in their YAML so e.g. ``t2m`` for a heatwave
diagnostic gets land-masked while ``z500`` / ``msl`` stay unmasked::

    diagnostics:
      variables: [t2m, z500, msl]
      free_end_states:
        mask: {t2m: land}            # variables not listed default to 'none'

When variables in one dispatch invocation need different masks, the
dispatcher submits one qsub per distinct mask value (the bash submit
script applies a single ``--mask`` to all variables it iterates over).
The CLI ``--mask`` overrides the YAML for *all* variables.

Usage::

    python dispatch_free_end_states.py sandy
    python dispatch_free_end_states.py sandy --dry-run
    python dispatch_free_end_states.py sandy --mode end
    python dispatch_free_end_states.py sandy --variables z500,msl,t850
    python dispatch_free_end_states.py sandy --bbox -100 -50 20 50
    python dispatch_free_end_states.py sandy --view-bbox -120 -20 15 60
    python dispatch_free_end_states.py pnw_heatwave --mask land
    python dispatch_free_end_states.py --list
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
    ensemble_zarr_path,
    era5_zarr_path,
    iter_modes,
    list_cases,
    load_case,
    region_box,
    require_region,
    resolve_timezone,
    start_time_iso_from_case,
    submit_qsub,
    validate_zarrs_exist,
)

SRC_SCRIPT = str(SCRIPT_DIR / "submission_scripts" / "submit_free_end_states.sh")
AGG_SCRIPT = str(
    SCRIPT_DIR / "submission_scripts" / "submit_aggregate_free_end_states.sh"
)
SRC_LOG_DIR = str(SCRIPT_DIR / "submission_scripts" / "free_end_states_logs")

DEFAULT_H_RT = "2:00:00"
AGG_H_RT = "0:30:00"
DEFAULT_VARIABLES = ["z500", "msl"]
DEFAULT_TOP_K = 3
# Field-of-view fallback for cases without a ``regions.synoptic`` box: pad the
# impact box outward by this fraction of its width/height per side so the
# dashed box sits inside some surrounding context instead of hugging the frame.
DEFAULT_VIEW_PAD_FRAC = 0.75
SUPPORTED_MODES = ("start", "end")
SUPPORTED_MASKS = ("none", "land", "sea")


def _resolve_variables(
    case: Dict[str, Any], override: Optional[List[str]]
) -> List[str]:
    """Pick the variable list for the diagnostic.

    Priority: explicit ``--variables`` > ``diagnostics.variables`` in the
    case YAML > the hardcoded fallback ``["z500", "msl"]``.
    """
    if override:
        return list(override)
    diag = case.get("diagnostics", {}) or {}
    if "variables" in diag:
        return list(diag["variables"])
    return list(DEFAULT_VARIABLES)


def _resolve_var_masks(
    case: Dict[str, Any],
    variables: List[str],
    override: Optional[str],
) -> Dict[str, str]:
    """Resolve the {variable -> mask_kind} mapping for this dispatch.

    Priority: explicit ``--mask`` CLI (applies to every variable) >
    ``diagnostics.free_end_states.mask`` in the case YAML > default of
    ``none`` for each variable.  The YAML form is a dict so e.g. a
    heatwave case can mask only ``t2m`` to land and leave ``z500`` /
    ``msl`` unmasked.  Unknown mask values raise; variables absent from
    the YAML dict default to ``none``.
    """
    if override is not None:
        if override not in SUPPORTED_MASKS:
            raise SystemExit(
                f"ERROR: --mask must be one of {list(SUPPORTED_MASKS)}; got {override!r}"
            )
        return {v: override for v in variables}

    diag = case.get("diagnostics", {}) or {}
    fes = diag.get("free_end_states", {}) or {}
    yaml_mask = fes.get("mask")
    if yaml_mask is None:
        return {v: "none" for v in variables}
    if not isinstance(yaml_mask, dict):
        raise SystemExit(
            f"ERROR: case '{case['name']}' diagnostics.free_end_states.mask "
            f"must be a mapping of variable -> mask_kind; got "
            f"{type(yaml_mask).__name__}."
        )
    out: Dict[str, str] = {}
    for v in variables:
        m = str(yaml_mask.get(v, "none"))
        if m not in SUPPORTED_MASKS:
            raise SystemExit(
                f"ERROR: case '{case['name']}' diagnostics.free_end_states."
                f"mask.{v} = {m!r}; must be one of {list(SUPPORTED_MASKS)}."
            )
        out[v] = m
    unknown_keys = [k for k in yaml_mask if k not in variables]
    if unknown_keys:
        print(
            f"[{case['name']}] WARNING: diagnostics.free_end_states.mask has "
            f"entries for variables not in this dispatch: {unknown_keys}"
        )
    return out


def _group_variables_by_mask(var_masks: Dict[str, str]) -> Dict[str, List[str]]:
    """Group variables by their resolved mask value, preserving input order.

    Returned dict maps ``mask_kind -> [var, var, ...]``.  The submit
    script's single ``--mask`` applies uniformly to every variable in
    its variable-loop iteration, so a single dispatch call may need
    multiple qsub submissions when the masks differ across variables.
    """
    groups: Dict[str, List[str]] = {}
    for v, m in var_masks.items():
        groups.setdefault(m, []).append(v)
    return groups


def _resolve_bbox(case: Dict[str, Any], override: Optional[List[float]]) -> List[float]:
    """Pick the lat/lon box for the diagnostic.

    Priority: explicit ``--bbox`` > ``regions.impact`` in the case YAML.
    ``regions.impact`` is the single effect region -- the same box (and mask)
    the synoptic-PCA scalar impact averages over -- so the two diagnostics
    report one impact average, never two divergent ones.  Errors out if
    neither is available so the domain is explicit rather than a silent
    global default.
    """
    if override:
        if len(override) != 4:
            raise SystemExit(
                f"ERROR: --bbox needs four numbers "
                f"(lon_min lon_max lat_min lat_max); got {override!r}"
            )
        return list(override)
    return [float(x) for x in require_region(case, "impact")]


def _pad_bbox(bbox: List[float], frac: float) -> List[float]:
    """Expand ``bbox`` outward by ``frac`` of its width/height per side.

    Clamped to the valid lon/lat range.  Used as the field-of-view fallback
    for cases without a ``regions.synoptic`` box.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    dlon = (lon_max - lon_min) * frac
    dlat = (lat_max - lat_min) * frac
    return [
        max(lon_min - dlon, -180.0),
        min(lon_max + dlon, 180.0),
        max(lat_min - dlat, -90.0),
        min(lat_max + dlat, 90.0),
    ]


def _resolve_view_bbox(
    case: Dict[str, Any],
    impact_bbox: List[float],
    override: Optional[List[float]],
) -> List[float]:
    """Pick the plotting field of view for the spatial panels.

    Priority: explicit ``--view-bbox`` > ``regions.synoptic`` in the case
    YAML > the impact box padded by :data:`DEFAULT_VIEW_PAD_FRAC`.  Using
    ``regions.synoptic`` makes the free-end panels share the synoptic-PCA
    field of view (the impact box is drawn dashed inside it); cases without
    a synoptic box fall back to a padded impact box so the box still sits
    inside some surrounding context rather than hugging the panel frame.
    """
    if override:
        if len(override) != 4:
            raise SystemExit(
                f"ERROR: --view-bbox needs four numbers "
                f"(lon_min lon_max lat_min lat_max); got {override!r}"
            )
        return list(override)
    synoptic = region_box(case, "synoptic")
    if synoptic is not None:
        return [float(x) for x in synoptic]
    return _pad_bbox(impact_bbox, DEFAULT_VIEW_PAD_FRAC)


def submit_for_mode(
    case: Dict[str, Any],
    mode: str,
    *,
    ensemble_size: int,
    variables: List[str],
    mask: str,
    bbox: List[float],
    view_bbox: List[float],
    top_k: int,
    diag_output_root: str,
    h_rt: str,
    dry_run: bool,
) -> Optional[str]:
    """Submit the per-mode free-end-state compute job.

    Every variable in ``variables`` is computed with the same ``mask``
    in this submission.  The caller is responsible for grouping
    variables by mask and calling :func:`submit_for_mode` once per group.
    """
    base = case["base"]
    ens_zarr = ensemble_zarr_path(base, mode, ensemble_size)
    era5_zarr = era5_zarr_path(base, mode)

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
        "--mask",
        mask,
        "--case-name",
        case["display_name"],
        "--start-time",
        start_time_iso_from_case(case),
        "--timezone",
        resolve_timezone(case),
        "--top-k",
        str(top_k),
        "--bbox",
        *(str(b) for b in bbox),
        "--view-bbox",
        *(str(b) for b in view_bbox),
    ]

    # Tag the job name with the mask kind when it's non-default so two
    # qsub jobs for the same (case, mode) — one per mask group — don't
    # collide in the SGE listing or in -hold_jid wiring downstream.
    job_suffix = f"_{mask}" if mask != "none" else ""
    cmd = build_qsub_cmd(
        script=SRC_SCRIPT,
        job_name=f"FreeEnd_{case['name']}_{mode}{job_suffix}",
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
) -> Optional[str]:
    """Submit the cross-mode aggregator that renders the final figure."""
    script_args = [
        "--output-root",
        diag_output_root,
        "--variables",
        ",".join(variables),
    ]
    cmd = build_qsub_cmd(
        script=AGG_SCRIPT,
        job_name=f"AggFreeEnd_{case['name']}",
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
    mask_override: Optional[str] = None,
    bbox: Optional[List[float]] = None,
    view_bbox: Optional[List[float]] = None,
    top_k: int = DEFAULT_TOP_K,
    ensemble_size_override: Optional[int] = None,
    output_root_override: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Optional[str]]:
    """Submit free-end-state jobs for a case.

    Returns ``{<mode>[/<mask>]: jid_or_None, "aggregate": jid_or_None}``.
    When a mode has more than one mask group, the per-job result keys
    are suffixed with ``/<mask>`` so the caller can recover the mapping.
    """
    case = load_case(case_name)
    ensemble_size = case_ensemble_size(case, override=ensemble_size_override)
    resolved_variables = _resolve_variables(case, variables)
    resolved_bbox = _resolve_bbox(case, bbox)
    resolved_view_bbox = _resolve_view_bbox(case, resolved_bbox, view_bbox)
    var_masks = _resolve_var_masks(case, resolved_variables, mask_override)
    mask_groups = _group_variables_by_mask(var_masks)

    output_root = output_root_override or f"{case['base']}/diagnostics"
    diag_output_root = f"{output_root}/free_end_states"

    # ``both`` has no free end, so we filter it out unconditionally
    # rather than failing loudly when the case YAML includes it.
    available = [m for m, _ in iter_modes(case) if m in SUPPORTED_MODES]
    if only_modes:
        bad = [m for m in only_modes if m not in SUPPORTED_MODES]
        if bad:
            raise SystemExit(
                f"ERROR: unsupported --mode value(s) {bad}; "
                f"valid: {list(SUPPORTED_MODES)} (the 'both' mode has no free end)."
            )
        modes_to_run = [m for m in available if m in only_modes]
    else:
        modes_to_run = available

    if not modes_to_run:
        raise SystemExit(
            f"ERROR: case '{case['name']}' has no supported modes after "
            f"filtering (available in case: "
            f"{[m for m, _ in iter_modes(case)]}, supported: {list(SUPPORTED_MODES)})."
        )

    n_jobs = len(modes_to_run) * len(mask_groups)
    print(
        f"[{case['name']}] submitting {n_jobs} free-end-state job(s): "
        f"modes={modes_to_run}, mask-groups={dict(mask_groups)} "
        f"(bbox={resolved_bbox}, view={resolved_view_bbox}, top-k={top_k})"
    )

    validate_zarrs_exist(case, modes_to_run, ensemble_size)

    results: Dict[str, Optional[str]] = {}
    for mode in modes_to_run:
        for mask_kind, group_vars in mask_groups.items():
            jid = submit_for_mode(
                case,
                mode,
                ensemble_size=ensemble_size,
                variables=group_vars,
                mask=mask_kind,
                bbox=resolved_bbox,
                view_bbox=resolved_view_bbox,
                top_k=top_k,
                diag_output_root=diag_output_root,
                h_rt=DEFAULT_H_RT,
                dry_run=dry_run,
            )
            key = mode if len(mask_groups) == 1 else f"{mode}/{mask_kind}"
            results[key] = jid

    hold_jids = [j for j in results.values() if j]
    results["aggregate"] = submit_aggregator(
        case,
        diag_output_root=diag_output_root,
        variables=resolved_variables,
        hold_jids=hold_jids,
        dry_run=dry_run,
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch free-end-state ranking jobs for a named case.",
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
        choices=list(SUPPORTED_MODES),
        help="Restrict to one or more conditioning modes (repeatable; "
        "'both' is intentionally not a valid choice).",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default=None,
        help=f"Comma-separated variable list (default: "
        f"diagnostics.variables in the case YAML, falling back to "
        f"{','.join(DEFAULT_VARIABLES)}).",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Impact box: the RMSE ranking + bbox-mean timeseries domain, "
        "drawn as a dashed box on the plots (default: regions.impact in the "
        "case YAML).",
    )
    parser.add_argument(
        "--view-bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Plotting field of view for the spatial panels, with the impact "
        "box drawn dashed inside it (default: regions.synoptic in the case "
        "YAML, else the impact box padded outward).",
    )
    parser.add_argument(
        "--mask",
        choices=list(SUPPORTED_MASKS),
        default=None,
        help=(
            "Apply a single land/sea mask to every variable in this dispatch. "
            "Overrides any per-variable mask configured in the case YAML "
            "under diagnostics.free_end_states.mask.  Default: read the YAML, "
            "fall back to 'none' for variables not configured."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of low/high (coolest/warmest) members per tail "
        f"(default: {DEFAULT_TOP_K}).",
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
        mask_override=args.mask,
        bbox=args.bbox,
        view_bbox=args.view_bbox,
        top_k=args.top_k,
        ensemble_size_override=args.ensemble_size,
        output_root_override=args.output_root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
