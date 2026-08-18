#!/usr/bin/env python3
"""Numeric health screen for ensemble members, to run BEFORE rendering videos.

A member video that comes out as one flat colour is ambiguous by construction:
the panels are drawn on a fixed ``[vmin, vmax]`` from ``cases/_palettes.yaml``,
so *every* field whose values all sit below ``vmin`` renders identically —
whether it is an unwritten zarr chunk (fill value 0.0), a field left in
normalized units, a diverged diffusion sample, or a genuine but off-scale
atmosphere.  Eyeballing MP4s cannot separate those; the numbers can, in
seconds.

Statistics are taken over the pixels inside the case's ``plots.view`` box —
the domain the videos actually draw — because that is the domain the colour
range was chosen for.  Screening the global field against a range tuned to a
regional box flags every member of any case whose box is a quiet corner of a
noisy planet: `equator_indian_ocean` sets msl to [100000, 101400] Pa for a
band of shallow equatorial lows, and the same 12 frames contain extratropical
lows near 94500 Pa and highs near 104600 Pa elsewhere on the globe, so a
global screen returns MOSTLY_OFF_SCALE for all 100 members and means nothing
by it.  ``--global-field`` opts out, for hunting pipeline damage anywhere.

This script reads the raw HEALPix values straight out of the ensemble zarr (no
regrid, no matplotlib) and reports, per member and per frame:

  * min / max / mean / std, and the NaN / non-finite / exact-zero fractions
  * the fraction of values outside the plot's colour range, i.e. how much of
    the panel is saturated at one end
  * a verdict per member (see ``VERDICTS`` below)
  * duplicate detection — members whose field bytes are identical.  Chunked
    jobs derive their per-micro-batch seed from ``--member-offset``, so a
    misaligned offset silently yields correlated members and would corrupt
    any "n of N members behave this way" sentence.

Because the pinned frames are conditioning data rather than model output,
per-frame stats separate the two failure geographies that matter: a run whose
frames 0 and 11 are clean while the interior is degenerate is the model
declining the pair, whereas a member that is degenerate at every frame
(pinned frames included) is a pipeline artifact and says nothing about physics.

Usage
-----
Screen every member of a case (all modes in the case YAML)::

    python member_field_qc.py --event teleport_atlantic

Restrict members / variables / modes::

    python member_field_qc.py --event teleport_gulf --members 0-19
    python member_field_qc.py --event equator_indian_ocean --variables msl
    python member_field_qc.py --event teleport_gulf --mode both

Outputs (under ``--output-dir``, default ``output/<event>_qc``)::

    member_field_qc_<mode>_members.csv   one row per (variable, member)
    member_field_qc_<mode>_frames.csv    one row per (variable, member, frame)

and a summary table on stdout naming every member that is not OK.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import zarr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "_shared"))
sys.path.insert(0, SHARED_DIR)

# _dispatch_lib deliberately carries no torch / matplotlib import chain, and
# this screen needs neither — it reads raw zarr values and prints numbers, so
# it stays runnable on a head node (or anywhere without the GPU environment).
from _dispatch_lib import (  # noqa: E402
    case_ensemble_size,
    ensemble_zarr_path,
    iter_modes,
    list_cases,
    load_case,
    load_palettes,
    parse_member_spec,
    resolve_view,
    zarr_variable_index,
)


def _hpx_level_from_root(root: Any) -> int:
    """HEALPix level of an ensemble store, from its attr or its pixel count."""
    level = root.attrs.get("hpx_level")
    if level is not None:
        return int(level)
    return int(np.log2(root["hpx"].shape[0] // 12) / 2)


def view_box_pixels(root: Any, box: Sequence[float]) -> np.ndarray:
    """NEST pixel indices inside ``box`` = ``(lon_min, lon_max, lat_min, lat_max)``.

    Imported lazily: ``custom_utils.diagnostics`` pulls in torch and
    earth2grid, which the global-field screen has no need for.  Delegating
    here rather than re-deriving the mask is deliberate — that module is the
    repo's single source of truth for which pixels a box selects
    (``scripts/_shared/README.md``).
    """
    try:
        from custom_utils.diagnostics import domain_hpx_indices
    except ImportError as exc:
        raise SystemExit(
            f"view-box screening needs custom_utils.diagnostics (torch + "
            f"earth2grid), which failed to import: {exc}\n"
            f"Activate the e2s-custom environment, or pass --global-field to "
            f"screen the whole globe without them (off-scale verdicts then "
            f"compare against a range not chosen for the global field)."
        ) from exc

    lon_min, lon_max, lat_min, lat_max = box
    return domain_hpx_indices(
        _hpx_level_from_root(root), lon_min, lon_max, lat_min, lat_max
    )


# Stored components behind each derived plot variable.  The screen runs on
# stored fields only: a member that is broken is broken in its components, and
# resolving a derived field would mean pulling in the regrid path this script
# exists to avoid.
DERIVED_COMPONENTS: Dict[str, List[str]] = {
    "wspd10m": ["u10m", "v10m"],
    "wspd850": ["u850", "v850"],
    "vort850": ["u850", "v850"],
}

# Verdicts, most severe first.  The first match wins.
VERDICTS = (
    "EMPTY",  # every value exactly 0.0 -> chunk never written (zarr fill)
    "NAN",  # any NaN present
    "NONFINITE",  # any +/-inf present
    "CONSTANT",  # finite, non-zero, but zero variance
    "FLAT_OFF_SCALE",  # entirely outside [vmin, vmax] -> renders one colour
    "MOSTLY_OFF_SCALE",  # >50% of values saturated at one end
    "OK",
)

OFF_SCALE_WARN_FRAC = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Numeric health screen for ensemble members "
        "(run before rendering member videos)."
    )
    parser.add_argument(
        "--event",
        type=str,
        default=None,
        help="Case name (e.g. 'teleport_atlantic'). Reads cases/<event>.yaml.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available cases and exit."
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        choices=["start", "end", "both"],
        help="Restrict to one or more conditioning modes (repeatable).",
    )
    parser.add_argument(
        "--members",
        type=str,
        default="all",
        help="Members to screen: 'all' (default), or a spec like "
        "'0-99', '5,6,7', '0-3,10,20-25'.",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default=None,
        help="Comma-separated stored variables to screen. Default: the "
        "stored variables behind the case's plots.variables.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=None,
        help="Override ensemble.size when resolving the zarr path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the CSVs (default: output/<event>_qc).",
    )
    parser.add_argument(
        "--global-field",
        action="store_true",
        help="Screen the whole global field instead of the plots.view box.  "
        "The colour ranges in the case YAML are chosen for the view box, so "
        "the off-scale verdicts are only meaningful inside it; use this to "
        "hunt for pipeline damage anywhere on the globe.",
    )
    parser.add_argument(
        "--no-csv", action="store_true", help="Print the summary but write no CSVs."
    )
    args = parser.parse_args()

    if not args.list and args.event is None:
        parser.error("--event is required (or pass --list)")
    return args


def resolve_screen_variables(
    case: Dict[str, Any], override: Optional[str] = None
) -> List[str]:
    """Stored variables to screen for a case.

    Derived plot variables are expanded to the stored components they are
    built from, so ``plots.variables: [msl, wspd850, vort850]`` screens
    ``msl``, ``u850``, ``v850``.
    """
    if override:
        return [v.strip() for v in override.split(",") if v.strip()]

    plots_cfg = case.get("plots") or {}
    entries = plots_cfg.get("variables") or []
    names = [e["name"] for e in entries if e.get("name")]
    if not names:
        return ["msl"]

    stored: List[str] = []
    for name in names:
        for comp in DERIVED_COMPONENTS.get(name, [name]):
            if comp not in stored:
                stored.append(comp)
    return stored


def plot_range(
    case: Dict[str, Any], palettes: Dict[str, Any], var: str
) -> Tuple[Optional[float], Optional[float]]:
    """The colour range a member video would draw ``var`` on.

    Case-level ``plots.variables`` entries override the palette defaults, the
    same precedence ``plot_ensemble_member.py`` applies.  Returns
    ``(None, None)`` for a variable with no configured range.
    """
    defaults = (palettes.get("variable_defaults") or {}).get(var, {})
    vmin = defaults.get("vmin")
    vmax = defaults.get("vmax")

    for entry in (case.get("plots") or {}).get("variables") or []:
        if entry.get("name") == var:
            vmin = entry.get("vmin", vmin)
            vmax = entry.get("vmax", vmax)
            break

    return (
        None if vmin is None else float(vmin),
        None if vmax is None else float(vmax),
    )


def field_stats(
    field: np.ndarray, vmin: Optional[float], vmax: Optional[float]
) -> Dict[str, float]:
    """Summary statistics for one field slab.

    ``field`` is any shape; statistics are taken over every element.  The
    off-scale fractions are computed over finite values only, so a field that
    is half NaN does not read as half in-range.
    """
    flat = field.reshape(-1)
    n = flat.size
    finite = np.isfinite(flat)
    n_finite = int(finite.sum())

    stats: Dict[str, float] = {
        "n": float(n),
        "nan_frac": float(np.isnan(flat).sum()) / n,
        "nonfinite_frac": float(n - n_finite) / n,
        "zero_frac": float((flat == 0.0).sum()) / n,
    }

    if n_finite == 0:
        stats.update(
            {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "below_vmin_frac": float("nan"),
                "above_vmax_frac": float("nan"),
            }
        )
        return stats

    vals = flat[finite]
    stats.update(
        {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "below_vmin_frac": (
                float((vals < vmin).sum()) / n_finite if vmin is not None else 0.0
            ),
            "above_vmax_frac": (
                float((vals > vmax).sum()) / n_finite if vmax is not None else 0.0
            ),
        }
    )
    return stats


def verdict(
    stats: Dict[str, float], vmin: Optional[float], vmax: Optional[float]
) -> str:
    """Classify a member's field.  See ``VERDICTS`` for the ordering."""
    if stats["zero_frac"] == 1.0:
        return "EMPTY"
    if stats["nan_frac"] > 0.0:
        return "NAN"
    if stats["nonfinite_frac"] > 0.0:
        return "NONFINITE"
    if stats["std"] == 0.0:
        return "CONSTANT"

    off_scale = stats["below_vmin_frac"] + stats["above_vmax_frac"]
    if vmin is not None and vmax is not None:
        if stats["max"] < vmin or stats["min"] > vmax:
            return "FLAT_OFF_SCALE"
        if off_scale > OFF_SCALE_WARN_FRAC:
            return "MOSTLY_OFF_SCALE"
    return "OK"


def screen_variable(
    root: Any,
    var: str,
    members: Sequence[int],
    vmin: Optional[float],
    vmax: Optional[float],
    pix: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Screen one variable across ``members``.

    Returns ``(member_rows, frame_rows)``.  Reads the raw HEALPix slab
    ``data[member, 0, :, var_idx, :]`` — no regrid, so the whole screen costs
    a few MB per member.  ``pix`` restricts the statistics to a subset of
    HEALPix pixels (the view box); ``None`` screens the global field.
    """
    var_idx = zarr_variable_index(root, var)
    n_leads = root["data"].shape[2]

    member_rows: List[Dict[str, Any]] = []
    frame_rows: List[Dict[str, Any]] = []
    digests: Dict[str, List[int]] = {}

    for m in members:
        slab = np.asarray(root["data"][m, 0, :, var_idx, :], dtype=np.float64)
        if pix is not None:
            slab = slab[:, pix]

        stats = field_stats(slab, vmin, vmax)
        row: Dict[str, Any] = {"variable": var, "member": m}
        row.update(stats)
        row["verdict"] = verdict(stats, vmin, vmax)
        member_rows.append(row)

        # Exact byte-level identity: catches seed reuse across chunk jobs,
        # which no statistical summary would reliably separate from two
        # members that merely landed close together.
        digest = hashlib.blake2b(
            np.ascontiguousarray(slab, dtype=np.float32).tobytes(), digest_size=16
        ).hexdigest()
        digests.setdefault(digest, []).append(m)

        for f in range(n_leads):
            f_stats = field_stats(slab[f], vmin, vmax)
            f_row: Dict[str, Any] = {"variable": var, "member": m, "frame": f}
            f_row.update(f_stats)
            f_row["verdict"] = verdict(f_stats, vmin, vmax)
            frame_rows.append(f_row)

    for row in member_rows:
        for group in digests.values():
            if len(group) > 1 and row["member"] in group:
                row["duplicate_of"] = ",".join(
                    str(g) for g in group if g != row["member"]
                )
                break
        else:
            row["duplicate_of"] = ""

    return member_rows, frame_rows


def write_csv(path: str, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  wrote {path}")


def print_summary(
    mode: str,
    member_rows: List[Dict[str, Any]],
    frame_rows: List[Dict[str, Any]],
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
) -> None:
    """Print the per-variable verdict tally and every member that is not OK."""
    variables = sorted({row["variable"] for row in member_rows})

    for var in variables:
        rows = [r for r in member_rows if r["variable"] == var]
        vmin, vmax = ranges[var]
        range_txt = (
            f"[{vmin:g}, {vmax:g}]"
            if vmin is not None and vmax is not None
            else "unset"
        )
        tally = {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICTS}
        tally_txt = ", ".join(f"{k}={v}" for k, v in tally.items() if v)

        print(f"\n[{mode}] {var} — plot range {range_txt}, {len(rows)} members")
        print(f"  verdicts: {tally_txt}")

        # The in-box extremes across the whole ensemble are exactly the numbers
        # a shared colour scale wants, and they are already computed here.  A
        # range much wider than this spends most of the colormap on values the
        # panels never contain, which is what makes a 12-panel grid unreadable.
        # Healthy members only: one member left in normalized units drags the
        # suggested minimum to ~-4 and would hand back a range that hides
        # every real field in the top pixel of the colorbar.
        healthy = [
            r
            for r in rows
            if r["verdict"] in ("OK", "MOSTLY_OFF_SCALE")
            and np.isfinite(r["min"])
            and np.isfinite(r["max"])
        ]
        if healthy:
            obs_min = min(r["min"] for r in healthy)
            obs_max = max(r["max"] for r in healthy)
            note = "" if len(healthy) == len(rows) else f" ({len(healthy)} healthy)"
            print(
                f"  in-box range across members{note}: "
                f"[{obs_min:.6g}, {obs_max:.6g}]   <- candidate vmin/vmax"
            )

        bad = [r for r in rows if r["verdict"] != "OK"]
        if bad:
            print(
                f"  {'member':>6}  {'verdict':<18} {'min':>14} {'max':>14} "
                f"{'mean':>14} {'std':>12}"
            )
            for r in bad:
                print(
                    f"  {r['member']:>6}  {r['verdict']:<18} {r['min']:>14.4g} "
                    f"{r['max']:>14.4g} {r['mean']:>14.4g} {r['std']:>12.4g}"
                )

            # Which frames are affected decides the reading: pinned frames
            # clean + interior broken is a model result; everything broken is
            # a pipeline artifact.
            bad_members = {r["member"] for r in bad}
            for m in sorted(bad_members):
                frames = [
                    fr["frame"]
                    for fr in frame_rows
                    if fr["variable"] == var
                    and fr["member"] == m
                    and fr["verdict"] != "OK"
                ]
                n_leads = max(
                    (fr["frame"] for fr in frame_rows if fr["variable"] == var),
                    default=-1,
                )
                span = "all frames" if len(frames) == n_leads + 1 else str(frames)
                print(f"    member {m}: affected frames {span}")

        dups = [r for r in rows if r.get("duplicate_of")]
        if dups:
            groups = sorted(
                {
                    ",".join(
                        sorted(
                            [str(r["member"])] + r["duplicate_of"].split(","), key=int
                        )
                    )
                    for r in dups
                }
            )
            print(f"  DUPLICATE members (identical fields): {groups}")


def screen_case(
    event: str,
    *,
    only_modes: Optional[Sequence[str]] = None,
    member_spec: str = "all",
    variables_override: Optional[str] = None,
    ensemble_size_override: Optional[int] = None,
    output_dir: Optional[str] = None,
    write_csvs: bool = True,
    global_field: bool = False,
) -> int:
    """Screen every requested (mode, variable, member).  Returns a bad count."""
    case = load_case(event)
    palettes = load_palettes()
    size = case_ensemble_size(case, override=ensemble_size_override)
    variables = resolve_screen_variables(case, variables_override)
    out_dir = output_dir or f"output/{case['name']}_qc"

    # The case YAML's colour ranges are chosen for the view box, so screening
    # the global field against them flags every member of a case whose box is
    # a quiet corner of a noisy planet.  Default to the box the videos
    # actually show; --global-field opts out for pipeline hunting.
    box = None if global_field else resolve_view(case)
    if not global_field and box is None:
        print(
            "  NOTE: no plots.view and no regions.impact — screening the "
            "global field.  Off-scale verdicts compare against a range that "
            "was not chosen for it."
        )

    print(f"Screening case '{case['name']}' ({case['display_name']})")
    print(f"  variables: {', '.join(variables)}")
    if box is not None:
        print(
            f"  domain:    view box lon [{box[0]:g}, {box[1]:g}], "
            f"lat [{box[2]:g}, {box[3]:g}]"
        )
    else:
        print("  domain:    global field")

    total_bad = 0
    for mode, _frames in iter_modes(case):
        if only_modes and mode not in only_modes:
            continue

        zarr_path = ensemble_zarr_path(case["base"], mode, size)
        if not os.path.exists(zarr_path):
            print(f"\n[{mode}] SKIP — no ensemble zarr at {zarr_path}")
            continue

        root = zarr.open_group(zarr_path, mode="r")
        n_members = root["data"].shape[0]
        members = parse_member_spec(member_spec, n_members)
        print(f"\n[{mode}] {zarr_path}")
        print(f"  {n_members} members on disk, screening {len(members)}")

        # ensemble_interpolation.py sets attrs["complete"] only after its last
        # micro-batch lands, so a job killed at the wall clock leaves the flag
        # off and the unwritten tail at zarr's fill value.  merge_ensemble_zarrs
        # enforces this, but only for chunked submissions -- an unpartitioned
        # run (any --quick smoke test) has no consumer that checks it, which is
        # how a truncated store stays silently renderable.
        if not root.attrs.get("complete", False):
            print(
                "  WARNING: store is NOT marked complete — the writing job did "
                "not finish.\n"
                "           Trailing members are unwritten and read back as "
                "fill value 0.0."
            )

        pix = None
        if box is not None:
            pix = view_box_pixels(root, box)
            if pix.size == 0:
                raise SystemExit(
                    f"case '{case['name']}': the view box {tuple(box)} selects "
                    f"no HEALPix pixels at level {_hpx_level_from_root(root)}.  "
                    f"Check the [-180, 180] longitude convention."
                )
            print(f"  {pix.size} of {root['data'].shape[4]} pixels in the view box")

        all_member_rows: List[Dict[str, Any]] = []
        all_frame_rows: List[Dict[str, Any]] = []
        ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

        for var in variables:
            vmin, vmax = plot_range(case, palettes, var)
            ranges[var] = (vmin, vmax)
            m_rows, f_rows = screen_variable(root, var, members, vmin, vmax, pix)
            all_member_rows.extend(m_rows)
            all_frame_rows.extend(f_rows)

        print_summary(mode, all_member_rows, all_frame_rows, ranges)
        total_bad += sum(1 for r in all_member_rows if r["verdict"] != "OK")

        if write_csvs:
            os.makedirs(out_dir, exist_ok=True)
            member_fields = [
                "variable",
                "member",
                "verdict",
                "duplicate_of",
                "min",
                "max",
                "mean",
                "std",
                "nan_frac",
                "nonfinite_frac",
                "zero_frac",
                "below_vmin_frac",
                "above_vmax_frac",
                "n",
            ]
            frame_fields = ["variable", "member", "frame", "verdict"] + [
                f
                for f in member_fields
                if f not in ("variable", "member", "verdict", "duplicate_of")
            ]
            write_csv(
                os.path.join(out_dir, f"member_field_qc_{mode}_members.csv"),
                all_member_rows,
                member_fields,
            )
            write_csv(
                os.path.join(out_dir, f"member_field_qc_{mode}_frames.csv"),
                all_frame_rows,
                frame_fields,
            )

    return total_bad


def main() -> None:
    args = parse_args()

    if args.list:
        print("Available cases:")
        for n in list_cases():
            print(f"  {n}")
        return

    n_bad = screen_case(
        args.event,
        only_modes=args.mode,
        member_spec=args.members,
        variables_override=args.variables,
        ensemble_size_override=args.ensemble_size,
        output_dir=args.output_dir,
        write_csvs=not args.no_csv,
        global_field=args.global_field,
    )
    print(f"\nDone — {n_bad} (variable, member) pair(s) flagged.")


if __name__ == "__main__":
    main()
