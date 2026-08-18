"""Quantify free-end precursor spread, start- vs end-conditioning.

A companion to ``plot_diversity_metrics`` in ``aggregate_synoptic_pca.py``
that turns panel (a) of ``synoptic_pca_diversity.png`` into numbers a
paper can quote.  The motivating question is the start-vs-end
organization asymmetry: if pinning the end state constrains the
antecedent large-scale flow (the persistence/range-restriction
mechanism), the free-end precursor ensemble variance under
end-conditioning should be substantially smaller than under
start-conditioning, and the low end-mode PC -> impact R^2 partly
reflects restricted predictor variance rather than weak coupling.  If
instead the two variances are comparable, the weak end-mode organization
is a statement about the coupling itself (or about model
under-dispersion of conditioned large scales).

Reads, per conditioning mode, the ``synoptic_pca_<mode>.npz`` written by
``compute_synoptic_pca.py`` and reports:

1.  **Physical free-end spread per variable** -- the per-pixel-mean
    ensemble variance of the raw free-end field over the PCA domain
    (``physical_variance_mean_per_var``: computed upstream before the
    z-score, unbiased over members, equal-area HPX pixels), converted to
    display units (geopotential -> height), plus its square root, and
    the end/start ratios of both.  The std ratio is the headline
    range-restriction number.

2.  **Spectrum shape per mode** -- leading-EOF EVR, cumulative EVR over
    the leading ``--n-pc`` modes, participation ratio, spectral
    perplexity, and the count of North (1982)-separable leading modes.
    Distinguishes "less total variance" from "same variance concentrated
    in fewer independent modes".

Modes are compared on the same PCA domain by construction (the
dispatcher passes one domain to both jobs); the domain box is echoed and
checked for agreement.

Outputs (in --output-dir):
    synoptic_pca_diversity_stats.csv   per-variable cross-mode table
    synoptic_pca_diversity_stats.txt   full human-readable report

Usage:
    python synoptic_pca_diversity_stats.py sandy
    python synoptic_pca_diversity_stats.py pnw_heatwave --n-pc 8
    python synoptic_pca_diversity_stats.py --list
    # --output-dir stays as an explicit override for a non-standard layout:
    python synoptic_pca_diversity_stats.py \\
        --output-dir /path/to/diagnostics/synoptic_pca
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent.parent / "_shared"
sys.path.insert(0, str(SHARED_DIR))

from _dispatch_lib import list_cases, load_case  # noqa: E402
from var_metadata import to_display_units  # noqa: E402
from var_metadata import units as var_units  # noqa: E402

# Modes whose free end is uniquely defined (mirrors dispatch_synoptic_pca).
STAT_MODES: Tuple[str, ...] = ("end", "start")

DEFAULT_N_PC = 8


# ---------------------------------------------------------------------------
# Spectrum-shape metrics (mirrors plot_diversity_metrics in
# aggregate_synoptic_pca.py so the txt report and the figure agree)
# ---------------------------------------------------------------------------


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """Effective # of modes: (Σλ)² / Σλ²  (1 if one mode dominates, d if flat)."""
    e = np.asarray(eigenvalues, dtype=np.float64)
    denom = float((e**2).sum())
    return float((e.sum() ** 2) / denom) if denom > 0 else 0.0


def spectral_perplexity(eigenvalues: np.ndarray) -> float:
    """Entropy-based effective # of modes: exp(-Σ pᵢ ln pᵢ), pᵢ = λᵢ/Σλ."""
    e = np.asarray(eigenvalues, dtype=np.float64)
    s = float(e.sum())
    if s <= 0:
        return 0.0
    p = e[e > 0] / s
    return float(np.exp(-(p * np.log(p)).sum()))


def n_separable_modes(eigenvalues: np.ndarray, n_members: int) -> int:
    """Leading modes (contiguous from the top) that clear North's separation:
    ``λᵢ − λ_{i+1} ≥ δλᵢ`` with ``δλᵢ ≈ λᵢ·√(2/N)``.
    """
    e = np.asarray(eigenvalues, dtype=np.float64)
    dl = e * np.sqrt(2.0 / float(n_members))
    k = 0
    for i in range(e.size - 1):
        if (e[i] - e[i + 1]) >= dl[i]:
            k = i + 1
        else:
            break
    return k


# ---------------------------------------------------------------------------
# Per-mode npz loading
# ---------------------------------------------------------------------------


def _scalar(arr: np.ndarray) -> object:
    """Unwrap a 0-d npz array to a python scalar/str."""
    return np.asarray(arr).item()


def load_mode(output_dir: Path, mode: str) -> Optional[Dict[str, object]]:
    """Load the diversity-relevant slice of one mode's npz (None if absent)."""
    npz_path = output_dir / f"synoptic_pca_{mode}.npz"
    if not npz_path.exists():
        print(f"[diversity-stats] {npz_path} missing; skipping mode '{mode}'.")
        return None
    with np.load(npz_path, allow_pickle=False) as data:
        return {
            "case_name": str(_scalar(data["case_name"])),
            "variables": [str(v) for v in np.asarray(data["variables"])],
            "n_members": int(_scalar(data["n_members"])),
            "free_end_frame": int(_scalar(data["free_end_frame"])),
            "domain": tuple(
                float(_scalar(data[k]))
                for k in ("lon_min", "lon_max", "lat_min", "lat_max")
            ),
            "phys_var_mean": np.asarray(
                data["physical_variance_mean_per_var"], dtype=np.float64
            ),
            "evr": np.asarray(data["explained_variance_ratio"], dtype=np.float64),
            "eigenvalues": np.asarray(data["eigenvalues"], dtype=np.float64),
        }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def variance_table(per_mode: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    """Per-variable free-end spread, one row per variable, modes as columns.

    Variance is converted to display units (geopotential m²/s² -> height
    m, ``power=2``); the std columns are its square root.  When both
    modes are present, ``end/start`` ratio columns quantify the
    range restriction of the end-conditioned precursor.
    """
    modes = [m for m in STAT_MODES if m in per_mode]
    variables = per_mode[modes[0]]["variables"]  # type: ignore[index]
    rows: List[Dict[str, object]] = []
    for vi, var in enumerate(variables):  # type: ignore[arg-type]
        row: Dict[str, object] = {"variable": var, "units": var_units(var) or "-"}
        for m in modes:
            raw = float(per_mode[m]["phys_var_mean"][vi])  # type: ignore[index]
            disp = float(to_display_units(var, np.float64(raw), power=2))
            row[f"var_pp_{m}"] = disp
            row[f"std_pp_{m}"] = float(np.sqrt(disp))
        if "end" in modes and "start" in modes:
            row["var_ratio_end_over_start"] = (
                row["var_pp_end"] / row["var_pp_start"]  # type: ignore[operator]
                if row["var_pp_start"]
                else float("nan")
            )
            row["std_ratio_end_over_start"] = float(
                np.sqrt(row["var_ratio_end_over_start"])  # type: ignore[arg-type]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def spectrum_table(per_mode: Dict[str, Dict[str, object]], n_pc: int) -> pd.DataFrame:
    """Per-mode spectrum-shape summary (one row per mode)."""
    rows: List[Dict[str, object]] = []
    for m in [m for m in STAT_MODES if m in per_mode]:
        evr = np.asarray(per_mode[m]["evr"], dtype=np.float64)
        eig = np.asarray(per_mode[m]["eigenvalues"], dtype=np.float64)
        n_mem = int(per_mode[m]["n_members"])  # type: ignore[arg-type]
        k = min(n_pc, evr.size)
        rows.append(
            {
                "mode": m,
                "n_members": n_mem,
                "evr1_pct": 100.0 * float(evr[0]) if evr.size else float("nan"),
                f"evr{k}_cum_pct": 100.0 * float(evr[:k].sum()),
                "participation_ratio": participation_ratio(eig),
                "spectral_perplexity": spectral_perplexity(eig),
                "n_north_separable": n_separable_modes(eig, n_mem),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    per_mode: Dict[str, Dict[str, object]],
    output_dir: Path,
    n_pc: int,
) -> None:
    """Write the txt + csv report and echo the txt to stdout."""
    modes = [m for m in STAT_MODES if m in per_mode]
    first = per_mode[modes[0]]
    var_df = variance_table(per_mode)
    spec_df = spectrum_table(per_mode, n_pc)

    domains = {m: per_mode[m]["domain"] for m in modes}
    domain_note = ""
    if len({d for d in domains.values()}) > 1:
        domain_note = (
            "WARNING: PCA domains differ across modes -- the cross-mode "
            f"variance comparison is not apples-to-apples: {domains}\n"
        )

    lon_min, lon_max, lat_min, lat_max = first["domain"]  # type: ignore[misc]
    lines: List[str] = []
    lines.append(f"Synoptic-PCA free-end diversity report -- {first['case_name']}")
    lines.append(
        f"modes={modes}, variables={first['variables']}, "
        f"domain: lon in [{lon_min:g}, {lon_max:g}], "
        f"lat in [{lat_min:g}, {lat_max:g}]"
    )
    lines.append(
        "free-end frame per mode: "
        + ", ".join(f"{m}={per_mode[m]['free_end_frame']}" for m in modes)
    )
    if domain_note:
        lines.append(domain_note.rstrip())
    lines.append("")
    lines.append(
        "Physical free-end spread (per-pixel-mean ensemble variance of the "
        "raw field\nover the PCA domain; display units; std = sqrt(var)):"
    )
    lines.append(var_df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if "std_ratio_end_over_start" in var_df.columns:
        lines.append("")
        lines.append(
            "std_ratio_end_over_start < 1 means the end-conditioned "
            "antecedent precursor\nenvelope is narrower than the "
            "start-conditioned forecast envelope at the same\nlead "
            "(range restriction of the PC predictors)."
        )
    lines.append("")
    lines.append(f"Spectrum shape (leading {n_pc} PCs; full-spectrum metrics):")
    lines.append(spec_df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    lines.append("")
    text = "\n".join(lines) + "\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "synoptic_pca_diversity_stats.txt"
    txt_path.write_text(text)
    csv_path = output_dir / "synoptic_pca_diversity_stats.csv"
    var_df.to_csv(csv_path, index=False)
    print(text)
    print(f"[diversity-stats] wrote {txt_path}")
    print(f"[diversity-stats] wrote {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Free-end precursor spread, start- vs end-conditioning.",
    )
    parser.add_argument("case", nargs="?", help="Case name (e.g. sandy, pnw_heatwave)")
    parser.add_argument(
        "--list", action="store_true", help="List available cases and exit."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory holding synoptic_pca_<mode>.npz (outputs land here). "
        "Default: <case base>/diagnostics/synoptic_pca.",
    )
    parser.add_argument(
        "--n-pc",
        type=int,
        default=DEFAULT_N_PC,
        help=f"Leading PCs for the cumulative-EVR column (default "
        f"{DEFAULT_N_PC}, matching the paper's regressions).",
    )
    args = parser.parse_args()

    if args.list:
        print("Available cases:")
        for n in list_cases():
            print(f"  {n}")
        return

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        if not args.case:
            parser.error("case name required (or pass --output-dir / --list)")
        case = load_case(args.case)
        output_dir = Path(case["base"]) / "diagnostics" / "synoptic_pca"

    per_mode: Dict[str, Dict[str, object]] = {}
    for mode in STAT_MODES:
        diag = load_mode(output_dir, mode)
        if diag is not None:
            per_mode[mode] = diag

    if not per_mode:
        raise SystemExit(
            f"ERROR: no synoptic_pca_<mode>.npz found in {output_dir}; "
            f"run the synoptic-PCA diagnostic first "
            f"(dispatch_synoptic_pca.py)."
        )
    if len(per_mode) < 2:
        print(
            "[diversity-stats] only one mode present -- per-mode numbers "
            "reported, cross-mode ratios skipped."
        )

    write_report(per_mode, output_dir, n_pc=args.n_pc)


if __name__ == "__main__":
    main()
