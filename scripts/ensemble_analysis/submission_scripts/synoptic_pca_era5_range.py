"""Is ERA5's free-end PC state within the members' range?  One table, all cases.

For each case and free-end conditioning mode, reads the
``synoptic_pca_<mode>.npz`` written by ``compute_synoptic_pca.py`` and runs
``pca_stats.ellipsoid_test`` on ERA5's leading ``--n-pc`` PC scores against
all members' (``pc_scores``; every member, since this is a statistic of the
PCA field, not of a storm track).  Columns:

    N        members
    m        members at least as far out as ERA5
    d2_era5  ERA5's squared Mahalanobis distance from the members' mean
    d2_max   the farthest member's
    p        (m+1)/(N+1): if ERA5 were one more draw like the members, the
             chance it would rank at least this far out
    ERA5     inside (m >= 1) or outside

This is the source the paper cites.  A start-conditioned row
is a coverage fact, not a test of the model: those cases were chosen for
their outcomes, so ERA5 can fall outside even under a perfect model.

Usage:
    python synoptic_pca_era5_range.py sandy ian pnw_heatwave
    python synoptic_pca_era5_range.py sandy --modes end
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent.parent / "_shared"
sys.path.insert(0, str(SHARED_DIR))

from _dispatch_lib import load_case  # noqa: E402
from pca_stats import ellipsoid_test  # noqa: E402

# Modes with a free end (``both`` pins both ends).
FREE_END_MODES = ("end", "start")

# Matches the paper's multi-PC regressions.
DEFAULT_N_PC = 8


def range_row(npz_path: Path, n_pc: int) -> Dict[str, object]:
    """One table row: ERA5's ellipsoid test against all members in ``npz_path``."""
    with np.load(npz_path, allow_pickle=False) as data:
        scores = np.asarray(data["pc_scores"], dtype=np.float64)
        era5 = np.asarray(data["era5_pc_score"], dtype=np.float64)
        variables = ",".join(str(v) for v in np.asarray(data["variables"]))
    if era5.size < n_pc or scores.shape[1] < n_pc:
        raise SystemExit(
            f"ERROR: {npz_path} has {era5.size} ERA5 PC scores and "
            f"{scores.shape[1]} member PC scores; this test needs {n_pc}. "
            f"Rerun compute_synoptic_pca.py (it now saves all --max-d PCs)."
        )
    result = ellipsoid_test(scores[:, :n_pc], era5[:n_pc])
    return {
        "vars": variables,
        "N": result.n,
        "m": result.m,
        "d2_era5": result.d2_point,
        "d2_max": result.d2_max,
        "p": result.p_value,
        "ERA5": "inside" if result.inside else "outside",
    }


def range_table(cases: Sequence[str], modes: Sequence[str], n_pc: int) -> pd.DataFrame:
    """The table over ``cases`` x ``modes``; a missing npz is an error."""
    rows: List[Dict[str, object]] = []
    for case_name in cases:
        npz_dir = Path(load_case(case_name)["base"]) / "diagnostics" / "synoptic_pca"
        for mode in modes:
            npz_path = npz_dir / f"synoptic_pca_{mode}.npz"
            if not npz_path.exists():
                raise SystemExit(
                    f"ERROR: {npz_path} missing. Run dispatch_synoptic_pca.py "
                    f"for {case_name}, or leave out modes it does not run (--modes)."
                )
            rows.append({"case": case_name, "mode": mode, **range_row(npz_path, n_pc)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ERA5's free-end PC state vs the members' range, per case and mode.",
    )
    parser.add_argument(
        "cases", nargs="+", help="Case names (e.g. sandy ian pnw_heatwave)"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(FREE_END_MODES),
        choices=FREE_END_MODES,
        help="Free-end conditioning modes (default: end start).",
    )
    parser.add_argument(
        "--n-pc",
        type=int,
        default=DEFAULT_N_PC,
        help=f"Leading PCs (default {DEFAULT_N_PC}, matching the regressions).",
    )
    args = parser.parse_args()
    if args.n_pc < 1:
        parser.error("--n-pc must be at least 1")

    table = range_table(args.cases, args.modes, args.n_pc)
    print(
        f"ERA5 free-end PC state vs all members ({args.n_pc} PCs, Mahalanobis; "
        f"m = members at least as far out, p = (m+1)/(N+1))"
    )
    print(
        table.to_string(
            index=False,
            formatters={
                "d2_era5": "{:.1f}".format,
                "d2_max": "{:.1f}".format,
                "p": "{:.3g}".format,
            },
        )
    )


if __name__ == "__main__":
    main()
