"""Aggregate ageostrophic-fraction plots across conditioning modes.

Reads the per-mode ``ageostrophic_fraction_<mode>.npz`` files and produces
a single 4-row × N-column figure (where N is the number of available
modes).  Each row is one diagnostic (ratio, |V|, |V_g|, |V_a|); each
column is one conditioning mode, ordered end → start → both so the
end-conditioned column (the paper's headline mode) sits leftmost.
Every cell shows the full per-member spaghetti, the ensemble mean, the
ERA5 reference, the ERA5 HPX round-trip reference, and the mode's
conditioning-frame markers — giving a direct side-by-side view of how
the ensemble spread (and where it pinches) varies with conditioning.

Outputs (in --output-dir):
    ageostrophic_fraction.png   - 4×N grid; one column per conditioning mode

Usage:
    python aggregate_ageostrophic.py \\
        --output-dir /path/to/diagnostics/ageostrophic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent.parent / "_shared"
sys.path.insert(0, str(SHARED_DIR))

from _dispatch_lib import MODE_LABELS_SHORT  # noqa: E402
from axis_limits import apply_limits  # noqa: E402
from local_time_axis import format_local_time_axis  # noqa: E402
from plot_style import (  # noqa: E402
    AXIS_LABEL_SIZE_DENSE,
    LEGEND_SIZE_DENSE,
    SUPTITLE_SIZE_DENSE,
    TICK_LABEL_SIZE_DENSE,
    TITLE_SIZE_DENSE,
    add_subplot_labels,
)

# Fixed per-element color scheme.  Mode identity is carried by the column
# layout, so the same colors are reused in every column — keeping the
# visual mapping (blue = ensemble, black = ERA5, green = ERA5 round-trip,
# gray = conditioning bookkeeping) stable across the whole figure.
ENS_COLOR = "tab:blue"
ERA_COLOR = "black"
ERA_RT_COLOR = "tab:green"
COND_COLOR = "gray"


def _scalar(arr) -> object:
    return arr.item() if isinstance(arr, np.ndarray) and arr.shape == () else arr


def _load_mode(output_dir: Path, mode: str) -> dict | None:
    path = output_dir / f"ageostrophic_fraction_{mode}.npz"
    if not path.exists():
        print(f"WARNING: missing {path}, skipping mode '{mode}'")
        return None
    return dict(np.load(path, allow_pickle=True))


def _plot_cell(
    ax,
    diag: dict,
    ens_key: str,
    era_key: str,
    era_rt_key: str,
    lead_h: np.ndarray,
    *,
    ylabel: str | None,
    show_legend: bool,
) -> None:
    """Render one (mode, panel) cell.

    Order of plotting is chosen so the labelled lines sit on top of the
    spaghetti (which is drawn first at low alpha).  Conditioning-frame
    markers are added last so they remain visible across every layer.
    """
    ens = np.asarray(diag[ens_key])  # (n_members, n_leads)
    n_members = int(ens.shape[0])
    alpha = max(0.02, min(0.3, 30.0 / max(n_members, 1)))
    for i in range(n_members):
        ax.plot(lead_h, ens[i], color=ENS_COLOR, alpha=alpha, linewidth=0.5)

    ax.plot(
        lead_h,
        ens.mean(axis=0),
        color=ENS_COLOR,
        linewidth=2.0,
        label=f"Ens mean (N={n_members})",
    )
    ax.plot(
        lead_h,
        np.asarray(diag[era_key]),
        color=ERA_COLOR,
        linewidth=1.6,
        linestyle="--",
        label="ERA5",
    )
    ax.plot(
        lead_h,
        np.asarray(diag[era_rt_key]),
        color=ERA_RT_COLOR,
        linewidth=1.6,
        linestyle=":",
        label="ERA5 (HPX round-trip)",
    )

    cond = np.asarray(diag.get("conditioning_frames", np.array([], dtype=int)))
    for frame in cond.tolist():
        if 0 <= frame < len(lead_h):
            ax.axvline(
                lead_h[frame],
                color=COND_COLOR,
                linestyle="--",
                linewidth=1.4,
                alpha=0.75,
                label=f"f={frame} (conditioned)",
            )

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE_DENSE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE_DENSE)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=LEGEND_SIZE_DENSE, loc="best")


def plot_fraction_timeseries(per_mode: dict[str, dict], out_path: Path) -> None:
    modes = list(per_mode)
    any_mode = next(iter(per_mode.values()))
    lead_h = np.asarray(any_mode["lead_hours"])
    case_name = str(_scalar(any_mode["case_name"]))
    band_label = str(_scalar(any_mode["band_label"]))
    level = int(_scalar(any_mode["level"]))
    start_time_iso = str(_scalar(any_mode["start_time"]))
    timezone_name = str(_scalar(any_mode["timezone"]))

    panels = [
        ("ratio_ens", "ratio_era", "ratio_era_rt", r"$|V_a| / |V|$"),
        ("mean_v_ens", "mean_v_era", "mean_v_era_rt", r"mean $|V|$ (m/s)"),
        ("mean_vg_ens", "mean_vg_era", "mean_vg_era_rt", r"mean $|V_g|$ (m/s)"),
        ("mean_va_ens", "mean_va_era", "mean_va_era_rt", r"mean $|V_a|$ (m/s)"),
    ]
    n_rows = len(panels)
    n_cols = len(modes)
    # ~5" per column keeps the local-time x-tick labels legible; 3" per row
    # matches the original 4-row layout's vertical rhythm.
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 3.2 * n_rows),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for j, mode in enumerate(modes):
        diag = per_mode[mode]
        axes[0, j].set_title(
            MODE_LABELS_SHORT.get(mode, mode), fontsize=TITLE_SIZE_DENSE
        )
        for i, (ens_key, era_key, era_rt_key, ylabel) in enumerate(panels):
            _plot_cell(
                axes[i, j],
                diag,
                ens_key,
                era_key,
                era_rt_key,
                lead_h,
                ylabel=ylabel if j == 0 else None,
                show_legend=(i == 0 and j == 0),
            )

    # Inside placement avoids overlapping the column-header titles on
    # row 0 and the suptitle banner; outside placement on a 4xN sharex
    # grid would also collide with the row above's tick labels.
    add_subplot_labels(axes, placement="inside")

    fig.suptitle(
        f"{case_name} - ageostrophic-fraction diagnostics at {level} hPa\n"
        f"band: {band_label}",
        fontsize=SUPTITLE_SIZE_DENSE,
    )

    # Floor the ratio panel at 0.1, below every observed value, so the
    # conditioning-frame pinch and the ensemble-vs-ERA5 gap stay easy to read.
    # (This comment previously said "anchor at 0", which the line below has
    # never done; the code is the intended behavior and the comment was wrong.)
    # A fallback only -- axis_limits.yaml pins this row and is applied after,
    # so it wins whenever an entry exists.
    axes[0, 0].set_ylim(bottom=0.1)

    # Cross-case comparable magnitude rows (scripts/_shared/README.md, "Adding
    # a new aggregator?" item 5).  S13-S15 are one float per case, pages apart,
    # and every case reduces over the SAME global band -- so a reader is meant
    # to compare row 2 of one against row 2 of another.  Autoscaled that
    # comparison is wrong by construction: PNW's |V| envelope sits ~1.5 m/s
    # below Sandy's, which unpinned looks like a difference in the figure
    # rather than in the data.
    #
    # sharey="row" means constraining column 0 constrains the whole row.
    # Applied AFTER the floor above so a future `ratio` entry in the YAML wins
    # over it; today there is none, so the ratio row keeps that floor.  Missing
    # keys autoscale, and a limit in force warns when data exceeds it.
    for i, (ens_key, era_key, era_rt_key, _ylabel) in enumerate(panels):
        row_data = np.concatenate(
            [
                np.asarray(per_mode[m][k], dtype=float).ravel()
                for m in modes
                for k in (ens_key, era_key, era_rt_key)
            ]
        )
        apply_limits(
            axes[i, 0],
            "ageostrophic",
            f"uvz{level}",
            ens_key.removesuffix("_ens"),
            y_data=row_data,
            log_tag="aggregate_ageostrophic",
        )

    for j in range(n_cols):
        format_local_time_axis(axes[-1, j], lead_h, start_time_iso, timezone_name)
        axes[-1, j].xaxis.label.set_fontsize(AXIS_LABEL_SIZE_DENSE)
        axes[-1, j].tick_params(axis="x", labelsize=TICK_LABEL_SIZE_DENSE)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"ERROR: --output-dir does not exist: {output_dir}")

    per_mode: dict[str, dict] = {}
    # Columns are ordered end -> start -> both so the end-conditioned run
    # (the headline mode for the paper) sits in the leftmost, eye-first
    # column.  start and both follow in the canonical order.
    for mode in ("end", "start", "both"):
        d = _load_mode(output_dir, mode)
        if d is not None:
            per_mode[mode] = d

    if not per_mode:
        raise SystemExit(f"ERROR: no per-mode npz files found in {output_dir}")

    print(f"[aggregate_ageostrophic] modes: {list(per_mode)}")
    out_path = output_dir / "ageostrophic_fraction.png"
    plot_fraction_timeseries(per_mode, out_path)
    print(f"[aggregate_ageostrophic] wrote {out_path}")


if __name__ == "__main__":
    main()
