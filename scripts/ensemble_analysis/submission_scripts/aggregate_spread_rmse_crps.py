"""Aggregate spread/RMSE/CRPS plots across conditioning modes.

Reads the three per-mode ``spread_rmse_crps_<mode>.npz`` files that
``compute_spread_rmse_crps.py`` writes, and produces a single combined
plot that stacks the spread/skill calibration ratio on top of the
signed ensemble-mean error on top of the spread / per-member RMSE / CRPS
curves, with start/end/both overlaid as different curves on every panel.

Outputs (in --output-dir):
    spread_rmse_crps.png      - 3-row figure: row 1 is the spread/skill
                                calibration ratio (one line per mode),
                                with a horizontal target line drawn at
                                sqrt(N/(N+1)) following Fortin et al.
                                (MWR 2014); row 2 is the signed
                                ensemble-mean error (one line per mode,
                                symmetric linear y-axis around zero);
                                row 3 is a 3-panel
                                spread / per-member RMSE / CRPS plot
                                (one line per mode in each panel).

Usage:
    python aggregate_spread_rmse_crps.py \\
        --output-dir /path/to/diagnostics/spread_rmse_crps/<variable>
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent.parent / "_shared"
sys.path.insert(0, str(SHARED_DIR))

from _dispatch_lib import (  # noqa: E402
    MODE_COLORS_TAB,
    MODE_FRAMES,
    MODE_LABELS_SHORT,
)
from axis_limits import apply_limits as apply_axis_limits  # noqa: E402
from local_time_axis import format_local_time_axis  # noqa: E402
from plot_style import (  # noqa: E402
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    SUPTITLE_SIZE,
    TICK_LABEL_SIZE,
    TITLE_SIZE,
    add_subplot_labels,
)
from var_metadata import axis_label, to_display_units  # noqa: E402
from var_metadata import long_name as var_long_name  # noqa: E402

MODE_MARKERS = {"start": "o", "end": "s", "both": "D"}


def _load_mode(output_dir: Path, mode: str, suffix: str = "") -> dict | None:
    path = output_dir / f"spread_rmse_crps_{mode}{suffix}.npz"
    if not path.exists():
        print(f"WARNING: missing {path}, skipping mode '{mode}'")
        return None
    return dict(np.load(path, allow_pickle=True))


def _domain_label(diag: dict) -> str:
    """Human-readable domain for titles.

    Falls back to "global" for npz written before ``--region`` existed, so
    the pre-region figures keep rendering with the label they had.  A mask
    is named too: a land-masked average over a box is a different number
    from an unmasked one, and the figure has to say which it is.
    """
    name = diag.get("region_name")
    label = str(_scalar(name)) if name is not None else "global"
    mask = diag.get("mask_kind")
    mask_kind = str(_scalar(mask)) if mask is not None else "none"
    if mask_kind != "none":
        label += f", {mask_kind}-masked"
    return label


def _scalar(arr: np.ndarray) -> Any:
    return arr.item() if isinstance(arr, np.ndarray) and arr.shape == () else arr


def _plot_spread_rmse_crps_panels(
    per_mode: dict[str, dict],
    axes: Sequence[Axes],
) -> None:
    """Populate three axes with spread, per-member RMSE, and CRPS curves."""
    any_mode = next(iter(per_mode.values()))
    var_name = str(_scalar(any_mode["var_name"]))
    start_time_iso = str(_scalar(any_mode["start_time"]))
    timezone_name = str(_scalar(any_mode["timezone"]))

    for ax_idx, (ax, (key, label, stat)) in enumerate(
        zip(
            axes,
            [
                ("spread", "Ensemble spread", "spread"),
                # Suffix gets a leading newline: "RMSE (per member)"
                # plus the variable long name + units overflows the
                # panel height as a single rotated line, so the
                # qualifier+units drop to a second line. Matches the
                # treatment of the row-2 signed-error label.
                ("member_rmse", "RMSE (per member)", "\nRMSE (per member)"),
                ("crps", "CRPS", "CRPS"),
            ],
        )
    ):
        all_y: list[np.ndarray] = []
        all_x: list[np.ndarray] = []
        for mode, diag in per_mode.items():
            hours = np.asarray(diag["lead_hours"])
            # spread / member_rmse / crps are all field-unit quantities;
            # convert to display units (geopotential m^2/s^2 -> m) to match
            # the var_label axis label.
            y = to_display_units(var_name, np.asarray(diag[key]))
            color = MODE_COLORS_TAB.get(mode, "gray")
            marker = MODE_MARKERS.get(mode, "o")
            ax.plot(
                hours,
                y,
                marker=marker,
                markersize=3,
                color=color,
                label=f"{MODE_LABELS_SHORT.get(mode, mode)} conditioning",
            )
            cond = MODE_FRAMES.get(mode, [])
            cond_valid = [f for f in cond if 0 <= f < len(hours)]
            if cond_valid:
                ax.plot(
                    hours[cond_valid],
                    y[cond_valid],
                    marker=marker,
                    markersize=8,
                    color=color,
                    linestyle="none",
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    zorder=5,
                )
            all_y.append(y)
            all_x.append(hours)

        ax.set_title(label, fontsize=TITLE_SIZE)
        format_local_time_axis(ax, all_x[0], start_time_iso, timezone_name)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
        ax.set_ylabel(axis_label(var_name, stat), fontsize=AXIS_LABEL_SIZE)
        apply_axis_limits(
            ax,
            "spread_rmse_crps",
            var_name,
            key,
            y_data=np.concatenate(all_y),
            x_data=all_x[0],
            log_tag="spread_rmse_crps",
        )
        ax.grid(True, alpha=0.3)
        # The calibration panel's legend already maps each color to its
        # conditioning mode; show the conditioning legend on just one
        # bottom panel to avoid repeating it three times.
        if ax_idx == 0:
            ax.legend(fontsize=LEGEND_SIZE, loc="best")


def _plot_calibration_panel(
    per_mode: dict[str, dict],
    ax: Axes,
) -> None:
    """Populate one axis with the spread/skill calibration ratio.

    Denominator is the RMSE of the ensemble mean (the ``skill`` field),
    which is the literature-standard calibration test (Fortin et al.,
    MWR 2014).  For a reliable ensemble of N members iid with truth the
    expected ratio is ``sqrt(N/(N+1))``, tending to 1 for large N.
    """
    any_mode = next(iter(per_mode.values()))
    start_time_iso = str(_scalar(any_mode["start_time"]))
    timezone_name = str(_scalar(any_mode["timezone"]))

    # All modes share the same ensemble size in this pipeline; use any
    # mode's N to draw the finite-N target.  At N=1000 the correction
    # from 1 is ~5e-4 and visually invisible, but we draw the exact
    # derived value so the math is reproducible.
    N_target = int(_scalar(any_mode["n_members"]))
    target_ratio = float(np.sqrt(N_target / (N_target + 1)))

    hours_any: np.ndarray | None = None
    all_ratios: list[np.ndarray] = []
    for mode, diag in per_mode.items():
        hours = np.asarray(diag["lead_hours"])
        spread = np.asarray(diag["spread"])
        skill = np.asarray(diag["skill"])
        ratio = spread / np.where(skill == 0, np.nan, skill)
        color = MODE_COLORS_TAB.get(mode, "gray")
        marker = MODE_MARKERS.get(mode, "o")
        N = int(_scalar(diag["n_members"]))
        ax.plot(
            hours,
            ratio,
            marker=marker,
            color=color,
            linewidth=1.5,
            label=f"{MODE_LABELS_SHORT.get(mode, mode)} (N={N})",
        )
        cond = MODE_FRAMES.get(mode, [])
        cond_valid = [f for f in cond if 0 <= f < len(hours)]
        if cond_valid:
            ax.plot(
                hours[cond_valid],
                ratio[cond_valid],
                marker=marker,
                markersize=8,
                color=color,
                linestyle="none",
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=5,
            )
        all_ratios.append(ratio)
        if hours_any is None:
            hours_any = hours

    ax.axhline(
        target_ratio,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label="calibration target",
    )
    ax.set_title("spread / RMSE (ens. mean) calibration", fontsize=TITLE_SIZE)
    format_local_time_axis(ax, hours_any, start_time_iso, timezone_name)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    ax.set_ylabel("spread / RMSE (ens. mean)", fontsize=AXIS_LABEL_SIZE)
    # Dimensionless ratio bounded below by zero.  Floor the upper bound
    # at 1.1 so the calibration target (~1 for large N) sits visibly
    # inside the plot rather than being clipped at the top edge, and
    # auto-expand if any mode actually goes overdispersive (ratio > 1).
    ratios_finite = np.concatenate(
        [r[np.isfinite(r)] for r in all_ratios] if all_ratios else [np.array([])]
    )
    ymax = 1.1
    if ratios_finite.size:
        ymax = max(ymax, 1.05 * float(ratios_finite.max()))
    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE, loc="best")


def _plot_mean_error_panel(
    per_mode: dict[str, dict],
    ax: Axes,
) -> None:
    r"""Populate one axis with the signed pixel-mean ensemble-mean error.

    The plotted quantity is
    :math:`\langle\overline{x}_\mathrm{ens}-x_\mathrm{ERA5}\rangle_\mathrm{pix}`:
    members are averaged into the ensemble mean, the per-pixel error
    against ERA5 is averaged over every HEALPix pixel (so the reduction
    is global, equal-area), and the sign is preserved.  Positive values
    mean the ensemble mean exceeds ERA5 on average over the domain.

    The y-axis is symmetric-linear around zero so equal hot and cold
    deviations read with the same eye-level.
    """
    any_mode = next(iter(per_mode.values()))
    var_name = str(_scalar(any_mode["var_name"]))
    start_time_iso = str(_scalar(any_mode["start_time"]))
    timezone_name = str(_scalar(any_mode["timezone"]))

    hours_any: np.ndarray | None = None
    all_y: list[np.ndarray] = []
    for mode, diag in per_mode.items():
        hours = np.asarray(diag["lead_hours"])
        me_key = "mean_error" if "mean_error" in diag else "bias"
        mean_error = to_display_units(var_name, np.asarray(diag[me_key]))
        color = MODE_COLORS_TAB.get(mode, "gray")
        marker = MODE_MARKERS.get(mode, "o")
        ax.plot(
            hours,
            mean_error,
            marker=marker,
            markersize=3,
            color=color,
            linewidth=1.5,
            label=f"{MODE_LABELS_SHORT.get(mode, mode)} conditioning",
        )
        cond = MODE_FRAMES.get(mode, [])
        cond_valid = [f for f in cond if 0 <= f < len(hours)]
        if cond_valid:
            ax.plot(
                hours[cond_valid],
                mean_error[cond_valid],
                marker=marker,
                markersize=8,
                color=color,
                linestyle="none",
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=5,
            )
        all_y.append(mean_error)
        if hours_any is None:
            hours_any = hours

    # Reference line at zero (no error).
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)

    ax.set_title(
        f"Signed ensemble-mean error ({_domain_label(any_mode)})",
        fontsize=TITLE_SIZE,
    )
    format_local_time_axis(ax, hours_any, start_time_iso, timezone_name)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    ax.set_ylabel(axis_label(var_name, "\nmean error"), fontsize=AXIS_LABEL_SIZE)

    # Symmetric linear y-axis around zero so equal hot/cold deviations
    # read at the same eye-level.  Compute the span from the actual data
    # and add a 5% margin; matplotlib autoscale would give an asymmetric
    # window when error has a consistent sign.
    y_all = np.concatenate(all_y)
    y_finite = y_all[np.isfinite(y_all)]
    if y_finite.size:
        y_abs = float(np.max(np.abs(y_finite)))
        if y_abs > 0:
            ax.set_ylim(-1.05 * y_abs, 1.05 * y_abs)

    ax.grid(True, alpha=0.3)
    # No legend: the calibration panel above already maps each color to
    # its conditioning mode.


def make_combined_plot(
    per_mode: dict[str, dict],
    out_path: Path,
) -> None:
    """Combined figure: calibration, mean error, and spread / member-RMSE / CRPS.

    Rows 1 and 2 (calibration ratio and signed ensemble-mean error) each
    span the full figure width so that the local-time axis lines up
    cleanly with the 3-panel metric row at the bottom.  Case name and
    variable are pulled into a two-row suptitle so individual subplot
    titles can show only the per-panel metric.
    """
    any_mode = next(iter(per_mode.values()))
    case_name = str(_scalar(any_mode["case_name"]))
    pretty_var = var_long_name(str(_scalar(any_mode["var_name"])))
    domain = _domain_label(any_mode)
    # Name the domain in the suptitle whenever it is not the whole globe:
    # a regional figure that does not say so invites the reader to compare
    # its spread ratios against the paper's global-sounding numbers.
    if domain != "global":
        region = np.asarray(any_mode.get("region", []), dtype=float)
        if region.size == 4:
            lon0, lon1, lat0, lat1 = region
            pretty_var += (
                f"\n{domain} domain: "
                f"lon [{lon0:g}, {lon1:g}], lat [{lat0:g}, {lat1:g}]"
            )
        else:
            pretty_var += f"\n{domain} domain"

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.25, top=0.92)
    ax_calib = fig.add_subplot(gs[0, :])
    ax_me = fig.add_subplot(gs[1, :])
    axes_metrics = [fig.add_subplot(gs[2, c]) for c in range(3)]
    _plot_calibration_panel(per_mode, ax_calib)
    _plot_mean_error_panel(per_mode, ax_me)
    _plot_spread_rmse_crps_panels(per_mode, axes_metrics)
    labelled_axes = [ax_calib, ax_me, *axes_metrics]

    add_subplot_labels(labelled_axes)
    suptitle = fig.suptitle(
        f"{case_name}\n{pretty_var}",
        fontsize=SUPTITLE_SIZE,
        y=0.985,
    )
    # The suptitle is anchored at its top and grows downward, so adding the
    # domain line takes a regional figure from two lines to three and pushes
    # the last line straight through the calibration panel's own title.  The
    # fixed ``top=0.92`` above was tuned for the two-line global case.
    #
    # Measure what was actually rendered rather than subtracting a constant
    # per line: the line count depends on whether the domain is global and on
    # whether the region box is present, and the line height depends on
    # SUPTITLE_SIZE, none of which this function should have to predict.  The
    # pad clears the top panel's TITLE_SIZE title; ``min`` keeps the two-line
    # case exactly where it was, so the global figures do not move.
    fig.canvas.draw()
    suptitle_bottom = (
        suptitle.get_window_extent().transformed(fig.transFigure.inverted()).y0
    )
    gs.update(top=min(0.92, suptitle_bottom - 0.035))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing spread_rmse_crps_{start,end,both}.npz; "
        "the aggregated PNG is written here too.",
    )
    parser.add_argument(
        "--region-name",
        type=str,
        default=None,
        help="Aggregate the regional run written with this --region-name "
        "(reads spread_rmse_crps_<mode>_<name>.npz and writes "
        "spread_rmse_crps_<name>.png). Default aggregates the global run.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default="none",
        choices=["none", "land", "sea"],
        help="Aggregate the masked run written with this --mask (default: "
        "none, i.e. the unmasked run). Must match what "
        "compute_spread_rmse_crps.py was given, since the mask is part of "
        "the npz filename.",
    )
    args = parser.parse_args()

    suffix = f"_{args.region_name}" if args.region_name else ""
    if args.mask != "none":
        suffix += f"_{args.mask}"

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"ERROR: --output-dir does not exist: {output_dir}")

    per_mode: dict[str, dict] = {}
    for mode in ("end", "start", "both"):
        d = _load_mode(output_dir, mode, suffix)
        if d is not None:
            per_mode[mode] = d

    if not per_mode:
        raise SystemExit(f"ERROR: no per-mode npz files found in {output_dir}")

    print(f"[aggregate_spread_rmse_crps] modes: {list(per_mode)}")
    combined_path = output_dir / f"spread_rmse_crps{suffix}.png"
    make_combined_plot(per_mode, combined_path)
    print(f"[aggregate_spread_rmse_crps] wrote {combined_path}")


if __name__ == "__main__":
    main()
