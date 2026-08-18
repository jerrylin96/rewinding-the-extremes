#!/usr/bin/env python3
"""Plot globally-averaged diagnostics with individual ensemble members visible.

For each diagnostic variable, produces a single plot with all three conditioning
modes (start, end, both) overlaid.  Individual ensemble members are drawn as
thin semi-transparent lines; the ensemble mean is a thick opaque line; ERA5 is
a dashed black line.  Vertical markers indicate which frames are conditioned
for each mode.

Diagnostic variables
--------------------
* **tcwv** — total column water vapour (moisture budget proxy)
* **z500** — 500 hPa geopotential height (mid-troposphere circulation)
* **tke**  — column-averaged kinetic energy, derived as
  ``0.5 * (u² + v²)`` averaged across three wind levels (10 m, 850 hPa,
  500 hPa).  This captures bulk wind energy across the troposphere.

Alternative / additional variables worth considering:
  msl  — mean sea level pressure (large-scale mass field)
  t850 — 850 hPa temperature (lower-troposphere thermodynamics)

All global means are latitude-weighted (cosine weighting).

Output
------
Per-variable PNGs::

    <output_dir>/tcwv_ensemble_members.png
    <output_dir>/z500_ensemble_members.png
    <output_dir>/tke_ensemble_members.png

Combined stacked figure::

    <output_dir>/ensemble_members_combined.png

Usage
-----
Event mode (reads case YAML for paths)::

    python ensemble_member_diagnostics.py --event katrina

CLI mode (explicit paths)::

    python ensemble_member_diagnostics.py \\
        --base-dir /path/to/case/root \\
        --ensemble-size 64

Options::

    --n-members N       Limit the number of members plotted (default: all)
    --variables tcwv,z500,tke   Override the variable list
    --modes start,end,both      Override conditioning modes
    --output-dir /path          Override output directory
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "_shared"))
sys.path.insert(0, SHARED_DIR)
sys.path.insert(0, SCRIPT_DIR)

from _dispatch_lib import (  # noqa: E402
    MODE_COLORS_TAB as MODE_COLORS,
    MODE_FRAMES,
    MODE_LABELS_SHORT as MODE_LABELS,
    case_ensemble_size,
    ensemble_zarr_path,
    era5_zarr_path,
    iter_modes,
    lead_hours,
    list_cases,
    load_case,
    parse_mode_list,
)
from custom_utils.ensemble_loading import (  # noqa: E402
    load_var_ensemble,
    load_var_era5,
)
from plot_style import (  # noqa: E402
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    SUPTITLE_SIZE,
    TICK_LABEL_SIZE,
    TITLE_SIZE,
    add_subplot_labels,
)

# ── Constants ─────────────────────────────────────────────────────────────

# Conditioning frame positions (hours) for vertical line markers.
# frame index * 6h gives offset from t=0.
FRAME_DT_HOURS = 6

# Default diagnostic variables.
DEFAULT_VARIABLES = ["tcwv", "z500", "tke"]

# Wind components needed to derive TKE.
TKE_COMPONENTS = {
    "10m": ("u10m", "v10m"),
    "850": ("u850", "v850"),
    "500": ("u500", "v500"),
}

# Variable display metadata.
VAR_UNITS = {
    "tcwv": "kg m$^{-2}$",
    "z500": "m$^2$ s$^{-2}$",
    "tke": "m$^2$ s$^{-2}$",
    "msl": "Pa",
    "t850": "K",
}

VAR_LONG_NAMES = {
    "tcwv": "Total Column Water Vapour",
    "z500": "500 hPa Geopotential Height",
    "tke": "Column-Averaged Kinetic Energy",
    "msl": "Mean Sea Level Pressure",
    "t850": "850 hPa Temperature",
}


# ── Data loading ──────────────────────────────────────────────────────────

def _cos_lat_weights(lat: np.ndarray) -> torch.Tensor:
    """Cosine-latitude weights for global averaging."""
    return torch.tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)


def _global_mean(data: torch.Tensor, cos_lat: torch.Tensor) -> torch.Tensor:
    """Latitude-weighted global mean over the last two (lat, lon) axes.

    Parameters
    ----------
    data : torch.Tensor
        Shape ``[..., n_lat, n_lon]``.
    cos_lat : torch.Tensor
        Shape ``[n_lat]``.

    Returns
    -------
    torch.Tensor
        Shape ``[...]`` — the spatial dimensions are collapsed.
    """
    # Weight by cos(lat), sum over lat, then mean over lon.
    weighted = (data * cos_lat[..., :, None]).sum(dim=-2) / cos_lat.sum()
    return weighted.mean(dim=-1)


def load_global_mean_series(
    ens_zarr_path: str,
    era5_zarr_path_str: str,
    var_name: str,
    n_members: Optional[int] = None,
) -> dict:
    """Load globally-averaged time series for a single stored variable.

    Returns
    -------
    dict with keys:
        hours        : np.ndarray [n_leads]
        ens_members  : np.ndarray [N, n_leads]   — per-member global means
        ens_mean     : np.ndarray [n_leads]       — ensemble mean
        era5         : np.ndarray [n_leads]       — ERA5 global mean
        cond_frames  : list[int] | None
        n_members    : int
    """
    ens_root = zarr.open_group(ens_zarr_path, mode="r")

    lat = np.array(ens_root["lat"][:])
    cos_lat = _cos_lat_weights(lat)
    lead_times = np.array(ens_root["lead_time"][:])
    hours = lead_hours(lead_times)
    cond_frames = ens_root.attrs.get("conditioning_frames", None)

    # Ensemble: [N, n_leads, n_lat, n_lon]
    ens_data = load_var_ensemble(ens_zarr_path, var_name, n_members)
    # ERA5: [n_leads, n_lat, n_lon]
    era5_data = load_var_era5(era5_zarr_path_str, var_name)

    # Global means
    ens_gm = _global_mean(ens_data, cos_lat)   # [N, n_leads]
    era5_gm = _global_mean(era5_data, cos_lat)  # [n_leads]

    return {
        "hours": hours,
        "ens_members": ens_gm.numpy(),
        "ens_mean": ens_gm.mean(dim=0).numpy(),
        "era5": era5_gm.numpy(),
        "cond_frames": cond_frames,
        "n_members": int(ens_gm.shape[0]),
    }


def load_tke_global_mean_series(
    ens_zarr_path: str,
    era5_zarr_path_str: str,
    n_members: Optional[int] = None,
) -> dict:
    """Derive column-averaged TKE and return globally-averaged time series.

    TKE = 0.5 * (u² + v²) averaged over three wind levels (10m, 850, 500).
    Each level contributes equally (simple arithmetic mean of the three KE
    values).

    Returns the same dict structure as :func:`load_global_mean_series`.
    """
    ens_root = zarr.open_group(ens_zarr_path, mode="r")

    lat = np.array(ens_root["lat"][:])
    cos_lat = _cos_lat_weights(lat)
    lead_times = np.array(ens_root["lead_time"][:])
    hours = lead_hours(lead_times)
    cond_frames = ens_root.attrs.get("conditioning_frames", None)

    n_levels = len(TKE_COMPONENTS)
    ens_tke_accum = None
    era5_tke_accum = None

    for _level, (u_name, v_name) in TKE_COMPONENTS.items():
        u_ens = load_var_ensemble(ens_zarr_path, u_name, n_members)
        v_ens = load_var_ensemble(ens_zarr_path, v_name, n_members)
        ke_ens = 0.5 * (u_ens ** 2 + v_ens ** 2)  # [N, n_leads, n_lat, n_lon]

        u_era5 = load_var_era5(era5_zarr_path_str, u_name)
        v_era5 = load_var_era5(era5_zarr_path_str, v_name)
        ke_era5 = 0.5 * (u_era5 ** 2 + v_era5 ** 2)  # [n_leads, n_lat, n_lon]

        if ens_tke_accum is None:
            ens_tke_accum = ke_ens
            era5_tke_accum = ke_era5
        else:
            ens_tke_accum = ens_tke_accum + ke_ens
            era5_tke_accum = era5_tke_accum + ke_era5

        # Free intermediate tensors
        del u_ens, v_ens, ke_ens, u_era5, v_era5, ke_era5

    # Average over levels
    ens_tke = ens_tke_accum / n_levels   # [N, n_leads, n_lat, n_lon]
    era5_tke = era5_tke_accum / n_levels  # [n_leads, n_lat, n_lon]

    # Global means
    ens_gm = _global_mean(ens_tke, cos_lat)   # [N, n_leads]
    era5_gm = _global_mean(era5_tke, cos_lat)  # [n_leads]

    return {
        "hours": hours,
        "ens_members": ens_gm.numpy(),
        "ens_mean": ens_gm.mean(dim=0).numpy(),
        "era5": era5_gm.numpy(),
        "cond_frames": cond_frames,
        "n_members": int(ens_gm.shape[0]),
    }


def load_all_modes(
    base_dir: str,
    modes: List[str],
    ensemble_size: int,
    var_name: str,
    n_members: Optional[int] = None,
) -> Dict[str, dict]:
    """Load globally-averaged series for all conditioning modes.

    Returns {mode: series_dict}.  Skips modes whose zarr stores are missing.
    """
    result = {}
    for mode in modes:
        ens_path = ensemble_zarr_path(base_dir, mode, ensemble_size)
        era5_path = era5_zarr_path(base_dir, mode)

        if not os.path.exists(ens_path):
            print(f"  WARNING: ensemble zarr not found for mode '{mode}': {ens_path}")
            continue
        if not os.path.exists(era5_path):
            print(f"  WARNING: ERA5 zarr not found for mode '{mode}': {era5_path}")
            continue

        print(f"  Loading {var_name} — mode={mode} ...")
        if var_name == "tke":
            series = load_tke_global_mean_series(ens_path, era5_path, n_members)
        else:
            series = load_global_mean_series(ens_path, era5_path, var_name, n_members)

        result[mode] = series
    return result


# ── Plotting ──────────────────────────────────────────────────────────────

def _conditioning_frame_hours(mode: str) -> List[float]:
    """Return the lead-time hours at which a mode is conditioned."""
    return [f * FRAME_DT_HOURS for f in MODE_FRAMES[mode]]


def plot_variable(
    mode_series: Dict[str, dict],
    var_name: str,
    output_dir: str,
    *,
    member_alpha: float = 0.15,
    member_lw: float = 0.5,
) -> str:
    """Plot a single variable with all modes overlaid.

    Individual members are thin semi-transparent lines, ensemble means are
    thick opaque lines, ERA5 is a dashed black line.  Conditioning frames
    are shown as vertical dotted lines.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # ERA5 is the same underlying truth for all modes — plot once from the
    # first available mode.
    first_mode = next(iter(mode_series))
    first = mode_series[first_mode]
    ax.plot(
        first["hours"], first["era5"],
        color="black", linewidth=2.0, linestyle="--", label="ERA5", zorder=10,
    )

    for mode, series in mode_series.items():
        color = MODE_COLORS.get(mode, "gray")
        label = MODE_LABELS.get(mode, mode)
        hours = series["hours"]
        members = series["ens_members"]  # [N, n_leads]
        mean = series["ens_mean"]

        # Individual members
        for i in range(members.shape[0]):
            ax.plot(
                hours, members[i],
                color=color, alpha=member_alpha, linewidth=member_lw,
                zorder=2,
            )

        # Ensemble mean
        ax.plot(
            hours, mean,
            color=color, linewidth=2.5, label=f"{label} mean",
            zorder=5,
        )

        # Conditioning frame vertical markers
        cond_hours = _conditioning_frame_hours(mode)
        for ch in cond_hours:
            ax.axvline(
                ch, color=color, linestyle=":", linewidth=1.0,
                alpha=0.6, zorder=1,
            )

    long_name = VAR_LONG_NAMES.get(var_name, var_name)
    units = VAR_UNITS.get(var_name, "")
    ylabel = f"{long_name}" + (f" [{units}]" if units else "")

    ax.set_xlabel("Lead time (h)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_title(
        f"Global-Mean {long_name} — Ensemble Members by Conditioning Mode",
        fontsize=TITLE_SIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{var_name}_ensemble_members.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_combined(
    all_var_series: Dict[str, Dict[str, dict]],
    variables: List[str],
    output_dir: str,
    *,
    member_alpha: float = 0.15,
    member_lw: float = 0.5,
) -> str:
    """Stacked multi-panel figure with one row per variable."""
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 4.5 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    for ax, var_name in zip(axes, variables):
        mode_series = all_var_series[var_name]

        # ERA5 from first available mode
        first_mode = next(iter(mode_series))
        first = mode_series[first_mode]
        ax.plot(
            first["hours"], first["era5"],
            color="black", linewidth=2.0, linestyle="--", label="ERA5", zorder=10,
        )

        for mode, series in mode_series.items():
            color = MODE_COLORS.get(mode, "gray")
            label = MODE_LABELS.get(mode, mode)
            hours = series["hours"]
            members = series["ens_members"]
            mean = series["ens_mean"]

            for i in range(members.shape[0]):
                ax.plot(
                    hours, members[i],
                    color=color, alpha=member_alpha, linewidth=member_lw,
                    zorder=2,
                )

            ax.plot(
                hours, mean,
                color=color, linewidth=2.5, label=f"{label} mean",
                zorder=5,
            )

            cond_hours = _conditioning_frame_hours(mode)
            for ch in cond_hours:
                ax.axvline(
                    ch, color=color, linestyle=":", linewidth=1.0,
                    alpha=0.6, zorder=1,
                )

        long_name = VAR_LONG_NAMES.get(var_name, var_name)
        units = VAR_UNITS.get(var_name, "")
        ylabel = f"{long_name}" + (f"\n[{units}]" if units else "")
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
        ax.set_title(f"Global-Mean {long_name}", fontsize=TITLE_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        ax.grid(True, alpha=0.3)

    # Legend on top panel only
    axes[0].legend(fontsize=LEGEND_SIZE, loc="best")
    axes[-1].set_xlabel("Lead time (h)", fontsize=AXIS_LABEL_SIZE)
    add_subplot_labels(axes)
    fig.suptitle(
        "Ensemble Members by Conditioning Mode — Global Diagnostics",
        fontsize=SUPTITLE_SIZE, y=1.0,
    )
    fig.tight_layout()

    out_path = os.path.join(output_dir, "ensemble_members_combined.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot globally-averaged ensemble member diagnostics "
                    "across conditioning modes.",
    )

    # Event mode
    parser.add_argument(
        "--event", type=str, default=None,
        help="Case name (e.g. 'katrina'). Reads cases/<event>.yaml.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available cases and exit.",
    )

    # CLI mode
    parser.add_argument(
        "--base-dir", type=str, default=None,
        help="Root directory containing {start,end,both}/ sub-dirs with zarrs.",
    )
    parser.add_argument(
        "--ensemble-size", type=int, default=None,
        help="Ensemble size (required in CLI mode, optional override in event mode).",
    )

    # Shared options
    parser.add_argument(
        "--variables", type=str, default=None,
        help="Comma-separated list of variables (default: tcwv,z500,tke).",
    )
    parser.add_argument(
        "--modes", type=str, default=None,
        help="Comma-separated conditioning modes (default: start,end,both).",
    )
    parser.add_argument(
        "--n-members", type=int, default=None,
        help="Limit the number of ensemble members plotted (default: all).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: <base>/diagnostics/_ensemble_members).",
    )
    parser.add_argument(
        "--member-alpha", type=float, default=0.15,
        help="Alpha (opacity) for individual member lines (default: 0.15).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("Available cases:")
        for n in list_cases():
            print(f"  {n}")
        return

    # ── Resolve paths and parameters ─────────────────────────────────
    if args.event:
        case = load_case(args.event)
        base_dir = case["base"]
        ensemble_size = case_ensemble_size(case, override=args.ensemble_size)
        modes = parse_mode_list(args.modes, [m for m, _ in iter_modes(case)])
    elif args.base_dir:
        base_dir = args.base_dir
        if args.ensemble_size is None:
            print("ERROR: --ensemble-size is required in CLI mode.",
                  file=sys.stderr)
            sys.exit(1)
        ensemble_size = args.ensemble_size
        modes = parse_mode_list(args.modes, ["start", "end", "both"])
    else:
        print("ERROR: specify --event or --base-dir.", file=sys.stderr)
        sys.exit(1)

    variables = (
        [v.strip() for v in args.variables.split(",")]
        if args.variables
        else DEFAULT_VARIABLES
    )

    output_dir = args.output_dir or os.path.join(
        base_dir, "diagnostics", "_ensemble_members"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Ensemble Member Diagnostics")
    print("=" * 70)
    print(f"Base dir       : {base_dir}")
    print(f"Ensemble size  : {ensemble_size}")
    print(f"Modes          : {modes}")
    print(f"Variables      : {variables}")
    print(f"N members plot : {args.n_members or 'all'}")
    print(f"Output dir     : {output_dir}")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────
    all_var_series: Dict[str, Dict[str, dict]] = {}
    plotted_vars: List[str] = []

    for var_name in variables:
        print(f"\n[{var_name}]")
        mode_series = load_all_modes(
            base_dir, modes, ensemble_size, var_name, args.n_members,
        )
        if not mode_series:
            print(f"  No data loaded for '{var_name}', skipping.")
            continue
        all_var_series[var_name] = mode_series
        plotted_vars.append(var_name)

    if not plotted_vars:
        print("\nERROR: no data loaded for any variable.", file=sys.stderr)
        sys.exit(1)

    # ── Generate per-variable plots ──────────────────────────────────
    print("\nGenerating plots ...")
    for var_name in plotted_vars:
        path = plot_variable(
            all_var_series[var_name], var_name, output_dir,
            member_alpha=args.member_alpha,
        )
        print(f"  {path}")

    # ── Combined stacked plot ────────────────────────────────────────
    if len(plotted_vars) > 1:
        path = plot_combined(
            all_var_series, plotted_vars, output_dir,
            member_alpha=args.member_alpha,
        )
        print(f"  {path}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
