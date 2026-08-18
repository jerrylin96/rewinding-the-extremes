"""Aggregate zonal-power-spectrum plots across conditioning modes.

Reads the three per-mode ``power_spectra_<mode>.npz`` files and produces
a single zonal-spectrum figure with one ensemble-mean curve per mode
overlaid on a shared ERA5 reference.

Outputs (in --output-dir):
    power_spectra_zonal.png   - 4 panels (lead times), one ensemble curve per mode

Usage:
    python aggregate_power_spectra.py \\
        --output-dir /path/to/diagnostics/power_spectra/<variable>
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

from _dispatch_lib import MODE_COLORS_TAB, MODE_LABELS_SHORT  # noqa: E402
from local_time_axis import format_local_time_title  # noqa: E402
from plot_style import (  # noqa: E402
    AXIS_LABEL_SIZE,
    LEGEND_SIZE_DENSE,
    SUPTITLE_SIZE,
    TICK_LABEL_SIZE,
    TITLE_SIZE,
    add_subplot_labels,
)
from var_metadata import is_geopotential, to_display_units  # noqa: E402
from var_metadata import long_name as var_long_name  # noqa: E402
from var_metadata import units_tex as var_units  # noqa: E402


def _load_mode(output_dir: Path, mode: str) -> dict | None:
    path = output_dir / f"power_spectra_{mode}.npz"
    if not path.exists():
        print(f"WARNING: missing {path}, skipping mode '{mode}'")
        return None
    diag = dict(np.load(path, allow_pickle=True))
    _to_display_units(diag)
    return diag


def _scalar(arr) -> object:
    return arr.item() if isinstance(arr, np.ndarray) and arr.shape == () else arr


# Spectral-density keys: zonal power ~ field^2, so for geopotential the
# m^2/s^2 -> m display conversion enters squared (power=2).  Converted once at
# load so the ERA5 and ensemble curves and the y-axis label stay consistent.
_SPECTRUM_KEYS = (
    "zonal_spectrum_era5_rt",
    "zonal_spectrum_era5_raw",
    "zonal_spectrum_ens",
)


def _to_display_units(diag: dict) -> None:
    """In place, convert this mode's spectral arrays to display units."""
    var_name = str(_scalar(diag["var_name"]))
    if not is_geopotential(var_name):
        return
    for key in _SPECTRUM_KEYS:
        if key in diag:
            diag[key] = to_display_units(var_name, np.asarray(diag[key]), power=2)


def plot_zonal_spectra(per_mode: dict[str, dict], out_path: Path) -> None:
    any_mode = next(iter(per_mode.values()))
    var_name = str(_scalar(any_mode["var_name"]))
    case_name = str(_scalar(any_mode["case_name"]))
    start_time_iso = str(_scalar(any_mode["start_time"]))
    timezone_name = str(_scalar(any_mode["timezone"]))
    hours = np.asarray(any_mode["lead_hours"])
    wn_z = np.asarray(any_mode["zonal_wavenumbers"])
    era5_rt = np.asarray(any_mode["zonal_spectrum_era5_rt"])
    era5_raw = np.asarray(any_mode["zonal_spectrum_era5_raw"])

    n_leads = len(hours)
    idxs = np.linspace(0, n_leads - 1, min(4, n_leads), dtype=int)
    valid = wn_z >= 1
    k_z = wn_z[valid]
    pretty_var = var_long_name(var_name)
    unit_str = var_units(var_name)
    power_units = f"m $\\cdot$ ({unit_str})$^2$" if unit_str else r"m $\cdot$ field$^2$"

    fig, axes = plt.subplots(
        1, len(idxs), figsize=(5.0 * len(idxs), 4.5), squeeze=False
    )
    for ax, li in zip(axes[0], idxs):
        # ERA5 lines are shared across modes; draw them once.
        ax.loglog(
            k_z,
            era5_rt[li, valid],
            label="ERA5 (HPX round-trip)",
            color="black",
            linewidth=1.2,
        )
        ax.loglog(
            k_z,
            era5_raw[li, valid],
            label="ERA5 (raw)",
            color="tab:gray",
            linestyle="--",
            linewidth=0.8,
        )
        for mode, diag in per_mode.items():
            ens = np.asarray(diag["zonal_spectrum_ens"])
            ax.loglog(
                k_z,
                ens[li, valid],
                label=f"Ensemble mean ({MODE_LABELS_SHORT.get(mode, mode)})",
                color=MODE_COLORS_TAB.get(mode, "gray"),
                linewidth=1.1,
            )
        ax.set_title(
            format_local_time_title(start_time_iso, float(hours[li]), timezone_name),
            fontsize=TITLE_SIZE,
        )
        ax.set_xlabel("Zonal wavenumber", fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel(f"Power ({power_units})", fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_SIZE)
        ax.grid(True, which="both", alpha=0.3)
        # 5 entries in a narrow log-log panel — keep the legend tight.
        ax.legend(fontsize=LEGEND_SIZE_DENSE)

    fig.suptitle(
        f"{case_name} - {pretty_var} - zonal power spectrum",
        fontsize=SUPTITLE_SIZE,
        y=1.02,
    )
    add_subplot_labels(axes)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"ERROR: --output-dir does not exist: {output_dir}")

    per_mode: dict[str, dict] = {}
    for mode in ("end", "start", "both"):
        d = _load_mode(output_dir, mode)
        if d is not None:
            per_mode[mode] = d

    if not per_mode:
        raise SystemExit(f"ERROR: no per-mode npz files found in {output_dir}")

    print(f"[aggregate_power_spectra] modes: {list(per_mode)}")
    out_path = output_dir / "power_spectra_zonal.png"
    plot_zonal_spectra(per_mode, out_path)
    print(f"[aggregate_power_spectra] wrote {out_path}")


if __name__ == "__main__":
    main()
