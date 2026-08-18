"""Combine per-case TC-track parquet outputs into a single multi-row figure.

Reads the per-mode ``tc_tracks_<mode>.parquet`` files written by
``plot_ensemble_tc_tracks.py`` for each case and renders an ``N x 3``
grid where each row corresponds to a case and each column corresponds
to a conditioning mode (end / start / both, matching
``aggregate_tc_tracks.py``).  Row labels show the case display name,
column titles appear only on the top row, and subplot labels (``a.)``,
``b.)``, ...) are placed inside the top-left of each panel.

Each row uses its own case's ``regions.track`` lat/lon extent so the
individual maps remain visually identical to the corresponding panels
in the per-case ``tc_tracks.png``.

Usage:
    python combine_tc_tracks.py \\
        --case sandy --case ian \\
        --tracker minmsl \\
        --output-path /path/to/combined/tc_tracks_sandy_ian.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "_shared"))

from _dispatch_lib import (  # noqa: E402
    load_case,
    require_region,
    tc_tracks_parquet_path,
    wrap_lon_0_360,
)
from plot_style import (  # noqa: E402
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    SUPTITLE_SIZE,
    TITLE_SIZE,
    add_subplot_labels,
)
from tc_tracks_io import load_tracks_parquet  # noqa: E402

# Mode ordering (left -> right) and human labels are taken straight from
# ``aggregate_tc_tracks.py`` so the per-row layout matches the per-case
# ``tc_tracks.png`` it summarizes.
MODE_ORDER: tuple[str, ...] = ("end", "start", "both")
MODE_LABELS: dict[str, str] = {
    "start": "Start Conditioning",
    "end": "End Conditioning",
    "both": "Both-End Conditioning",
}

MEMBER_TRACK_CMAP = plt.cm.jet
MIN_PATH_LENGTH = 3


def _load_case_tracks(
    case: dict[str, Any], tracker: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray | None]]:
    """Load each mode's ensemble + ERA5 tracks for a case."""
    diag_root = f"{case['base']}/diagnostics"
    mode_tracks: dict[str, np.ndarray] = {}
    mode_era5: dict[str, np.ndarray | None] = {}
    for mode in MODE_ORDER:
        parquet = Path(tc_tracks_parquet_path(diag_root, mode, tracker))
        if not parquet.exists():
            print(f"WARNING: missing {parquet}; skipping ({case['name']}, {mode})")
            continue
        tracks = load_tracks_parquet(str(parquet))
        ens = tracks.get("ensemble")
        if ens is None:
            print(f"WARNING: no ensemble rows in {parquet}; skipping")
            continue
        mode_tracks[mode] = ens
        mode_era5[mode] = tracks.get("era5")
    return mode_tracks, mode_era5


def _plot_panel(
    ax: plt.Axes,
    *,
    ensemble_tracks: np.ndarray,
    era5_tracks: np.ndarray | None,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> int:
    """Render one ``(case, mode)`` panel onto ``ax`` and return # member tracks plotted."""
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="#555555")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.4, color="#888888")
    ax.add_feature(cfeature.STATES, linewidth=0.2, alpha=0.3, color="#aaaaaa")
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.4,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 11}
    gl.ylabel_style = {"size": 11}
    ax.set_extent(
        [wrap_lon_0_360(lon_min), wrap_lon_0_360(lon_max), lat_min, lat_max],
        crs=ccrs.PlateCarree(),
    )

    n_members = ensemble_tracks.shape[0]
    n_plotted = 0
    for path_id in range(ensemble_tracks.shape[1]):
        for member_idx in range(n_members):
            tc_lats = ensemble_tracks[member_idx, path_id, :, 0]
            tc_lons = ensemble_tracks[member_idx, path_id, :, 1]
            valid = ~np.isnan(tc_lats) & ~np.isnan(tc_lons)
            if valid.sum() < MIN_PATH_LENGTH:
                continue
            color = MEMBER_TRACK_CMAP(member_idx / max(1, n_members - 1))
            ax.plot(
                tc_lons[valid],
                tc_lats[valid],
                color=color,
                linewidth=0.8,
                alpha=0.3,
                transform=ccrs.PlateCarree(),
                zorder=2,
            )
            ax.plot(
                tc_lons[valid][-1],
                tc_lats[valid][-1],
                "o",
                color=color,
                markersize=2,
                alpha=0.4,
                transform=ccrs.PlateCarree(),
                zorder=2,
            )
            n_plotted += 1

    if era5_tracks is not None:
        for path_id in range(era5_tracks.shape[1]):
            era5_lats = era5_tracks[0, path_id, :, 0]
            era5_lons = era5_tracks[0, path_id, :, 1]
            valid_era5 = ~np.isnan(era5_lats) & ~np.isnan(era5_lons)
            if valid_era5.sum() < MIN_PATH_LENGTH:
                continue
            label = "ERA5" if path_id == 0 else None
            ax.plot(
                era5_lons[valid_era5],
                era5_lats[valid_era5],
                color="black",
                linewidth=2.0,
                linestyle="--",
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                zorder=5,
                label=label,
            )
            ax.plot(
                era5_lons[valid_era5],
                era5_lats[valid_era5],
                "D",
                color="black",
                markersize=4,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )

    return n_plotted


def plot_combined(
    cases: list[dict[str, Any]],
    case_data: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray | None]]],
    *,
    out_path: Path,
    title: str | None,
) -> None:
    """Render the ``len(cases) x len(MODE_ORDER)`` combined figure."""
    n_rows = len(cases)
    n_cols = len(MODE_ORDER)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 6 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    # axes is 1-D when n_rows == 1; force 2-D for uniform indexing.
    axes = np.atleast_2d(axes)

    for row, (case, (mode_tracks, mode_era5)) in enumerate(zip(cases, case_data)):
        lon_min, lon_max, lat_min, lat_max = require_region(case, "track")

        for col, mode in enumerate(MODE_ORDER):
            ax = axes[row, col]
            ens = mode_tracks.get(mode)
            if ens is None:
                # Keep the panel for layout consistency but mark it empty
                # so the grid remains visually aligned.
                ax.set_facecolor("#f5f5f5")
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="#888888",
                )
                ax.set_extent(
                    [
                        wrap_lon_0_360(lon_min),
                        wrap_lon_0_360(lon_max),
                        lat_min,
                        lat_max,
                    ],
                    crs=ccrs.PlateCarree(),
                )
                continue

            n_plotted = _plot_panel(
                ax,
                ensemble_tracks=ens,
                era5_tracks=mode_era5.get(mode),
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
            )
            print(f"  [{case['name']}/{mode}] plotted {n_plotted} member tracks")

            # Column titles on the top row only; the case row label
            # carries the case identity, so repeating mode labels on
            # every row would just add visual noise.
            if row == 0:
                ax.set_title(
                    MODE_LABELS.get(mode, mode.title()),
                    fontsize=TITLE_SIZE,
                    fontweight="bold",
                )

            # Legend on the top-left panel only — every panel shares the
            # same ERA5 line style, so one entry tells the whole story.
            if row == 0 and col == 0:
                ax.legend(loc="lower left", fontsize=LEGEND_SIZE, framealpha=0.8)

    # Row labels: rotated text in axes-fraction coords on the leftmost
    # panel of each row.  Negative x places the label well outside the
    # panel spine so it clears cartopy's gridline labels; bbox_inches
    # ``tight`` (set at savefig time) captures the off-axes text.
    for row, case in enumerate(cases):
        axes[row, 0].text(
            -0.18,
            0.5,
            case["display_name"],
            rotation=90,
            transform=axes[row, 0].transAxes,
            ha="center",
            va="center",
            fontsize=AXIS_LABEL_SIZE + 3,
            fontweight="bold",
        )

    add_subplot_labels(axes, placement="inside")

    if title:
        fig.suptitle(title, fontsize=SUPTITLE_SIZE, y=1.02)

    os.makedirs(out_path.parent, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Case name (e.g. sandy).  Repeatable; one row per case.",
    )
    parser.add_argument("--tracker", default="minmsl")
    parser.add_argument(
        "--output-path",
        required=True,
        help="Full path (including filename) for the combined PNG.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure suptitle.  Omit for no suptitle.",
    )
    args = parser.parse_args()

    cases = [load_case(name) for name in args.case]
    print(
        f"[combine_tc_tracks] cases={[c['name'] for c in cases]} tracker={args.tracker}"
    )

    case_data: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray | None]]] = []
    for case in cases:
        mode_tracks, mode_era5 = _load_case_tracks(case, args.tracker)
        if not mode_tracks:
            raise SystemExit(
                f"ERROR: no parquet files found for case '{case['name']}' "
                f"under {case['base']}/diagnostics/tc_tracks/{args.tracker}/"
            )
        print(
            f"  [{case['name']}] loaded modes: {list(mode_tracks)} "
            f"(ensemble shapes: "
            f"{[ {m: t.shape} for m, t in mode_tracks.items() ]})"
        )
        case_data.append((mode_tracks, mode_era5))

    plot_combined(
        cases,
        case_data,
        out_path=Path(args.output_path),
        title=args.title,
    )


if __name__ == "__main__":
    main()
