"""Dispatch the end-conditioned landfall-split TC diagnostic (Hurricane Ian).

This is a single-case, ``end``-only companion to ``dispatch_tc_tracks.py``.
It reads the ``end``-mode ``tc_tracks_end.parquet`` already produced by a
prior ``dispatch_tc_tracks.py`` run and submits one short aggregator job
that renders ``tc_landfall_split.png`` next to the case's ``tc_tracks.png``.

The figure splits the end-conditioned members by where they made landfall
— Florida vs Florida-bypass-then-South-Carolina — to show how much
stronger an impact the free-IC backward trajectories drive through
Florida even though every member is anchored to the same weak SC landfall.
See ``submission_scripts/tc_landfall_split.py`` for the figure details.

The two landfall classification boxes come from the case YAML's
``regions.<fl-region>`` / ``regions.<sc-region>`` (default ``landfall_fl``
/ ``landfall_sc``); the track-map extent + domain restriction come from
``regions.track``.  Because the boxes are Ian-specific physics, the
diagnostic is Ian-defaulted but stays case-driven so a future two-landfall
case can reuse it by declaring the same regions.

With ``--animate-residual`` it also submits an ``animate_ensemble.py`` job
that renders the MSL field of the inspect-set members — bypass members with
no SC-box fix plus dropped (no-coherent-track) members, i.e. the ones that
fall outside the clean FL/SC story — one MP4 each, to
``<base>/diagnostics/tc_tracks/<tracker>/residual_msl_anim/``.

Output:
    <base>/diagnostics/tc_tracks/<tracker>/tc_landfall_split.png
    <base>/diagnostics/tc_tracks/<tracker>/residual_msl_anim/  (--animate-residual)

Usage:
    python dispatch_tc_landfall_split.py                 # defaults to ian
    python dispatch_tc_landfall_split.py ian --dry-run
    python dispatch_tc_landfall_split.py ian --tracker tempest
    python dispatch_tc_landfall_split.py ian --animate-residual
    python dispatch_tc_landfall_split.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "_shared"))

from _dispatch_lib import (  # noqa: E402
    build_qsub_cmd,
    case_ensemble_size,
    ensemble_zarr_path,
    list_cases,
    load_case,
    require_region,
    resolve_landfall_frame,
    resolve_timezone,
    start_time_iso_from_case,
    submit_qsub,
    tc_tracks_dir,
    tc_tracks_parquet_path,
    trackers_from_case,
)

SPLIT_SCRIPT = str(SCRIPT_DIR / "submission_scripts" / "submit_tc_landfall_split.sh")
ANIM_SCRIPT = str(SCRIPT_DIR / "submission_scripts" / "submit_animate_ensemble.sh")
SPLIT_LOG_DIR = str(SCRIPT_DIR / "submission_scripts" / "tc_tracks_logs")

DEFAULT_CASE = "ian"
H_RT = "0:30:00"
ANIM_H_RT = "2:00:00"

# Great-circle anchor radius (km); matches tc_landfall_split.py's
# --anchor-radius-km default (track_qc.ANCHOR_RADIUS_KM).  Kept as a local
# literal so this dispatcher stays free of the numpy-backed track_qc import on
# its default (non-animating) path.
DEFAULT_ANCHOR_RADIUS_KM = 500.0

# Default MSL color range (Pa) for the residual animation; matches the
# msl defaults used elsewhere in the ensemble pipeline.
ANIM_MSL_VMIN = 96000.0
ANIM_MSL_VMAX = 102000.0


def _resolve_tracker(case: dict, override: str | None) -> str:
    """Pick a single configured tracker for the case (first one by default)."""
    trackers = trackers_from_case(case, override=[override] if override else None)
    return trackers[0]


def _submit_inspect_animation(
    case: dict,
    *,
    parquet: Path,
    tracker_dir: str,
    track_box: tuple[float, float, float, float],
    landfall_frame: int,
    anchor_radius_km: float,
    ensemble_size: int,
    vmin: float,
    vmax: float,
    dry_run: bool,
) -> str | None:
    """Animate the MSL field of the inspect-set members.

    The inspect set is the unified-QC *excluded* members (no_track /
    unanchored / insufficient_fixes) -- the ones outside the clean FL/SC
    story.  Resolves the (already-existing) end-mode parquet with
    ``track_qc``, then submits an ``animate_ensemble.py`` job that writes one
    ``msl`` MP4 per member to ``<tracker_dir>/residual_msl_anim``.  Animating
    the field (not the track) is the point: several excluded members have
    little or no usable track, so the field shows whether a storm is present
    that the tracker missed.  Returns the SGE job id, or ``None`` when there
    is nothing to animate.  The numpy/pandas-backed resolution is imported
    lazily so the default (non-animating) dispatch path stays free of those
    imports.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    import numpy as np  # noqa: E402

    from track_qc import compute_track_qc  # noqa: E402

    qc = compute_track_qc(
        str(parquet),
        track_box=track_box,
        anchor_frame=landfall_frame,
        free_end_frame=0,
        anchor_radius_km=anchor_radius_km,
    )
    members = np.flatnonzero(~qc.analyzable)
    if members.size == 0:
        print(
            "[landfall_split] inspect set is empty (every member is "
            "analyzable); skipping animation."
        )
        return None

    ens_zarr = ensemble_zarr_path(case["base"], "end", ensemble_size)
    if not Path(ens_zarr).exists():
        print(
            f"[landfall_split] WARNING: ensemble zarr not found ({ens_zarr}); "
            "cannot animate inspect-set members."
        )
        return None

    anim_out = f"{tracker_dir}/residual_msl_anim"
    lon_min, lon_max, lat_min, lat_max = track_box
    anim_args = [
        "--ensemble-zarr",
        ens_zarr,
        "--variable",
        "msl",
        "--members",
        *[str(m) for m in members.tolist()],
        "--extent",
        str(lon_min),
        str(lon_max),
        str(lat_min),
        str(lat_max),
        "--vmin",
        str(vmin),
        "--vmax",
        str(vmax),
        "--output-dir",
        anim_out,
    ]
    print(
        f"[landfall_split] animating {members.size} inspect-set member(s) "
        f"{members.tolist()}\n  -> {anim_out}/ensemble_msl_member_<m>.mp4"
    )
    cmd = build_qsub_cmd(
        script=ANIM_SCRIPT,
        job_name=f"AnimInspect_{case['name']}",
        h_rt=ANIM_H_RT,
        script_args=anim_args,
    )
    return submit_qsub(
        cmd,
        dry_run=dry_run,
        cwd=str(SCRIPT_DIR / "submission_scripts"),
        log_dir=SPLIT_LOG_DIR,
    )


def dispatch(
    case_name: str,
    *,
    tracker_override: str | None = None,
    fl_region: str = "landfall_fl",
    sc_region: str = "landfall_sc",
    fl_label: str = "Florida",
    sc_label: str = "South Carolina",
    output_root_override: str | None = None,
    animate_residual: bool = False,
    anim_vmin: float = ANIM_MSL_VMIN,
    anim_vmax: float = ANIM_MSL_VMAX,
    dry_run: bool = False,
) -> str | None:
    """Submit the landfall-split figure job (+ optional residual animation).

    Returns the figure job's SGE job id.  When ``animate_residual`` is set,
    also submits an independent ``animate_ensemble.py`` job that renders the
    MSL field of the bypass members with no SC-box fix.
    """
    case = load_case(case_name)
    tracker = _resolve_tracker(case, tracker_override)

    track_box = require_region(case, "track")
    fl_box = require_region(case, fl_region)
    sc_box = require_region(case, sc_region)

    # The end-conditioned split resolves each member's storm anchor-first, so
    # it needs the pinned-end (landfall) frame the ensemble was conditioned on.
    landfall_frame = resolve_landfall_frame(case)
    if landfall_frame < 0:
        raise SystemExit(
            f"ERROR: case '{case_name}' has no tc_tracks.landfall_frame; the "
            "end-conditioned landfall split needs a pinned-end anchor frame."
        )

    output_root = output_root_override or f"{case['base']}/diagnostics"
    tracker_dir = tc_tracks_dir(output_root, tracker)

    # Friendly pre-flight: the end-mode parquet must already exist.
    parquet = Path(tc_tracks_parquet_path(output_root, "end", tracker))
    if not parquet.exists():
        raise SystemExit(
            f"ERROR: {parquet} not found.  Run dispatch_tc_tracks.py "
            f"{case_name} --mode end first (the landfall-split figure reads "
            "the end-mode tracks it produces)."
        )

    lon_min, lon_max, lat_min, lat_max = track_box
    fl_lon_min, fl_lon_max, fl_lat_min, fl_lat_max = fl_box
    sc_lon_min, sc_lon_max, sc_lat_min, sc_lat_max = sc_box

    script_args = [
        "--output-dir",
        tracker_dir,
        "--lon-min",
        str(lon_min),
        "--lon-max",
        str(lon_max),
        "--lat-min",
        str(lat_min),
        "--lat-max",
        str(lat_max),
        "--fl-lon-min",
        str(fl_lon_min),
        "--fl-lon-max",
        str(fl_lon_max),
        "--fl-lat-min",
        str(fl_lat_min),
        "--fl-lat-max",
        str(fl_lat_max),
        "--sc-lon-min",
        str(sc_lon_min),
        "--sc-lon-max",
        str(sc_lon_max),
        "--sc-lat-min",
        str(sc_lat_min),
        "--sc-lat-max",
        str(sc_lat_max),
        "--fl-label",
        fl_label,
        "--sc-label",
        sc_label,
        "--landfall-frame",
        str(landfall_frame),
        "--title",
        f"{case['display_name']} — End-Conditioned Landfall Split",
        "--start-time",
        start_time_iso_from_case(case),
        "--timezone",
        resolve_timezone(case),
    ]

    print(
        f"[landfall_split] case={case_name} tracker={tracker}\n"
        f"  -> {tracker_dir}/tc_landfall_split.png"
    )

    cmd = build_qsub_cmd(
        script=SPLIT_SCRIPT,
        job_name=f"TcLandfallSplit_{case_name}",
        h_rt=H_RT,
        script_args=script_args,
    )
    fig_jid = submit_qsub(
        cmd,
        dry_run=dry_run,
        cwd=str(SCRIPT_DIR / "submission_scripts"),
        log_dir=SPLIT_LOG_DIR,
    )

    if animate_residual:
        _submit_inspect_animation(
            case,
            parquet=parquet,
            tracker_dir=tracker_dir,
            track_box=track_box,
            landfall_frame=landfall_frame,
            anchor_radius_km=DEFAULT_ANCHOR_RADIUS_KM,
            ensemble_size=case_ensemble_size(case),
            vmin=anim_vmin,
            vmax=anim_vmax,
            dry_run=dry_run,
        )

    return fig_jid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch the end-conditioned landfall-split TC diagnostic.",
    )
    parser.add_argument(
        "case",
        nargs="?",
        default=DEFAULT_CASE,
        help=f"Case name (default: {DEFAULT_CASE}).",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available cases and exit."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--tracker",
        default=None,
        choices=["wuduan", "minmsl", "tempest"],
        help="Tracker to read (default: the first tracker configured in the "
        "case YAML's tc_tracks.trackers list).",
    )
    parser.add_argument(
        "--fl-region",
        default="landfall_fl",
        help="regions.<name> for the Florida landfall box (default: landfall_fl).",
    )
    parser.add_argument(
        "--sc-region",
        default="landfall_sc",
        help="regions.<name> for the SC landfall box (default: landfall_sc).",
    )
    parser.add_argument("--fl-label", default="Florida")
    parser.add_argument("--sc-label", default="South Carolina")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root output directory (default: <base>/diagnostics).",
    )
    parser.add_argument(
        "--animate-residual",
        action="store_true",
        help="Also submit an animate_ensemble.py job that renders the MSL "
        "field of the inspect-set members — bypass members with no SC-box fix "
        "plus dropped (no-coherent-track) members — one MP4 each, to "
        "<tracker_dir>/residual_msl_anim.",
    )
    parser.add_argument(
        "--anim-vmin",
        type=float,
        default=ANIM_MSL_VMIN,
        help=f"MSL colorbar min (Pa) for --animate-residual (default {ANIM_MSL_VMIN}).",
    )
    parser.add_argument(
        "--anim-vmax",
        type=float,
        default=ANIM_MSL_VMAX,
        help=f"MSL colorbar max (Pa) for --animate-residual (default {ANIM_MSL_VMAX}).",
    )
    args = parser.parse_args()

    if args.list:
        print("Available cases:")
        for n in list_cases():
            print(f"  {n}")
        return

    dispatch(
        args.case,
        tracker_override=args.tracker,
        fl_region=args.fl_region,
        sc_region=args.sc_region,
        fl_label=args.fl_label,
        sc_label=args.sc_label,
        output_root_override=args.output_root,
        animate_residual=args.animate_residual,
        anim_vmin=args.anim_vmin,
        anim_vmax=args.anim_vmax,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
