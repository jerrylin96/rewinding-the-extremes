"""TC-track the makani backward ensemble with TempestExtremes (Text S3).

Re-tracks the makani-native ensemble through the paper's EXACT TempestExtremes
pipeline -- `DetectNodes` + `StitchNodes` with the Zarzycki & Ullrich (2017)
warm-core thickness contour and the Mu-Ting orography-aware stitch thresholds --
so the makani tracks are directly comparable to the cBottle tracks the submitted
paper produced.  We reuse `tempest_extremes_runner` verbatim and only supply the
makani side: native 0.25 deg fields (no regrid) streamed member-by-member and the
ERA5 reference track from makani's own bundled truth.

Key points:
* **makani is native lat/lon** (721x1440, lat 90->-90, lon 0->359.75) -- the exact
  WB2-ERA5 grid the tracker's orography fetch expects, so nothing is regridded.
* **Both models are tracked at 0.25 deg** (cBottle regridded, makani native), so the
  *tracking grid* is equal: track POSITION is a clean head-to-head, and central
  pressure (intensity) is where makani's finer native field content should show --
  reported noting cBottle's field was upsampled from HEALPix L6.
* **The pinned present (forward frame -1) is already ERA5 truth** in makani
  (`pred == truth` at the anchor), so no ERA5 frame-replacement is needed.
* **Tracking is global**; all paths are saved. The Ian track-box restriction happens
  at consumption (the synoptic-PCA `--tc-domain`, the aggregator's domain).
* Output is a parquet in the paper's schema (`save_tracks_parquet`), so it feeds the
  **Ian synoptic-PCA impact target** and the TC-track aggregator unchanged.

Heavy env (TempestExtremes binaries on PATH from e2s-custom; earth2studio for the
orography fetch, which hits WB2 -- pre-fetch once on a login node so the compute
job reads it from EARTH2STUDIO_CACHE).

Usage (in e2s-custom, after a login-node orography pre-fetch):
    python tc_tracks_compare.py --makani-dir .../ian --case ian \\
        --out-dir ./out/ian/tc_tracks
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence

import numpy as np

BW_DIR = Path(__file__).resolve().parent
SHARED = BW_DIR.parent / "_shared"
EA_DIR = BW_DIR.parent / "ensemble_analysis"
# Our own dirs go first; the paper's ensemble_analysis dir (for the tempest
# runner / parquet IO) is APPENDED so it can never shadow a same-named module of
# ours (e.g. its own dispatch_tc_tracks.py vs our dispatch_tc_tracks_compare.py).
for _p in (str(BW_DIR), str(SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(EA_DIR) not in sys.path:
    sys.path.append(str(EA_DIR))

import _dispatch_lib as dl  # noqa: E402
from makani_source import MakaniBackwardSource  # noqa: E402


def build_member_slab(fields: dict, variables: Sequence[str]) -> np.ndarray:
    """Stack a per-member ``{name: (T,H,W)}`` dict into a ``[T, V, H, W]`` slab.

    Variable order follows ``variables`` (the TempestExtremes list), which the
    detect/stitch flags reference by name.  Pure numpy -- the light-env-testable
    seam between the makani source and the tracker.
    """
    missing = [v for v in variables if v not in fields]
    if missing:
        raise KeyError(f"member is missing tracker channels {missing}")
    return np.stack([np.asarray(fields[v]) for v in variables], axis=1)


def _tempest_variables(warm_core: bool) -> List[str]:
    from earth2studio.models.dx.tc_tracking import (
        VARIABLES_TEMPEST_BASE,
        VARIABLES_TEMPEST_WARM_CORE_EXTRAS,
    )
    base = list(VARIABLES_TEMPEST_BASE)
    return base + list(VARIABLES_TEMPEST_WARM_CORE_EXTRAS) if warm_core else base


def _track_one(slab, variables, lats, lons, step_h, member_idx, work_dir,
               orography, warm_core):
    from tempest_extremes_runner import track_member_with_tempest
    return track_member_with_tempest(
        slab, variables=variables, lats=lats, lons=lons,
        timestep_hours=step_h, member_idx=member_idx, work_dir=work_dir,
        orography=orography, warm_core=warm_core)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--makani-dir", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-frames", type=int, default=12)
    p.add_argument("--warm-core", dest="warm_core", action="store_true", default=None,
                   help="Z&U 2017 warm-core thickness filter (default: the case "
                   "tc_tracks.warm_core, true for Ian).")
    p.add_argument("--no-warm-core", dest="warm_core", action="store_false")
    p.add_argument("--no-orography", dest="orography", action="store_false", default=True,
                   help="Skip the WB2 surface-elevation (zs<=150m) stitch filter.")
    p.add_argument("--max-members", type=int, default=None,
                   help="Track only the first N members (smoke / debugging).")
    p.add_argument("--work-dir", default=None,
                   help="Scratch dir for staged NetCDFs (default: a temp dir).")
    args = p.parse_args(argv)

    case = dl.load_case(args.case)
    tc = case.get("tc_tracks") or {}
    warm_core = bool(tc.get("warm_core", False)) if args.warm_core is None else args.warm_core

    makani = MakaniBackwardSource(args.makani_dir, n_frames=args.n_frames)
    lats, lons = makani.lat, makani.lon
    step_h = float(abs(makani.step_hours))
    variables = _tempest_variables(warm_core)
    missing = [v for v in variables if v not in makani.channels]
    if missing:
        raise SystemExit(f"makani dumps lack tracker channels {missing}; "
                         f"have {makani.channels}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) if args.work_dir else Path(
        tempfile.mkdtemp(prefix="tc_tracks_makani_", dir=str(out)))
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tc-cmp] {args.case}: tracker=tempest warm_core={warm_core} "
          f"orography={args.orography} vars={variables} step={step_h}h")
    print(f"[tc-cmp] grid {lats.shape[0]}x{lons.shape[0]} lat[{lats[0]:.2f}..{lats[-1]:.2f}] "
          f"lon[{lons[0]:.2f}..{lons[-1]:.2f}]  n_members={makani.n_members}")

    orography = None
    if args.orography:
        from tempest_extremes_runner import fetch_surface_orography
        print("[tc-cmp] fetching WB2 surface orography (once) ...", flush=True)
        orography = fetch_surface_orography(np.asarray(lats), np.asarray(lons))
        print(f"[tc-cmp]   orography ready {orography.shape} "
              f"[{orography.min():.0f}, {orography.max():.0f}] m")

    # --- ensemble: stream members, track each ---------------------------------
    per_member: List[np.ndarray] = []
    limit = args.max_members
    n_done = 0
    for gid, fields in makani.iter_member_fields(variables):
        slab = build_member_slab(fields, variables)  # [T, V, H, W]
        tracks = _track_one(slab, variables, lats, lons, step_h, gid, work_dir,
                            orography, warm_core)      # [P, T, 4]
        per_member.append(tracks)
        n_done += 1
        if n_done % 25 == 0 or (limit and n_done >= limit):
            print(f"[tc-cmp]   tracked {n_done}"
                  f"{'/' + str(limit) if limit else ''} members", flush=True)
        if limit and n_done >= limit:
            break

    from tempest_extremes_runner import pad_and_stack_tracks
    ens_arr = pad_and_stack_tracks(per_member, len(per_member), args.n_frames)  # [M,P,T,4]

    # --- ERA5 reference: makani's own native truth ----------------------------
    print("[tc-cmp] tracking ERA5 reference (makani native truth) ...", flush=True)
    tf = makani.truth_fields(variables)
    truth_slab = build_member_slab(tf, variables)  # [T, V, H, W]
    era5_tracks = _track_one(truth_slab, variables, lats, lons, step_h, -1, work_dir,
                             orography, warm_core)  # [P, T, 4]
    era5_arr = era5_tracks[None]  # [1, P, T, 4]

    from tc_tracks_io import save_tracks_parquet
    parquet = out / f"tc_tracks_makani_{args.case}_end.parquet"
    save_tracks_parquet(str(parquet), ensemble_arr=ens_arr, era5_arr=era5_arr)

    n_tracked = int(np.isfinite(ens_arr[:, :, :, 0]).any(axis=(1, 2)).sum())
    print(f"[tc-cmp] ensemble tracks {ens_arr.shape}  members with >=1 fix: "
          f"{n_tracked}/{ens_arr.shape[0]}")
    print(f"[tc-cmp] ERA5 reference paths: {era5_arr.shape[1]} "
          f"(finite fixes: {int(np.isfinite(era5_arr[..., 0]).sum())})")
    print(f"[tc-cmp] wrote {parquet}")


if __name__ == "__main__":
    main()
