"""Quick interactive smoke test of the cBottle -> lat/lon regrid glue.

``cbottle_source.CBottleLatLonSource`` is the one piece of this comparison that
cannot be exercised in the light test env (it needs earth2grid / torch / zarr), so
it is untested until it runs on the analysis HPC.  Rather than wait ~6 h for a full
compare job to find a wiring bug, run THIS on a login node (or a short interactive
session) with a handful of members: it exercises the exact glue -- zarr path
resolution, the ``hpx`` / ``data`` / ``variable`` array names, the HPX->lat/lon
regrid call, the ERA5 round-trip, and the output shapes -- in seconds.

What it checks:
  * the end-mode ensemble + era5_reference zarr paths resolve and open;
  * ``load_region`` returns ``ens (N, L, P)`` and ``truth (L, P)`` for the region;
  * every returned value is finite and physically plausible for the variable;
  * the pinned frame (forward -1) is finite (the regrid preserved the IC).

Exit status is 0 on PASS, non-zero on the first failed check, so it doubles as a
pre-flight gate before the dispatcher fans out cBottle-overlay jobs.

Usage (in e2s-custom):
    python smoke_cbottle_source.py --case pnw_heatwave --variable z500
    python smoke_cbottle_source.py --case ian --variable msl --n-members 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
SHARED = Path(__file__).resolve().parents[1] / "_shared"
sys.path.insert(0, str(SHARED))

import _dispatch_lib as dl  # noqa: E402
from cbottle_source import CBottleLatLonSource  # noqa: E402
from latlon_metrics import select_pixels  # noqa: E402

# Loose physical sanity ranges (SI, as stored): geopotential m^2/s^2, K, Pa, m/s.
_PLAUSIBLE = {
    "z500": (4.0e4, 6.0e4), "z300": (7.0e4, 1.05e5),
    "t2m": (180.0, 340.0), "msl": (9.0e4, 1.08e5),
    "u500": (-120.0, 120.0), "v500": (-120.0, 120.0),
    "u10m": (-80.0, 80.0), "v10m": (-80.0, 80.0),
}


def _check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"[smoke] FAIL: {msg}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[smoke]   ok: {msg}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", required=True)
    p.add_argument("--variable", default="z500")
    p.add_argument("--region-name", default="synoptic")
    p.add_argument("--n-members", type=int, default=2,
                   help="Members to regrid (small = fast; the glue is the same).")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--size", type=int, default=None,
                   help="cBottle ensemble size for the zarr path (default: case size).")
    args = p.parse_args(argv)

    case = dl.load_case(args.case)
    print(f"[smoke] case={args.case} variable={args.variable} "
          f"region={args.region_name} n_members={args.n_members}")

    src = CBottleLatLonSource(case, mode="end", size=args.size,
                              n_members=args.n_members, batch_size=args.batch_size)
    s = src.summary()
    print(f"[smoke] source summary: {s}")
    _check(args.variable in src.channels,
           f"'{args.variable}' in cBottle channels {src.channels}")
    _check(tuple((int(src.lat.shape[0]), int(src.lon.shape[0]))) == (721, 1440),
           f"grid is 721x1440 (got {src.lat.shape[0]}x{src.lon.shape[0]})")

    box = dl.region_box(case, args.region_name)
    _check(box is not None, f"region '{args.region_name}' resolves from the case YAML")
    flat, _w = select_pixels(src.lat, src.lon, box)
    _check(flat.size > 0, f"region selected {flat.size} pixels")

    print(f"[smoke] regridding {args.n_members} members x {src.n_frames} frames "
          f"HPX(L{s['hpx_level']})->lat/lon and subsetting {flat.size} pixels ...")
    ens, truth = src.load_region(args.variable, flat)
    _check(ens.shape == (src.n_members, src.n_frames, flat.size),
           f"ens shape {ens.shape} == (N, L, P)")
    _check(truth.shape == (src.n_frames, flat.size),
           f"truth shape {truth.shape} == (L, P)")
    _check(np.isfinite(ens).all(), "all ensemble values finite (regrid produced no NaNs)")
    _check(np.isfinite(truth).all(), "all round-tripped-ERA5 truth values finite")
    _check(np.isfinite(ens[:, -1, :]).all(), "pinned frame (forward -1) is finite")

    lo, hi = _PLAUSIBLE.get(args.variable, (-np.inf, np.inf))
    emin, emax = float(ens.min()), float(ens.max())
    tmin, tmax = float(truth.min()), float(truth.max())
    print(f"[smoke]   ens range   [{emin:.4g}, {emax:.4g}]")
    print(f"[smoke]   truth range [{tmin:.4g}, {tmax:.4g}]  (plausible {lo:g}..{hi:g})")
    _check(lo <= emin and emax <= hi,
           f"ensemble within plausible {args.variable} range")
    _check(lo <= tmin and tmax <= hi,
           f"round-tripped ERA5 truth within plausible {args.variable} range")

    print("[smoke] PASS: cBottle regrid glue works end-to-end for this "
          f"({args.case}, {args.variable}, {args.region_name}).")


if __name__ == "__main__":
    main()
