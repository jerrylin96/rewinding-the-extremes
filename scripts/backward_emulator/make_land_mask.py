"""Produce the shared ERA5 land mask on the 721x1440 lat/lon grid.

The comparison masks PNW ``t2m`` to land (ocean t2m is essentially the prescribed,
member-identical SST and barely moves, so it dilutes the heatwave signal).  Both
models are scored on the same 721x1440 grid, so ONE mask array masks identical
pixels for makani (native) and cBottle (regridded), which is the point.

Land convention -- **verbatim from the paper**: ERA5's SST is NaN over land in the
reference store, so ``isnan(sst)`` identifies land.  ``custom_utils.diagnostics.
hpx_land_mask_from_sst_hpx`` does exactly this in HPX space; here we do it on the
native lat/lon grid (before any HPX round-trip) because the comparison lives in
lat/lon.  We read the SST frame straight from the case's ``era5_reference.zarr``
(the same store cBottle's truth comes from), so the land/sea boundary matches the
reference both models are compared against.

Only ``zarr`` + ``numpy`` are needed (no earth2grid / torch): reading one SST frame
and testing ``isnan`` is cheap, so this runs interactively on a login node in
seconds -- it is the quick check that unblocks the ``t2m`` land jobs.

Usage (in e2s-custom):
    python make_land_mask.py --case pnw_heatwave --out ./out/pnw/land_mask_pnw_heatwave_721x1440.npy
    python make_land_mask.py --case pnw_heatwave --inspect      # print, don't write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

SHARED = Path(__file__).resolve().parents[1] / "_shared"
sys.path.insert(0, str(SHARED))

import _dispatch_lib as dl  # noqa: E402

TARGET_GRID = (721, 1440)  # the grid makani is native on and cBottle regrids to


def _decode(raw) -> List[str]:
    return [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in raw]


def land_from_sst(sst_latlon: np.ndarray) -> np.ndarray:
    """Boolean land mask from an SST field: True where SST is missing (land).

    The paper's exact convention (``isnan(sst)``), factored out so it is unit
    tested without a zarr store.  A pixel is land iff SST is non-finite there.
    """
    return ~np.isfinite(np.asarray(sst_latlon))


def read_sst_latlon(era5_zarr: str, *, sst_channel: str = "sst") -> np.ndarray:
    """Read the first SST frame ``(H, W)`` from an ERA5 reference zarr group."""
    import zarr  # heavy-env only

    era5 = zarr.open_group(era5_zarr, mode="r")
    channels = _decode(np.array(era5["variable"][:]))
    if sst_channel not in channels:
        raise SystemExit(
            f"'{sst_channel}' not in {era5_zarr} (have {channels}); the land mask "
            f"is derived from SST (NaN over land). Pass --sst-channel if it differs."
        )
    si = channels.index(sst_channel)
    # era5['data'] is [time, lead, variable, lat, lon]; take frame 0 (land is static).
    return np.array(era5["data"][0, 0, si, :, :], dtype=np.float32)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", required=True, help="cBottle case YAML name (e.g. pnw_heatwave).")
    p.add_argument("--mode", default="end", help="Conditioning mode whose era5_reference to read (default end).")
    p.add_argument("--sst-channel", default="sst")
    p.add_argument("--out", default=None,
                   help="Output .npy (default ./land_mask_<case>_<H>x<W>.npy).")
    p.add_argument("--inspect", action="store_true",
                   help="Print grid + land fraction and exit without writing.")
    args = p.parse_args(argv)

    case = dl.load_case(args.case)
    era5_zarr = dl.era5_zarr_path(case["base"], args.mode)
    sst = read_sst_latlon(era5_zarr, sst_channel=args.sst_channel)
    land = land_from_sst(sst)
    H, W = land.shape
    frac = float(land.mean())

    print(f"[land_mask] case={args.case} mode={args.mode}")
    print(f"[land_mask]   era5={era5_zarr}")
    print(f"[land_mask]   grid=({H}, {W})  land fraction={frac:.3f} "
          f"({int(land.sum())}/{land.size} px)")
    if (H, W) != TARGET_GRID:
        print(f"[land_mask]   WARNING: grid ({H}, {W}) != target {TARGET_GRID}; "
              f"the comparison driver expects the 721x1440 grid. Check the "
              f"era5_reference resolution before using this mask.", file=sys.stderr)
    if not (0.05 < frac < 0.6):
        print(f"[land_mask]   WARNING: land fraction {frac:.3f} is outside the "
              f"~0.1-0.4 expected for a global grid; verify the SST/NaN convention.",
              file=sys.stderr)

    if args.inspect:
        print("[land_mask]   --inspect: not writing.")
        return

    out = Path(args.out) if args.out else Path(
        f"./land_mask_{args.case}_{H}x{W}.npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, land)
    print(f"[land_mask] wrote {out}")


if __name__ == "__main__":
    main()
