"""Compute zonal power spectra for one mode's ensemble.

The ensemble is stored on the native HEALPix grid (cBottle's training
representation).  Zonal spectra require a regular longitude axis, so each
batch is regridded HPX -> lat/lon at compute time.  ERA5 is provided on
lat/lon and round-tripped (lat/lon -> HPX -> lat/lon) before its zonal
spectrum is computed; this enforces the same effective HPX resolution on
both fields, which is the fairness fix for the ensemble-vs-ERA5
resolution mismatch that the model itself was trained against.

This script writes the numerics for ONE conditioning mode. The aggregated
plot across all three modes is produced separately by
``aggregate_power_spectra.py`` after all three per-mode jobs finish.

Outputs (in --output-dir):
    power_spectra_<mode>.npz
        zonal_wavenumbers        : (n_kz,) int
        zonal_spectrum_ens       : (n_leads, n_kz) float32 - ensemble-mean
        zonal_spectrum_era5_rt   : (n_leads, n_kz) float32 - ERA5 round-tripped
        zonal_spectrum_era5_raw  : (n_leads, n_kz) float32 - ERA5 on lat/lon
        zonal_member_spectra     : (N, n_leads, n_kz) float32
        lead_hours               : (n_leads,) float64
        case_name                : () str
        var_name                 : () str
        mode                     : () str
        start_time               : () str
        timezone                 : () str

Usage:
    python compute_power_spectra.py \\
        --ensemble-zarr /path/to/ensemble_f11_n1000.zarr \\
        --era5-zarr     /path/to/era5_reference.zarr \\
        --output-dir    /path/to/diagnostics/power_spectra/<variable> \\
        --mode          end \\
        --var-name      msl
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent.parent / "_shared"
REPO_DIR = SCRIPT_DIR.parent.parent.parent
for _p in (str(SHARED_DIR), str(REPO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _dispatch_lib import lead_hours  # noqa: E402

from custom_utils.diagnostics import zonal_spectrum  # noqa: E402
from custom_utils.interpolation_utils import CBottleUtils  # noqa: E402


def _round_trip_era5(
    era5: torch.Tensor,
    lead_times: np.ndarray,
    var_name: str,
    lat: np.ndarray,
    lon: np.ndarray,
    hpx_level: int,
) -> tuple[torch.Tensor, OrderedDict]:
    """Project ERA5 onto the HPX-resolvable subspace, then back to lat/lon."""
    era5_4d = era5.unsqueeze(0).unsqueeze(2)  # [1, n_leads, 1, n_lat, n_lon]
    coords = OrderedDict(
        {
            "time": np.array([0]),
            "lead_time": lead_times,
            "variable": np.array([var_name]),
            "lat": lat,
            "lon": lon,
        }
    )
    hpx_data, hpx_coords = CBottleUtils.regrid_latlon_to_healpix_nest(
        era5_4d,
        coords,
        hpx_level=hpx_level,
    )
    rt_data, rt_coords = CBottleUtils.regrid_healpix_nest_to_latlon(
        hpx_data,
        hpx_coords,
        hpx_level=hpx_level,
    )
    rt_era5 = rt_data[0, :, 0, :, :]  # [n_leads, n_lat, n_lon]
    rt_era5_coords = OrderedDict(
        {
            "lead_time": rt_coords["lead_time"],
            "lat": rt_coords["lat"],
            "lon": rt_coords["lon"],
        }
    )
    return rt_era5, rt_era5_coords


def compute_zonal_spectra_batched(
    ens_root: zarr.Group,
    var_idx: int,
    n_members: int,
    batch_size: int,
    hpx_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-member zonal spectra in batches."""
    n_leads = ens_root["data"].shape[2]
    data = ens_root["data"]
    lead_times = np.array(ens_root["lead_time"][:])

    zonal_spectra: np.ndarray | None = None
    zonal_wn: np.ndarray | None = None

    t_start = time.time()
    for batch_start in range(0, n_members, batch_size):
        batch_end = min(batch_start + batch_size, n_members)
        bs = batch_end - batch_start

        ens_batch = torch.from_numpy(
            np.array(data[batch_start:batch_end, 0, :, var_idx, :])
        ).float()

        ens_4d = ens_batch.unsqueeze(2)
        hpx_coords = OrderedDict(
            {
                "time": np.arange(bs),
                "lead_time": lead_times,
                "variable": np.array(["_"]),
                "hpx": np.arange(ens_batch.shape[-1]),
            }
        )
        latlon_data, latlon_coords = CBottleUtils.regrid_healpix_nest_to_latlon(
            ens_4d,
            hpx_coords,
            hpx_level=hpx_level,
        )
        ens_latlon = latlon_data[:, :, 0, :, :].cpu()
        del ens_batch, ens_4d, latlon_data

        flat = ens_latlon.reshape(
            bs * n_leads, ens_latlon.shape[-2], ens_latlon.shape[-1]
        )
        flat_coords = OrderedDict(
            {
                "batch": np.arange(bs * n_leads),
                "lat": latlon_coords["lat"],
                "lon": latlon_coords["lon"],
            }
        )
        wn_b, flat_zonal, _ = zonal_spectrum(flat, flat_coords)
        batch_zon = flat_zonal.reshape(bs, n_leads, -1).cpu().numpy().astype(np.float32)
        del flat, flat_zonal, ens_latlon

        if zonal_spectra is None:
            zonal_wn = wn_b
            n_kz = batch_zon.shape[-1]
            zonal_spectra = np.empty((n_members, n_leads, n_kz), dtype=np.float32)

        zonal_spectra[batch_start:batch_end] = batch_zon

        elapsed = time.time() - t_start
        done = batch_end
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (n_members - done) / rate if rate > 0 else float("nan")
        print(
            f"  [{done:4d}/{n_members}]  batch={bs:3d}  "
            f"elapsed={elapsed:6.1f}s  rate={rate:5.2f} mem/s  eta={eta:6.1f}s",
            flush=True,
        )

    if zonal_spectra is None or zonal_wn is None:
        raise RuntimeError("No batches were processed; cannot return spectra.")
    return zonal_wn, zonal_spectra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ensemble-zarr", required=True)
    parser.add_argument("--era5-zarr", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["start", "end", "both"],
        help="Conditioning mode for this ensemble; encoded in the output filename.",
    )
    parser.add_argument(
        "--var-name", default="msl", help="Variable to analyse (default: msl)."
    )
    parser.add_argument("--case-name", type=str, default="Hurricane Sandy")
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Ensemble IC as an ISO 8601 UTC string (forwarded to the aggregator).",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="IANA timezone for panel-title timestamps (forwarded to the aggregator).",
    )
    parser.add_argument(
        "--n-members",
        type=int,
        default=None,
        help="Number of members to process (default: all in the zarr).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Members per batch (memory/throughput knob).",
    )
    parser.add_argument(
        "--k-range",
        type=str,
        default=None,
        help="Wavenumber range for spectral slope fits (forwarded by the "
        "dispatcher; currently unused by this script).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[spec] case        : {args.case_name}")
    print(f"[spec] mode        : {args.mode}")
    print(f"[spec] ensemble    : {args.ensemble_zarr}")
    print(f"[spec] era5        : {args.era5_zarr}")
    print(f"[spec] output dir  : {output_dir}")
    print(f"[spec] variable    : {args.var_name}")

    ens_root = zarr.open_group(args.ensemble_zarr, mode="r")
    era5_root = zarr.open_group(args.era5_zarr, mode="r")

    if "hpx" not in ens_root.array_keys():
        raise SystemExit(
            f"ERROR: ensemble zarr is not on HEALPix (no 'hpx' coord). "
            f"Available arrays: {sorted(ens_root.array_keys())}"
        )
    hpx_level = ens_root.attrs.get("hpx_level")
    if hpx_level is None:
        n_pix_arr = int(ens_root["hpx"].shape[0])
        hpx_level = int(np.log2(n_pix_arr // 12) / 2)

    ens_vars = list(np.array(ens_root["variable"][:]))
    era5_vars = list(np.array(era5_root["variable"][:]))
    if args.var_name not in ens_vars:
        raise SystemExit(
            f"ERROR: '{args.var_name}' not in ensemble zarr. Available: {ens_vars}"
        )
    if args.var_name not in era5_vars:
        raise SystemExit(
            f"ERROR: '{args.var_name}' not in ERA5 zarr. Available: {era5_vars}"
        )

    ens_var_idx = ens_vars.index(args.var_name)
    era5_var_idx = era5_vars.index(args.var_name)

    era5_lat = np.array(era5_root["lat"][:])
    era5_lon = np.array(era5_root["lon"][:])
    lead_times = np.array(ens_root["lead_time"][:])
    hours = lead_hours(lead_times)
    n_leads = len(lead_times)

    total_members = int(ens_root["data"].shape[0])
    n_members = (
        total_members if args.n_members is None else min(args.n_members, total_members)
    )
    print(f"[spec] hpx level   : {hpx_level} ({12 * 4**hpx_level} pixels)")
    print(f"[spec] members     : {n_members} / {total_members} in zarr")
    print(f"[spec] batch size  : {args.batch_size}")
    print(f"[spec] n_leads     : {n_leads}")

    print(
        "[spec] loading ERA5 and computing zonal spectra (raw + HPX round-trip)...",
        flush=True,
    )
    t0 = time.time()
    era5 = torch.from_numpy(
        np.array(era5_root["data"][:, 0, era5_var_idx, :, :])
    ).float()
    raw_era5_coords = OrderedDict(
        {
            "lead_time": lead_times,
            "lat": era5_lat,
            "lon": era5_lon,
        }
    )
    zonal_wn, zonal_era5_raw_t, _ = zonal_spectrum(era5, raw_era5_coords)
    zonal_era5_raw = zonal_era5_raw_t.cpu().numpy().astype(np.float32)

    rt_era5, rt_era5_coords = _round_trip_era5(
        era5,
        lead_times,
        args.var_name,
        era5_lat,
        era5_lon,
        hpx_level,
    )
    _, zonal_era5_rt_t, _ = zonal_spectrum(rt_era5, rt_era5_coords)
    zonal_era5_rt = zonal_era5_rt_t.cpu().numpy().astype(np.float32)
    del era5, rt_era5
    print(f"[spec] ERA5 done in {time.time() - t0:.1f}s")

    print(
        "[spec] computing ensemble zonal spectra (batched HPX -> lat/lon)...",
        flush=True,
    )
    t0 = time.time()
    zonal_wn, zonal_member_spectra = compute_zonal_spectra_batched(
        ens_root,
        var_idx=ens_var_idx,
        n_members=n_members,
        batch_size=args.batch_size,
        hpx_level=hpx_level,
    )
    print(f"[spec] ensemble done in {time.time() - t0:.1f}s")

    zonal_spectrum_ens = zonal_member_spectra.mean(axis=0).astype(np.float32)

    npz_path = output_dir / f"power_spectra_{args.mode}.npz"
    np.savez(
        npz_path,
        zonal_wavenumbers=zonal_wn,
        zonal_spectrum_ens=zonal_spectrum_ens,
        zonal_spectrum_era5_rt=zonal_era5_rt,
        zonal_spectrum_era5_raw=zonal_era5_raw,
        zonal_member_spectra=zonal_member_spectra,
        lead_hours=np.asarray(hours, dtype=np.float64),
        case_name=np.array(args.case_name),
        var_name=np.array(args.var_name),
        mode=np.array(args.mode),
        start_time=np.array(args.start_time or "1970-01-01T00:00:00Z"),
        timezone=np.array(args.timezone),
    )
    print(f"[spec] wrote {npz_path}")
    print("[spec] done.")


if __name__ == "__main__":
    main()
